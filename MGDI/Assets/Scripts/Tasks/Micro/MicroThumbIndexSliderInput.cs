using UnityEngine;

public class MicroThumbIndexSliderInput : MonoBehaviour
{
    public enum AxisMode
    {
        X = 0,
        Y = 1,
        Z = 2
    }

    [Header("References")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("Index axis (projection)")]
    [Tooltip("If true, thumb position is projected to index MCP->TIP axis (recommended).")]
    [SerializeField] private bool useProjection = true;

    [Tooltip("If true, capture neutral t at the moment the task starts (Enable).")]
    [SerializeField] private bool captureNeutralOnEnable = true;

    [Tooltip("If true, capture neutral t right after a mode switch (recommended).")]
    [SerializeField] private bool recenterOnModeSwitch = true;

    [Header("Slider shaping")]
    [Tooltip("Dead-zone around neutral (in normalized t units).")]
    [SerializeField] private float deadZoneT = 0.12f;

    [Tooltip("Max range from neutral mapped to full output (in normalized t units).")]
    [SerializeField] private float fullScaleT = 0.40f;

    [Tooltip("Exponent for response curve (>1 makes small moves less sensitive).")]
    [SerializeField] private float gamma = 1.6f;

    [Tooltip("Output smoothing (higher = snappier).")]
    [SerializeField] private float outputLerp = 14f;

    [Tooltip("Clamp per-second change of output to reduce jumps.")]
    [SerializeField] private float maxOutputRatePerSec = 6f;

    [Header("Tap detection (Thumb-Index touch)")]
    [Tooltip("Touch is considered DOWN if distance <= this for debounceSeconds.")]
    [SerializeField] private float touchOnMeters = 0.030f;

    [Tooltip("Touch is considered UP if distance >= this for debounceSeconds.")]
    [SerializeField] private float touchOffMeters = 0.045f;

    [Tooltip("Debounce time for DOWN/UP (seconds).")]
    [SerializeField] private float debounceSeconds = 0.06f;

    [Tooltip("Max time between two taps to count as double tap (seconds).")]
    [SerializeField] private float doubleTapWindowSec = 0.30f;

    [Tooltip("Cooldown after accepting a tap sequence (seconds).")]
    [SerializeField] private float tapCooldownSec = 0.18f;

    [Header("Mode switching rules")]
    [Tooltip("Single tap toggles between X and Y.")]
    [SerializeField] private bool singleTapTogglesXY = true;

    [Tooltip("Double tap toggles Z mode (enter/exit). Exiting returns to X.")]
    [SerializeField] private bool doubleTapTogglesZ = true;

    [Header("Debug")]
    [SerializeField] private bool debug = false;
    [SerializeField] private float debug_t = -1f;
    [SerializeField] private float debug_touchDist = -1f;
    [SerializeField] private string debug_state = "";

    // Outputs
    public AxisMode Mode { get; private set; } = AxisMode.X;

    // Signed output in [-1..1]. Positive means toward TIP direction, negative toward MCP direction.
    public float AxisValue { get; private set; } = 0f;

    public bool SingleTapThisFrame { get; private set; } = false;
    public bool DoubleTapThisFrame { get; private set; } = false;

    // Internal: neutral point in t
    float _tNeutral = 0.5f;
    bool _neutralCaptured = false;

    // Output smoothing state
    float _axisSm = 0f;

    // Tap FSM
    enum TouchFSM { Idle, DownLatched }
    TouchFSM _touchFsm = TouchFSM.Idle;
    float _downHeld = 0f;
    float _upHeld = 0f;

    int _tapCount = 0;
    float _tapWindowUntil = 0f;
    float _tapCooldownUntil = 0f;

    void OnEnable()
    {
        if (captureNeutralOnEnable)
        {
            _neutralCaptured = false; // recapture on first Update
        }
    }

    void ResetFramePulses()
    {
        SingleTapThisFrame = false;
        DoubleTapThisFrame = false;
    }

    void Update()
    {
        ResetFramePulses();

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (!HasRequiredJoints())
        {
            AxisValue = 0f;
            _axisSm = 0f;
            if (debug) debug_state = "Missing joints";
            return;
        }

        // 1) Compute slider position t (0..1)
        float t = ComputeProjectedT();
        debug_t = t;

        if (!_neutralCaptured)
        {
            _tNeutral = t;
            _neutralCaptured = true;
        }

        // 2) Convert t -> signed axis value in [-1..1]
        float rawAxis = TToAxisValue(t, _tNeutral);

        // 3) Smooth + rate limit
        float k = 1f - Mathf.Exp(-Mathf.Max(0.01f, outputLerp) * dt);
        float target = Mathf.Lerp(_axisSm, rawAxis, k);

        // rate limit on output
        if (maxOutputRatePerSec > 0f)
        {
            float maxStep = maxOutputRatePerSec * dt;
            float d = target - _axisSm;
            d = Mathf.Clamp(d, -maxStep, maxStep);
            target = _axisSm + d;
        }

        _axisSm = target;
        AxisValue = _axisSm;

        // 4) Tap detection (thumb-index touch)
        HandleTapFSM(dt);

        // 5) Apply mode rules when tap sequence completes
        if (SingleTapThisFrame)
        {
            if (singleTapTogglesXY)
                ToggleXY();
        }

        if (DoubleTapThisFrame)
        {
            if (doubleTapTogglesZ)
                ToggleZ();
        }
    }

    bool HasRequiredJoints()
    {
        if (remoteHand == null) return false;
        if (remoteHand.remoteByIndex == null) return false;
        if (remoteHand.remoteByIndex.Length < 9) return false;

        // needed: thumb tip 4, index mcp 5, index tip 8
        if (remoteHand.remoteByIndex[4] == null) return false;
        if (remoteHand.remoteByIndex[5] == null) return false;
        if (remoteHand.remoteByIndex[8] == null) return false;

        return true;
    }

    float ComputeProjectedT()
    {
        // thumb tip and index axis
        Vector3 thumb = remoteHand.remoteByIndex[4].position;
        Vector3 mcp = remoteHand.remoteByIndex[5].position;
        Vector3 tip = remoteHand.remoteByIndex[8].position;

        Vector3 axis = tip - mcp;
        float denom = axis.sqrMagnitude;
        if (denom < 1e-8f) return 0.5f;

        float t = Vector3.Dot(thumb - mcp, axis) / denom;
        return Mathf.Clamp01(t);
    }

    float TToAxisValue(float t, float t0)
    {
        // signed distance from neutral
        float d = t - t0; // [-1..1] roughly, but depends on t0
        float ad = Mathf.Abs(d);

        float dz = Mathf.Max(0f, deadZoneT);
        float fs = Mathf.Max(dz + 1e-4f, fullScaleT);

        if (ad < dz) return 0f;

        // map [dz..fs] -> [0..1]
        float x = Mathf.Clamp01((ad - dz) / (fs - dz));

        // response curve
        if (gamma > 0.01f && Mathf.Abs(gamma - 1f) > 1e-3f)
            x = Mathf.Pow(x, gamma);

        return Mathf.Sign(d) * x;
    }

    void HandleTapFSM(float dt)
    {
        // Use fast tip distances if available (snappier), otherwise fallback to transforms
        float dTI;
        if (remoteHand != null && remoteHand.fastTipsReady)
            dTI = Vector3.Distance(remoteHand.thumbTipFast, remoteHand.indexTipFast);
        else
            dTI = Vector3.Distance(remoteHand.remoteByIndex[4].position, remoteHand.remoteByIndex[8].position);

        debug_touchDist = dTI;

        if (Time.time < _tapCooldownUntil)
        {
            if (debug) debug_state = "Tap: cooldown";
            return;
        }

        bool downCond = dTI <= touchOnMeters;
        bool upCond = dTI >= touchOffMeters;

        if (_touchFsm == TouchFSM.Idle)
        {
            _upHeld = 0f;

            if (downCond)
            {
                _downHeld += dt;
                if (_downHeld >= debounceSeconds)
                {
                    _touchFsm = TouchFSM.DownLatched;
                    _downHeld = 0f;
                    if (debug) debug_state = "Tap: DOWN latched";
                }
            }
            else
            {
                _downHeld = 0f;
            }
        }
        else // DownLatched
        {
            _downHeld = 0f;

            if (upCond)
            {
                _upHeld += dt;
                if (_upHeld >= debounceSeconds)
                {
                    _upHeld = 0f;
                    _touchFsm = TouchFSM.Idle;

                    RegisterTap();
                }
            }
            else
            {
                _upHeld = 0f;
            }
        }

        // finalize single tap if window expires
        if (_tapCount == 1 && Time.time >= _tapWindowUntil)
        {
            _tapCount = 0;
            _tapWindowUntil = 0f;
            SingleTapThisFrame = true;
            _tapCooldownUntil = Time.time + Mathf.Max(0f, tapCooldownSec);

            if (debug) debug_state = "Tap: SINGLE";
        }
    }

    void RegisterTap()
    {
        if (_tapCount == 0)
        {
            _tapCount = 1;
            _tapWindowUntil = Time.time + Mathf.Max(0.05f, doubleTapWindowSec);
            if (debug) debug_state = "Tap: first";
            return;
        }

        if (_tapCount == 1 && Time.time <= _tapWindowUntil)
        {
            // double tap
            _tapCount = 0;
            _tapWindowUntil = 0f;
            DoubleTapThisFrame = true;
            _tapCooldownUntil = Time.time + Mathf.Max(0f, tapCooldownSec);

            if (debug) debug_state = "Tap: DOUBLE";
            return;
        }

        // If second tap came too late, treat as new first tap
        _tapCount = 1;
        _tapWindowUntil = Time.time + Mathf.Max(0.05f, doubleTapWindowSec);
        if (debug) debug_state = "Tap: first (late reset)";
    }

    void ToggleXY()
    {
        if (Mode == AxisMode.Z)
        {
            // If currently in Z, single tap does nothing (keeps rules simple)
            return;
        }

        Mode = (Mode == AxisMode.X) ? AxisMode.Y : AxisMode.X;

        if (recenterOnModeSwitch)
        {
            _neutralCaptured = false; // capture neutral next frame
            _axisSm = 0f;
            AxisValue = 0f;
        }
    }

    void ToggleZ()
    {
        if (Mode == AxisMode.Z)
        {
            // exiting Z returns to X
            Mode = AxisMode.X;
        }
        else
        {
            Mode = AxisMode.Z;
        }

        if (recenterOnModeSwitch)
        {
            _neutralCaptured = false;
            _axisSm = 0f;
            AxisValue = 0f;
        }
    }
}
