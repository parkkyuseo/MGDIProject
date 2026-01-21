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

    [Header("Neutral (t0) handling")]
    [Tooltip("If true, neutral t0 is fixed at 0.5 (recommended for stability).")]
    [SerializeField] private bool useFixedNeutralMidpoint = true;

    [SerializeField] private float fixedNeutralT = 0.50f;

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

    [Tooltip("Invert the axis if direction feels reversed.")]
    [SerializeField] private bool invertAxis = false;

/*     [Header("Tap detection (Thumb-Index touch)")]
 *     [Tooltip("Touch is considered DOWN if distance <= this for debounceSeconds.")]
 *     [SerializeField] private float touchOnMeters = 0.030f;
 *
 *     [Tooltip("Touch is considered UP if distance >= this for debounceSeconds.")]
 *     [SerializeField] private float touchOffMeters = 0.045f; */
    [Header("Tap detection (Thumb-Index touch) - normalized")]
    [Tooltip("Touch DOWN if (thumb-index distance / index length) <= this.")]
    [SerializeField] private float touchOnRatio = 0.55f;

    [Tooltip("Touch UP if (thumb-index distance / index length) >= this. Must be > touchOnRatio.")]
    [SerializeField] private float touchOffRatio = 0.85f;

    [Tooltip("Minimum index length to avoid divide-by-zero (meters).")]
    [SerializeField] private float minIndexLenMeters = 0.03f;

    [Tooltip("Debounce time for DOWN/UP (seconds).")]
    [SerializeField] private float debounceSeconds = 0.06f;

    [Tooltip("Max time between two taps to count as double tap (seconds).")]
    [SerializeField] private float doubleTapWindowSec = 0.30f;

    [Tooltip("Cooldown after accepting a tap sequence (seconds).")]
    [SerializeField] private float tapCooldownSec = 0.18f;

    [Header("Tap gating (prevents toggles while sliding)")]
    [Tooltip("If true, taps are only recognized near the center zone and when the slider is not moving.")]
    [SerializeField] private bool gateTapToCenterZone = true;

    [Tooltip("Require |t - 0.5| <= this to accept a tap.")]
    [SerializeField] private float tapCenterBand = 0.08f;

    [Tooltip("Require |AxisValue| <= this to accept a tap (prevents taps during sliding).")]
    [SerializeField] private float tapAxisGate = 0.10f;

    [Tooltip("Extra minimum interval between accepted taps (seconds).")]
    [SerializeField] private float tapMinIntervalSec = 0.12f;

    [Header("Mode switching rules")]
    [Tooltip("Single tap toggles between X and Y.")]
    [SerializeField] private bool singleTapTogglesXY = true;

    [Tooltip("Double tap toggles Z mode (enter/exit). Exiting returns to X.")]
    [SerializeField] private bool doubleTapTogglesZ = true;

    [Header("Debug (read-only)")]
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

    // Debug getters for HUD
    public float Debug_t => debug_t;
    public float Debug_touchDist => debug_touchDist;
    public string Debug_state => debug_state;

    public float Debug_tNeutral => _tNeutral;
    public int Debug_tapCount => _tapCount;
    public float Debug_tapWindowRemaining => Mathf.Max(0f, _tapWindowUntil - Time.time);
    public bool Debug_touchDownLatched => (_touchFsm == TouchFSM.DownLatched);
    public bool Debug_inTapCooldown => (Time.time < _tapCooldownUntil);
    public float Debug_lastTapAcceptedAge => Mathf.Max(0f, Time.time - _lastTapAcceptedTime);

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

    float _lastTapAcceptedTime = -999f;

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

        // 2) Neutral handling
        if (useFixedNeutralMidpoint)
        {
            _tNeutral = fixedNeutralT;
            _neutralCaptured = true;
        }
        else
        {
            if (!_neutralCaptured)
            {
                _tNeutral = t;
                _neutralCaptured = true;
            }
        }

        // 3) Convert t -> signed axis value in [-1..1]
        float rawAxis = TToAxisValue(t, _tNeutral);
        if (invertAxis) rawAxis = -rawAxis;

        // 4) Smooth + rate limit
        float k = 1f - Mathf.Exp(-Mathf.Max(0.01f, outputLerp) * dt);
        float target = Mathf.Lerp(_axisSm, rawAxis, k);

        if (maxOutputRatePerSec > 0f)
        {
            float maxStep = maxOutputRatePerSec * dt;
            float d = target - _axisSm;
            d = Mathf.Clamp(d, -maxStep, maxStep);
            target = _axisSm + d;
        }

        _axisSm = target;
        AxisValue = _axisSm;

        // 5) Tap detection + mode switching
        HandleTapFSM(dt);

        if (SingleTapThisFrame && singleTapTogglesXY)
            ToggleXY();

        if (DoubleTapThisFrame && doubleTapTogglesZ)
            ToggleZ();
    }

    bool HasRequiredJoints()
    {
        if (remoteHand == null) return false;
        if (remoteHand.remoteByIndex == null) return false;
        if (remoteHand.remoteByIndex.Length < 9) return false;

        return remoteHand.remoteByIndex[4] != null && // thumb tip
               remoteHand.remoteByIndex[5] != null && // index mcp
               remoteHand.remoteByIndex[8] != null;   // index tip
    }

    float ComputeProjectedT()
    {
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
        float d = t - t0;
        float ad = Mathf.Abs(d);

        float dz = Mathf.Max(0f, deadZoneT);
        float fs = Mathf.Max(dz + 1e-4f, fullScaleT);

        if (ad < dz) return 0f;

        float x = Mathf.Clamp01((ad - dz) / (fs - dz));

        if (gamma > 0.01f && Mathf.Abs(gamma - 1f) > 1e-3f)
            x = Mathf.Pow(x, gamma);

        return Mathf.Sign(d) * x;
    }

    void HandleTapFSM(float dt)
    {
        // Gate taps so sliding does not trigger toggles
        if (gateTapToCenterZone)
        {
            if (Mathf.Abs(AxisValue) > tapAxisGate)
            {
                if (debug) debug_state = "Tap: blocked (axis moving)";
                return;
            }

            float center = 0.5f;
            if (Mathf.Abs(debug_t - center) > tapCenterBand)
            {
                if (debug) debug_state = "Tap: blocked (not in center band)";
                return;
            }
        }

        // extra min interval between accepted taps
        if (Time.time - _lastTapAcceptedTime < tapMinIntervalSec)
        {
            if (debug) debug_state = "Tap: blocked (min interval)";
            return;
        }

        if (Time.time < _tapCooldownUntil)
        {
            if (debug) debug_state = "Tap: cooldown";
            return;
        }

        // -----------------------------------------
        // Normalized touch distance (depth-robust)
        // dNorm = dist(thumbTip, indexTip) / dist(indexMCP, indexTip)
        // -----------------------------------------

        // thumb tip / index tip
        Vector3 thumbPos, indexTipPos;

        if (remoteHand != null && remoteHand.fastTipsReady)
        {
            thumbPos = remoteHand.thumbTipFast;
            indexTipPos = remoteHand.indexTipFast;
        }
        else
        {
            thumbPos = remoteHand.remoteByIndex[4].position;
            indexTipPos = remoteHand.remoteByIndex[8].position;
        }

        // index MCP is always from transforms (fast path not provided)
        Vector3 indexMcpPos = remoteHand.remoteByIndex[5].position;

        float dTI = Vector3.Distance(thumbPos, indexTipPos);
        float indexLen = Vector3.Distance(indexMcpPos, indexTipPos);
        indexLen = Mathf.Max(minIndexLenMeters, indexLen);

        float dNorm = dTI / indexLen;

        // For HUD: store normalized value (more useful than meters under depth changes)
        debug_touchDist = dNorm;

        bool downCond = dNorm <= touchOnRatio;
        bool upCond = dNorm >= touchOffRatio;

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
            _lastTapAcceptedTime = Time.time;

            if (debug) debug_state = "Tap: SINGLE";
        }
    }

    void RegisterTap()
    {
        _lastTapAcceptedTime = Time.time;

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
            // keep rules simple: single tap does not affect Z
            return;
        }

        Mode = (Mode == AxisMode.X) ? AxisMode.Y : AxisMode.X;

        // recenter output (helps avoid surprise drift after switching)
        _axisSm = 0f;
        AxisValue = 0f;

        if (!useFixedNeutralMidpoint)
            _neutralCaptured = false;
    }

    void ToggleZ()
    {
        if (Mode == AxisMode.Z)
        {
            Mode = AxisMode.X; // exiting Z returns to X
        }
        else
        {
            Mode = AxisMode.Z;
        }

        _axisSm = 0f;
        AxisValue = 0f;

        if (!useFixedNeutralMidpoint)
            _neutralCaptured = false;
    }
}
