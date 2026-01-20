using UnityEngine;

public class MicroInputThumbDpadToggle : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("Engage toggle (Thumb-Index pinch)")]
    [Tooltip("Pinch considered DOWN if distance <= this for a minimum duration.")]
    [SerializeField] private float pinchOnMeters = 0.050f;

    [Tooltip("Pinch considered UP if distance >= this for a minimum duration. Must be > pinchOnMeters.")]
    [SerializeField] private float pinchOffMeters = 0.070f;

    [Tooltip("Minimum time the condition must hold to count as DOWN/UP (seconds).")]
    [SerializeField] private float debounceSeconds = 0.06f; // ~4 frames at 60fps

    [Tooltip("Cooldown after a successful toggle (seconds).")]
    [SerializeField] private float toggleCooldownSec = 0.20f;

    [Header("Optional Z-mode toggle (Thumb-Middle pinch)")]
    [Tooltip("If false, ZMode toggle is disabled for debugging Engage stability.")]
    [SerializeField] private bool enableZModeToggle = false;

    [SerializeField] private float zPinchOnMeters = 0.050f;
    [SerializeField] private float zPinchOffMeters = 0.070f;

    [Header("D-pad (thumb position)")]
    [SerializeField] private float dpadDeadZone = 0.22f;
    [SerializeField] private float dpadFullScale = 0.85f;
    [SerializeField] private float dpadOutputLerp = 12f;
    [SerializeField] private bool calibrateCenterOnEngage = true;
    [SerializeField] private bool requireEngagedForDpad = true;

    [Header("Axis options")]
    [SerializeField] private bool invertX = false;
    [SerializeField] private bool invertY = false;

    [Header("Debug (read-only)")]
    [SerializeField] private bool debug = true;
    [SerializeField] private float debug_dTI = -1f;
    [SerializeField] private float debug_dTM = -1f;
    [SerializeField] private string debug_state = "";

    // Outputs
    public bool IsEngaged { get; private set; } = false;
    public bool ZMode { get; private set; } = false;
    public Vector2 Dpad { get; private set; } = Vector2.zero;

    public bool EngageToggledThisFrame { get; private set; } = false;
    public bool ZModeToggledThisFrame { get; private set; } = false;

    enum PinchFSM { Idle, DownLatched }
    PinchFSM _engageFsm = PinchFSM.Idle;

    float _downHeldSec = 0f;
    float _upHeldSec = 0f;
    float _cooldownUntil = 0f;

    // Z toggle FSM (optional)
    PinchFSM _zFsm = PinchFSM.Idle;
    float _zDownHeldSec = 0f;
    float _zUpHeldSec = 0f;
    float _zCooldownUntil = 0f;

    // D-pad center calibration + smoothing
    Vector2 _uvCenter = Vector2.zero;
    bool _hasCenter = false;
    Vector2 _dpadSm = Vector2.zero;
    bool _prevEngaged = false;

    void ResetFrameOutputs()
    {
        EngageToggledThisFrame = false;
        ZModeToggledThisFrame = false;
    }

    void Update()
    {
        ResetFrameOutputs();

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (!HasMinJoints())
        {
            Dpad = Vector2.zero;
            _dpadSm = Vector2.zero;
            if (debug) debug_state = "Missing joints / remoteHand";
            return;
        }

        // Read distances
        Transform thumbTip = remoteHand.remoteByIndex[4];
        Transform indexTip = remoteHand.remoteByIndex[8];
        Transform middleTip = remoteHand.remoteByIndex[12];

        float dTI = Vector3.Distance(thumbTip.position, indexTip.position);
        float dTM = Vector3.Distance(thumbTip.position, middleTip.position);

        if (debug)
        {
            debug_dTI = dTI;
            debug_dTM = dTM;
        }

        // ----------------------------
        // 1) Engage toggle: Thumb-Index only
        // ----------------------------
        if (Time.time >= _cooldownUntil)
        {
            bool downCond = dTI <= pinchOnMeters;
            bool upCond = dTI >= pinchOffMeters;

            if (_engageFsm == PinchFSM.Idle)
            {
                _upHeldSec = 0f;

                if (downCond)
                {
                    _downHeldSec += dt;
                    if (_downHeldSec >= debounceSeconds)
                    {
                        _engageFsm = PinchFSM.DownLatched;
                        _downHeldSec = 0f;
                        if (debug) debug_state = "Engage: DOWN latched";
                    }
                }
                else
                {
                    _downHeldSec = 0f;
                }
            }
            else // DownLatched
            {
                _downHeldSec = 0f;

                if (upCond)
                {
                    _upHeldSec += dt;
                    if (_upHeldSec >= debounceSeconds)
                    {
                        _upHeldSec = 0f;
                        _engageFsm = PinchFSM.Idle;

                        IsEngaged = !IsEngaged;
                        EngageToggledThisFrame = true;
                        _cooldownUntil = Time.time + Mathf.Max(0f, toggleCooldownSec);

                        if (debug) debug_state = $"Engage: TOGGLED -> {(IsEngaged ? "ON" : "OFF")}";
                    }
                }
                else
                {
                    _upHeldSec = 0f;
                }
            }
        }
        else
        {
            if (debug) debug_state = "Engage: cooldown";
        }

        // ----------------------------
        // 2) Optional ZMode toggle: Thumb-Middle only
        // ----------------------------
        if (enableZModeToggle && IsEngaged) // recommend only allow while engaged
        {
            if (Time.time >= _zCooldownUntil)
            {
                bool zDown = dTM <= zPinchOnMeters;
                bool zUp = dTM >= zPinchOffMeters;

                if (_zFsm == PinchFSM.Idle)
                {
                    _zUpHeldSec = 0f;

                    if (zDown)
                    {
                        _zDownHeldSec += dt;
                        if (_zDownHeldSec >= debounceSeconds)
                        {
                            _zFsm = PinchFSM.DownLatched;
                            _zDownHeldSec = 0f;
                        }
                    }
                    else
                    {
                        _zDownHeldSec = 0f;
                    }
                }
                else
                {
                    _zDownHeldSec = 0f;

                    if (zUp)
                    {
                        _zUpHeldSec += dt;
                        if (_zUpHeldSec >= debounceSeconds)
                        {
                            _zUpHeldSec = 0f;
                            _zFsm = PinchFSM.Idle;

                            ZMode = !ZMode;
                            ZModeToggledThisFrame = true;
                            _zCooldownUntil = Time.time + Mathf.Max(0f, toggleCooldownSec);
                        }
                    }
                    else
                    {
                        _zUpHeldSec = 0f;
                    }
                }
            }
        }

        // ----------------------------
        // 3) D-pad center calibration edge
        // ----------------------------
        if (calibrateCenterOnEngage)
        {
            if (IsEngaged && !_prevEngaged)
            {
                _hasCenter = false; // capture next time
            }
            else if (!IsEngaged)
            {
                _hasCenter = false;
                _dpadSm = Vector2.zero;
            }
            _prevEngaged = IsEngaged;
        }

        // ----------------------------
        // 4) D-pad compute
        // ----------------------------
        UpdateDpad(dt);
    }

    bool HasMinJoints()
    {
        if (remoteHand == null) return false;
        if (remoteHand.remoteByIndex == null) return false;
        if (remoteHand.remoteByIndex.Length < 18) return false;

        return remoteHand.remoteByIndex[0] != null &&
               remoteHand.remoteByIndex[4] != null &&
               remoteHand.remoteByIndex[5] != null &&
               remoteHand.remoteByIndex[8] != null &&
               remoteHand.remoteByIndex[9] != null &&
               remoteHand.remoteByIndex[13] != null &&
               remoteHand.remoteByIndex[17] != null &&
               remoteHand.remoteByIndex[12] != null;
    }

    void UpdateDpad(float dt)
    {
        if (requireEngagedForDpad && !IsEngaged)
        {
            Dpad = Vector2.zero;
            _dpadSm = Vector2.zero;
            return;
        }

        Transform thumbTip = remoteHand.remoteByIndex[4];
        Transform wrist = remoteHand.remoteByIndex[0];

        Transform indexMcp = remoteHand.remoteByIndex[5];
        Transform middleMcp = remoteHand.remoteByIndex[9];
        Transform ringMcp = remoteHand.remoteByIndex[13];
        Transform pinkyMcp = remoteHand.remoteByIndex[17];

        Vector3 centerW = (indexMcp.position + middleMcp.position + ringMcp.position + pinkyMcp.position) * 0.25f;

        Vector3 uAxis = pinkyMcp.position - indexMcp.position;
        if (uAxis.sqrMagnitude < 1e-8f) return;
        uAxis.Normalize();

        Vector3 vAxis = centerW - wrist.position;
        vAxis = vAxis - Vector3.Dot(vAxis, uAxis) * uAxis;
        if (vAxis.sqrMagnitude < 1e-8f) return;
        vAxis.Normalize();

        Vector3 rel = thumbTip.position - centerW;

        float palmWidth = Mathf.Max(0.03f, Vector3.Distance(indexMcp.position, pinkyMcp.position));
        float palmHeight = Mathf.Max(0.04f, Vector3.Distance(wrist.position, centerW));

        float u = Vector3.Dot(rel, uAxis) / palmWidth;
        float v = Vector3.Dot(rel, vAxis) / palmHeight;

        if (invertX) u = -u;
        if (invertY) v = -v;

        Vector2 uv = new Vector2(u, v);

        if (calibrateCenterOnEngage && !_hasCenter)
        {
            _uvCenter = uv;
            _hasCenter = true;
        }

        Vector2 raw = calibrateCenterOnEngage ? (uv - _uvCenter) : uv;

        Vector2 outVec = Vector2.zero;
        float mag = raw.magnitude;

        if (mag >= dpadDeadZone)
        {
            float denom = Mathf.Max(dpadDeadZone + 1e-6f, dpadFullScale);
            float t = Mathf.Clamp01((mag - dpadDeadZone) / (denom - dpadDeadZone));
            outVec = raw.normalized * t;
        }

        float k = 1f - Mathf.Exp(-Mathf.Max(0.01f, dpadOutputLerp) * dt);
        _dpadSm = Vector2.Lerp(_dpadSm, outVec, k);
        Dpad = _dpadSm;
    }
}
