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
    [SerializeField] private float debounceSeconds = 0.06f;

    [Tooltip("Cooldown after a successful toggle (seconds).")]
    [SerializeField] private float toggleCooldownSec = 0.20f;

    [Tooltip("To reduce accidental OFF, require a longer pinch-hold before allowing OFF toggle (seconds).")]
    [SerializeField] private float offToggleMinHoldSec = 0.35f;

    [Header("Optional Z-mode toggle (Thumb-Middle pinch)")]
    [Tooltip("If false, ZMode toggle is disabled.")]
    [SerializeField] private bool enableZModeToggle = false;

    [SerializeField] private float zPinchOnMeters = 0.050f;
    [SerializeField] private float zPinchOffMeters = 0.070f;

    [Tooltip("To reduce accidental toggles, require a longer pinch-hold before allowing ZMode toggle (seconds).")]
    [SerializeField] private float zToggleMinHoldSec = 0.20f;

    [Header("Fingerpad (thumb position)")]
    [SerializeField] private float dpadDeadZone = 0.22f;
    [SerializeField] private float dpadFullScale = 0.85f;
    [SerializeField] private float dpadOutputLerp = 12f;

    [Tooltip("If true, capture center after Engage turns ON, not immediately.")]
    [SerializeField] private bool calibrateCenterOnEngage = true;

    [Tooltip("Delay after Engage ON before capturing center (seconds). Gives time to move thumb to the pad middle.")]
    [SerializeField] private float centerCaptureDelaySec = 0.20f;

    [Tooltip("Only capture center when thumb is near neutral (raw magnitude <= this).")]
    [SerializeField] private float centerCaptureMaxMag = 0.18f;

    [Tooltip("If true, fingerpad output is zero until center capture is done.")]
    [SerializeField] private bool zeroOutputUntilCenterCaptured = true;

    [Tooltip("If true, output is zero when not engaged.")]
    [SerializeField] private bool requireEngagedForDpad = true;

    [Header("Anti-accidental toggle while moving")]
    [Tooltip("Block Engage/Z toggles if fingerpad output magnitude is above this.")]
    [SerializeField] private float toggleMotionGate = 0.12f;

    [Header("Axis options")]
    [SerializeField] private bool invertX = false;
    [SerializeField] private bool invertY = false;

    [Header("Debug (read-only)")]
    [SerializeField] private bool debug = true;
    [SerializeField] private float debug_dTI = -1f;
    [SerializeField] private float debug_dTM = -1f;
    [SerializeField] private string debug_state = "";

    // ---- Debug public getters (for HUD) ----
    public float Debug_dTI => debug_dTI;
    public float Debug_dTM => debug_dTM;
    public string DebugState => debug_state;
    public bool DebugHasMinJoints => HasMinJoints();
    public bool DebugRemoteHandAssigned => (remoteHand != null);
    public int DebugRemoteByIndexLen => (remoteHand != null && remoteHand.remoteByIndex != null) ? remoteHand.remoteByIndex.Length : -1;
    public bool DebugThumbTipOk => JointOk(4);
    public bool DebugIndexTipOk => JointOk(8);
    public bool DebugMiddleTipOk => JointOk(12);
    public bool DebugWristOk => JointOk(0);
    public bool DebugIndexMcpOk => JointOk(5);
    public bool DebugMiddleMcpOk => JointOk(9);
    public bool DebugRingMcpOk => JointOk(13);
    public bool DebugPinkyMcpOk => JointOk(17);

    bool JointOk(int idx)
    {
        if (remoteHand == null) return false;
        if (remoteHand.remoteByIndex == null) return false;
        if (idx < 0 || idx >= remoteHand.remoteByIndex.Length) return false;
        return remoteHand.remoteByIndex[idx] != null;
    }

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

    // Track how long pinch has been held since latch (for OFF hardening)
    float _engageLatchedHoldSec = 0f;

    // Z toggle FSM (optional)
    PinchFSM _zFsm = PinchFSM.Idle;
    float _zDownHeldSec = 0f;
    float _zUpHeldSec = 0f;
    float _zCooldownUntil = 0f;
    float _zLatchedHoldSec = 0f;

    // Fingerpad center calibration + smoothing
    Vector2 _uvCenter = Vector2.zero;
    bool _hasCenter = false;
    Vector2 _dpadSm = Vector2.zero;
    bool _prevEngaged = false;

    // Center capture timing
    float _centerCaptureNotBefore = 0f;

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

        // ----- 0) Compute fingerpad first (so toggle gating can use it) -----
        // This also updates _dpadSm and Dpad.
        UpdateFingerpad(dt);

        bool blockToggleBecauseMoving = (_dpadSm.magnitude > Mathf.Max(0f, toggleMotionGate));
        bool blockToggleBecauseCenterWait = (IsEngaged && calibrateCenterOnEngage && Time.time < _centerCaptureNotBefore);

        // Read distances (pinch)
        Transform thumbTip = remoteHand.remoteByIndex[4];
        Transform indexTip = remoteHand.remoteByIndex[8];
        Transform middleTip = remoteHand.remoteByIndex[12];

        float dTI;
        float dTM;

        if (remoteHand.fastTipsReady)
        {
            dTI = Vector3.Distance(remoteHand.thumbTipFast, remoteHand.indexTipFast);
            dTM = Vector3.Distance(remoteHand.thumbTipFast, remoteHand.middleTipFast);
        }
        else
        {
            dTI = Vector3.Distance(thumbTip.position, indexTip.position);
            dTM = Vector3.Distance(thumbTip.position, middleTip.position);
        }

        if (debug)
        {
            debug_dTI = dTI;
            debug_dTM = dTM;
        }

        // ----------------------------
        // 1) Engage toggle (Thumb-Index)
        //    - Block toggles while fingerpad is moving
        //    - Make OFF harder with hold requirement
        // ----------------------------
        if (Time.time >= _cooldownUntil && !blockToggleBecauseMoving && !blockToggleBecauseCenterWait)
        {
            bool downCond = dTI <= pinchOnMeters;
            bool upCond = dTI >= pinchOffMeters;

            if (_engageFsm == PinchFSM.Idle)
            {
                _upHeldSec = 0f;
                _engageLatchedHoldSec = 0f;

                if (downCond)
                {
                    _downHeldSec += dt;
                    if (_downHeldSec >= debounceSeconds)
                    {
                        _engageFsm = PinchFSM.DownLatched;
                        _downHeldSec = 0f;
                        _engageLatchedHoldSec = 0f;

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
                _engageLatchedHoldSec += dt;

                // If currently engaged, require longer hold before allowing an OFF toggle
                bool offIsAllowed = (!IsEngaged) || (_engageLatchedHoldSec >= Mathf.Max(0f, offToggleMinHoldSec));

                if (upCond && offIsAllowed)
                {
                    _upHeldSec += dt;
                    if (_upHeldSec >= debounceSeconds)
                    {
                        _upHeldSec = 0f;
                        _engageFsm = PinchFSM.Idle;
                        _engageLatchedHoldSec = 0f;

                        bool newEngaged = !IsEngaged;
                        IsEngaged = newEngaged;
                        EngageToggledThisFrame = true;
                        _cooldownUntil = Time.time + Mathf.Max(0f, toggleCooldownSec);

                        if (debug) debug_state = $"Engage: TOGGLED -> {(IsEngaged ? "ON" : "OFF")}";

                        // Engage edge handling for center capture
                        if (calibrateCenterOnEngage && IsEngaged && !_prevEngaged)
                        {
                            _hasCenter = false;
                            _centerCaptureNotBefore = Time.time + Mathf.Max(0f, centerCaptureDelaySec);
                        }
                    }
                }
                else
                {
                    _upHeldSec = 0f;

                    if (IsEngaged && !offIsAllowed && debug)
                        debug_state = $"Engage: holding (OFF gated, hold={_engageLatchedHoldSec:F2}/{offToggleMinHoldSec:F2})";
                }
            }
        }
        else
        {
            if (debug)
            {
                if (Time.time < _cooldownUntil) debug_state = "Engage: cooldown";
                else if (blockToggleBecauseMoving) debug_state = "Engage: blocked (fingerpad moving)";
                else if (blockToggleBecauseCenterWait) debug_state = "Engage: blocked (center capture delay)";
            }
        }

        // ----------------------------
        // 2) Optional ZMode toggle (Thumb-Middle)
        //    - Only allow while engaged
        //    - Block while fingerpad moving
        // ----------------------------
        if (enableZModeToggle && IsEngaged && !blockToggleBecauseMoving && !blockToggleBecauseCenterWait)
        {
            if (Time.time >= _zCooldownUntil)
            {
                bool zDown = dTM <= zPinchOnMeters;
                bool zUp = dTM >= zPinchOffMeters;

                if (_zFsm == PinchFSM.Idle)
                {
                    _zUpHeldSec = 0f;
                    _zLatchedHoldSec = 0f;

                    if (zDown)
                    {
                        _zDownHeldSec += dt;
                        if (_zDownHeldSec >= debounceSeconds)
                        {
                            _zFsm = PinchFSM.DownLatched;
                            _zDownHeldSec = 0f;
                            _zLatchedHoldSec = 0f;
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
                    _zLatchedHoldSec += dt;

                    bool zAllowed = (_zLatchedHoldSec >= Mathf.Max(0f, zToggleMinHoldSec));

                    if (zUp && zAllowed)
                    {
                        _zUpHeldSec += dt;
                        if (_zUpHeldSec >= debounceSeconds)
                        {
                            _zUpHeldSec = 0f;
                            _zFsm = PinchFSM.Idle;
                            _zLatchedHoldSec = 0f;

                            ZMode = !ZMode;
                            ZModeToggledThisFrame = true;
                            _zCooldownUntil = Time.time + Mathf.Max(0f, toggleCooldownSec);

                            // Prevent sudden axis jump after mode switch
                            _dpadSm = Vector2.zero;
                            Dpad = Vector2.zero;

                            if (debug) debug_state = $"ZMode: TOGGLED -> {(ZMode ? "ON" : "OFF")}";
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
        // 3) Fingerpad center calibration edge bookkeeping
        // ----------------------------
        if (calibrateCenterOnEngage)
        {
            if (IsEngaged && !_prevEngaged)
            {
                // Engage ON happened (either this frame or earlier)
                _hasCenter = false;
                _centerCaptureNotBefore = Time.time + Mathf.Max(0f, centerCaptureDelaySec);
            }
            else if (!IsEngaged)
            {
                _hasCenter = false;
                _dpadSm = Vector2.zero;
            }

            _prevEngaged = IsEngaged;
        }
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

    void UpdateFingerpad(float dt)
    {
        // If fingerpad requires engage and we're not engaged, output zero.
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
        if (uAxis.sqrMagnitude < 1e-8f) { Dpad = Vector2.zero; return; }
        uAxis.Normalize();

        Vector3 vAxis = centerW - wrist.position;
        vAxis = vAxis - Vector3.Dot(vAxis, uAxis) * uAxis;
        if (vAxis.sqrMagnitude < 1e-8f) { Dpad = Vector2.zero; return; }
        vAxis.Normalize();

        Vector3 rel = thumbTip.position - centerW;

        float palmWidth = Mathf.Max(0.03f, Vector3.Distance(indexMcp.position, pinkyMcp.position));
        float palmHeight = Mathf.Max(0.04f, Vector3.Distance(wrist.position, centerW));

        float u = Vector3.Dot(rel, uAxis) / palmWidth;
        float v = Vector3.Dot(rel, vAxis) / palmHeight;

        if (invertX) u = -u;
        if (invertY) v = -v;

        Vector2 uv = new Vector2(u, v);

        // Center capture timing: wait a bit after Engage ON, then capture when near neutral.
        if (calibrateCenterOnEngage)
        {
            bool canCaptureNow = (Time.time >= _centerCaptureNotBefore);
            if (!_hasCenter && canCaptureNow)
            {
                // Require the thumb to be near neutral before capturing center (optional but very helpful)
                if (uv.magnitude <= Mathf.Max(0f, centerCaptureMaxMag))
                {
                    _uvCenter = uv;
                    _hasCenter = true;

                    if (debug) debug_state = "Fingerpad: center captured";
                }
            }

            // Optional: keep output zero until center is captured
            if (zeroOutputUntilCenterCaptured && !_hasCenter)
            {
                Dpad = Vector2.zero;
                _dpadSm = Vector2.zero;
                return;
            }
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
