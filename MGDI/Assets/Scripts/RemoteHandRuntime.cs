using UnityEngine;

public class RemoteHandRuntime : MonoBehaviour
{
    [Header("Remote driver joints (21)")]
    public Transform[] remoteByIndex = new Transform[21]; // 0..20 (WRIST..PINKY_TIP)

    [Header("Wrist aim target (position only)")]
    public Transform palmFwd; // Remote_PALM_FWD
    public Transform palmUp;  // Remote_PALM_UP
    [Tooltip("Distance from wrist to aim target in meters.")]
    public float palmAimDistance = 0.08f; // about 5-10 cm

    [Header("Options")]
    [Tooltip("True for left hand, false for right hand. Filled by UdpHandReceiver.")]
    public bool isLeft = false;
    [Tooltip("If true, incoming network data is ignored.")]
    public bool manualTestMode = false;

    // =========================================================
    // Interpolation buffer (network jitter smoothing)
    // =========================================================
    [Header("Interpolation buffer (network jitter smoothing)")]
    [Tooltip("If true, received poses are buffered and rendered with a small delay using interpolation.")]
    public bool useInterpolationBuffer = true;

    [Tooltip("Render time = now - delay. Increase = smoother but more lag (seconds).")]
    public float interpolationDelaySeconds = 0.06f;

    [Tooltip("How long samples are kept in the buffer (seconds). Must be >= interpolationDelaySeconds.")]
    public float bufferMaxSeconds = 0.50f;

    [Tooltip("Max number of buffered samples (ring buffer size).")]
    public int bufferMaxSamples = 90;

    // =========================================================
    // Smoothing (joint positions)
    // =========================================================
    [Header("Smoothing for joint positions (OneEuro + speed clamp)")]
    [Tooltip("Fallback LPF cutoff for non-tip joints (Hz) when OneEuro is OFF.")]
    public float cutoffHz = 10f;

    [Tooltip("Fallback LPF cutoff for tip joints (Hz) when OneEuro is OFF.")]
    public float cutoffHzTips = 6f;

    [Tooltip("Max joint speed for non-tip joints (meters/second).")]
    public float maxSpeedMps = 4.8f; // ~= 0.08 m/frame @ 60fps

    [Tooltip("Max joint speed for tip joints (meters/second).")]
    public float maxSpeedTipsMps = 0.9f; // ~= 0.015 m/frame @ 60fps

    [Header("Micro-jitter suppression (soft dead-zone)")]
    [Tooltip("If true, very small moves are softened (NOT hard-clamped).")]
    public bool useJitterDeadZone = true;

    [Tooltip("Soft dead-zone radius (meters). Smaller deltas are attenuated smoothly.")]
    public float jitterDeadZoneMeters = 0.003f; // 3 mm

    // =========================================================
    // Fast signals (filtered pose) + raw samples
    // =========================================================
    [Header("Fast signals (filtered pose) for discrete gestures")]
    [Tooltip("Updated from the filtered/applied pose (after SmoothAndApply). Stable but has smoothing/latency.")]
    public Vector3 thumbTipFast;
    public Vector3 indexTipFast;
    public Vector3 middleTipFast;
    public bool fastTipsReady = false;

    [Header("Raw tip samples (latest received, offset-applied, before smoothing)")]
    [Tooltip("Latest received sample (ApplyWorldPositions), with initial offset applied if enabled. Not smoothed.")]
    public Vector3 thumbTipRaw;
    public Vector3 indexTipRaw;
    public Vector3 middleTipRaw;
    public bool rawTipsReady = false;

    // =========================================================
    // Tracking IDs (for gesture gating)
    // =========================================================
    public int SampleId => _sampleId;
    public float LastSampleTime => _lastSampleTime;
    public int RenderFrameId => _renderFrameId;
    public float LastRenderTime => _lastRenderTime;

    int _sampleId = 0;
    float _lastSampleTime = -999f;
    int _renderFrameId = 0;
    float _lastRenderTime = -999f;

    // =========================================================
    // One Euro Filter (adaptive smoothing)
    // =========================================================
    [Header("One Euro Filter (adaptive smoothing)")]
    [Tooltip("If true, use One Euro Filter for joint smoothing (recommended with interpolation buffer).")]
    public bool useOneEuro = true;

    [Tooltip("Derivative cutoff (Hz). Higher = derivative reacts faster (usually 1~10).")]
    public float oneEuroDerivativeCutoffHz = 5.0f;

    [Tooltip("WRIST minCutoff (Hz). Lower = smoother at rest, more lag.")]
    public float oneEuroMinCutoffWristHz = 1.5f;

    [Tooltip("WRIST beta. Higher = less lag when moving fast.")]
    public float oneEuroBetaWrist = 0.7f;

    [Tooltip("Non-tip joints minCutoff (Hz).")]
    public float oneEuroMinCutoffHz = 2.0f;

    [Tooltip("Non-tip joints beta.")]
    public float oneEuroBeta = 0.5f;

    [Tooltip("TIP joints minCutoff (Hz). Lower = smoother tips.")]
    public float oneEuroMinCutoffTipsHz = 1.0f;

    [Tooltip("TIP joints beta. (tips: usually a bit lower for steadiness).")]
    public float oneEuroBetaTips = 0.4f;

    // =========================================================
    // Rig arming
    // =========================================================
    [Header("Rig arming")]
    [Tooltip("HandRigArmer that controls the rig weight.")]
    public HandRigArmer armer;
    [Tooltip("Minimum wrist-to-finger distance for a frame to be considered valid.")]
    public float firstValidDistance = 0.02f;
    [Tooltip("If true, rig weight is armed automatically on first valid frame.")]
    public bool autoArm = false;

    // =========================================================
    // Initial offset
    // =========================================================
    [Header("Initial offset from rig and camera")]
    [Tooltip("If true, capture offset on first frame so proxy hand stays near rWrist.")]
    public bool addInitialOffset = true;
    [Tooltip("Wrist bone of the proxy hand rig (for offset capture).")]
    public Transform rWrist;
    [Tooltip("If true, push proxy hand forward from the headset camera when capturing offset.")]
    public bool useExtraCameraOffset = true;
    [Tooltip("Extra forward offset from camera when capturing initial offset (meters).")]
    public float extraForwardMeters = 0.25f;
    [Tooltip("Extra vertical offset from camera when capturing initial offset (meters).")]
    public float extraUpMeters = 0.0f;

    // =========================================================
    // Aim settings
    // =========================================================
    [Header("Aim settings (position only)")]
    [Tooltip("If true, palmFwd is updated every frame to follow the hand direction.")]
    public bool computePalmFrame = true;
    [Tooltip("If true, aim direction is based on wrist velocity. If false, use wrist-to-middle vector.")]
    public bool aimFromVelocity = true;
    [Tooltip("Minimum wrist speed for velocity-based aiming (meters per second).")]
    public float aimVelMinSpeed = 0.05f;
    [Range(0f, 1f)]
    [Tooltip("Slerp factor for aim direction. (주의: 값이 클수록 더 즉각 반응)")]
    public float aimDirLerp = 0.20f;

    // =========================================================
    // Translation gain
    // =========================================================
    [Header("Translation gain (no remap)")]
    [Tooltip("Proxy translation gain. 1=1:1, 2=twice as far.")]
    public float translationGain = 1.0f;

    [Tooltip("If true, gain is applied around the first captured wrist position.")]
    public bool gainUseNeutralWrist = true;

    Vector3 _gainNeutralWristWorld = Vector3.zero;
    bool _gainNeutralCaptured = false;

    // =========================================================
    // Depth stabilization
    // =========================================================
    [Header("Depth (camera-forward) stabilization")]
    [Tooltip("If true, suppress jitter along camera-forward direction (depth axis).")]
    public bool stabilizeDepth = true;

    [Tooltip("If true, apply depth stabilization only to wrist joint (index 0). If false, apply to all joints.")]
    public bool stabilizeDepthWristOnly = true;

    [Tooltip("Extra dead-zone for depth component (meters).")]
    public float depthDeadZoneMeters = 0.010f; // 1 cm

    [Tooltip("Max depth speed when gated (meters/second).")]
    public float depthMaxSpeedMps = 0.30f; // ~= 5mm/frame @60fps

    [Tooltip("Extra low-pass cutoff for depth delta when gated (Hz). Lower = smoother, more lag.")]
    public float depthCutoffHz = 2.0f;

    // gating
    [Header("Depth stabilization gating (wrist intent detection)")]
    [Tooltip("If true, depth stabilization is applied only when lateral motion dominates.")]
    public bool depthUseGating = true;

    [Tooltip("Minimum non-depth speed to consider as 'lateral motion' (m/s).")]
    public float depthGateMinNonDepthSpeed = 0.24f; // ~= 4mm/frame @60fps

    [Tooltip("If |dDepth| <= ratio * nonDepthMag, treat depth as noise and stabilize it.")]
    public float depthGateDepthToNonDepthRatio = 0.35f; // 0.2~0.6

    [Tooltip("Max depth speed when NOT gated (user intends forward/back) (m/s).")]
    public float depthMaxSpeedMpsFree = 2.4f; // ~= 4cm/frame @60fps

    [Tooltip("Depth deadzone when NOT gated.")]
    public float depthDeadZoneMetersFree = 0.000f;

    [Tooltip("Depth cutoff when NOT gated (less smoothing).")]
    public float depthCutoffHzFree = 10f;

    // =========================================================
    // Twist (door knob style)
    // =========================================================
    [Header("Twist (door knob style)")]
    [Tooltip("Bone that should twist around its local forward axis (usually R_Wrist_Twist).")]
    public Transform wristTwist;
    [Tooltip("Maximum twist angle from neutral (degrees).")]
    public float twistMaxAbsDeg = 60f;
    [Tooltip("Angles smaller than this are treated as zero twist (degrees).")]
    public float twistDeadZoneDeg = 5f;
    [Tooltip("Maximum twist speed (degrees per second).")]
    public float twistMaxDegPerSec = 360f;
    [Range(0f, 1f)]
    [Tooltip("Extra smoothing for twist angle (0 = no smoothing, 1 = very slow).")]
    public float twistLerp = 0.2f;
    [Tooltip("Invert twist sign if the direction feels reversed.")]
    public bool twistInvertSign = false;

    public float TwistDegrees => _twistSmoothedDeg;
    public bool TwistReady => _twistNeutralReady;

    // =========================================================
    // Side-to-front remap
    // =========================================================
    [Header("Side-to-front remap")]
    public bool enableSideToFrontRemap = false;

    [Tooltip("Camera-local X (left/right) to workspace X scale.")]
    public float remapXScale = 1.0f;

    [Tooltip("Camera-local Z (forward) to workspace Y (up) scale.")]
    public float remapYFromZScale = 0.6f;
    public bool remapInvertYFromZ = false;

    [Tooltip("Camera-local Y (up) to workspace Z (forward/back) scale.")]
    public float remapZFromYScale = 0.6f;
    public bool remapInvertZFromY = false;

    [Range(0f, 1f)]
    [Tooltip("Smoothing factor for remap offset (0 = very stiff, 1 = no smoothing).")]
    public float remapLerp = 0.15f;

    [Tooltip("Max remap offset change per frame in camera space (meters).")]
    public float remapMaxStepMeters = 0.02f;

    [Tooltip("Max absolute workspace offset in camera-local space (meters).")]
    public Vector3 remapMaxOffsetCam = new Vector3(0.30f, 0.20f, 0.30f);

    [Tooltip("Camera-space dead-zone for remap offset changes (meters).")]
    public float remapDeadZoneMeters = 0.005f;

    Vector3 _remapNeutralCam = Vector3.zero;
    Vector3 _remapNeutralWorld = Vector3.zero;
    Vector3 _remapOffsetCamSm = Vector3.zero;
    bool _remapNeutralCaptured = false;

    // =========================================================
    // Joystick-style remap (experimental)
    // =========================================================
    [Header("Joystick-style remap (experimental)")]
    [Tooltip("If true, use joystick-style remap instead of position remap.")]
    public bool useJoystickRemap = false;

    [Tooltip("Half-size of the dead-zone box near your real hand (meters, world space).")]
    public Vector3 joyBoxHalfSizeWorld = new Vector3(0.10f, 0.10f, 0.10f); // 10cm cube

    [Tooltip("Max proxy speed along camera-right (X), camera-up (Y), camera-forward (Z) in m/s.")]
    public float joyMaxSpeedX = 0.4f; // real X -> proxy X
    public float joyMaxSpeedY = 0.4f; // real Z -> proxy Y
    public float joyMaxSpeedZ = 0.4f; // real Y -> proxy Z

    [Tooltip("Exponent for joystick response (1 = linear, 2 = softer near center).")]
    public float joyExpo = 1.0f;

    [Tooltip("Max workspace offset in camera-local space (meters).")]
    public Vector3 joyMaxOffsetCam = new Vector3(0.6f, 0.4f, 0.6f);

    [Tooltip("Invert proxy X when real hand moves left/right.")]
    public bool joyInvertX = false;

    [Tooltip("Invert proxy Y when real hand moves forward/back (real Z).")]
    public bool joyInvertYFromZ = false;

    [Tooltip("Invert proxy Z when real hand moves up/down (real Y).")]
    public bool joyInvertZFromY = false;

    Vector3 _joyOffsetCam = Vector3.zero;
    Vector3 _joyNeutralWorld = Vector3.zero;
    bool _joyNeutralCaptured = false;

    [Tooltip("Extra smoothing for joystick inputs (0 = no extra smoothing, 1 = very slow).")]
    [Range(0f, 1f)]
    public float joyInputLerp = 0.3f;

    float _joyInXSm = 0f;
    float _joyInYSm = 0f;
    float _joyInZSm = 0f;
    bool _joyHasInputPrev = false;

    // --- Driver joint base rotations (for workspace yaw offset) ---
    Quaternion[] _driverBaseRot = new Quaternion[21];
    bool _haveDriverBaseRot = false;

    Quaternion WorkspaceYawDelta()
    {
        if (!_haveWorkspaceBase) return Quaternion.identity;
        // deltaYaw = curYaw * inverse(baseYaw)
        return _wsCurYaw * Quaternion.Inverse(_wsBaseYaw);
    }

    // ---- Workspace offset (for side-body visual alignment) ----
    [SerializeField] private Transform workspaceOffsetAnchor; // WorkspaceAnchorController.workspaceAnchor
    private bool _haveWorkspaceBase = false;
    private Vector3 _wsBasePos;
    private Quaternion _wsBaseYaw;   // yaw-only
    private Vector3 _wsCurPos;
    private Quaternion _wsCurYaw;    // yaw-only

    public void SetWorkspaceOffsetAnchor(Transform t)
    {
        // Same anchor: only refresh current pose
        if (workspaceOffsetAnchor == t)
        {
            UpdateWorkspaceCurrentFromAnchor();
            return;
        }

        workspaceOffsetAnchor = t;

        DebugHUD.Log("[RHR] SetWorkspaceOffsetAnchor CALLED: " + (t ? t.name : "null"));

        if (workspaceOffsetAnchor == null)
        {
            _haveWorkspaceBase = false;
            return;
        }

        // Anchor changed -> treat this as a new reference frame.
        CaptureWorkspaceBaseFromCurrentAnchor();   // sets _haveWorkspaceBase=true
        UpdateWorkspaceCurrentFromAnchor();

        // Make rotation follow the new baseline cleanly on next SmoothAndApply
        _haveDriverBaseRot = false;

        // Prevent a big jump due to previous smoothing history
        _havePrevPos = false;

        // If using interpolation buffer, clear it to avoid blending old-frame into new-frame
        ClearInterpolationBuffer();
    }

    // Call this when entering NearHead (front) so it becomes the baseline frame.
    public void CaptureWorkspaceBaseFromCurrentAnchor()
    {
        if (workspaceOffsetAnchor == null) { _haveWorkspaceBase = false; return; }
        _wsBasePos = workspaceOffsetAnchor.position;
        _wsBaseYaw = YawOnly(workspaceOffsetAnchor.rotation);
        _haveWorkspaceBase = true;

        // 추가: 다음 SmoothAndApply에서 baseline 기준 회전을 다시 캡처하도록 리셋
        _haveDriverBaseRot = false;

        DebugHUD.Log($"[RHR] BaseCaptured pos={_wsBasePos}");
    }

    // Call this every time after profiles are applied (near or side), so current pose is updated.
    public void UpdateWorkspaceCurrentFromAnchor()
    {
        if (workspaceOffsetAnchor == null) { DebugHUD.Log("[RHR] UpdateCurrent failed: anchor null"); return; }

        /* if (workspaceOffsetAnchor == null) return; */
        _wsCurPos = workspaceOffsetAnchor.position;
        _wsCurYaw = YawOnly(workspaceOffsetAnchor.rotation);

        if (Time.frameCount % 120 == 0)
                Debug.Log($"[RHR] Cur pos={_wsCurPos}  dPos={( _wsCurPos - _wsBasePos).magnitude:F3}");
    }

    // Apply baseline->current transform to an input world position (translation + yaw).
    private Vector3 ApplyWorkspaceOffsetToWorldPos(Vector3 worldPos)
    {
        if (!_haveWorkspaceBase)
        {
            if (Time.frameCount % 120 == 0) DebugHUD.Log("[RHR] No base -> passthrough");
            return worldPos;
        }


        if (workspaceOffsetAnchor == null || !_haveWorkspaceBase)
            return worldPos;

        // Convert worldPos into baseline frame local (yaw-only around base)
        Vector3 v = worldPos - _wsBasePos;
        Vector3 local = Quaternion.Inverse(_wsBaseYaw) * v;

        // Re-express in current frame
        Vector3 outPos = _wsCurPos + (_wsCurYaw * local);
        return outPos;
    }

    private static Quaternion YawOnly(Quaternion q)
    {
        Vector3 f = q * Vector3.forward;
        f.y = 0f;
        if (f.sqrMagnitude < 1e-8f) return Quaternion.identity;
        return Quaternion.LookRotation(f.normalized, Vector3.up);
    }

    // =========================================================
    // Preset helper
    // =========================================================
    public enum SmoothingPreset
    {
        Custom = 0,
        Balanced = 1,
        Smooth = 2,
        VerySmooth = 3
    }

    [Header("Smoothing Preset")]
    public SmoothingPreset smoothingPreset = SmoothingPreset.Custom;

    [Tooltip("If true, the selected preset is applied automatically at startup.")]
    public bool applyPresetOnAwake = true;

    [Tooltip("If true, applying preset clears interpolation buffer (recommended).")]
    public bool clearBufferOnPresetApply = true;

    [Tooltip("If true, applying preset resets filter/smoothing state (recommended).")]
    public bool resetStateOnPresetApply = true;

#if UNITY_EDITOR
    [Tooltip("If true, selecting a preset also writes values into the inspector in edit mode.")]
    public bool applyPresetInEditorOnValidate = false;
#endif

    bool _presetAppliedOnce = false;
    bool _applyingPresetGuard = false;

    // =========================================================
    // Internal state (position smoothing)
    // =========================================================
    Vector3[] _prevPos = new Vector3[21];
    bool _havePrevPos = false;

    // aim state
    Vector3 _wristPrev;
    bool _haveWristPrev = false;
    Vector3 _aimDirSm = Vector3.zero;

    // offset
    bool _offsetCaptured = false;
    Vector3 _initialOffset = Vector3.zero;
    Vector3 _lastPreOffsetWrist = Vector3.zero;

    // auto-arm
    bool _firstArmed = false;

    // twist state
    bool _twistNeutralReady = false;
    Vector3 _twistAxisNeutral;
    Vector3 _twistRefNeutral;
    float _twistSmoothedDeg = 0f;
    Quaternion _twistBaseLocalRot = Quaternion.identity;
    bool _twistBaseCaptured = false;

    // =========================================================
    // Interpolation buffer (ring)
    // =========================================================
    float[] _bufTimes = null;
    Vector3[][] _bufPos = null;  // [sampleIndex][jointIndex]
    int _bufHead = 0;            // next write index
    int _bufCount = 0;           // number of valid samples

    // working pose array (reused, no GC)
    Vector3[] _workPos = new Vector3[21];

    // =========================================================
    // One Euro filters (per joint)
    // =========================================================
    class OneEuroVec3
    {
        bool _init = false;
        Vector3 _xHat = Vector3.zero;   // filtered signal
        Vector3 _dxHat = Vector3.zero;  // filtered derivative

        static float Alpha(float cutoffHz, float dt)
        {
            cutoffHz = Mathf.Max(0.0001f, cutoffHz);
            float omega = 2f * Mathf.PI * cutoffHz;
            return (omega * dt) / (1f + omega * dt);
        }

        public void Invalidate()
        {
            _init = false;
            _xHat = Vector3.zero;
            _dxHat = Vector3.zero;
        }

        public void Reset(Vector3 x0)
        {
            _init = true;
            _xHat = x0;
            _dxHat = Vector3.zero;
        }

        public Vector3 Filter(Vector3 x, float dt, float minCutoffHz, float beta, float dCutoffHz)
        {
            dt = Mathf.Max(1e-4f, dt);

            if (!_init)
            {
                Reset(x);
                return x;
            }

            // derivative from filtered signal
            Vector3 dx = (x - _xHat) / dt;

            // filter derivative
            float aD = Alpha(dCutoffHz, dt);
            _dxHat = Vector3.Lerp(_dxHat, dx, Mathf.Clamp01(aD));

            // adaptive cutoff
            float cutoff = Mathf.Max(0.0001f, minCutoffHz + beta * _dxHat.magnitude);

            // filter signal
            float a = Alpha(cutoff, dt);
            _xHat = Vector3.Lerp(_xHat, x, Mathf.Clamp01(a));

            return _xHat;
        }
    }

    OneEuroVec3[] _oneEuro = new OneEuroVec3[21];

    // =========================================================
    // Unity lifecycle
    // =========================================================
    void Awake()
    {
        EnsureBufferAllocated();
        EnsureOneEuroAllocated();
        ApplyPresetIfNeeded();
    }

    void OnEnable()
    {
        EnsureBufferAllocated();
        EnsureOneEuroAllocated();
        ApplyPresetIfNeeded();
    }

    void ApplyPresetIfNeeded()
    {
        if (!applyPresetOnAwake) return;
        if (_presetAppliedOnce) return;
        if (smoothingPreset == SmoothingPreset.Custom) return;

        ApplyPreset(smoothingPreset, resetStateOnPresetApply, clearBufferOnPresetApply);
        _presetAppliedOnce = true;
    }

#if UNITY_EDITOR
    void OnValidate()
    {
        if (!applyPresetInEditorOnValidate) return;
        if (Application.isPlaying) return;
        if (smoothingPreset == SmoothingPreset.Custom) return;

        // 편집 중에는 상태 리셋/버퍼 클리어 불필요
        ApplyPreset(smoothingPreset, resetState: false, clearBuffer: false);
    }
#endif

    void Update()
    {
        if (manualTestMode) return;

        // If using interpolation buffer, we render from buffer every frame.
        if (useInterpolationBuffer)
        {
            RenderFromBuffer();
        }
        // else: ApplyWorldPositions() processes immediately (receiver-driven)
    }

    // =========================================================
    // Entry from UdpHandReceiver
    // =========================================================
    public void ApplyWorldPositions(Vector3[] worldPos)
    {
        if (manualTestMode || worldPos == null || worldPos.Length < 21) return;

        _sampleId++;
        _lastSampleTime = Time.time;

        // store wrist before applying offset
        _lastPreOffsetWrist = worldPos[0];

        bool capturedOffsetThisCall = false;

        // capture initial offset once so that remote hand aligns with proxy rig
        if (addInitialOffset && !_offsetCaptured && rWrist != null)
        {
            Vector3 anchorPos = rWrist.position;

            if (useExtraCameraOffset && Camera.main != null)
            {
                Transform cam = Camera.main.transform;
                anchorPos += cam.forward * extraForwardMeters;
                anchorPos += cam.up * extraUpMeters;
            }

            _initialOffset = anchorPos - _lastPreOffsetWrist;
            _offsetCaptured = true;
            capturedOffsetThisCall = true;

            if (workspaceOffsetAnchor != null)
            {
                CaptureWorkspaceBaseFromCurrentAnchor();
                UpdateWorkspaceCurrentFromAnchor();
            }

            // coordinate frame changed => clear buffer + reset smoothing
            ClearInterpolationBuffer();
            ResetSmoothingState(fullResetRemapAndGain: false);
        }

        // --- Raw tips (latest received, offset-applied, BEFORE smoothing) ---
        {
            Vector3 t = worldPos[4];
            Vector3 i = worldPos[8];
            Vector3 m = worldPos[12];

            if (addInitialOffset && _offsetCaptured)
            {
                t += _initialOffset;
                i += _initialOffset;
                m += _initialOffset;
            }

            thumbTipRaw = t;
            indexTipRaw = i;
            middleTipRaw = m;
            rawTipsReady = true;
        }

        if (useInterpolationBuffer)
        {
            // enqueue sample (copy into ring, apply initial offset if captured)
            EnqueueSample(worldPos);

            // NOTE:
            // thumbTipFast/... are updated from the filtered/applied pose in SmoothAndApply()
            // so we don't touch them here.
        }
        else
        {
            // immediate mode: copy -> apply offset -> process now
            CopyWithInitialOffset(worldPos, _workPos);
            ProcessFrame(_workPos);
        }
    }

    // =========================================================
    // Buffer helpers
    // =========================================================
    void EnsureBufferAllocated()
    {
        int N = Mathf.Max(2, bufferMaxSamples);
        if (_bufTimes != null && _bufPos != null && _bufTimes.Length == N && _bufPos.Length == N)
            return;

        bufferMaxSamples = N;
        _bufTimes = new float[N];
        _bufPos = new Vector3[N][];

        for (int i = 0; i < N; i++)
            _bufPos[i] = new Vector3[21];

        _bufHead = 0;
        _bufCount = 0;
    }

    void ClearInterpolationBuffer()
    {
        _bufHead = 0;
        _bufCount = 0;
    }

    int TailIndex(int N)
    {
        return (_bufHead - _bufCount + N) % N;
    }

    void PruneOldSamples(float minTime)
    {
        if (_bufTimes == null) return;

        int N = _bufTimes.Length;

        // Drop oldest while too old
        while (_bufCount > 0)
        {
            int tail = TailIndex(N);
            if (_bufTimes[tail] < minTime)
                _bufCount--;
            else
                break;
        }
    }

    void EnqueueSample(Vector3[] worldPos)
    {
        EnsureBufferAllocated();

        int N = _bufTimes.Length;
        int idx = _bufHead;

        _bufTimes[idx] = Time.time;

        Vector3[] dst = _bufPos[idx];
        for (int i = 0; i < 21; i++)
        {
            Vector3 v = worldPos[i];

            if (addInitialOffset && _offsetCaptured)
                v += _initialOffset;

            dst[i] = v;
        }

        _bufHead = (_bufHead + 1) % N;
        if (_bufCount < N) _bufCount++;

        // Ensure buffer retention covers interpolation delay
        float keep = Mathf.Max(bufferMaxSeconds, interpolationDelaySeconds + 0.05f);
        float minTime = Time.time - Mathf.Max(0.01f, keep);
        PruneOldSamples(minTime);
    }

    void CopyWithInitialOffset(Vector3[] src, Vector3[] dst)
    {
        for (int i = 0; i < 21; i++)
        {
            Vector3 v = src[i];
            if (addInitialOffset && _offsetCaptured) v += _initialOffset;
            dst[i] = v;
        }
    }

    void CopyPose(Vector3[] src, Vector3[] dst)
    {
        for (int i = 0; i < 21; i++)
            dst[i] = src[i];
    }

    void RenderFromBuffer()
    {
        EnsureBufferAllocated();
        if (_bufCount <= 0) return;

        float now = Time.time;
        float keep = Mathf.Max(bufferMaxSeconds, interpolationDelaySeconds + 0.05f);
        PruneOldSamples(now - Mathf.Max(0.01f, keep));

        if (_bufCount <= 0) return;

        float targetTime = now - Mathf.Max(0f, interpolationDelaySeconds);

        int N = _bufTimes.Length;
        int tail = TailIndex(N);

        // If only one sample, use it.
        if (_bufCount == 1)
        {
            CopyPose(_bufPos[tail], _workPos);
            ProcessFrame(_workPos);
            return;
        }

        // Find first sample with time >= targetTime
        int prev = tail;
        int next = tail;
        bool found = false;

        for (int k = 0; k < _bufCount; k++)
        {
            int idx = (tail + k) % N;
            float t = _bufTimes[idx];

            if (t >= targetTime)
            {
                next = idx;
                prev = (k == 0) ? idx : (tail + k - 1) % N;
                found = true;
                break;
            }
        }

        if (!found)
        {
            // target is newer than newest => use newest
            int newest = (tail + _bufCount - 1) % N;
            CopyPose(_bufPos[newest], _workPos);
            ProcessFrame(_workPos);
            return;
        }

        float tPrev = _bufTimes[prev];
        float tNext = _bufTimes[next];

        Vector3[] pPrev = _bufPos[prev];
        Vector3[] pNext = _bufPos[next];

        if (prev == next || Mathf.Abs(tNext - tPrev) < 1e-6f)
        {
            CopyPose(pPrev, _workPos);
        }
        else
        {
            float u = Mathf.InverseLerp(tPrev, tNext, targetTime);
            for (int i = 0; i < 21; i++)
                _workPos[i] = Vector3.Lerp(pPrev[i], pNext[i], u);
        }

        ProcessFrame(_workPos);
    }

    // =========================================================
    // Per-frame processing (remap/gain/smooth/apply/twist/aim/arm)
    // =========================================================
    void ProcessFrame(Vector3[] worldPos)
    {
        if (worldPos == null || worldPos.Length < 21) return;

        _renderFrameId++;
        _lastRenderTime = Time.time;

        // --- REMAP 단계 ---
        if (useJoystickRemap)
            RemapJoystickStyle(worldPos);
        else
            RemapSideToFront(worldPos);

        // --- Translation gain (amplify wrist translation) ---
        if (translationGain != 1.0f && translationGain > 0f)
        {
            Vector3 wrist = worldPos[0];

            if (gainUseNeutralWrist)
            {
                if (!_gainNeutralCaptured)
                {
                    _gainNeutralWristWorld = wrist;
                    _gainNeutralCaptured = true;
                }

                Vector3 d = wrist - _gainNeutralWristWorld;
                Vector3 wristG = _gainNeutralWristWorld + d * translationGain;
                Vector3 delta = wristG - wrist;

                for (int i = 0; i < 21; i++)
                    worldPos[i] += delta;
            }
        }

        // smooth and apply to remote driver joints
        SmoothAndApply(worldPos);

        // doorknob style twist on separate bone
        UpdateTwist();

        // aim target for Wrist_Aim (position only)
        if (computePalmFrame)
            UpdateAimPositionOnly();

        // auto-arm rig on first valid frame
        if (autoArm && !_firstArmed && FrameLooksValid(worldPos))
        {
            if (armer != null) armer.ArmNow();
            _firstArmed = true;
        }
    }

    // =========================================================
    // Smoothing and apply (OneEuro + m/s step clamp + soft dead-zone)
    // =========================================================
    void EnsureOneEuroAllocated()
    {
        if (_oneEuro == null || _oneEuro.Length != 21)
            _oneEuro = new OneEuroVec3[21];

        for (int i = 0; i < 21; i++)
        {
            if (_oneEuro[i] == null)
                _oneEuro[i] = new OneEuroVec3();
        }
    }

    float StableDt()
    {
        // Avoid dt=0 and avoid huge dt on hitches.
        return Mathf.Clamp(Time.deltaTime, 1e-4f, 0.05f);
    }

    void ResetSmoothingState(bool fullResetRemapAndGain)
    {
        _havePrevPos = false;
        _haveWristPrev = false;
        _aimDirSm = Vector3.zero;

        // reset fast/raw readiness
        fastTipsReady = false;
        rawTipsReady = false;

        if (_oneEuro != null)
        {
            for (int i = 0; i < 21; i++)
                _oneEuro[i]?.Invalidate();
        }

        if (fullResetRemapAndGain)
        {
            _gainNeutralCaptured = false;

            _remapNeutralCaptured = false;
            _remapOffsetCamSm = Vector3.zero;

            _joyNeutralCaptured = false;
            _joyOffsetCam = Vector3.zero;
            _joyHasInputPrev = false;
            _joyInXSm = _joyInYSm = _joyInZSm = 0f;
        }
    }

    void SmoothAndApply(Vector3[] inPos)
    {
        float dt = StableDt();

        // Camera-forward direction for "depth" stabilization
        Vector3 camFwd = Vector3.forward;
        if (Camera.main != null)
        {
            camFwd = Camera.main.transform.forward;
            if (camFwd.sqrMagnitude > 1e-8f) camFwd.Normalize();
            else camFwd = Vector3.forward;
        }

        // Update workspace-current pose once per frame (used for output mapping)
        UpdateWorkspaceCurrentFromAnchor();

        // Workspace yaw delta (baseline -> current). Identity if no base.
        Quaternion dyaw = WorkspaceYawDelta();

        // First frame init
        if (!_havePrevPos)
        {
            for (int i = 0; i < 21; i++)
            {
                Vector3 v = inPos[i];
                _prevPos[i] = v;

                if (useOneEuro && _oneEuro != null && _oneEuro[i] != null)
                    _oneEuro[i].Reset(v);

                if (remoteByIndex[i] != null)
                {
                    // Capture base driver rotations once (baseline pose)
                    if (!_haveDriverBaseRot)
                        _driverBaseRot[i] = remoteByIndex[i].rotation;

                    // Apply workspace offset ONLY to output pose
                    Vector3 outV = ApplyWorkspaceOffsetToWorldPos(v);
                    remoteByIndex[i].position = outV;

                    // Apply workspace yaw delta to driver joint rotation (visual alignment)
                    if (_haveWorkspaceBase)
                        remoteByIndex[i].rotation = dyaw * _driverBaseRot[i];
                }
            }

            _haveDriverBaseRot = true;

            // update filtered tips from applied pose (stable) — use OUTPUT frame
            thumbTipFast = ApplyWorkspaceOffsetToWorldPos(_prevPos[4]);
            indexTipFast = ApplyWorkspaceOffsetToWorldPos(_prevPos[8]);
            middleTipFast = ApplyWorkspaceOffsetToWorldPos(_prevPos[12]);
            fastTipsReady = true;

            _havePrevPos = true;
            return;
        }

        // If base rotations were never captured (e.g., remoteByIndex was null at init),
        // try capturing now once.
        if (!_haveDriverBaseRot)
        {
            bool any = false;
            for (int i = 0; i < 21; i++)
            {
                if (remoteByIndex[i] == null) continue;
                _driverBaseRot[i] = remoteByIndex[i].rotation;
                any = true;
            }
            _haveDriverBaseRot = any;
        }

        for (int i = 0; i < 21; i++)
        {
            Vector3 target = inPos[i];

            // -----------------------------------------------------
            // 0) speed-based step clamp (m/s -> m/frame)
            // -----------------------------------------------------
            float maxSpeed = IsTip(i) ? maxSpeedTipsMps : maxSpeedMps;
            float stepCap = Mathf.Max(0f, maxSpeed) * dt;

            Vector3 candidate = target;
            if (stepCap > 0f)
            {
                Vector3 d = candidate - _prevPos[i];
                float mag = d.magnitude;
                if (mag > stepCap && mag > 1e-8f)
                    candidate = _prevPos[i] + d * (stepCap / mag);
            }

            // -----------------------------------------------------
            // 1) optional depth stabilization (camera-forward component)
            // -----------------------------------------------------
            bool applyDepth = stabilizeDepth && (!stabilizeDepthWristOnly || i == 0);
            if (applyDepth)
            {
                Vector3 delta = candidate - _prevPos[i];

                float dDepth = Vector3.Dot(delta, camFwd);
                Vector3 deltaNonDepth = delta - (dDepth * camFwd);
                float nonDepthMag = deltaNonDepth.magnitude;
                float nonDepthSpeed = nonDepthMag / Mathf.Max(1e-4f, dt);

                bool gateDepth = !depthUseGating ? true : false;

                if (depthUseGating)
                {
                    if (nonDepthSpeed >= depthGateMinNonDepthSpeed &&
                        Mathf.Abs(dDepth) <= depthGateDepthToNonDepthRatio * nonDepthMag)
                    {
                        gateDepth = true;
                    }
                    else
                    {
                        gateDepth = false;
                    }
                }

                float dzDepth = gateDepth ? depthDeadZoneMeters : depthDeadZoneMetersFree;

                float maxDepthSpeed = gateDepth ? depthMaxSpeedMps : depthMaxSpeedMpsFree;
                float maxDepthStep = Mathf.Max(0f, maxDepthSpeed) * dt;

                float depthCutoffLocalHz = gateDepth ? depthCutoffHz : depthCutoffHzFree;
                depthCutoffLocalHz = Mathf.Max(0.01f, depthCutoffLocalHz);

                float depthOmega = 2f * Mathf.PI * depthCutoffLocalHz;
                float depthAlpha = (depthOmega * dt) / (1f + depthOmega * dt);

                // soft dead-zone on depth component (still hard threshold, but depth only)
                if (dzDepth > 0f && Mathf.Abs(dDepth) < dzDepth)
                    dDepth = 0f;

                // depth step clamp (speed-based)
                if (maxDepthStep > 0f)
                    dDepth = Mathf.Clamp(dDepth, -maxDepthStep, maxDepthStep);

                // depth delta attenuation (acts like low-pass on delta)
                float dDepthSm = Mathf.Lerp(0f, dDepth, Mathf.Clamp01(depthAlpha));

                candidate = _prevPos[i] + deltaNonDepth + (dDepthSm * camFwd);
            }

            // -----------------------------------------------------
            // 2) One Euro filter (adaptive smoothing)
            // -----------------------------------------------------
            Vector3 filtered;

            if (useOneEuro && _oneEuro != null && _oneEuro[i] != null)
            {
                float minCutoff = oneEuroMinCutoffHz;
                float beta = oneEuroBeta;

                if (i == 0)
                {
                    minCutoff = oneEuroMinCutoffWristHz;
                    beta = oneEuroBetaWrist;
                }
                else if (IsTip(i))
                {
                    minCutoff = oneEuroMinCutoffTipsHz;
                    beta = oneEuroBetaTips;
                }

                filtered = _oneEuro[i].Filter(
                    candidate,
                    dt,
                    minCutoff,
                    beta,
                    oneEuroDerivativeCutoffHz
                );
            }
            else
            {
                // fallback: simple LPF
                float baseCutoffHz = IsTip(i) ? cutoffHzTips : cutoffHz;
                baseCutoffHz = Mathf.Max(0.01f, baseCutoffHz);
                float omega = 2f * Mathf.PI * baseCutoffHz;
                float alpha = (omega * dt) / (1f + omega * dt);
                alpha = IsTip(i) ? Mathf.Clamp(alpha, 0.02f, 0.30f) : Mathf.Clamp01(alpha);

                filtered = Vector3.Lerp(_prevPos[i], candidate, alpha);
            }

            // -----------------------------------------------------
            // 3) soft dead-zone (replace old hard dead-zone)
            // -----------------------------------------------------
            if (useJitterDeadZone)
            {
                float dz = Mathf.Max(0f, jitterDeadZoneMeters);
                if (dz > 0f)
                {
                    Vector3 d = filtered - _prevPos[i];
                    float mag = d.magnitude;

                    if (mag > 1e-8f && mag < dz)
                    {
                        float t = Mathf.Clamp01(mag / dz);
                        // Smoothstep: 0->0, 1->1, continuous derivative
                        float s = t * t * (3f - 2f * t);
                        filtered = _prevPos[i] + d * s;
                    }
                }
            }

            // Apply workspace offset ONLY to output pose (visual hand),
            // but keep filter state in original world space for stability.
            Vector3 outPos = ApplyWorkspaceOffsetToWorldPos(filtered);

            if (remoteByIndex[i] != null)
            {
                remoteByIndex[i].position = outPos;

                // Apply workspace yaw delta to driver joint rotation (visual alignment)
                if (_haveWorkspaceBase && _haveDriverBaseRot)
                    remoteByIndex[i].rotation = dyaw * _driverBaseRot[i];
            }

            _prevPos[i] = filtered;
        }

        // update filtered tips from applied pose (stable) — use OUTPUT frame
        thumbTipFast = ApplyWorkspaceOffsetToWorldPos(_prevPos[4]);
        indexTipFast = ApplyWorkspaceOffsetToWorldPos(_prevPos[8]);
        middleTipFast = ApplyWorkspaceOffsetToWorldPos(_prevPos[12]);
        fastTipsReady = true;
    }

    // =========================================================
    // Aim target (position only, no roll / no up)
    // =========================================================
    void UpdateAimPositionOnly()
    {
        Transform tWrist = remoteByIndex[0];
        if (tWrist == null) return;

        Vector3 w = tWrist.position;
        Vector3 dir;

        // 1) choose direction source
        if (aimFromVelocity)
        {
            if (!_haveWristPrev)
            {
                _wristPrev = w;
                _haveWristPrev = true;
            }

            float dt = Mathf.Max(1e-4f, Time.deltaTime);
            Vector3 v = (w - _wristPrev) / dt;
            _wristPrev = w;

            if (v.magnitude >= Mathf.Max(1e-4f, aimVelMinSpeed))
            {
                dir = v;
            }
            else
            {
                Transform tMid = remoteByIndex[9]; // MIDDLE_MCP
                if (tMid == null) return;
                dir = tMid.position - w;
            }
        }
        else
        {
            Transform tMid = remoteByIndex[9]; // MIDDLE_MCP
            if (tMid == null) return;
            dir = tMid.position - w;
        }

        if (dir.sqrMagnitude < 1e-8f) return;

        // 2) smooth direction
        Vector3 dN = dir.normalized;
        if (_aimDirSm == Vector3.zero)
            _aimDirSm = dN;
        else
            _aimDirSm = Vector3.Slerp(_aimDirSm, dN, Mathf.Clamp01(aimDirLerp));

        // 3) place aim target position only
        float dist = Mathf.Max(0.01f, palmAimDistance);
        Vector3 aimPos = w + _aimDirSm * dist;

        if (palmFwd != null)
            palmFwd.position = aimPos;

        // UpType is None in the constraint, so rotation is ignored.
        if (palmUp != null)
            palmUp.position = w + Vector3.up * dist;
    }

    // =========================================================
    // Door knob twist on a separate wristTwist bone
    // =========================================================
    void UpdateTwist()
    {
        if (wristTwist == null) return;
        if (remoteByIndex == null || remoteByIndex.Length < 10) return;

        Transform tWrist = remoteByIndex[0]; // WRIST
        Transform tMid = remoteByIndex[9];   // MIDDLE_MCP
        Transform tIdx = remoteByIndex[5];   // INDEX_MCP

        if (tWrist == null || tMid == null || tIdx == null) return;

        Vector3 wristPos = tWrist.position;
        Vector3 mcpMPos = tMid.position;
        Vector3 mcpIPos = tIdx.position;

        // 1) twist axis: roughly wrist -> middle MCP
        Vector3 axis = mcpMPos - wristPos;
        if (axis.sqrMagnitude < 1e-8f) return;
        axis.Normalize();

        // 2) reference vector in palm plane: middle -> index
        Vector3 refVec = mcpIPos - mcpMPos;
        if (refVec.sqrMagnitude < 1e-8f) return;
        refVec = Vector3.ProjectOnPlane(refVec, axis);
        if (refVec.sqrMagnitude < 1e-8f) return;
        refVec.Normalize();

        // capture base local rotation once
        if (!_twistBaseCaptured)
        {
            _twistBaseLocalRot = wristTwist.localRotation;
            _twistBaseCaptured = true;
        }

        // capture neutral once
        if (!_twistNeutralReady)
        {
            _twistAxisNeutral = axis;
            _twistRefNeutral = refVec;
            _twistNeutralReady = true;
            _twistSmoothedDeg = 0f;
            wristTwist.localRotation = _twistBaseLocalRot;
            return;
        }

        // 3) roll around neutral axis
        Vector3 refNow = Vector3.ProjectOnPlane(refVec, _twistAxisNeutral);
        Vector3 ref0 = Vector3.ProjectOnPlane(_twistRefNeutral, _twistAxisNeutral);

        if (refNow.sqrMagnitude < 1e-8f || ref0.sqrMagnitude < 1e-8f)
            return;

        refNow.Normalize();
        ref0.Normalize();

        float rawRollDeg = Vector3.SignedAngle(ref0, refNow, _twistAxisNeutral);

        // 4) clamp + dead zone
        rawRollDeg = Mathf.Clamp(rawRollDeg, -twistMaxAbsDeg, twistMaxAbsDeg);
        if (Mathf.Abs(rawRollDeg) < twistDeadZoneDeg)
            rawRollDeg = 0f;

        // 5) limit speed
        float dt = StableDt();
        float maxStep = twistMaxDegPerSec * dt;
        float delta = rawRollDeg - _twistSmoothedDeg;

        if (delta > maxStep) delta = maxStep;
        else if (delta < -maxStep) delta = -maxStep;

        float targetDeg = _twistSmoothedDeg + delta;

        // 6) extra low-pass
        float s = 1f - Mathf.Pow(1f - Mathf.Clamp01(twistLerp), dt * 60f);
        _twistSmoothedDeg = Mathf.Lerp(_twistSmoothedDeg, targetDeg, s);

        // 7) apply around local forward axis (Z)
        float finalDeg = twistInvertSign ? -_twistSmoothedDeg : _twistSmoothedDeg;
        Quaternion twistRot = Quaternion.AngleAxis(finalDeg, Vector3.forward);
        wristTwist.localRotation = _twistBaseLocalRot * twistRot;
    }

    // =========================================================
    // Side-to-front remap
    // =========================================================
    void RemapSideToFront(Vector3[] joints)
    {
        if (!enableSideToFrontRemap) return;
        if (joints == null || joints.Length < 1) return;
        if (Camera.main == null) return;

        Transform cam = Camera.main.transform;

        Vector3 wristWorld = joints[0];
        Vector3 wristCam = cam.InverseTransformPoint(wristWorld);

        // 1) neutral capture
        if (!_remapNeutralCaptured)
        {
            _remapNeutralCam = wristCam;
            _remapNeutralWorld = wristWorld;
            _remapOffsetCamSm = Vector3.zero;
            _remapNeutralCaptured = true;
            return;
        }

        // 2) camera local delta
        Vector3 dCam = wristCam - _remapNeutralCam;
        Vector3 dWorld = wristWorld - _remapNeutralWorld;

        float xOff = dWorld.x * remapXScale;

        float yOff = dCam.z * remapYFromZScale;
        if (remapInvertYFromZ) yOff = -yOff;

        float zOff = dCam.y * remapZFromYScale;
        if (remapInvertZFromY) zOff = -zOff;

        Vector3 targetOffsetCam = new Vector3(xOff, yOff, zOff);

        // 4) workspace clamp
        if (remapMaxOffsetCam.x > 0f)
            targetOffsetCam.x = Mathf.Clamp(targetOffsetCam.x, -remapMaxOffsetCam.x, remapMaxOffsetCam.x);
        if (remapMaxOffsetCam.y > 0f)
            targetOffsetCam.y = Mathf.Clamp(targetOffsetCam.y, -remapMaxOffsetCam.y, remapMaxOffsetCam.y);
        if (remapMaxOffsetCam.z > 0f)
            targetOffsetCam.z = Mathf.Clamp(targetOffsetCam.z, -remapMaxOffsetCam.z, remapMaxOffsetCam.z);

        // 5) dead-zone (remap offset)
        float dz = Mathf.Max(0f, remapDeadZoneMeters);
        if (dz > 0f)
        {
            Vector3 diff = targetOffsetCam - _remapOffsetCamSm;
            if (diff.sqrMagnitude < dz * dz)
            {
                targetOffsetCam = _remapOffsetCamSm;
            }
        }

        // 6) smoothing + step clamp (camera space)
        float dt = StableDt();
        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(remapLerp), dt * 60f);
        Vector3 candidate = Vector3.Lerp(_remapOffsetCamSm, targetOffsetCam, k);

        float maxStep = Mathf.Max(0f, remapMaxStepMeters);
        if (maxStep > 0f)
        {
            Vector3 step = candidate - _remapOffsetCamSm;
            float stepMag = step.magnitude;
            if (stepMag > maxStep && stepMag > 1e-8f)
                candidate = _remapOffsetCamSm + step * (maxStep / stepMag);
        }

        _remapOffsetCamSm = candidate;

        // 7) back to world -> apply delta to all joints
        Vector3 newWristWorld = _remapNeutralWorld + cam.TransformVector(_remapOffsetCamSm);
        Vector3 deltaWorld = newWristWorld - wristWorld;

        for (int i = 0; i < joints.Length; i++)
            joints[i] += deltaWorld;
    }

    // =========================================================
    // Joystick-style remap helpers
    // =========================================================
    float JoyAxisInput(float delta, float halfSize)
    {
        float hs = Mathf.Abs(halfSize);
        if (hs <= 1e-5f)
            return 0f;

        float absD = Mathf.Abs(delta);

        if (absD <= hs)
            return 0f;

        float over = absD - hs;
        float range = hs;
        float t = Mathf.Clamp01(over / range);

        if (joyExpo > 0.0f && Mathf.Abs(joyExpo - 1.0f) > 1e-3f)
            t = Mathf.Pow(t, joyExpo);

        return Mathf.Sign(delta) * t;
    }

    void RemapJoystickStyle(Vector3[] joints)
    {
        if (!useJoystickRemap) return;
        if (joints == null || joints.Length < 1) return;
        if (Camera.main == null) return;

        Transform cam = Camera.main.transform;
        float dt = StableDt();

        Vector3 wristWorld = joints[0];

        if (!_joyNeutralCaptured)
        {
            _joyNeutralWorld = wristWorld;
            _joyOffsetCam = Vector3.zero;
            _joyNeutralCaptured = true;
            return;
        }

        Vector3 dWorld = wristWorld - _joyNeutralWorld;

        float rawInX = JoyAxisInput(dWorld.x, joyBoxHalfSizeWorld.x);
        float rawInY = JoyAxisInput(dWorld.y, joyBoxHalfSizeWorld.y);
        float rawInZ = JoyAxisInput(dWorld.z, joyBoxHalfSizeWorld.z);

        // input smoothing
        if (!_joyHasInputPrev)
        {
            _joyInXSm = rawInX;
            _joyInYSm = rawInY;
            _joyInZSm = rawInZ;
            _joyHasInputPrev = true;
        }

        float kIn = 1f - Mathf.Pow(1f - Mathf.Clamp01(joyInputLerp), dt * 60f);
        _joyInXSm = Mathf.Lerp(_joyInXSm, rawInX, kIn);
        _joyInYSm = Mathf.Lerp(_joyInYSm, rawInY, kIn);
        _joyInZSm = Mathf.Lerp(_joyInZSm, rawInZ, kIn);

        float inX = _joyInXSm;
        float inY = _joyInYSm;
        float inZ = _joyInZSm;

        float vxCam = inX * joyMaxSpeedX;
        float vyCam = inZ * joyMaxSpeedY;
        float vzCam = inY * joyMaxSpeedZ;

        if (joyInvertX) vxCam = -vxCam;
        if (joyInvertYFromZ) vyCam = -vyCam;
        if (joyInvertZFromY) vzCam = -vzCam;

        Vector3 velCam = new Vector3(vxCam, vyCam, vzCam);

        _joyOffsetCam += velCam * dt;

        // workspace clamp
        if (joyMaxOffsetCam.x > 0f)
            _joyOffsetCam.x = Mathf.Clamp(_joyOffsetCam.x, -joyMaxOffsetCam.x, joyMaxOffsetCam.x);
        if (joyMaxOffsetCam.y > 0f)
            _joyOffsetCam.y = Mathf.Clamp(_joyOffsetCam.y, -joyMaxOffsetCam.y, joyMaxOffsetCam.y);
        if (joyMaxOffsetCam.z > 0f)
            _joyOffsetCam.z = Mathf.Clamp(_joyOffsetCam.z, -joyMaxOffsetCam.z, joyMaxOffsetCam.z);

        Vector3 offsetWorld = cam.TransformVector(_joyOffsetCam);
        Vector3 newWristWorld = _joyNeutralWorld + offsetWorld;

        Vector3 deltaWorld = newWristWorld - wristWorld;
        for (int i = 0; i < joints.Length; i++)
            joints[i] += deltaWorld;
    }

    // =========================================================
    // Helpers
    // =========================================================
    bool FrameLooksValid(Vector3[] pos)
    {
        if (pos == null || pos.Length < 10) return false;
        float d1 = Vector3.Distance(pos[0], pos[5]);  // WRIST to INDEX_MCP
        float d2 = Vector3.Distance(pos[0], pos[9]);  // WRIST to MIDDLE_MCP
        return (d1 > firstValidDistance && d2 > firstValidDistance);
    }

    bool IsTip(int i)
    {
        return (i == 4 || i == 8 || i == 12 || i == 16 || i == 20);
    }

    // =========================================================
    // Context menus
    // =========================================================
    [ContextMenu("Offset / Clear and re-arm")]
    public void ContextClearAndRearm()
    {
        _offsetCaptured = false;
        _initialOffset = Vector3.zero;
        _firstArmed = false;

        _gainNeutralCaptured = false;

        ResetSmoothingState(fullResetRemapAndGain: true);
        ClearInterpolationBuffer();
    }

    [ContextMenu("Offset / Recapture now (use last pre-offset wrist)")]
    public void ContextRecaptureNow()
    {
        if (rWrist == null)
        {
            Debug.Log("[RemoteHandRuntime] rWrist is null, cannot recapture offset.");
            return;
        }

        _initialOffset = rWrist.position - _lastPreOffsetWrist;
        _offsetCaptured = true;

        _gainNeutralCaptured = false;

        ResetSmoothingState(fullResetRemapAndGain: true);
        ClearInterpolationBuffer();
    }

    [ContextMenu("Preset/Apply Selected Preset Now")]
    public void ContextApplySelectedPresetNow()
    {
        if (smoothingPreset == SmoothingPreset.Custom) return;
        ApplyPreset(smoothingPreset, resetStateOnPresetApply, clearBufferOnPresetApply);
    }

    public void ApplyPreset(SmoothingPreset preset, bool resetState = true, bool clearBuffer = true)
    {
        if (_applyingPresetGuard) return;
        _applyingPresetGuard = true;

        // 공통: 프리셋은 기본적으로 '부드럽게' 목표라서 기능 ON
        useInterpolationBuffer = true;
        useJitterDeadZone = true;
        stabilizeDepth = true;
        stabilizeDepthWristOnly = true;
        depthUseGating = true;

        // A 버전 기준: OneEuro도 프리셋이면 ON
        useOneEuro = true;

        switch (preset)
        {
            case SmoothingPreset.Balanced:
            {
                // --- Interpolation buffer ---
                interpolationDelaySeconds = 0.06f;
                bufferMaxSeconds = 0.60f;
                bufferMaxSamples = 120;

                // --- One Euro ---
                oneEuroDerivativeCutoffHz = 5.0f;
                oneEuroMinCutoffWristHz = 1.5f;
                oneEuroBetaWrist = 0.70f;
                oneEuroMinCutoffHz = 1.6f;
                oneEuroBeta = 0.50f;
                oneEuroMinCutoffTipsHz = 0.9f;
                oneEuroBetaTips = 0.35f;

                // --- Speed clamp (m/s) ---
                maxSpeedMps = 4.8f;
                maxSpeedTipsMps = 0.90f;

                // --- Soft dead-zone ---
                jitterDeadZoneMeters = 0.0020f;

                // --- Depth stabilization ---
                depthDeadZoneMeters = 0.010f;
                depthMaxSpeedMps = 0.30f;
                depthCutoffHz = 2.0f;

                depthGateMinNonDepthSpeed = 0.24f;
                depthGateDepthToNonDepthRatio = 0.35f;

                depthMaxSpeedMpsFree = 2.4f;
                depthDeadZoneMetersFree = 0.0f;
                depthCutoffHzFree = 10.0f;

                // --- Aim (값이 작을수록 더 부드러움) ---
                aimDirLerp = 0.20f;
                break;
            }

            case SmoothingPreset.Smooth:
            {
                interpolationDelaySeconds = 0.09f;
                bufferMaxSeconds = 0.75f;
                bufferMaxSamples = 120;

                oneEuroDerivativeCutoffHz = 3.0f;
                oneEuroMinCutoffWristHz = 1.0f;
                oneEuroBetaWrist = 0.45f;
                oneEuroMinCutoffHz = 1.2f;
                oneEuroBeta = 0.30f;
                oneEuroMinCutoffTipsHz = 0.6f;
                oneEuroBetaTips = 0.22f;

                maxSpeedMps = 3.5f;
                maxSpeedTipsMps = 0.70f;

                jitterDeadZoneMeters = 0.0025f;

                depthDeadZoneMeters = 0.010f;
                depthMaxSpeedMps = 0.22f;
                depthCutoffHz = 1.5f;

                depthGateMinNonDepthSpeed = 0.14f;
                depthGateDepthToNonDepthRatio = 0.45f;

                depthMaxSpeedMpsFree = 2.0f;
                depthDeadZoneMetersFree = 0.0f;
                depthCutoffHzFree = 8.0f;

                aimDirLerp = 0.12f;
                break;
            }

            case SmoothingPreset.VerySmooth:
            {
                interpolationDelaySeconds = 0.12f;
                bufferMaxSeconds = 1.0f;
                bufferMaxSamples = 150;

                oneEuroDerivativeCutoffHz = 2.0f;
                oneEuroMinCutoffWristHz = 0.7f;
                oneEuroBetaWrist = 0.30f;
                oneEuroMinCutoffHz = 0.8f;
                oneEuroBeta = 0.20f;
                oneEuroMinCutoffTipsHz = 0.4f;
                oneEuroBetaTips = 0.15f;

                maxSpeedMps = 3.0f;
                maxSpeedTipsMps = 0.55f;

                jitterDeadZoneMeters = 0.0030f;

                depthDeadZoneMeters = 0.010f;
                depthMaxSpeedMps = 0.18f;
                depthCutoffHz = 1.0f;

                depthGateMinNonDepthSpeed = 0.10f;
                depthGateDepthToNonDepthRatio = 0.50f;

                depthMaxSpeedMpsFree = 1.6f;
                depthDeadZoneMetersFree = 0.0f;
                depthCutoffHzFree = 6.0f;

                aimDirLerp = 0.08f;
                break;
            }

            case SmoothingPreset.Custom:
            default:
                break;
        }

        // 안정장치: 버퍼 유지시간이 delay보다 짧으면 이상해지므로 보정
        bufferMaxSeconds = Mathf.Max(bufferMaxSeconds, interpolationDelaySeconds + 0.05f);

        // bufferMaxSamples 변경됐을 수 있으니 재할당 체크
        EnsureBufferAllocated();

        if (clearBuffer)
            ClearInterpolationBuffer();

        if (resetState)
            ResetSmoothingState(fullResetRemapAndGain: false);

        _applyingPresetGuard = false;
    }

    public void SetSideToFrontRemap(bool enable)
    {
        if (enableSideToFrontRemap == enable)
            return; // 이미 같은 상태면 아무 것도 안 함

        enableSideToFrontRemap = enable;

        // 켤 때는 반드시 neutral을 다시 잡아야 튐이 없음
        if (enable)
        {
            _remapNeutralCaptured = false;
            _remapOffsetCamSm = Vector3.zero;
        }

        // 상태가 바뀌었으므로 안정성 확보
        _gainNeutralCaptured = false;
        ClearInterpolationBuffer();
        ResetSmoothingState(fullResetRemapAndGain: false);
    }

    public void RebaseWorkspaceOffsetAfterAnchorJump()
    {
        if (workspaceOffsetAnchor == null) return;

        // baseline을 현재 anchor로 다시 잡아 Δ=0으로 만든다
        CaptureWorkspaceBaseFromCurrentAnchor();
        UpdateWorkspaceCurrentFromAnchor();

        // 다음 프레임에 드라이버 회전도 새 기준으로 다시 캡처
        _haveDriverBaseRot = false;

        // 스무딩 상태도 리셋해서 sudden jump를 흡수
        _havePrevPos = false;

        // (선택) 버퍼를 쓰면 같이 비우는 게 안전
        ClearInterpolationBuffer();
    }
}
