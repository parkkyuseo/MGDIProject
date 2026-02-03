using UnityEngine;

public class RemoteHandRuntime : MonoBehaviour
{
    // =========================================================
    // Remote driver joints
    // =========================================================
    [Header("Remote driver joints (21)")]
    public Transform[] remoteByIndex = new Transform[21]; // 0..20 (WRIST..PINKY_TIP)

    // =========================================================
    // Wrist aim target (position only)
    // =========================================================
    [Header("Wrist aim target (position only)")]
    public Transform palmFwd; // Remote_PALM_FWD
    public Transform palmUp;  // Remote_PALM_UP
    [Tooltip("Distance from wrist to aim target in meters.")]
    public float palmAimDistance = 0.08f;

    // =========================================================
    // Options
    // =========================================================
    [Header("Options")]
    [Tooltip("True for left hand, false for right hand. (Optional)")]
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
    public float maxSpeedMps = 4.8f;

    [Tooltip("Max joint speed for tip joints (meters/second).")]
    public float maxSpeedTipsMps = 0.9f;

    [Header("Micro-jitter suppression (soft dead-zone)")]
    [Tooltip("If true, very small moves are softened (NOT hard-clamped).")]
    public bool useJitterDeadZone = true;

    [Tooltip("Soft dead-zone radius (meters). Smaller deltas are attenuated smoothly.")]
    public float jitterDeadZoneMeters = 0.003f;

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
    [Tooltip("If true, palmFwd/palmUp are updated every frame to follow the hand direction.")]
    public bool computePalmFrame = true;

    [Range(0f, 1f)]
    [Tooltip("Slerp factor for palm forward direction (higher = more responsive).")]
    public float aimDirLerp = 0.20f;

    [Range(0f, 1f)]
    [Tooltip("Slerp factor for palm up direction (higher = more responsive).")]
    public float aimUpLerp = 0.20f;

    [Tooltip("If true, flips palm forward direction (useful when side condition makes hand face backwards).")]
    public bool invertPalmForward = false;

    [Tooltip("Extra yaw offset applied to the AIM direction ONLY (degrees).")]
    public float visualYawOffsetDeg = 0f;

    // =========================================================
    // Translation gain
    // =========================================================
    [Header("Translation gain (optional)")]
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
    public float depthDeadZoneMeters = 0.010f;

    [Tooltip("Max depth speed when gated (meters/second).")]
    public float depthMaxSpeedMps = 0.30f;

    [Tooltip("Extra low-pass cutoff for depth delta when gated (Hz). Lower = smoother, more lag.")]
    public float depthCutoffHz = 2.0f;

    [Header("Depth stabilization gating (wrist intent detection)")]
    [Tooltip("If true, depth stabilization is applied only when lateral motion dominates.")]
    public bool depthUseGating = true;

    [Tooltip("Minimum non-depth speed to consider as 'lateral motion' (m/s).")]
    public float depthGateMinNonDepthSpeed = 0.24f;

    [Tooltip("If |dDepth| <= ratio * nonDepthMag, treat depth as noise and stabilize it.")]
    public float depthGateDepthToNonDepthRatio = 0.35f;

    [Tooltip("Max depth speed when NOT gated (user intends forward/back) (m/s).")]
    public float depthMaxSpeedMpsFree = 2.4f;

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
    // Side-to-front remap (absolute axis permutation, neutral-based)
    // =========================================================
    [Header("Side-to-front remap (absolute axis permutation)")]
    [Tooltip("Enable remap for side condition. Uses neutral-based absolute axis permutation.")]
    public bool enableSideToFrontRemap = false;

    [Tooltip("Basis frame for axis permutation. If null, uses Camera.main.")]
    public Transform axisPermFrame;

    [Tooltip("Gain applied to neutral-based displacement after permutation. 1 = no gain.")]
    public float axisPermGain = 1.0f;

    [Tooltip("Dead-zone radius (meters) applied to NEUTRAL displacement in frame-local space.")]
    public float axisPermDeadZoneMeters = 0.0f;

    [Tooltip("Max absolute workspace offset in remap-frame local space. (<=0 means no clamp for that axis)")]
    public Vector3 axisPermMaxOffsetLocal = new Vector3(0.30f, 0.30f, 0.30f);

    // =========================================================
    // Joystick-style remap (experimental, kept for later)
    // =========================================================
    [Header("Joystick-style remap (experimental)")]
    [Tooltip("If true, use joystick-style remap instead of axis-permutation remap.")]
    public bool useJoystickRemap = false;

    [Tooltip("Half-size of the dead-zone box near your real hand (meters, world space).")]
    public Vector3 joyBoxHalfSizeWorld = new Vector3(0.10f, 0.10f, 0.10f);

    [Tooltip("Max proxy speed along camera-right (X), camera-up (Y), camera-forward (Z) in m/s.")]
    public float joyMaxSpeedX = 0.4f;
    public float joyMaxSpeedY = 0.4f;
    public float joyMaxSpeedZ = 0.4f;

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

    [Range(0f, 1f)]
    [Tooltip("Extra smoothing for joystick inputs (0 = no extra smoothing, 1 = very slow).")]
    public float joyInputLerp = 0.3f;

    Vector3 _joyOffsetCam = Vector3.zero;
    Vector3 _joyNeutralWorld = Vector3.zero;
    bool _joyNeutralCaptured = false;

    float _joyInXSm = 0f;
    float _joyInYSm = 0f;
    float _joyInZSm = 0f;
    bool _joyHasInputPrev = false;

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

    // aim state (forward + up continuity)
    Vector3 _aimDirSm = Vector3.zero;
    Vector3 _palmUpPrev = Vector3.up;
    bool _palmUpHavePrev = false;

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
    // Side-to-front remap (absolute) state
    // =========================================================
    bool _axisPermNeutralCaptured = false;
    Vector3 _axisPermNeutralWristLocal = Vector3.zero;
    Vector3 _axisPermNeutralWristWorld = Vector3.zero;

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

            Vector3 dx = (x - _xHat) / dt;

            float aD = Alpha(dCutoffHz, dt);
            _dxHat = Vector3.Lerp(_dxHat, dx, Mathf.Clamp01(aD));

            float cutoff = Mathf.Max(0.0001f, minCutoffHz + beta * _dxHat.magnitude);

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

        ApplyPreset(smoothingPreset, resetState: false, clearBuffer: false);
    }
#endif

    void Update()
    {
        if (manualTestMode) return;

        if (useInterpolationBuffer)
        {
            RenderFromBuffer();
        }
    }

    // =========================================================
    // Entry from UdpHandReceiver
    // =========================================================
    public void ApplyWorldPositions(Vector3[] worldPos)
    {
        if (manualTestMode || worldPos == null || worldPos.Length < 21) return;

        _sampleId++;
        _lastSampleTime = Time.time;

        _lastPreOffsetWrist = worldPos[0];

        // Capture initial offset once so remote hand aligns near proxy rig
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

            // Coordinate frame changed => reset buffer + smoothing + remap neutral
            _axisPermNeutralCaptured = false;
            _joyNeutralCaptured = false;

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
            EnqueueSample(worldPos);
        }
        else
        {
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
            if (addInitialOffset && _offsetCaptured) v += _initialOffset;
            dst[i] = v;
        }

        _bufHead = (_bufHead + 1) % N;
        if (_bufCount < N) _bufCount++;

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

        if (_bufCount == 1)
        {
            CopyPose(_bufPos[tail], _workPos);
            ProcessFrame(_workPos);
            return;
        }

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
        {
            RemapJoystickStyle(worldPos);
        }
        else if (enableSideToFrontRemap)
        {
            RemapAbsoluteAxisPermutation(worldPos);
        }

        // --- Translation gain (optional) ---
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
    // Side-to-front remap: ABSOLUTE axis permutation (neutral-based)
    // Desired mapping (frame-local):
    //   real X (left/right)  -> proxy Y
    //   real Y (up/down)     -> proxy Z
    //   real Z (forward)     -> proxy X
    // => dPerm = (dz, dx, dy)
    // =========================================================
    void RemapAbsoluteAxisPermutation(Vector3[] joints)
    {
        if (joints == null || joints.Length < 21) return;

        Transform frame = axisPermFrame != null ? axisPermFrame :
                          (Camera.main != null ? Camera.main.transform : null);

        if (frame == null) return;

        Vector3 wristWorld = joints[0];
        Vector3 wristLocal = frame.InverseTransformPoint(wristWorld);

        if (!_axisPermNeutralCaptured)
        {
            _axisPermNeutralWristLocal = wristLocal;
            _axisPermNeutralWristWorld = wristWorld;
            _axisPermNeutralCaptured = true;
            // Do not return: apply zero offset this frame to keep smoothing consistent.
        }

        Vector3 dLocal = wristLocal - _axisPermNeutralWristLocal;

        float dz = Mathf.Max(0f, axisPermDeadZoneMeters);
        if (dz > 0f && dLocal.sqrMagnitude < dz * dz)
            dLocal = Vector3.zero;

        Vector3 dPerm = new Vector3(dLocal.z, dLocal.x, dLocal.y);
        dPerm *= Mathf.Max(0f, axisPermGain);

        // per-axis clamp (<=0 means no clamp)
        if (axisPermMaxOffsetLocal.x > 0f)
            dPerm.x = Mathf.Clamp(dPerm.x, -axisPermMaxOffsetLocal.x, axisPermMaxOffsetLocal.x);
        if (axisPermMaxOffsetLocal.y > 0f)
            dPerm.y = Mathf.Clamp(dPerm.y, -axisPermMaxOffsetLocal.y, axisPermMaxOffsetLocal.y);
        if (axisPermMaxOffsetLocal.z > 0f)
            dPerm.z = Mathf.Clamp(dPerm.z, -axisPermMaxOffsetLocal.z, axisPermMaxOffsetLocal.z);

        Vector3 newWristWorld = _axisPermNeutralWristWorld + frame.TransformVector(dPerm);
        Vector3 deltaWorld = newWristWorld - wristWorld;

        for (int i = 0; i < 21; i++)
            joints[i] += deltaWorld;
    }

    // =========================================================
    // Smoothing and apply
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
        return Mathf.Clamp(Time.deltaTime, 1e-4f, 0.05f);
    }

    void ResetSmoothingState(bool fullResetRemapAndGain)
    {
        _havePrevPos = false;

        _aimDirSm = Vector3.zero;
        _palmUpHavePrev = false;
        _palmUpPrev = Vector3.up;

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

            _axisPermNeutralCaptured = false;

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
                    remoteByIndex[i].position = v;
            }

            thumbTipFast = _prevPos[4];
            indexTipFast = _prevPos[8];
            middleTipFast = _prevPos[12];
            fastTipsReady = true;

            _havePrevPos = true;
            return;
        }

        for (int i = 0; i < 21; i++)
        {
            Vector3 target = inPos[i];

            // 0) speed-based step clamp (m/s -> m/frame)
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

            // 1) optional depth stabilization (camera-forward component)
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

                if (dzDepth > 0f && Mathf.Abs(dDepth) < dzDepth)
                    dDepth = 0f;

                if (maxDepthStep > 0f)
                    dDepth = Mathf.Clamp(dDepth, -maxDepthStep, maxDepthStep);

                float dDepthSm = Mathf.Lerp(0f, dDepth, Mathf.Clamp01(depthAlpha));

                candidate = _prevPos[i] + deltaNonDepth + (dDepthSm * camFwd);
            }

            // 2) One Euro filter (adaptive smoothing)
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
                float baseCutoffHz = IsTip(i) ? cutoffHzTips : cutoffHz;
                baseCutoffHz = Mathf.Max(0.01f, baseCutoffHz);
                float omega = 2f * Mathf.PI * baseCutoffHz;
                float alpha = (omega * dt) / (1f + omega * dt);
                alpha = IsTip(i) ? Mathf.Clamp(alpha, 0.02f, 0.30f) : Mathf.Clamp01(alpha);

                filtered = Vector3.Lerp(_prevPos[i], candidate, alpha);
            }

            // 3) soft dead-zone
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
                        float s = t * t * (3f - 2f * t); // smoothstep
                        filtered = _prevPos[i] + d * s;
                    }
                }
            }

            if (remoteByIndex[i] != null)
                remoteByIndex[i].position = filtered;

            _prevPos[i] = filtered;
        }

        thumbTipFast = _prevPos[4];
        indexTipFast = _prevPos[8];
        middleTipFast = _prevPos[12];
        fastTipsReady = true;
    }

    // =========================================================
    // Aim target (position only)
    // - forward: wrist -> middle MCP (optionally inverted)
    // - up: palm normal from cross(idx-w, mid-w), stabilized to avoid flipping
    // =========================================================
    void UpdateAimPositionOnly()
    {
        if (remoteByIndex == null || remoteByIndex.Length < 13) return;

        Transform tWrist = remoteByIndex[0];   // WRIST
        Transform tIdxMcp = remoteByIndex[5];  // INDEX_MCP
        Transform tMidMcp = remoteByIndex[9];  // MIDDLE_MCP

        if (tWrist == null || tIdxMcp == null || tMidMcp == null) return;

        Vector3 w = tWrist.position;
        Vector3 idx = tIdxMcp.position;
        Vector3 mid = tMidMcp.position;

        // forward dir
        Vector3 dir = (mid - w);
        if (dir.sqrMagnitude < 1e-8f) return;
        dir.Normalize();

        if (invertPalmForward) dir = -dir;

        if (_aimDirSm == Vector3.zero) _aimDirSm = dir;
        else _aimDirSm = Vector3.Slerp(_aimDirSm, dir, Mathf.Clamp01(aimDirLerp));

        float dist = Mathf.Max(0.01f, palmAimDistance);

        Quaternion visualYaw = Quaternion.Euler(0f, visualYawOffsetDeg, 0f);
        Vector3 aimDirVis = visualYaw * _aimDirSm;

        if (palmFwd != null)
            palmFwd.position = w + aimDirVis * dist;

        // up dir (palm normal)
        Vector3 a = (idx - w);
        Vector3 b = (mid - w);
        if (a.sqrMagnitude < 1e-8f || b.sqrMagnitude < 1e-8f) return;

        Vector3 n = Vector3.Cross(a, b);
        if (n.sqrMagnitude < 1e-8f) n = Vector3.up;
        n.Normalize();

        // Stabilize flips: keep continuity
        if (!_palmUpHavePrev)
        {
            // initial sign: try to keep it generally aligned with camera up (if available)
            if (Camera.main != null)
            {
                Vector3 camUp = Camera.main.transform.up;
                if (Vector3.Dot(n, camUp) < 0f) n = -n;
            }

            _palmUpPrev = n;
            _palmUpHavePrev = true;
        }
        else
        {
            if (Vector3.Dot(_palmUpPrev, n) < 0f)
                n = -n;

            float k = Mathf.Clamp01(aimUpLerp);
            _palmUpPrev = Vector3.Slerp(_palmUpPrev, n, k);
            n = _palmUpPrev;
        }

        if (palmUp != null)
            palmUp.position = w + n * dist;
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

        Vector3 axis = mcpMPos - wristPos;
        if (axis.sqrMagnitude < 1e-8f) return;
        axis.Normalize();

        Vector3 refVec = mcpIPos - mcpMPos;
        if (refVec.sqrMagnitude < 1e-8f) return;
        refVec = Vector3.ProjectOnPlane(refVec, axis);
        if (refVec.sqrMagnitude < 1e-8f) return;
        refVec.Normalize();

        if (!_twistBaseCaptured)
        {
            _twistBaseLocalRot = wristTwist.localRotation;
            _twistBaseCaptured = true;
        }

        if (!_twistNeutralReady)
        {
            _twistAxisNeutral = axis;
            _twistRefNeutral = refVec;
            _twistNeutralReady = true;
            _twistSmoothedDeg = 0f;
            wristTwist.localRotation = _twistBaseLocalRot;
            return;
        }

        Vector3 refNow = Vector3.ProjectOnPlane(refVec, _twistAxisNeutral);
        Vector3 ref0 = Vector3.ProjectOnPlane(_twistRefNeutral, _twistAxisNeutral);

        if (refNow.sqrMagnitude < 1e-8f || ref0.sqrMagnitude < 1e-8f)
            return;

        refNow.Normalize();
        ref0.Normalize();

        float rawRollDeg = Vector3.SignedAngle(ref0, refNow, _twistAxisNeutral);

        rawRollDeg = Mathf.Clamp(rawRollDeg, -twistMaxAbsDeg, twistMaxAbsDeg);
        if (Mathf.Abs(rawRollDeg) < twistDeadZoneDeg)
            rawRollDeg = 0f;

        float dt = StableDt();
        float maxStep = twistMaxDegPerSec * dt;
        float delta = rawRollDeg - _twistSmoothedDeg;

        if (delta > maxStep) delta = maxStep;
        else if (delta < -maxStep) delta = -maxStep;

        float targetDeg = _twistSmoothedDeg + delta;

        float s = 1f - Mathf.Pow(1f - Mathf.Clamp01(twistLerp), dt * 60f);
        _twistSmoothedDeg = Mathf.Lerp(_twistSmoothedDeg, targetDeg, s);

        float finalDeg = twistInvertSign ? -_twistSmoothedDeg : _twistSmoothedDeg;
        Quaternion twistRot = Quaternion.AngleAxis(finalDeg, Vector3.forward);
        wristTwist.localRotation = _twistBaseLocalRot * twistRot;
    }

    // =========================================================
    // Joystick-style remap (kept as-is for later use)
    // =========================================================
    float JoyAxisInput(float delta, float halfSize)
    {
        float hs = Mathf.Abs(halfSize);
        if (hs <= 1e-5f) return 0f;

        float absD = Mathf.Abs(delta);
        if (absD <= hs) return 0f;

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
            // Do not return; zero-input results in zero delta anyway.
        }

        Vector3 dWorld = wristWorld - _joyNeutralWorld;

        float rawInX = JoyAxisInput(dWorld.x, joyBoxHalfSizeWorld.x);
        float rawInY = JoyAxisInput(dWorld.y, joyBoxHalfSizeWorld.y);
        float rawInZ = JoyAxisInput(dWorld.z, joyBoxHalfSizeWorld.z);

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
        _axisPermNeutralCaptured = false;

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
        _axisPermNeutralCaptured = false;

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

        useInterpolationBuffer = true;
        useJitterDeadZone = true;
        stabilizeDepth = true;
        stabilizeDepthWristOnly = true;
        depthUseGating = true;
        useOneEuro = true;

        switch (preset)
        {
            case SmoothingPreset.Balanced:
            {
                interpolationDelaySeconds = 0.06f;
                bufferMaxSeconds = 0.60f;
                bufferMaxSamples = 120;

                oneEuroDerivativeCutoffHz = 5.0f;
                oneEuroMinCutoffWristHz = 1.5f;
                oneEuroBetaWrist = 0.70f;
                oneEuroMinCutoffHz = 1.6f;
                oneEuroBeta = 0.50f;
                oneEuroMinCutoffTipsHz = 0.9f;
                oneEuroBetaTips = 0.35f;

                maxSpeedMps = 4.8f;
                maxSpeedTipsMps = 0.90f;

                jitterDeadZoneMeters = 0.0020f;

                depthDeadZoneMeters = 0.010f;
                depthMaxSpeedMps = 0.30f;
                depthCutoffHz = 2.0f;

                depthGateMinNonDepthSpeed = 0.24f;
                depthGateDepthToNonDepthRatio = 0.35f;

                depthMaxSpeedMpsFree = 2.4f;
                depthDeadZoneMetersFree = 0.0f;
                depthCutoffHzFree = 10.0f;

                aimDirLerp = 0.20f;
                aimUpLerp = 0.20f;
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
                aimUpLerp = 0.12f;
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
                aimUpLerp = 0.08f;
                break;
            }

            case SmoothingPreset.Custom:
            default:
                break;
        }

        bufferMaxSeconds = Mathf.Max(bufferMaxSeconds, interpolationDelaySeconds + 0.05f);
        EnsureBufferAllocated();

        if (clearBuffer)
            ClearInterpolationBuffer();

        if (resetState)
            ResetSmoothingState(fullResetRemapAndGain: false);

        _applyingPresetGuard = false;
    }

    // =========================================================
    // External control methods
    // =========================================================
    public void SetSideToFrontRemap(bool enable)
    {
        if (enableSideToFrontRemap == enable)
            return;

        enableSideToFrontRemap = enable;

        // ✅ 핵심: 토글 시 neutral을 리셋해서 "현재 손 위치"가 새 기준이 되게 함
        _axisPermNeutralCaptured = false;

        // Optional: gain neutral도 리셋 (조건 전환 시 튐 방지)
        _gainNeutralCaptured = false;

        // Stability: clear buffer + reset smoothing
        ClearInterpolationBuffer();
        ResetSmoothingState(fullResetRemapAndGain: false);
    }

    public void RecenterRemapNeutralNow()
    {
        // Next frame will capture current wrist as neutral
        _axisPermNeutralCaptured = false;
    }
}
