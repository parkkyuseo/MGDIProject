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
    // Smoothing (One Euro + speed-based clamp + soft deadzone)
    // =========================================================
    [Header("Smoothing for joint positions (improved)")]

    [Tooltip("If true, use One Euro filter (adaptive low-pass) for joint positions.")]
    public bool useOneEuroFilter = true;

    [Tooltip("One Euro min cutoff for non-tip joints (Hz). Lower = smoother when still, more lag.")]
    public float oneEuroMinCutoffHz = 3.0f;

    [Tooltip("One Euro beta for non-tip joints. Higher = less lag during fast motion, but can pass more jitter.")]
    public float oneEuroBeta = 0.25f;

    [Tooltip("One Euro min cutoff for tip joints (Hz).")]
    public float oneEuroMinCutoffHzTips = 2.0f;

    [Tooltip("One Euro beta for tip joints.")]
    public float oneEuroBetaTips = 0.15f;

    [Tooltip("One Euro derivative cutoff (Hz). Usually ~1Hz.")]
    public float oneEuroDerivCutoffHz = 1.0f;

    // --- Legacy fixed LPF params (used only if useOneEuroFilter == false) ---
    [Tooltip("Legacy fixed low-pass cutoff for non-tip joints (Hz). Used only when One Euro is OFF.")]
    public float cutoffHz = 10f;

    [Tooltip("Legacy fixed low-pass cutoff for tip joints (Hz). Used only when One Euro is OFF.")]
    public float cutoffHzTips = 6f;

    // --- Step clamp (converted to speed using reference fps) ---
    [Tooltip("Reference FPS used to convert maxStepMeters/maxStepTips (meters per frame) into a per-second speed cap. " +
             "At dt=1/referenceFPS, behavior matches the old per-frame step clamp.")]
    public float stepClampReferenceFps = 90f;

    [Tooltip("Max position step per frame (at reference FPS) for non-tip joints (meters).")]
    public float maxStepMeters = 0.08f;

    [Tooltip("Max position step per frame (at reference FPS) for tip joints (meters).")]
    public float maxStepTips = 0.015f;

    // --- Jitter dead-zone: soft + hysteresis (to reduce stick-slip) ---
    [Tooltip("If true, small frame-to-frame moves are suppressed.")]
    public bool useJitterDeadZone = true;

    [Tooltip("Dead-zone radius (meters). Small moves are treated as noise.")]
    public float jitterDeadZoneMeters = 0.003f; // 3 mm

    [Tooltip("If true, use soft dead-zone + hysteresis to reduce stick-slip. If false, use a hard threshold.")]
    public bool useSoftJitterDeadZone = true;

    [Tooltip("Hysteresis half-width around the dead-zone boundary (meters). 0 = hard dead-zone.")]
    public float jitterHysteresisMeters = 0.0015f; // 1.5 mm

    [Header("Rig arming")]
    [Tooltip("HandRigArmer that controls the rig weight.")]
    public HandRigArmer armer;
    [Tooltip("Minimum wrist-to-finger distance for a frame to be considered valid.")]
    public float firstValidDistance = 0.02f;
    [Tooltip("If true, rig weight is armed automatically on first valid frame.")]
    public bool autoArm = false;

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

    [Header("Aim settings (position only)")]
    [Tooltip("If true, palmFwd is updated every frame to follow the hand direction.")]
    public bool computePalmFrame = true;
    [Tooltip("If true, aim direction is based on wrist velocity. If false, use wrist-to-middle vector.")]
    public bool aimFromVelocity = true;
    [Tooltip("Minimum wrist speed for velocity-based aiming (meters per second).")]
    public float aimVelMinSpeed = 0.05f;
    [Range(0f, 1f)]
    [Tooltip("Smoothing factor for aim direction (0 = no smoothing, 1 = very slow).")]
    public float aimDirLerp = 0.20f;

    [Header("Translation gain (no remap)")]
    [Tooltip("Proxy translation gain. 1=1:1, 2=twice as far.")]
    public float translationGain = 1.0f;

    [Tooltip("If true, gain is applied around the first captured wrist position.")]
    public bool gainUseNeutralWrist = true;

    Vector3 _gainNeutralWristWorld = Vector3.zero;
    bool _gainNeutralCaptured = false;

    [Header("Depth (camera-forward) stabilization")]
    [Tooltip("If true, suppress jitter along camera-forward direction (depth axis).")]
    public bool stabilizeDepth = true;

    [Tooltip("If true, apply depth stabilization only to wrist joint (index 0). If false, apply to all joints.")]
    public bool stabilizeDepthWristOnly = true;

    [Tooltip("Extra dead-zone for depth component (meters).")]
    public float depthDeadZoneMeters = 0.010f; // 1 cm

    [Tooltip("Max change per frame along camera-forward (meters, at reference FPS).")]
    public float depthMaxStepMeters = 0.005f; // 5 mm/frame

    [Tooltip("Extra low-pass cutoff for depth component (Hz). Lower = smoother, more lag.")]
    public float depthCutoffHz = 2.0f;

    [Header("Depth stabilization gating (wrist intent detection)")]
    [Tooltip("If true, depth stabilization is applied only when lateral motion dominates.")]
    public bool depthUseGating = true;

    [Tooltip("Minimum non-depth movement per frame to consider as 'lateral motion' (meters).")]
    public float depthGateMinNonDepthStep = 0.004f; // 4 mm/frame

    [Tooltip("If |dDepth| <= ratio * nonDepthMag, treat depth as noise and stabilize it.")]
    public float depthGateDepthToNonDepthRatio = 0.35f; // 0.2~0.6

    [Tooltip("When depth is NOT gated (user intends forward/back), allow larger depth per frame (meters, at reference FPS).")]
    public float depthMaxStepMetersFree = 0.04f; // 4 cm/frame (large)

    [Tooltip("When depth is NOT gated, use weaker deadzone for depth.")]
    public float depthDeadZoneMetersFree = 0.000f;

    [Tooltip("When depth is NOT gated, use higher cutoff (less smoothing) for depth.")]
    public float depthCutoffHzFree = 10f;

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

    // Side-to-front remap settings
    [Header("Side-to-front remap")]
    public bool enableSideToFrontRemap = false;

    [Tooltip("Camera-local X (left/right) to workspace X scale.")]
    public float remapXScale = 1.0f;

    [Tooltip("Camera-local Z (forward) to workspace Y (up) scale.")]
    public float remapYFromZScale = 0.6f;
    public bool remapInvertYFromZ = false;

    [Tooltip("Camera-local Y (up/down) to workspace Z (forward/back) scale.")]
    public float remapZFromYScale = 0.6f;
    public bool remapInvertZFromY = false;

    [Range(0f, 1f)]
    [Tooltip("Smoothing factor for remap offset (0 = very stiff, 1 = no smoothing).")]
    public float remapLerp = 0.15f;

    [Tooltip("Max remap offset change per frame in camera space (meters, at reference FPS).")]
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

    // =========================================================
    // Internal state
    // =========================================================

    // (1) Network reception buffer: last received raw world positions (no processing)
    Vector3[] _netRawPos = new Vector3[21];
    bool _haveNetRaw = false;

    // (2) Per-frame working buffer (processed target positions)
    Vector3[] _workPos = new Vector3[21];

    // internal state for position smoothing (output)
    Vector3[] _prevPos = new Vector3[21];
    bool _havePrevPos = false;

    // One Euro filters (per joint)
    OneEuroFilterVec3[] _oneEuro = new OneEuroFilterVec3[21];

    // internal state for aim
    Vector3 _wristPrev;
    bool _haveWristPrev = false;
    Vector3 _aimDirSm = Vector3.zero;

    // internal state for offset
    bool _offsetCaptured = false;
    Vector3 _initialOffset = Vector3.zero;
    Vector3 _lastPreOffsetWrist = Vector3.zero;

    // auto-arm state
    bool _firstArmed = false;

    // internal state for twist
    bool _twistNeutralReady = false;
    Vector3 _twistAxisNeutral;
    Vector3 _twistRefNeutral;
    float _twistSmoothedDeg = 0f;
    Quaternion _twistBaseLocalRot = Quaternion.identity;
    bool _twistBaseCaptured = false;

    void Awake()
    {
        for (int i = 0; i < 21; i++)
            _oneEuro[i] = new OneEuroFilterVec3();
    }

    // =========================================================
    // Main entry from UdpHandReceiver
    //   - Now: only stores the latest raw target.
    //   - Processing + smoothing happens every render frame in LateUpdate().
    // =========================================================
    public void ApplyWorldPositions(Vector3[] worldPos)
    {
        if (manualTestMode || worldPos == null || worldPos.Length < 21) return;

        // store raw wrist for ContextRecaptureNow()
        _lastPreOffsetWrist = worldPos[0];

        // store latest raw positions
        for (int i = 0; i < 21; i++)
            _netRawPos[i] = worldPos[i];

        _haveNetRaw = true;
    }

    void LateUpdate()
    {
        if (manualTestMode) return;
        if (!_haveNetRaw) return;

        ProcessAndApplyPerFrame();
    }

    // =========================================================
    // Per-frame processing (render-rate)
    //   offset -> remap -> translation gain -> smoothing -> twist/aim -> auto-arm
    // =========================================================
    void ProcessAndApplyPerFrame()
    {
        // copy raw -> work
        for (int i = 0; i < 21; i++)
            _workPos[i] = _netRawPos[i];

        // keep latest raw wrist for context recapture (in case ApplyWorldPositions stops)
        _lastPreOffsetWrist = _workPos[0];

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

            ResetSmoothingState(); // IMPORTANT: prevent slow catch-up after new offset
        }

        // apply offset to all joints
        if (addInitialOffset && _offsetCaptured)
        {
            for (int i = 0; i < 21; i++)
                _workPos[i] += _initialOffset;
        }

        // --- REMAP 단계 (per-frame now) ---
        if (useJoystickRemap)
        {
            RemapJoystickStyle(_workPos);
        }
        else
        {
            RemapSideToFront(_workPos);
        }

        // --- Translation gain (amplify wrist translation) ---
        if (translationGain != 1.0f && translationGain > 0f)
        {
            Vector3 wrist = _workPos[0];

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
                    _workPos[i] += delta;
            }
        }

        // smooth and apply to remote driver joints (per render frame)
        SmoothAndApply(_workPos);

        // doorknob style twist on separate bone
        UpdateTwist();

        // aim target for Wrist_Aim (position only)
        if (computePalmFrame)
            UpdateAimPositionOnly();

        // auto-arm rig on first valid frame
        if (autoArm && !_firstArmed && FrameLooksValid(_workPos))
        {
            if (armer != null) armer.ArmNow();
            _firstArmed = true;
        }
    }

    // =========================================================
    // Position smoothing and speed-based step clamp
    // =========================================================
    void SmoothAndApply(Vector3[] inPos)
    {
        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);

        // Reference FPS for converting "meters per frame" -> "m/s"
        float refFps = Mathf.Max(1f, stepClampReferenceFps);

        // Camera-forward direction for "depth" stabilization
        Vector3 camFwd = Vector3.forward;
        if (Camera.main != null)
        {
            camFwd = Camera.main.transform.forward;
            if (camFwd.sqrMagnitude > 1e-8f) camFwd.Normalize();
            else camFwd = Vector3.forward;
        }

        for (int i = 0; i < 21; i++)
        {
            Vector3 v = inPos[i];

            // first frame init
            if (!_havePrevPos)
            {
                _prevPos[i] = v;
                if (_oneEuro[i] != null) _oneEuro[i].Reset();
            }

            // 1) base smoothing: One Euro (adaptive) or legacy fixed LPF
            Vector3 raw;
            if (useOneEuroFilter && _oneEuro[i] != null)
            {
                float minC = Mathf.Max(0.01f, IsTip(i) ? oneEuroMinCutoffHzTips : oneEuroMinCutoffHz);
                float beta = IsTip(i) ? oneEuroBetaTips : oneEuroBeta;
                float dCut = Mathf.Max(0.01f, oneEuroDerivCutoffHz);

                raw = _oneEuro[i].Filter(v, dt, minC, beta, dCut);
            }
            else
            {
                // legacy fixed LPF (kept for fallback)
                float baseCutoffHz = IsTip(i) ? cutoffHzTips : cutoffHz;
                baseCutoffHz = Mathf.Max(0.01f, baseCutoffHz);
                float baseOmega = 2f * Mathf.PI * baseCutoffHz;
                float baseAlpha = baseOmega * dt / (1f + baseOmega * dt);

                if (IsTip(i))
                    baseAlpha = Mathf.Clamp(baseAlpha, 0.02f, 0.30f);
                else
                    baseAlpha = Mathf.Clamp01(baseAlpha);

                raw = Vector3.Lerp(_prevPos[i], v, baseAlpha);
            }

            // 2) speed-based step clamp (framerate-independent)
            float stepCapPerFrame = IsTip(i) ? maxStepTips : maxStepMeters;
            if (stepCapPerFrame > 0f)
            {
                // convert old "meters per frame" into "meters per second"
                float maxSpeed = stepCapPerFrame * refFps;     // m/s
                float stepCap = maxSpeed * dt;                // m for this frame

                Vector3 dVec = raw - _prevPos[i];
                float m = dVec.magnitude;
                if (m > stepCap && m > 1e-8f)
                    raw = _prevPos[i] + dVec * (stepCap / m);
            }

            // 3) optional depth stabilization (camera-forward component only)
            bool applyDepth = stabilizeDepth && (!stabilizeDepthWristOnly || i == 0);
            if (applyDepth)
            {
                Vector3 delta = raw - _prevPos[i];

                // Depth component along camera forward
                float dDepth = Vector3.Dot(delta, camFwd);
                Vector3 deltaNonDepth = delta - (dDepth * camFwd);
                float nonDepthMag = deltaNonDepth.magnitude;

                // Decide whether to gate (stabilize) depth strongly
                bool gateDepth = !depthUseGating ? true : false;

                if (depthUseGating)
                {
                    if (nonDepthMag >= depthGateMinNonDepthStep &&
                        Mathf.Abs(dDepth) <= depthGateDepthToNonDepthRatio * nonDepthMag)
                    {
                        gateDepth = true;   // strong stabilize
                    }
                    else
                    {
                        gateDepth = false;  // user likely intends forward/back
                    }
                }

                float dzDepth = gateDepth ? depthDeadZoneMeters : depthDeadZoneMetersFree;

                // depth step clamp: convert old per-frame value (at reference FPS) -> per-second -> per-dt
                float maxStepDepthPerFrame = gateDepth ? depthMaxStepMeters : depthMaxStepMetersFree;
                float maxStepDepth = 0f;
                if (maxStepDepthPerFrame > 0f)
                {
                    float maxSpeedDepth = maxStepDepthPerFrame * refFps; // m/s
                    maxStepDepth = maxSpeedDepth * dt;                  // m this frame
                }

                float depthCutoffLocalHz = gateDepth ? depthCutoffHz : depthCutoffHzFree;
                depthCutoffLocalHz = Mathf.Max(0.01f, depthCutoffLocalHz);
                float depthOmega = 2f * Mathf.PI * depthCutoffLocalHz;
                float depthAlpha = depthOmega * dt / (1f + depthOmega * dt); // 0..1

                // Dead-zone on depth (keep as hard; depth already has LPF + clamp)
                if (dzDepth > 0f && Mathf.Abs(dDepth) < dzDepth)
                    dDepth = 0f;

                // Step clamp on depth (now dt-based)
                if (maxStepDepth > 0f)
                    dDepth = Mathf.Clamp(dDepth, -maxStepDepth, maxStepDepth);

                // Extra "LPF-like" attenuation on depth scalar
                float dDepthSm = Mathf.Lerp(0f, dDepth, Mathf.Clamp01(depthAlpha));

                raw = _prevPos[i] + deltaNonDepth + (dDepthSm * camFwd);
            }

            // 4) jitter dead-zone: soft + hysteresis (reduces stick-slip)
            if (useJitterDeadZone)
            {
                float dz = Mathf.Max(0f, jitterDeadZoneMeters);
                if (dz > 0f)
                {
                    Vector3 d = raw - _prevPos[i];

                    if (useSoftJitterDeadZone)
                    {
                        float h = Mathf.Max(0f, jitterHysteresisMeters);
                        float g = DeadzoneGain(d.magnitude, dz, h); // 0..1
                        raw = _prevPos[i] + d * g;
                    }
                    else
                    {
                        if (d.sqrMagnitude < dz * dz)
                            raw = _prevPos[i];
                    }
                }
            }

            if (remoteByIndex[i] != null)
                remoteByIndex[i].position = raw;

            _prevPos[i] = raw;
        }

        _havePrevPos = true;
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

        if (aimFromVelocity)
        {
            if (!_haveWristPrev)
            {
                _wristPrev = w;
                _haveWristPrev = true;
            }

            float dt = Mathf.Max(Time.deltaTime, 1e-4f);
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

        Vector3 dN = dir.normalized;
        if (_aimDirSm == Vector3.zero)
            _aimDirSm = dN;
        else
            _aimDirSm = Vector3.Slerp(_aimDirSm, dN, Mathf.Clamp01(aimDirLerp));

        float dist = Mathf.Max(0.01f, palmAimDistance);
        Vector3 aimPos = w + _aimDirSm * dist;

        if (palmFwd != null)
            palmFwd.position = aimPos;

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

        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);
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

    void RemapSideToFront(Vector3[] joints)
    {
        if (!enableSideToFrontRemap) return;
        if (joints == null || joints.Length < 1) return;
        if (Camera.main == null) return;

        Transform cam = Camera.main.transform;

        Vector3 wristWorld = joints[0];
        Vector3 wristCam = cam.InverseTransformPoint(wristWorld);

        if (!_remapNeutralCaptured)
        {
            _remapNeutralCam = wristCam;
            _remapNeutralWorld = wristWorld;
            _remapOffsetCamSm = Vector3.zero;
            _remapNeutralCaptured = true;
            return;
        }

        Vector3 dCam = wristCam - _remapNeutralCam;
        Vector3 dWorld = wristWorld - _remapNeutralWorld;

        float xOff = dWorld.x * remapXScale;

        float yOff = dCam.z * remapYFromZScale;
        if (remapInvertYFromZ) yOff = -yOff;

        float zOff = dCam.y * remapZFromYScale;
        if (remapInvertZFromY) zOff = -zOff;

        Vector3 targetOffsetCam = new Vector3(xOff, yOff, zOff);

        if (remapMaxOffsetCam.x > 0f)
            targetOffsetCam.x = Mathf.Clamp(targetOffsetCam.x, -remapMaxOffsetCam.x, remapMaxOffsetCam.x);
        if (remapMaxOffsetCam.y > 0f)
            targetOffsetCam.y = Mathf.Clamp(targetOffsetCam.y, -remapMaxOffsetCam.y, remapMaxOffsetCam.y);
        if (remapMaxOffsetCam.z > 0f)
            targetOffsetCam.z = Mathf.Clamp(targetOffsetCam.z, -remapMaxOffsetCam.z, remapMaxOffsetCam.z);

        // remap dead-zone (kept hard; remap already has smoothing)
        float dz = Mathf.Max(0f, remapDeadZoneMeters);
        if (dz > 0f)
        {
            Vector3 diff = targetOffsetCam - _remapOffsetCamSm;
            if (diff.sqrMagnitude < dz * dz)
                targetOffsetCam = _remapOffsetCamSm;
        }

        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);
        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(remapLerp), dt * 60f);
        Vector3 candidate = Vector3.Lerp(_remapOffsetCamSm, targetOffsetCam, k);

        // speed-based step clamp (framerate-independent)
        float refFps = Mathf.Max(1f, stepClampReferenceFps);
        float maxStepPerFrame = Mathf.Max(0f, remapMaxStepMeters);
        if (maxStepPerFrame > 0f)
        {
            float maxSpeed = maxStepPerFrame * refFps; // m/s
            float maxStep = maxSpeed * dt;            // m this frame

            Vector3 step = candidate - _remapOffsetCamSm;
            float stepMag = step.magnitude;
            if (stepMag > maxStep && stepMag > 1e-8f)
                candidate = _remapOffsetCamSm + step * (maxStep / stepMag);
        }

        _remapOffsetCamSm = candidate;

        Vector3 newWristWorld = _remapNeutralWorld + cam.TransformVector(_remapOffsetCamSm);
        Vector3 deltaWorld = newWristWorld - wristWorld;

        for (int i = 0; i < joints.Length; i++)
            joints[i] += deltaWorld;
    }

    // =========================================================
    // Joystick-style remap helper: one axis [-1..1]
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
        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);

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

    void ResetSmoothingState()
    {
        _havePrevPos = false;

        _haveWristPrev = false;
        _aimDirSm = Vector3.zero;

        // Also reset One Euro filters so we don't "catch up" slowly after a big target jump
        if (_oneEuro != null)
        {
            for (int i = 0; i < _oneEuro.Length; i++)
                if (_oneEuro[i] != null) _oneEuro[i].Reset();
        }
    }

    [ContextMenu("Offset / Clear and re-arm")]
    public void ContextClearAndRearm()
    {
        _offsetCaptured = false;
        _initialOffset = Vector3.zero;
        _firstArmed = false;

        _gainNeutralCaptured = false;
        _gainNeutralWristWorld = Vector3.zero;

        ResetSmoothingState();

        // remap 관련 상태도 리셋
        _remapNeutralCaptured = false;
        _remapOffsetCamSm = Vector3.zero;

        // joystick remap 상태도 같이 리셋
        _joyNeutralCaptured = false;
        _joyOffsetCam = Vector3.zero;
        _joyHasInputPrev = false;
        _joyInXSm = _joyInYSm = _joyInZSm = 0f;

        // (optional) keep last net raw; if you want to fully stop until new data, uncomment:
        // _haveNetRaw = false;
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

        // recapture implies target jump; reset smoothing + gain neutral
        _gainNeutralCaptured = false;
        _gainNeutralWristWorld = Vector3.zero;

        ResetSmoothingState();

        // remap neutral is in absolute space; safer to recapture
        _remapNeutralCaptured = false;
        _remapOffsetCamSm = Vector3.zero;

        _joyNeutralCaptured = false;
        _joyOffsetCam = Vector3.zero;
        _joyHasInputPrev = false;
        _joyInXSm = _joyInYSm = _joyInZSm = 0f;
    }

    // =========================================================
    // Soft dead-zone gain with hysteresis band
    //   mag <= (dz - h) : 0
    //   mag >= (dz + h) : 1
    //   between: smoothstep
    // =========================================================
    static float DeadzoneGain(float mag, float dz, float h)
    {
        dz = Mathf.Max(0f, dz);
        h = Mathf.Max(0f, h);

        if (dz <= 1e-8f) return 1f;

        float a = Mathf.Max(0f, dz - h);
        float b = dz + h;

        if (mag <= a) return 0f;
        if (mag >= b) return 1f;

        float t = (mag - a) / Mathf.Max(1e-8f, (b - a)); // 0..1
        // smoothstep(0..1)
        return t * t * (3f - 2f * t);
    }

    // =========================================================
    // One Euro Filter (Vector3)
    // =========================================================
    class OneEuroFilterVec3
    {
        bool _initialized = false;
        Vector3 _xPrev = Vector3.zero;
        Vector3 _xHat = Vector3.zero;
        Vector3 _dxHat = Vector3.zero;

        public void Reset()
        {
            _initialized = false;
            _xPrev = Vector3.zero;
            _xHat = Vector3.zero;
            _dxHat = Vector3.zero;
        }

        static float Alpha(float cutoffHz, float dt)
        {
            cutoffHz = Mathf.Max(0.01f, cutoffHz);
            float omega = 2f * Mathf.PI * cutoffHz;
            return omega * dt / (1f + omega * dt);
        }

        public Vector3 Filter(Vector3 x, float dt, float minCutoffHz, float beta, float dCutoffHz)
        {
            dt = Mathf.Max(1e-5f, dt);

            if (!_initialized)
            {
                _initialized = true;
                _xPrev = x;
                _xHat = x;
                _dxHat = Vector3.zero;
                return _xHat;
            }

            // 1) derivative
            Vector3 dx = (x - _xPrev) / dt;
            _xPrev = x;

            // 2) low-pass filter derivative
            float aD = Alpha(dCutoffHz, dt);
            _dxHat = Vector3.Lerp(_dxHat, dx, Mathf.Clamp01(aD));

            // 3) adaptive cutoff
            float cutoff = Mathf.Max(0.01f, minCutoffHz + beta * _dxHat.magnitude);

            // 4) low-pass filter signal
            float aX = Alpha(cutoff, dt);
            _xHat = Vector3.Lerp(_xHat, x, Mathf.Clamp01(aX));

            return _xHat;
        }
    }
}
