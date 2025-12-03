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

    [Header("Smoothing for joint positions")]
    [Tooltip("Low-pass cutoff for non-tip joints (Hz).")]
    public float cutoffHz = 10f;
    [Tooltip("Max position step per frame for non-tip joints (meters).")]
    public float maxStepMeters = 0.08f;
    [Tooltip("Low-pass cutoff for tip joints (Hz).")]
    public float cutoffHzTips = 6f;
    [Tooltip("Max position step per frame for tip joints (meters).")]
    public float maxStepTips = 0.015f;

    [Tooltip("If true, very small frame-to-frame moves are ignored (dead-zone).")]
    public bool useJitterDeadZone = true;
    [Tooltip("Movements smaller than this are treated as noise and dropped (meters).")]
    public float jitterDeadZoneMeters = 0.003f; // 3 mm

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

    // Side-to-front remap settings
    [Header("Side-to-front remap")]
    public bool enableSideToFrontRemap = false;

    // camera-local X -> world X scale
    [Tooltip("Camera-local X (left/right) to workspace X scale.")]
    public float remapXScale = 1.0f;

    // camera-local Z (forward/back) -> workspace Y (up/down)
    [Tooltip("Camera-local Z (forward) to workspace Y (up) scale.")]
    public float remapYFromZScale = 0.6f;
    public bool remapInvertYFromZ = false;

    // camera-local Y (up/down) -> workspace Z (forward/back)
    [Tooltip("Camera-local Y (up) to workspace Z (forward) scale.")]
    public float remapZFromYScale = 0.6f;
    public bool remapInvertZFromY = false;

    // remap smoothing
    [Range(0f, 1f)]
    [Tooltip("Smoothing factor for remap offset (0 = very stiff, 1 = no smoothing).")]
    public float remapLerp = 0.15f;

    [Tooltip("Max remap offset change per frame in camera space (meters).")]
    public float remapMaxStepMeters = 0.02f;

    // workspace 범위 제한
    [Tooltip("Max absolute workspace offset in camera-local space (meters).")]
    public Vector3 remapMaxOffsetCam = new Vector3(0.30f, 0.20f, 0.30f);

    // remap 전용 dead-zone
    [Tooltip("Camera-space dead-zone for remap offset changes (meters).")]
    public float remapDeadZoneMeters = 0.005f;

    // 내부 상태
    Vector3 _remapNeutralCam = Vector3.zero;
    Vector3 _remapNeutralWorld = Vector3.zero;
    Vector3 _remapOffsetCamSm = Vector3.zero;
    bool _remapNeutralCaptured = false;





    // internal state for position smoothing
    Vector3[] _prevPos = new Vector3[21];
    bool _havePrevPos = false;

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

    // ======================================================================
    // Main entry from UdpHandReceiver
    // ======================================================================
    public void ApplyWorldPositions(Vector3[] worldPos)
    {
        if (manualTestMode || worldPos == null || worldPos.Length < 21) return;

        // store wrist before applying offset
        _lastPreOffsetWrist = worldPos[0];

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
            _havePrevPos = false; // reset smoothing after new offset
        }

        // apply offset to all joints
        if (addInitialOffset && _offsetCaptured)
        {
            for (int i = 0; i < 21; i++)
                worldPos[i] += _initialOffset;
        }

        RemapSideToFront(worldPos);

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

    // ======================================================================
    // Position smoothing and step clamp
    // ======================================================================
    void SmoothAndApply(Vector3[] inPos)
    {
        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);

        for (int i = 0; i < 21; i++)
        {
            float cutoff = IsTip(i) ? cutoffHzTips : cutoffHz;
            float omega = 2f * Mathf.PI * cutoff;
            float alpha = omega * dt / (1f + omega * dt);

            if (IsTip(i))
                alpha = Mathf.Clamp(alpha, 0.02f, 0.30f);
            else
                alpha = Mathf.Clamp01(alpha);

            Vector3 v = inPos[i];

            if (!_havePrevPos)
                _prevPos[i] = v;

            Vector3 raw = Vector3.Lerp(_prevPos[i], v, alpha);

            float stepCap = IsTip(i) ? maxStepTips : maxStepMeters;
            Vector3 d = raw - _prevPos[i];
            float m = d.magnitude;

            if (m > stepCap)
                raw = _prevPos[i] + d.normalized * stepCap;

            if (useJitterDeadZone)
            {
                float dz = Mathf.Max(0f, jitterDeadZoneMeters);
                if (dz > 0f && (raw - _prevPos[i]).sqrMagnitude < dz * dz)
                {
                    // treat as pure noise: keep previous value
                    raw = _prevPos[i];
                }
            }

            if (remoteByIndex[i] != null)
                remoteByIndex[i].position = raw;

            _prevPos[i] = raw;
        }

        _havePrevPos = true;
    }

    // ======================================================================
    // Aim target (position only, no roll / no up)
    // ======================================================================
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
        // palmUp is only moved for debug.
        if (palmUp != null)
            palmUp.position = w + Vector3.up * dist;
    }

    // ======================================================================
    // Door knob twist on a separate wristTwist bone
    // ======================================================================
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

        // capture base local rotation of wristTwist once
        if (!_twistBaseCaptured)
        {
            _twistBaseLocalRot = wristTwist.localRotation;
            _twistBaseCaptured = true;
        }

        // capture neutral frame once
        if (!_twistNeutralReady)
        {
            _twistAxisNeutral = axis;
            _twistRefNeutral = refVec;
            _twistNeutralReady = true;
            _twistSmoothedDeg = 0f;
            wristTwist.localRotation = _twistBaseLocalRot;
            return;
        }

        // 3) compute roll angle around neutral axis
        Vector3 refNow = Vector3.ProjectOnPlane(refVec, _twistAxisNeutral);
        Vector3 ref0 = Vector3.ProjectOnPlane(_twistRefNeutral, _twistAxisNeutral);

        if (refNow.sqrMagnitude < 1e-8f || ref0.sqrMagnitude < 1e-8f)
            return;

        refNow.Normalize();
        ref0.Normalize();

        float rawRollDeg = Vector3.SignedAngle(ref0, refNow, _twistAxisNeutral);

        // 4) clamp and dead zone
        rawRollDeg = Mathf.Clamp(rawRollDeg, -twistMaxAbsDeg, twistMaxAbsDeg);

        if (Mathf.Abs(rawRollDeg) < twistDeadZoneDeg)
            rawRollDeg = 0f;

        // 5) limit speed (deg per second)
        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);
        float maxStep = twistMaxDegPerSec * dt;
        float delta = rawRollDeg - _twistSmoothedDeg;
        if (delta > maxStep) delta = maxStep;
        else if (delta < -maxStep) delta = -maxStep;

        float targetDeg = _twistSmoothedDeg + delta;

        // 6) extra low-pass filter
        float s = 1f - Mathf.Pow(1f - Mathf.Clamp01(twistLerp), dt * 60f);
        _twistSmoothedDeg = Mathf.Lerp(_twistSmoothedDeg, targetDeg, s);

        // 7) apply around local forward axis (Z)
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

        // 0: wrist
        Vector3 wristWorld = joints[0];
        Vector3 wristCam = cam.InverseTransformPoint(wristWorld);

        // 1) 중립 캡처 (한번만)
        if (!_remapNeutralCaptured)
        {
            _remapNeutralCam = wristCam;
            _remapNeutralWorld = wristWorld;
            _remapOffsetCamSm = Vector3.zero;
            _remapNeutralCaptured = true;
            return;
        }

        // 2) 카메라 로컬 델타
        Vector3 dCam = wristCam - _remapNeutralCam;
        Vector3 dWorld = wristWorld - _remapNeutralWorld;

        float xOff = dWorld.x * remapXScale;

        // 3) 축 remap: X 그대로, Z->Y, Y->Z
        // float xOff = dCam.x * remapXScale;

        float yOff = dCam.z * remapYFromZScale; // forward/back -> up/down
        if (remapInvertYFromZ) yOff = -yOff;

        float zOff = dCam.y * remapZFromYScale; // up/down -> forward/back
        if (remapInvertZFromY) zOff = -zOff;

        Vector3 targetOffsetCam = new Vector3(xOff, yOff, zOff);

        // 4) workspace 크기 제한
        if (remapMaxOffsetCam.x > 0f)
            targetOffsetCam.x = Mathf.Clamp(targetOffsetCam.x,
                                            -remapMaxOffsetCam.x, remapMaxOffsetCam.x);
        if (remapMaxOffsetCam.y > 0f)
            targetOffsetCam.y = Mathf.Clamp(targetOffsetCam.y,
                                            -remapMaxOffsetCam.y, remapMaxOffsetCam.y);
        if (remapMaxOffsetCam.z > 0f)
            targetOffsetCam.z = Mathf.Clamp(targetOffsetCam.z,
                                            -remapMaxOffsetCam.z, remapMaxOffsetCam.z);

        // 5) remap 전용 dead-zone
        float dz = Mathf.Max(0f, remapDeadZoneMeters);
        if (dz > 0f)
        {
            Vector3 diff = targetOffsetCam - _remapOffsetCamSm;
            if (diff.sqrMagnitude < dz * dz)
            {
                targetOffsetCam = _remapOffsetCamSm;
            }
        }

        // 6) remap offset smoothing + step clamp (camera space)
        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);
        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(remapLerp), dt * 60f);
        Vector3 candidate = Vector3.Lerp(_remapOffsetCamSm, targetOffsetCam, k);

        float maxStep = Mathf.Max(0f, remapMaxStepMeters);
        if (maxStep > 0f)
        {
            Vector3 step = candidate - _remapOffsetCamSm;
            float stepMag = step.magnitude;
            if (stepMag > maxStep)
                candidate = _remapOffsetCamSm + step.normalized * maxStep;
        }

        _remapOffsetCamSm = candidate;

        // 7) 다시 world로 → 전체 joint에 동일 delta 적용
        Vector3 newWristWorld = _remapNeutralWorld + cam.TransformVector(_remapOffsetCamSm);
        Vector3 deltaWorld = newWristWorld - wristWorld;

        for (int i = 0; i < joints.Length; i++)
            joints[i] += deltaWorld;
    }





    // ======================================================================
    // Helpers
    // ======================================================================
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

    [ContextMenu("Offset / Clear and re-arm")]
    public void ContextClearAndRearm()
    {
        _offsetCaptured = false;
        _initialOffset = Vector3.zero;
        _firstArmed = false;
        _havePrevPos = false;
        _haveWristPrev = false;

            // remap 관련 상태도 리셋
        _remapNeutralCaptured = false;
        _remapOffsetCamSm = Vector3.zero;
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
        _havePrevPos = false;
    }
}
