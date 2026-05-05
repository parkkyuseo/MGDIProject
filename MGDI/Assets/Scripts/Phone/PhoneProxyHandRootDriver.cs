using UnityEngine;

public class PhoneProxyHandRootDriver : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhonePoseStreamReceiver phoneRx;
    [SerializeField] private Transform handRoot; // Remote_Wrist or HandRoot target
    [SerializeField] private Transform cameraTransform; // default: Camera.main
    [SerializeField] private ProxyHandGrabber grabber;
    [Tooltip("Optional explicit grip/contact pivot used for macro position gain. If empty, falls back to ProxyHandGrabber.grabAnchor.")]
    [SerializeField] private Transform gripPivot;

    [Header("Mapping")]
    [SerializeField] private float positionGain = 1.0f;
    [SerializeField] private bool applyRotation = true;

    [Header("Smoothing")]
    [SerializeField] private float posLerp = 24f;
    [SerializeField] private float rotLerp = 24f;
    [Tooltip("If true, macro position is solved around the grip/contact pivot. Disable to keep wrist twist from translating the proxy hand root in an arc.")]
    [SerializeField] private bool useGripPivotForPositionSolve = false;
    [Tooltip("If true, position solve uses the actual smoothed rotation applied this frame instead of the full desired rotation target. This reduces sideways drift caused by rotation-induced wrist compensation, especially in Macro+Side. Disable to restore the previous behavior.")]
    [SerializeField] private bool useAppliedRotationForPositionSolve = false;

    [Header("Stability")]
    [SerializeField] private bool driveInLateUpdate = true;
    [Tooltip("Maximum applied rotation speed. Set <= 0 to disable limiting.")]
    [SerializeField] private float maxRotDegPerSec = 360f;

    [Header("Short-Gap Prediction")]
    [Tooltip("If true, briefly extrapolates phone pose across short receive gaps so proxy-hand motion does not visibly freeze on 1-3 dropped frames.")]
    [SerializeField] private bool predictShortPoseGaps = true;
    [SerializeField] private float predictionMaxGapSeconds = 0.05f;
    [SerializeField] private float predictionMaxLeadSeconds = 0.025f;
    [SerializeField] private float predictionMaxLinearSpeedMetersPerSec = 1.1f;
    [SerializeField] private float predictionMaxAngularSpeedDegPerSec = 540f;
    [SerializeField] private bool predictRotationDuringShortGaps = false;

    [Header("Rotation-Dominant Translation Suppression")]
    [Tooltip("If true, absorbs translation drift while the phone is rotating quickly but not translating much. This prevents proxy-hand position creep during repeated wrist twisting.")]
    [SerializeField] private bool suppressTranslationDuringHighRotation = true;
    [SerializeField] private float translationSuppressionAngularSpeedDegPerSec = 180f;
    [SerializeField] private float translationSuppressionLinearSpeedMetersPerSec = 0.12f;
    [SerializeField] private float translationSuppressionEnterHoldSeconds = 0.04f;
    [SerializeField] private float translationSuppressionExitHoldSeconds = 0.12f;
    [SerializeField] private float translationSuppressionMaxMotionAgeSeconds = 0.08f;

    [Header("Baseline")]
    [SerializeField] private bool autoRecenterOnFirstPose = true;

    [Header("Grip Neutralization")]
    [Tooltip("After first pose baseline, recenter input once after a short delay so comfortable grip can become neutral.")]
    [SerializeField] private bool autoNeutralizeAfterConnect = true;
    [SerializeField] private float neutralizeDelaySeconds = 1.2f;
    [SerializeField] private bool neutralizeOnlyOnce = true;
    [Tooltip("If true, hand rotation is held fixed while waiting for neutralization.")]
    [SerializeField] private bool lockRotationUntilNeutralized = true;
    [Tooltip("If true, neutralization triggers when phone motion becomes stable instead of a strict fixed-time trigger.")]
    [SerializeField] private bool neutralizeWhenPhoneStable = true;
    [SerializeField] private float neutralizeStableAngularSpeedDegPerSec = 25f;
    [SerializeField] private float neutralizeStableHoldSeconds = 0.35f;
    [SerializeField] private float neutralizeMaxWaitSeconds = 8f;

    [Header("Position Offset (meters)")]
    [SerializeField] private Vector3 positionOffset = new Vector3(0f, 0f, 0.25f); // 25cm forward (root local)

    [Header("Offset Frame")]
    [Tooltip("If true, Near uses camera yaw frame for positionOffset. If false, Near uses proxy-root frame (reduces head-coupled drift).")]
    [SerializeField] private bool useCameraYawOffsetInNear = false;
    [Tooltip("If true, Side uses camera yaw frame for positionOffset.")]
    [SerializeField] private bool useCameraYawOffsetInSide = true;

    [Header("Rotation Offset (fix forward direction)")]
    [SerializeField] private Vector3 rotationOffsetEuler = new Vector3(0f, 180f, 0f);

    [Header("Condition Offset Profiles")]
    [Tooltip("If true, Near/Side hand-location profiles can override offsets on condition changes.")]
    [SerializeField] private bool useHandLocationOffsetProfiles = true;

    [Tooltip("Offset profile used when condition hand location is Side.")]
    [SerializeField] private Vector3 sidePositionOffset = Vector3.zero;

    [Tooltip("Rotation offset profile used when condition hand location is Side.")]
    [SerializeField] private Vector3 sideRotationOffsetEuler = Vector3.zero;

    [Header("Side→Front Remap (Macro + Side only)")]
    [Tooltip("If true, translation delta is remapped so up/down tends to become forward/back (reduces diagonal feel).")]
    [SerializeField] private bool enableSideToFrontRemap = false;

    [Tooltip("Additional translation gain used while Macro+Near is active.")]
    [SerializeField] private float nearGain = 1.0f;

    [Tooltip("Extra translation gain applied only while Side→Front remap is enabled.")]
    [SerializeField] private float sideRemapGain = 1.0f;

    [Tooltip("If true, flips the remap direction (use this if forward/back feels inverted).")]
    [SerializeField] private bool invertSideToFront = false;

    [Header("Remap Frame (QR Workspace)")]
    [Tooltip("Workspace transform aligned by QR lock. Side remap basis uses this yaw.")]
    [SerializeField] private Transform remapFrameTransform;

    [Header("Translation Frame Stabilization")]
    [Tooltip("If true, phone translation delta is yaw-aligned to the QR workspace basis captured at recenter/rebaseline.")]
    [SerializeField] private bool alignTranslationToWorkspaceYaw = true;
    [Tooltip("If true, only XZ is yaw-aligned and the original Y delta is preserved.")]
    [SerializeField] private bool alignTranslationPlanarOnly = true;
    [Tooltip("Yaw offset applied on top of the QR workspace basis before translation alignment. Use this to compensate for an intentionally rotated QR marker setup.")]
    [SerializeField] private float translationYawOffsetDeg = -90f;
    [Tooltip("If true, the aligned planar X/Z translation axes are swapped so phone left-right can drive proxy forward-back and phone forward-back can drive proxy left-right when the workspace basis is rotated.")]
    [SerializeField] private bool swapAlignedPlanarTranslationAxes = true;
    [Tooltip("If true, the mapped forward-back planar axis is inverted after the X/Z swap.")]
    [SerializeField] private bool invertMappedPlanarForwardBack = true;
    [Tooltip("If true, translation yaw alignment is refreshed on RecenterInputOnly/RebaselineKeepWorldPose. Disable to keep the initial calibrated axis mapping stable.")]
    [SerializeField] private bool recaptureTranslationYawOnInputRecenter = false;
    [Tooltip("If true, skips the extra planar axis swap/invert stage when the legacy Macro+Side remap is active. This avoids double-remapping planar motion before the Side-specific axis permutation runs.")]
    [SerializeField] private bool skipAlignedPlanarAxisMappingWhenUsingLegacySideRemap = true;

    [Header("Phone QR Translation Frame")]
    [Tooltip("If true and the phone stream includes QR delta or marker pose fields, translation uses phone motion expressed in the phone-seen QR frame instead of raw phone AR-session coordinates.")]
    [SerializeField] private bool usePhoneQrRelativeTranslation = true;
    [Tooltip("If true, macro translation waits until the phone has reported QR delta or QR/marker pose at least once. If false, it falls back to raw phone coordinates until QR data appears.")]
    [SerializeField] private bool requirePhoneQrRelativeTranslation = false;
    [Tooltip("If true, logs whether macro translation is currently using raw phone coordinates or QR-relative phone coordinates.")]
    [SerializeField] private bool logTranslationFrameSource = true;
    [Tooltip("If true, swaps QR-relative planar X/Z before applying the HoloLens workspace yaw. Use when phone left-right arrives as QR Z and phone forward-back arrives as QR X.")]
    [SerializeField] private bool swapQrRelativePlanarTranslationAxes = true;
    [Tooltip("If true, flips QR-relative lateral motion after the optional X/Z swap.")]
    [SerializeField] private bool invertQrRelativeLateral = false;
    [Tooltip("If true, flips QR-relative forward-back motion after the optional X/Z swap.")]
    [SerializeField] private bool invertQrRelativeForwardBack = false;

    [Header("Side Mapping Mode")]
    [Tooltip("If false, Side condition keeps the same axis mapping as Near and only applies sideRemapGain.")]
    [SerializeField] private bool useLegacySideAxisRemap = false;
    [Tooltip("Deprecated. Macro+Side remap now always preserves lateral motion and swaps vertical with forward/back.")]
    [SerializeField] private bool swapLegacySideLateralAndVerticalOutputs = true;

    private bool _hasBaseline;
    private Pose _phone0;
    private Pose _root0;
    private Vector3 _positionPivot0World;
    private Vector3 _rootToPositionPivotLocal = Vector3.zero;
    private bool _hasPositionPivotCalibration;
    private bool _nearOffsetsCaptured;
    private Vector3 _nearPositionOffset;
    private Vector3 _nearRotationOffsetEuler;
    private bool _nearStartPoseCaptured;
    private Vector3 _nearStartWorldPos;
    private Quaternion _nearStartWorldRot = Quaternion.identity;
    private bool _neutralizeQueued;
    private bool _neutralizeDone;
    private float _neutralizeAtTime = -1f;
    private float _neutralizeQueuedAtTime = -1f;
    private float _neutralizeStableAccum = 0f;
    private Quaternion _neutralizePrevPhoneRot = Quaternion.identity;
    private float _neutralizePrevSampleTime = -1f;
    private bool _neutralizeHasPrevSample = false;
    private Quaternion _lockedWorldRotation = Quaternion.identity;
    private bool _hasLockedWorldRotation = false;
    private Quaternion _translationYawAlign = Quaternion.identity;
    private bool _hasTranslationYawAlign = false;
    private Quaternion _offsetYawBasis = Quaternion.identity;
    private bool _hasOffsetYawBasis = false;
    private Vector3 _translationPhone0Position;
    private bool _hasTranslationPhone0Position;
    private bool _translationBaselineUsesQrRelative;
    private bool _hasLoggedTranslationFrameSource;
    private bool _loggedTranslationFrameUsesQrRelative;
    private bool _hasWarnedMissingQrTranslation;
    private bool _translationSuppressionActive;
    private float _translationSuppressionEnterAccum;
    private float _translationSuppressionExitAccum;
    private Vector3 _translationSuppressedRawDp = Vector3.zero;

    public bool UseLegacySideAxisRemap => useLegacySideAxisRemap;
    public Transform HandRootTransform => handRoot;
    public bool IsGripNeutralizationQueued => _neutralizeQueued;
    public bool IsGripNeutralizationReady
    {
        get
        {
            if (!autoNeutralizeAfterConnect)
                return true;
            if (!_hasBaseline)
                return false;
            return !_neutralizeQueued && _neutralizeDone;
        }
    }

    void Start()
    {
        if (phoneRx == null) phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();
        if (cameraTransform == null && Camera.main != null) cameraTransform = Camera.main.transform;
        if (grabber == null) grabber = FindFirstObjectByType<ProxyHandGrabber>();
        if (remapFrameTransform == null)
        {
            GameObject go = GameObject.Find("WorkshopEnvironment");
            if (go != null) remapFrameTransform = go.transform;
        }
        CaptureNearOffsetsFromCurrent();
        CaptureCurrentAsNearStartPose();
    }

    void Update()
    {
        if (driveInLateUpdate) return;
        TickDriver();
    }

    void LateUpdate()
    {
        if (!driveInLateUpdate) return;
        TickDriver();
    }

    private void TickDriver()
    {
        if (phoneRx == null || handRoot == null) return;
        if (!phoneRx.HasPhonePose) return;

        Pose phone = GetDrivenPhonePose();

        if (!_hasBaseline)
        {
            if (!autoRecenterOnFirstPose) return;
            Recenter();
            if (!_hasBaseline) return;
            QueueNeutralizeIfNeeded();
        }

        TryRunQueuedNeutralize();

        if (!TryGetTranslationSamplePosition(phone, out Vector3 translationPosition, out bool usingQrRelativeTranslation))
            return;

        LogTranslationFrameSourceIfChanged(usingQrRelativeTranslation);

        if (_hasTranslationPhone0Position && usingQrRelativeTranslation != _translationBaselineUsesQrRelative)
        {
            CaptureTranslationYawAlignment(usingQrRelativeTranslation);
            CaptureTranslationPositionBaseline(translationPosition, usingQrRelativeTranslation);
            Debug.Log($"[PhoneProxyHandRootDriver] Translation frame switched to {GetTranslationFrameName(usingQrRelativeTranslation)}; translation baseline recaptured.");
        }

        bool sideConditionActive = enableSideToFrontRemap;
        bool hasRemapBasis = TryGetRemapBasisRotation(out Quaternion remapBasisRot);
        bool applyLegacySideAxisRemap = sideConditionActive && useLegacySideAxisRemap && hasRemapBasis;

        // Phone translation delta
        bool skipAlignedPlanarAxisMapping = applyLegacySideAxisRemap && skipAlignedPlanarAxisMappingWhenUsingLegacySideRemap;
        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        UpdateRotationDominantTranslationSuppression(translationPosition, dt);

        Vector3 translationPhone0Pos = _hasTranslationPhone0Position ? _translationPhone0Position : translationPosition;
        Vector3 rawDp = translationPosition - translationPhone0Pos;
        rawDp = AlignTranslationDelta(
            rawDp,
            applyPlanarAxisMapping: !usingQrRelativeTranslation && !skipAlignedPlanarAxisMapping);
        Vector3 dp = rawDp * positionGain;

        // Optional: legacy Side->Front axis remap (translation only)
        if (applyLegacySideAxisRemap)
        {
            dp = RemapSideToFront(
                dp,
                remapBasisRot,
                invertSideToFront,
                swapLegacySideLateralAndVerticalOutputs);
        }

        // Side condition can use an extra gain even when axis remap is disabled.
        float conditionGain = sideConditionActive ? sideRemapGain : nearGain;
        dp *= Mathf.Max(0f, conditionGain);

        Vector3 offsetWorld = ComputeOffsetWorld(sideConditionActive, _root0.rotation);

        // Phone rotation delta (unchanged)
        Quaternion dq = Quaternion.identity;
        if (applyRotation)
            dq = phone.rotation * Quaternion.Inverse(_phone0.rotation);

        Quaternion desiredRot = dq * _root0.rotation;
        Quaternion rotOffset = Quaternion.Euler(rotationOffsetEuler);
        desiredRot = desiredRot * rotOffset;

        if (ShouldLockRotationForNeutralize())
        {
            if (!_hasLockedWorldRotation)
            {
                _lockedWorldRotation = handRoot.rotation;
                _hasLockedWorldRotation = true;
            }
            desiredRot = _lockedWorldRotation;
        }

        float aPos = 1f - Mathf.Exp(-posLerp * dt);
        float aRot = 1f - Mathf.Exp(-rotLerp * dt);

        Quaternion currentRot = handRoot.rotation;
        Quaternion nextRot = currentRot;
        if (applyRotation)
        {
            nextRot = Quaternion.Slerp(currentRot, desiredRot, aRot);

            if (maxRotDegPerSec > 0f)
            {
                float maxStep = maxRotDegPerSec * dt;
                float stepDeg = Quaternion.Angle(currentRot, nextRot);
                if (stepDeg > maxStep && maxStep > 0f)
                    nextRot = Quaternion.Slerp(currentRot, nextRot, maxStep / stepDeg);
            }
        }

        Vector3 desiredPos;
        if (useGripPivotForPositionSolve)
        {
            Quaternion positionSolveRot = applyRotation
                ? (useAppliedRotationForPositionSolve ? nextRot : desiredRot)
                : currentRot;
            Vector3 desiredPivotWorld = _positionPivot0World + dp + offsetWorld;
            desiredPos = desiredPivotWorld - (positionSolveRot * _rootToPositionPivotLocal);
        }
        else
        {
            desiredPos = _root0.position + dp + offsetWorld;
        }

        handRoot.position = Vector3.Lerp(handRoot.position, desiredPos, aPos);
        if (applyRotation)
        {
            handRoot.rotation = nextRot;
        }
    }

    /// <summary>
    /// Call this when toggling remap on/off, or when switching Side L/R, to avoid a jump.
    /// </summary>
    public void Recenter()
    {
        if (phoneRx == null || handRoot == null) return;
        if (!phoneRx.HasPhonePose) return;

        _phone0 = phoneRx.LatestPhonePose;
        if (!TryGetTranslationSamplePosition(_phone0, out Vector3 translationBaseline, out bool usingQrRelativeTranslation))
            return;

        CaptureTranslationYawAlignment(usingQrRelativeTranslation);
        CaptureOffsetYawBasis();
        CaptureTranslationPositionBaseline(translationBaseline, usingQrRelativeTranslation);
        Quaternion rotOffset = Quaternion.Euler(rotationOffsetEuler);
        Quaternion rootRot = handRoot.rotation * Quaternion.Inverse(rotOffset);
        bool sideConditionActive = enableSideToFrontRemap;
        Vector3 offsetWorld = ComputeOffsetWorld(sideConditionActive, rootRot);
        CapturePositionPivotBaseline(offsetWorld);

        _root0 = new Pose(
            handRoot.position - offsetWorld,
            rootRot
        );
        _hasBaseline = true;

        Debug.Log("[PhoneProxyHandRootDriver] Recenter baseline captured.");
    }

    public void RecenterInputOnly()
    {
        if (phoneRx == null) return;
        if (!phoneRx.HasPhonePose) return;

        _phone0 = phoneRx.LatestPhonePose;
        if (!TryGetTranslationSamplePosition(_phone0, out Vector3 translationBaseline, out bool usingQrRelativeTranslation))
            return;

        if (!_hasTranslationYawAlign || recaptureTranslationYawOnInputRecenter)
            CaptureTranslationYawAlignment(usingQrRelativeTranslation);
        CaptureOffsetYawBasis();
        CaptureTranslationPositionBaseline(translationBaseline, usingQrRelativeTranslation);
        _hasBaseline = true;

        Debug.Log("[PhoneProxyHandRootDriver] RecenterInputOnly (dp=0).");
    }

    public void BeginTaskStartNeutralization()
    {
        if (!autoNeutralizeAfterConnect)
        {
            _neutralizeQueued = false;
            _neutralizeDone = true;
            _hasLockedWorldRotation = false;
            return;
        }

        if (phoneRx == null)
            phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();
        if (handRoot == null)
            return;

        if (!_hasBaseline && autoRecenterOnFirstPose && phoneRx != null && phoneRx.HasPhonePose)
            Recenter();

        _neutralizeDone = false;
        QueueNeutralize(force: true);
    }

    public void CompleteGripNeutralizationNow()
    {
        if (phoneRx == null)
            phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();

        if (phoneRx != null && phoneRx.HasPhonePose)
            RecenterInputOnly();

        _neutralizeQueued = false;
        _neutralizeDone = true;
        _hasLockedWorldRotation = false;
        _neutralizeStableAccum = 0f;
        _neutralizeHasPrevSample = false;
    }

    /// <summary>
    /// StudyFlowController can call this when condition changes.
    /// </summary>
    public void SetSideToFrontRemap(bool enabled, bool invert, bool forceRecenter = false)
    {
        bool changed = (enableSideToFrontRemap != enabled) || (invertSideToFront != invert);
        enableSideToFrontRemap = enabled;
        invertSideToFront = invert;

        if (changed && forceRecenter)
            Recenter();
    }

    public void ApplyHandLocationOffsets(bool isSide, bool keepWorldPose = true)
    {
        if (!useHandLocationOffsetProfiles)
            return;

        if (!_nearOffsetsCaptured)
            CaptureNearOffsetsFromCurrent();

        Vector3 prevRotationOffsetEuler = rotationOffsetEuler;

        if (isSide)
        {
            positionOffset = sidePositionOffset;
            rotationOffsetEuler = sideRotationOffsetEuler;
        }
        else
        {
            positionOffset = _nearPositionOffset;
            rotationOffsetEuler = _nearRotationOffsetEuler;
        }

        // When switching profile with keepWorldPose=false, apply the offset delta as an explicit
        // orientation change so profile rotation visibly affects the start pose.
        if (!keepWorldPose && handRoot != null)
        {
            Quaternion prevOff = Quaternion.Euler(prevRotationOffsetEuler);
            Quaternion nextOff = Quaternion.Euler(rotationOffsetEuler);
            Quaternion delta = nextOff * Quaternion.Inverse(prevOff);
            handRoot.rotation = handRoot.rotation * delta;
        }

        if (keepWorldPose)
            RebaselineKeepWorldPose();
    }

    public void ApplyNearToSideRotationDelta(bool rebaseline = false)
    {
        if (handRoot == null)
            return;

        if (!_nearOffsetsCaptured)
            CaptureNearOffsetsFromCurrent();

        Quaternion nearOff = Quaternion.Euler(_nearRotationOffsetEuler);
        Quaternion sideOff = Quaternion.Euler(sideRotationOffsetEuler);
        Quaternion delta = sideOff * Quaternion.Inverse(nearOff);
        handRoot.rotation = handRoot.rotation * delta;

        if (rebaseline)
            RebaselineKeepWorldPose();
    }

    public void CaptureCurrentAsNearOffsets()
    {
        _nearPositionOffset = positionOffset;
        _nearRotationOffsetEuler = rotationOffsetEuler;
        _nearOffsetsCaptured = true;
    }

    public void CaptureCurrentAsNearStartPose()
    {
        if (handRoot == null) return;
        _nearStartWorldPos = handRoot.position;
        _nearStartWorldRot = handRoot.rotation;
        _nearStartPoseCaptured = true;
    }

    public bool SnapToNearStartPose(bool keepCurrentRotation = true, bool rebaseline = false)
    {
        if (handRoot == null || !_nearStartPoseCaptured)
            return false;

        if (keepCurrentRotation)
            SnapToWorldPosition(_nearStartWorldPos, rebaseline);
        else
            SnapToWorldPose(_nearStartWorldPos, _nearStartWorldRot, rebaseline);

        return true;
    }

    public bool TryGetNearStartPose(out Vector3 worldPos, out Quaternion worldRot)
    {
        if (_nearStartPoseCaptured)
        {
            worldPos = _nearStartWorldPos;
            worldRot = _nearStartWorldRot;
            return true;
        }

        worldPos = Vector3.zero;
        worldRot = Quaternion.identity;
        return false;
    }

    public void SnapToWorldPosition(Vector3 worldPos, bool rebaseline = true)
    {
        if (handRoot == null) return;
        handRoot.position = worldPos;
        if (rebaseline)
            RebaselineKeepWorldPose();
    }

    public void SnapToWorldPose(Vector3 worldPos, Quaternion worldRot, bool rebaseline = true)
    {
        if (handRoot == null) return;
        handRoot.SetPositionAndRotation(worldPos, worldRot);
        if (rebaseline)
            RebaselineKeepWorldPose();
    }

    private static Vector3 RemapSideToFront(
        Vector3 dpWorld,
        Quaternion basisRot,
        bool invert,
        bool legacySwapLateralAndVerticalOutputs)
    {
        _ = legacySwapLateralAndVerticalOutputs;
        Vector3 local = Quaternion.Inverse(basisRot) * dpWorld;

        // Macro+Side keeps phone left/right as proxy left/right, and swaps phone vertical
        // with phone forward/back so side-body motion maps onto the workspace-facing axes.
        float x = local.x;
        float y = -local.z;
        float zSign = invert ? 1f : -1f;
        float z = local.y * zSign;

        Vector3 remapLocal = new Vector3(x, y, z);
        return basisRot * remapLocal;
    }

    public void RebaselineKeepWorldPose()
    {
        if (phoneRx == null || handRoot == null) return;
        if (!phoneRx.HasPhonePose) return;

        // 현재 phone을 baseline으로
        _phone0 = phoneRx.LatestPhonePose;
        if (!TryGetTranslationSamplePosition(_phone0, out Vector3 translationBaseline, out bool usingQrRelativeTranslation))
            return;

        if (!_hasTranslationYawAlign || recaptureTranslationYawOnInputRecenter)
            CaptureTranslationYawAlignment(usingQrRelativeTranslation);
        CaptureOffsetYawBasis();
        CaptureTranslationPositionBaseline(translationBaseline, usingQrRelativeTranslation);

        // Keep the macro root rotation baseline so dq==I preserves the current handRoot rotation.
        Quaternion rotOffset = Quaternion.Euler(rotationOffsetEuler);
        Quaternion rootRot = handRoot.rotation * Quaternion.Inverse(rotOffset);
        bool sideConditionActive = enableSideToFrontRemap;

        Vector3 offsetWorld = ComputeOffsetWorld(sideConditionActive, rootRot);
        CapturePositionPivotBaseline(offsetWorld);

        _root0 = new Pose(
            handRoot.position - offsetWorld,
            rootRot
        );

        _hasBaseline = true;
        Debug.Log("[PhoneProxyHandRootDriver] RebaselineKeepWorldPose (no jump).");
    }

    private void CaptureNearOffsetsFromCurrent()
    {
        _nearPositionOffset = positionOffset;
        _nearRotationOffsetEuler = rotationOffsetEuler;
        _nearOffsetsCaptured = true;
    }

    private Vector3 ComputeOffsetWorld(bool sideConditionActive, Quaternion rootRot)
    {
        bool useCameraYawOffset = sideConditionActive ? useCameraYawOffsetInSide : useCameraYawOffsetInNear;
        if (useCameraYawOffset)
        {
            Quaternion yawBasis = _hasOffsetYawBasis
                ? _offsetYawBasis
                : (cameraTransform != null ? Quaternion.Euler(0f, cameraTransform.eulerAngles.y, 0f) : Quaternion.identity);
            return yawBasis * new Vector3(positionOffset.x, 0f, positionOffset.z) + Vector3.up * positionOffset.y;
        }

        return rootRot * positionOffset;
    }

    private void CaptureOffsetYawBasis()
    {
        if (cameraTransform == null && Camera.main != null)
            cameraTransform = Camera.main.transform;

        if (cameraTransform == null)
        {
            _offsetYawBasis = Quaternion.identity;
            _hasOffsetYawBasis = false;
            return;
        }

        _offsetYawBasis = Quaternion.Euler(0f, cameraTransform.eulerAngles.y, 0f);
        _hasOffsetYawBasis = true;
    }

    private void CaptureTranslationPositionBaseline(Vector3 translationPosition, bool usingQrRelativeTranslation)
    {
        _translationPhone0Position = translationPosition;
        _hasTranslationPhone0Position = true;
        _translationBaselineUsesQrRelative = usingQrRelativeTranslation;
        _translationSuppressionActive = false;
        _translationSuppressionEnterAccum = 0f;
        _translationSuppressionExitAccum = 0f;
        _translationSuppressedRawDp = Vector3.zero;
    }

    private bool TryGetTranslationSamplePosition(Pose fallbackPhonePose, out Vector3 translationPosition, out bool usingQrRelativeTranslation)
    {
        if (usePhoneQrRelativeTranslation && phoneRx != null)
        {
            if (phoneRx.HasQrDeltaPose)
            {
                translationPosition = MapQrRelativeTranslationPosition(phoneRx.LatestQrDeltaPose.position);
                usingQrRelativeTranslation = true;
                _hasWarnedMissingQrTranslation = false;
                return true;
            }

            if (phoneRx.HasQrRelativePhonePose)
            {
                translationPosition = MapQrRelativeTranslationPosition(phoneRx.LatestQrRelativePhonePose.position);
                usingQrRelativeTranslation = true;
                _hasWarnedMissingQrTranslation = false;
                return true;
            }
        }

        if (usePhoneQrRelativeTranslation && requirePhoneQrRelativeTranslation)
        {
            translationPosition = Vector3.zero;
            usingQrRelativeTranslation = false;
            if (!_hasWarnedMissingQrTranslation)
            {
                Debug.LogWarning("[PhoneProxyHandRootDriver] Waiting for phone QR-relative translation. Phone packets must include qrCalibrated/dx_qr/dy_qr/dz_qr or mvis/mx/my/mz/mqx/mqy/mqz/mqw.");
                _hasWarnedMissingQrTranslation = true;
            }
            return false;
        }

        translationPosition = fallbackPhonePose.position;
        usingQrRelativeTranslation = false;
        return true;
    }

    private Vector3 MapQrRelativeTranslationPosition(Vector3 qrPosition)
    {
        Vector3 mapped = qrPosition;

        if (swapQrRelativePlanarTranslationAxes)
            mapped = new Vector3(qrPosition.z, qrPosition.y, qrPosition.x);

        if (invertQrRelativeLateral)
            mapped.x = -mapped.x;

        if (invertQrRelativeForwardBack)
            mapped.z = -mapped.z;

        return mapped;
    }

    private void LogTranslationFrameSourceIfChanged(bool usingQrRelativeTranslation)
    {
        if (!logTranslationFrameSource)
            return;

        if (_hasLoggedTranslationFrameSource && _loggedTranslationFrameUsesQrRelative == usingQrRelativeTranslation)
            return;

        _hasLoggedTranslationFrameSource = true;
        _loggedTranslationFrameUsesQrRelative = usingQrRelativeTranslation;
        Debug.Log($"[PhoneProxyHandRootDriver] Translation frame: {GetTranslationFrameName(usingQrRelativeTranslation)}.");
    }

    private static string GetTranslationFrameName(bool usingQrRelativeTranslation)
    {
        return usingQrRelativeTranslation ? "phone QR-relative" : "raw phone AR-session";
    }

    private Transform ResolvePositionPivotTransform()
    {
        if (gripPivot != null)
            return gripPivot;

        if (grabber == null)
            grabber = FindFirstObjectByType<ProxyHandGrabber>();

        if (grabber != null && grabber.grabAnchor != null)
            return grabber.grabAnchor;

        return handRoot;
    }

    private void CapturePositionPivotBaseline(Vector3 offsetWorld)
    {
        Transform positionPivot = ResolvePositionPivotTransform();
        if (handRoot == null || positionPivot == null)
        {
            _positionPivot0World = handRoot != null ? handRoot.position - offsetWorld : Vector3.zero;
            _rootToPositionPivotLocal = Vector3.zero;
            _hasPositionPivotCalibration = false;
            return;
        }

        Vector3 pivotWorld;
        if (_hasPositionPivotCalibration)
        {
            pivotWorld = handRoot.position + (handRoot.rotation * _rootToPositionPivotLocal);
        }
        else
        {
            pivotWorld = positionPivot.position;

            Vector3 rootToPivotWorld = pivotWorld - handRoot.position;
            _rootToPositionPivotLocal = Quaternion.Inverse(handRoot.rotation) * rootToPivotWorld;
            _hasPositionPivotCalibration = true;
        }

        _positionPivot0World = pivotWorld - offsetWorld;
    }

    private bool TryGetRemapBasisRotation(out Quaternion basis)
    {
        if (remapFrameTransform == null)
        {
            GameObject go = GameObject.Find("WorkshopEnvironment");
            if (go != null) remapFrameTransform = go.transform;
        }

        if (remapFrameTransform == null)
        {
            basis = Quaternion.identity;
            return false;
        }

        basis = Quaternion.Euler(0f, remapFrameTransform.eulerAngles.y, 0f);
        return true;
    }

    private Vector3 AlignTranslationDelta(Vector3 rawDelta, bool applyPlanarAxisMapping = true)
    {
        if (!alignTranslationToWorkspaceYaw || !_hasTranslationYawAlign)
            return rawDelta;

        if (alignTranslationPlanarOnly)
        {
            Vector3 planar = new Vector3(rawDelta.x, 0f, rawDelta.z);
            Vector3 alignedPlanar = _translationYawAlign * planar;
            if (applyPlanarAxisMapping)
                alignedPlanar = ApplyAlignedPlanarTranslationMapping(alignedPlanar);
            return new Vector3(alignedPlanar.x, rawDelta.y, alignedPlanar.z);
        }

        Vector3 aligned = _translationYawAlign * rawDelta;
        Vector3 alignedPlanarMapped = new Vector3(aligned.x, 0f, aligned.z);
        if (applyPlanarAxisMapping)
            alignedPlanarMapped = ApplyAlignedPlanarTranslationMapping(alignedPlanarMapped);
        return new Vector3(alignedPlanarMapped.x, aligned.y, alignedPlanarMapped.z);
    }

    private Vector3 ApplyAlignedPlanarTranslationMapping(Vector3 planar)
    {
        Vector3 mapped = planar;
        if (swapAlignedPlanarTranslationAxes)
            mapped = new Vector3(planar.z, 0f, planar.x);

        if (invertMappedPlanarForwardBack)
            mapped.z = -mapped.z;

        return mapped;
    }

    private void CaptureTranslationYawAlignment(bool usingQrRelativeTranslation = false)
    {
        if (!alignTranslationToWorkspaceYaw)
        {
            _translationYawAlign = Quaternion.identity;
            _hasTranslationYawAlign = false;
            return;
        }

        if (!TryGetRemapBasisRotation(out Quaternion workspaceYaw))
        {
            _translationYawAlign = Quaternion.identity;
            _hasTranslationYawAlign = false;
            return;
        }

        if (!usingQrRelativeTranslation)
            workspaceYaw = workspaceYaw * Quaternion.Euler(0f, translationYawOffsetDeg, 0f);

        _translationYawAlign = workspaceYaw;
        _hasTranslationYawAlign = true;
    }

    private void QueueNeutralizeIfNeeded()
    {
        QueueNeutralize(force: false);
    }

    private void QueueNeutralize(bool force)
    {
        if (!autoNeutralizeAfterConnect)
            return;
        if (!force && neutralizeOnlyOnce && _neutralizeDone)
            return;

        _neutralizeQueued = true;
        _neutralizeQueuedAtTime = Time.unscaledTime;
        _neutralizeAtTime = Time.unscaledTime + Mathf.Max(0f, neutralizeDelaySeconds);
        _neutralizeStableAccum = 0f;
        _neutralizeHasPrevSample = false;
        _neutralizePrevSampleTime = -1f;
        _lockedWorldRotation = handRoot != null ? handRoot.rotation : Quaternion.identity;
        _hasLockedWorldRotation = handRoot != null;
    }

    private void TryRunQueuedNeutralize()
    {
        if (!_neutralizeQueued)
            return;
        if (Time.unscaledTime < _neutralizeAtTime)
            return;
        if (grabber != null && grabber.IsHolding)
            return;
        if (phoneRx == null || !phoneRx.HasPhonePose)
            return;

        if (!neutralizeWhenPhoneStable)
        {
            RecenterInputOnly();
            _neutralizeQueued = false;
            _neutralizeDone = true;
            _hasLockedWorldRotation = false;
            return;
        }

        float now = Time.unscaledTime;
        Quaternion curRot = phoneRx.LatestPhonePose.rotation;

        if (!_neutralizeHasPrevSample)
        {
            _neutralizeHasPrevSample = true;
            _neutralizePrevPhoneRot = curRot;
            _neutralizePrevSampleTime = now;
            return;
        }

        float dt = Mathf.Max(1e-4f, now - _neutralizePrevSampleTime);
        float angDeg = Quaternion.Angle(_neutralizePrevPhoneRot, curRot);
        float angSpeedDegPerSec = angDeg / dt;

        if (angSpeedDegPerSec <= Mathf.Max(0f, neutralizeStableAngularSpeedDegPerSec))
            _neutralizeStableAccum += dt;
        else
            _neutralizeStableAccum = 0f;

        _neutralizePrevPhoneRot = curRot;
        _neutralizePrevSampleTime = now;

        float stableHold = Mathf.Max(0.05f, neutralizeStableHoldSeconds);
        float maxWait = Mathf.Max(stableHold, neutralizeMaxWaitSeconds);
        bool stableReady = _neutralizeStableAccum >= stableHold;
        bool timeoutReady = (now - _neutralizeQueuedAtTime) >= maxWait;

        if (stableReady || timeoutReady)
        {
            RecenterInputOnly();
            _neutralizeQueued = false;
            _neutralizeDone = true;
            _hasLockedWorldRotation = false;
        }
    }

    private bool ShouldLockRotationForNeutralize()
    {
        if (!lockRotationUntilNeutralized)
            return false;
        if (!autoNeutralizeAfterConnect)
            return false;
        return _neutralizeQueued;
    }

    private Pose GetDrivenPhonePose()
    {
        Pose latest = phoneRx.LatestPhonePose;
        if (!predictShortPoseGaps)
            return latest;

        if (!phoneRx.TryGetPhoneMotionEstimate(
                out Pose latestPose,
                out Vector3 linearVel,
                out Vector3 angularVelDegPerSec,
                out float ageSec))
        {
            return latest;
        }

        float maxGap = Mathf.Max(0f, predictionMaxGapSeconds);
        if (ageSec <= 1e-4f || maxGap <= 1e-4f || ageSec > maxGap)
            return latestPose;

        float leadSec = Mathf.Min(ageSec, Mathf.Max(0f, predictionMaxLeadSeconds));
        float fade = 1f - Mathf.Clamp01(ageSec / maxGap);

        if (predictionMaxLinearSpeedMetersPerSec > 0f)
            linearVel = Vector3.ClampMagnitude(linearVel, predictionMaxLinearSpeedMetersPerSec);

        Vector3 predictedPos = latestPose.position + (linearVel * (leadSec * fade));
        Quaternion predictedRot = latestPose.rotation;

        if (predictRotationDuringShortGaps)
        {
            if (predictionMaxAngularSpeedDegPerSec > 0f)
                angularVelDegPerSec = Vector3.ClampMagnitude(angularVelDegPerSec, predictionMaxAngularSpeedDegPerSec);

            float angSpeed = angularVelDegPerSec.magnitude;
            if (angSpeed > 1e-3f)
            {
                Vector3 axis = angularVelDegPerSec / angSpeed;
                float angDeg = angSpeed * leadSec * fade;
                predictedRot = Quaternion.AngleAxis(angDeg, axis) * latestPose.rotation;
            }
        }

        return new Pose(predictedPos, predictedRot);
    }

    private void UpdateRotationDominantTranslationSuppression(Vector3 phonePosition, float dt)
    {
        if (!suppressTranslationDuringHighRotation || phoneRx == null)
            return;

        if (!_hasTranslationPhone0Position)
        {
            _translationPhone0Position = phonePosition;
            _hasTranslationPhone0Position = true;
        }

        bool hasEstimate = phoneRx.TryGetPhoneMotionEstimate(
            out _,
            out Vector3 linearVelocityMetersPerSec,
            out Vector3 angularVelocityDegPerSec,
            out float ageSec);

        if (!hasEstimate || ageSec > Mathf.Max(0.01f, translationSuppressionMaxMotionAgeSeconds))
        {
            _translationSuppressionEnterAccum = 0f;
            if (_translationSuppressionActive)
            {
                _translationSuppressionExitAccum += dt;
                if (_translationSuppressionExitAccum >= Mathf.Max(0.01f, translationSuppressionExitHoldSeconds))
                {
                    _translationSuppressionActive = false;
                    _translationSuppressionExitAccum = 0f;
                }
            }
            return;
        }

        float angularSpeedDegPerSec = angularVelocityDegPerSec.magnitude;
        float linearSpeedMetersPerSec = linearVelocityMetersPerSec.magnitude;

        bool shouldSuppressNow =
            angularSpeedDegPerSec >= Mathf.Max(0f, translationSuppressionAngularSpeedDegPerSec) &&
            linearSpeedMetersPerSec <= Mathf.Max(0f, translationSuppressionLinearSpeedMetersPerSec);

        if (shouldSuppressNow)
        {
            _translationSuppressionEnterAccum += dt;
            _translationSuppressionExitAccum = 0f;

            if (!_translationSuppressionActive &&
                _translationSuppressionEnterAccum >= Mathf.Max(0.01f, translationSuppressionEnterHoldSeconds))
            {
                _translationSuppressionActive = true;
                _translationSuppressedRawDp = phonePosition - _translationPhone0Position;
            }
        }
        else
        {
            _translationSuppressionEnterAccum = 0f;
            if (_translationSuppressionActive)
            {
                _translationSuppressionExitAccum += dt;
                if (_translationSuppressionExitAccum >= Mathf.Max(0.01f, translationSuppressionExitHoldSeconds))
                {
                    _translationSuppressionActive = false;
                    _translationSuppressionExitAccum = 0f;
                }
            }
        }

        if (_translationSuppressionActive)
            _translationPhone0Position = phonePosition - _translationSuppressedRawDp;
    }
}




