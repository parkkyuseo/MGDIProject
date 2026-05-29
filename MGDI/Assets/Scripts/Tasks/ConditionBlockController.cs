using System;
using System.Collections.Generic;
using UnityEngine;
using System.Collections;

public class ConditionBlockController : MonoBehaviour
{
    public enum Technique { Macro, Micro }
    public enum HandLocation { NearHead, Side } // "Side" means side-of-body input.

    [Serializable]
    public class Condition
    {
        public string label = "Cond";
        public Technique technique = Technique.Macro;
        public HandLocation handLocation = HandLocation.NearHead;

        [Tooltip("Only used for Macro+Side remap.")]
        public bool invertSideToFront = false;
    }

    [Header("Order (edit in Inspector)")]
    public List<Condition> conditions = new List<Condition>();

    [Header("Balanced Latin Square")]
    [SerializeField] private bool useBalancedLatinSquare = true;
    [SerializeField] private bool useParticipantIdForSequence = true;
    [SerializeField] private string participantIdPrefsKey = "participant_id";
    [Tooltip("Used when Participant ID has no numeric part or participant-based sequence is disabled.")]
    [SerializeField] private int manualSequenceNumber = 1;

    [Header("Refs")]
    [SerializeField] private WorkflowProgressionController workflow;
    [SerializeField] private ToolPlacementTaskManager placementTask;
    [SerializeField] private ToolRotationTaskManager rotationTask;
    [SerializeField] private ToolScalingTaskManager scalingTask;
    [SerializeField] private PhoneInputRouter phoneRouter;
    [SerializeField] private PhonePoseStreamReceiver phonePoseReceiver;
    [SerializeField] private PhoneTechniqueGate phoneTechniqueGate;
    [SerializeField] private PhoneProxyHandRootDriver phoneMacroPoseDriver;
    [SerializeField] private QRWorkspaceLock_OpenXR qrLock;
    [SerializeField] private ProxyHandGrabber grabber;
    [SerializeField] private InstructionHUD instructionHUD;

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

    [SerializeField] private TaskContextHUD taskContextHUD;
    [SerializeField] private BasketToolResetter basketResetter;

    [Header("Macro Start (Basket-based)")]
    [SerializeField] private bool useBasketBasedSideStart = true;
    [Tooltip("If empty, falls back to BasketToolResetter root.")]
    [SerializeField] private Transform sideStartReference;
    [SerializeField] private Vector3 nearStartLocalOffset = new Vector3(0f, 0.08f, 0.25f);
    [SerializeField] private Vector3 sideStartLocalOffset = new Vector3(0f, 0.08f, 0.25f);
    [SerializeField] private bool keepCurrentHandRotationOnSideStart = true;
    [SerializeField] private float maxAbsSideStartOffsetMeters = 1.0f;
    [Tooltip("If true, macro start snap always uses reference/captured start rotation instead of current rotation.")]
    [SerializeField] private bool forceReferenceRotationOnMacroStart = true;

    [Header("Macro Near Start (Initial Pose-based)")]
    [SerializeField] private bool useCapturedNearStartOnMacroNear = true;
    [SerializeField] private bool keepCurrentHandRotationOnNearStart = true;

    [Header("Per-Trial Hand Start Snap")]
    [SerializeField] private bool snapProxyHandPositionOnEachTrial = true;
    [SerializeField] private bool snapDuringMacroTrials = false;
    [SerializeField] private bool snapDuringMicroTrials = true;

    [Header("Condition Entry Posture Gate")]
    [SerializeField] private bool useConditionEntryPostureGate = true;
    [Header("Fast Test Timing")]
    [SerializeField] private bool useFastTestTiming = false;
    [SerializeField] private float fastPhoneAppReadyMessageSeconds = 1f;
    [SerializeField] private float fastConditionBreakSeconds = 5f;
    [SerializeField] private bool showPhoneAppReadyMessageOnFirstStart = true;
    [TextArea(2, 4)]
    [SerializeField] private string phoneAppReadyMessage = "Look around and check the basket and tool locations.\nGet familiar with the workspace before starting.";
    [SerializeField] private float phoneAppReadyMessageSeconds = 12f;
    [TextArea(2, 4)]
    [SerializeField] private string firstConditionEntryPostureText = "Open the phone app.\nScan the QR code with the phone.\nMatch the posture shown in the image.\nTriple tap to start.";
    [TextArea(2, 4)]
    [SerializeField] private string repeatConditionEntryPostureText = "Open the phone app.\nScan the QR code with the phone.\nMatch the posture shown in the image.\nTriple tap to start.";
    [TextArea(2, 4)]
    [SerializeField] private string conditionBreakText = "Condition complete.\nPlease hand the phone to the researcher.\nTake a short break and complete the questionnaire for this condition.";
    [SerializeField] private float conditionBreakSeconds = 5f;
    [SerializeField] private float conditionBreakPostReadSeconds = 2f;
    [SerializeField] private bool requireTripleTapToLeaveConditionBreak = true;
    [SerializeField] private float conditionEntryIntroSeconds = 4f;
    [SerializeField] private float conditionEntryHoldSeconds = 3f;
    [SerializeField] private bool showConditionEntryCountdown = true;
    [SerializeField] private int conditionEntryCountdownSeconds = 5;
    [SerializeField] private bool disableConditionEntryCountdown = true;
    [SerializeField] private bool useSpeechAwareGateTiming = true;
    [SerializeField] private float gateSpeechCharsPerSecond = 13f;
    [SerializeField] private float gateSpeechWordsPerSecond = 2.5f;
    [SerializeField] private float gateSpeechMinSeconds = 0.35f;
    [SerializeField] private bool waitForGateSpeechCompletion = true;
    [SerializeField] private bool restoreInitialRotationOnConditionEntry = true;
    [SerializeField] private bool rebaselineAfterConditionEntry = true;

    [Header("Phone Connection Gate")]
    [SerializeField] private bool requirePhoneConnectionBeforeTaskStart = true;
    [SerializeField] private float phoneConnectionFreshSeconds = 0.75f;
    [SerializeField] private float phoneConnectionReadyDelaySeconds = 2.5f;
    [Tooltip("<=0 keeps waiting until phone data arrives.")]
    [SerializeField] private float phoneConnectionMaxWaitSeconds = 0f;
    [TextArea(2, 4)]
    [SerializeField] private string waitingForPhoneConnectionText = "Waiting for phone connection.\nOpen the phone app if needed.";
    [TextArea(2, 4)]
    [SerializeField] private string readyTripleTapAcknowledgementText = "Triple tap detected.\nHold still.";

    [Header("Researcher Explanation Gates")]
    [SerializeField] private bool showConditionExplanationGate = true;
    [TextArea(3, 6)]
    [SerializeField] private string conditionExplanationTextTemplate = "Condition {condition_index}\n{condition_name}\nThis condition will be explained.\nAfter the explanation is finished, triple tap on the phone to continue.";

    [Header("Phone QR Gate")]
    [SerializeField] private bool requireFreshPhoneQrBeforeConditionReady = true;
    [SerializeField] private string waitingForPhoneQrText = "Scan the phone QR code first.";
    [SerializeField] private float waitingForPhoneQrReminderSeconds = 1.5f;

    [Header("Condition Entry Baseline Stability")]
    [SerializeField] private bool requirePhoneStillnessBeforeConditionBaseline = true;
    [SerializeField] private float conditionEntryStableAngularSpeedDegPerSec = 12f;
    [SerializeField] private float conditionEntryStableLinearSpeedMetersPerSec = 0.08f;
    [SerializeField] private float conditionEntryStableHoldSeconds = 0.35f;
    [SerializeField] private float conditionEntryStableTimeoutSeconds = 2f;
    [SerializeField] private bool logConditionEntryStillness = false;

    [Header("Condition Entry Example Images")]
    [SerializeField] private bool showConditionEntryExampleImage = true;
    [SerializeField] private bool hideConditionEntryExampleBeforeCountdown = true;
    [SerializeField] private float conditionEntryExampleMinViewSeconds = 12f;
    [SerializeField] private Sprite macroNearHeadExampleImage;
    [SerializeField] private Sprite macroSideOfBodyExampleImage;
    [SerializeField] private Sprite microNearHeadExampleImage;
    [SerializeField] private Sprite microSideOfBodyExampleImage;

    [Header("Study Complete")]
    [SerializeField] private bool showStudyCompletePanel = true;
    [TextArea(2, 4)]
    [SerializeField] private string studyCompleteText = "Experiment complete.\nThank you.";

    private int _condIndex = 0;
    private Condition _currentCondition;
    private bool _initialConditionApplied = false;
    private bool _nearStartPoseCaptured = false;
    private bool _macroEntryInitialized = false;
    private int _selectedSequenceIndex1Based = 1;
    private int _selectedSequenceCount = 1;
    private bool _initialStartRotationCaptured = false;
    private Quaternion _initialStartRotation = Quaternion.identity;
    private Coroutine _conditionEntryCoroutine;
    private int _conditionEntryToken = 0;
    private bool _warnedInstructionHudMissing = false;
    private bool _awaitingInitialConditionReady = false;
    private bool _initialConditionReady = false;
    private bool _startupPhoneAppMessageShown = false;
    private bool _hasCompletedConditionEntryOnce = false;
    private string _conditionEntryPhoneQrBaselineKey = "";
    private bool _conditionEntryPhoneQrBaselineSet = false;

    public Technique CurrentTechnique => _currentCondition != null ? _currentCondition.technique : Technique.Macro;
    public HandLocation CurrentHandLocation => _currentCondition != null ? _currentCondition.handLocation : HandLocation.NearHead;
    public bool HasCurrentCondition => _currentCondition != null;
    public int CurrentConditionIndex1Based => (_currentCondition != null && conditions != null && conditions.Count > 0)
        ? Mathf.Clamp(_condIndex + 1, 1, conditions.Count)
        : -1;
    public int ConditionCount => conditions != null ? conditions.Count : 0;
    public int SelectedSequenceIndex1Based => _selectedSequenceIndex1Based;
    public int SelectedSequenceCount => _selectedSequenceCount;
    public bool IsInitialConditionReady => _initialConditionReady;

    public string GetConditionLabel()
    {
        if (_currentCondition == null)
            return "Macro - Near Head";

        string tech = _currentCondition.technique == Technique.Micro ? "Micro" : "Macro";
        string location = _currentCondition.handLocation == HandLocation.Side ? "Side Of Body" : "Near Head";
        return $"{tech} - {location}";
    }

    public string GetConditionOrderLabel()
    {
        if (conditions == null || conditions.Count == 0)
            return string.Empty;

        var parts = new List<string>(conditions.Count);
        for (int i = 0; i < conditions.Count; i++)
        {
            Condition c = conditions[i];
            if (c == null)
            {
                parts.Add($"{i + 1}:Unknown");
                continue;
            }

            string label = string.IsNullOrWhiteSpace(c.label) ? $"Cond{i + 1}" : c.label.Trim();
            string tech = c.technique == Technique.Micro ? "Micro" : "Macro";
            string location = c.handLocation == HandLocation.Side ? "Side" : "Near";
            parts.Add($"{i + 1}:{label}[{tech}-{location}]");
        }

        return string.Join(" > ", parts);
    }

    void Start()
    {
        if (workflow == null) workflow = FindFirstObjectByType<WorkflowProgressionController>();
        if (placementTask == null) placementTask = FindFirstObjectByType<ToolPlacementTaskManager>();
        if (rotationTask == null) rotationTask = FindFirstObjectByType<ToolRotationTaskManager>();
        if (scalingTask == null) scalingTask = FindFirstObjectByType<ToolScalingTaskManager>();
        if (phoneRouter == null) phoneRouter = FindFirstObjectByType<PhoneInputRouter>();
        if (phonePoseReceiver == null) phonePoseReceiver = FindFirstObjectByType<PhonePoseStreamReceiver>();
        if (phoneTechniqueGate == null) phoneTechniqueGate = FindFirstObjectByType<PhoneTechniqueGate>();
        if (phoneMacroPoseDriver == null) phoneMacroPoseDriver = FindFirstObjectByType<PhoneProxyHandRootDriver>();
        if (qrLock == null) qrLock = FindFirstObjectByType<QRWorkspaceLock_OpenXR>();
        if (grabber == null) grabber = FindFirstObjectByType<ProxyHandGrabber>();
        if (taskContextHUD == null) taskContextHUD = FindFirstObjectByType<TaskContextHUD>();
        ResolveInstructionHudIfNeeded();

        if (workflow == null)
        {
            Debug.LogError("[Cond] WorkflowProgressionController not found.");
            return;
        }

        if (conditions == null || conditions.Count == 0)
            _initialConditionReady = true;

        workflow.OnAllCompleted += HandleBlockCompleted;
        SubscribeTrialStartEvents();

        ApplyBalancedLatinSquareOrderIfEnabled();
        _condIndex = Mathf.Clamp(_condIndex, 0, Mathf.Max(0, conditions.Count - 1));
        StartCoroutine(ApplyInitialConditionWhenWorkspaceReady());
    }

    void OnDestroy()
    {
        if (workflow != null)
            workflow.OnAllCompleted -= HandleBlockCompleted;
        UnsubscribeTrialStartEvents();
        if (_conditionEntryCoroutine != null)
            StopCoroutine(_conditionEntryCoroutine);
        if (phoneTechniqueGate != null)
            phoneTechniqueGate.SetInputFrozen(false);
        if (phoneRouter != null)
            phoneRouter.SetInputSuppressed(false);
    }

    private void HandleBlockCompleted()
    {
        if (conditions == null || conditions.Count == 0) return;

        _condIndex++;
        if (_condIndex >= conditions.Count)
        {
            if (logDebug) Debug.Log("[Cond] All conditions completed.");
            HandleStudyCompleted();
            return;
        }

        ApplyCurrentConditionAndRestartWorkflow();
    }

    private void HandleStudyCompleted()
    {
        if (_conditionEntryCoroutine != null)
        {
            StopCoroutine(_conditionEntryCoroutine);
            _conditionEntryCoroutine = null;
        }

        if (grabber != null)
            grabber.ForceRelease();

        if (phoneTechniqueGate != null)
            phoneTechniqueGate.SetInputFrozen(true);
        if (phoneRouter != null)
            phoneRouter.SetInputSuppressed(true);

        if (taskContextHUD != null)
        {
            taskContextHUD.SetPracticeText("");
            taskContextHUD.SetVisible(false);
        }

        ResolveInstructionHudIfNeeded();
        if (instructionHUD == null || !showStudyCompletePanel)
            return;

        instructionHUD.HideExample();
        instructionHUD.Show(studyCompleteText, float.PositiveInfinity);
    }

    private System.Collections.IEnumerator ApplyInitialConditionWhenWorkspaceReady()
    {
        while (qrLock == null)
        {
            qrLock = FindFirstObjectByType<QRWorkspaceLock_OpenXR>();
            if (qrLock != null) break;
            yield return null;
        }

        if (qrLock != null)
        {
            while (!qrLock.IsWorkspaceReady)
                yield return null;
        }

        if (_initialConditionApplied)
            yield break;

        if (phoneTechniqueGate != null)
            phoneTechniqueGate.SetInputFrozen(true);
        if (phoneRouter != null)
            phoneRouter.SetInputSuppressed(true);

        if (showPhoneAppReadyMessageOnFirstStart && !_startupPhoneAppMessageShown)
        {
            ResolveInstructionHudIfNeeded();

            float wait = GetPhoneAppReadyWaitSeconds();
            if (instructionHUD != null && !string.IsNullOrWhiteSpace(phoneAppReadyMessage))
            {
                wait = Mathf.Max(wait, instructionHUD.Show(phoneAppReadyMessage, wait));
            }

            _startupPhoneAppMessageShown = true;

            if (wait > 0f)
                yield return new WaitForSeconds(wait);
        }

        _initialConditionApplied = true;
        _awaitingInitialConditionReady = true;
        _initialConditionReady = false;
        ApplyCurrentConditionAndRestartWorkflow();
    }

    private void ApplyCurrentConditionAndRestartWorkflow()
    {
        if (conditions == null || conditions.Count == 0)
        {
            Debug.LogWarning("[Cond] No conditions defined.");
            return;
        }

        var c = conditions[_condIndex];
        _currentCondition = c;

        if (phoneMacroPoseDriver != null && !_nearStartPoseCaptured)
        {
            phoneMacroPoseDriver.CaptureCurrentAsNearStartPose();
            _nearStartPoseCaptured = true;
        }
        CaptureInitialStartRotationIfNeeded();

        if (phoneRouter != null)
        {
            if (c.technique == Technique.Micro) phoneRouter.SetModeMicro();
            else phoneRouter.SetModeMacro();
        }

        bool isMacro = (c.technique == Technique.Macro);
        bool isSide = (c.handLocation == HandLocation.Side);
        bool remapOn = isMacro && isSide;
        bool firstMacroEntry = isMacro && !_macroEntryInitialized;

        bool useBasketSideStartNow = isMacro && useBasketBasedSideStart;
        bool snappedToBasketSideStart = false;
        if (useBasketSideStartNow)
        {
            Vector3 localOffset = isSide ? sideStartLocalOffset : nearStartLocalOffset;
            string label = isSide ? "side" : "near";
            bool keepCurrentRotationForMode = true; // position-only move at macro condition start

            snappedToBasketSideStart = TrySnapProxyHandToBasketStart(localOffset, label, keepCurrentRotationForMode);
        }

        if (useBasketSideStartNow && !snappedToBasketSideStart && logDebug)
            Debug.LogWarning("[Cond] Basket-based macro start is enabled, but start reference is missing. Keeping current proxy hand pose.");

        bool useCapturedNearStartNow = isMacro && !isSide && useCapturedNearStartOnMacroNear && !useBasketSideStartNow;
        bool snappedToNearStart = false;
        if (useCapturedNearStartNow && phoneMacroPoseDriver != null)
        {
            if (grabber != null)
                grabber.ForceRelease();

            bool keepCurrentRotation = true; // position-only move at macro condition start

            snappedToNearStart = phoneMacroPoseDriver.SnapToNearStartPose(
                keepCurrentRotation: keepCurrentRotation,
                rebaseline: false);
        }

        if (useCapturedNearStartNow && !snappedToNearStart && logDebug)
            Debug.LogWarning("[Cond] Captured near-start is enabled, but near-start pose is unavailable. Keeping current proxy hand pose.");

        if (phoneMacroPoseDriver != null)
        {
            phoneMacroPoseDriver.SetSideToFrontRemap(remapOn, c.invertSideToFront, forceRecenter: false);

            if (isMacro)
            {
                bool applySideOffsetProfile = isSide;
                // Macro condition changes should not rotate the proxy hand.
                phoneMacroPoseDriver.ApplyHandLocationOffsets(applySideOffsetProfile, keepWorldPose: true);
                phoneMacroPoseDriver.RebaselineKeepWorldPose();

                if (firstMacroEntry)
                    _macroEntryInitialized = true;
            }
        }

        if (taskContextHUD != null)
        {
            taskContextHUD.SetVisible(false);

            string condText = GetConditionLabel();
            if ((c.technique == Technique.Macro) && (c.handLocation == HandLocation.Side))
                condText += c.invertSideToFront ? " - Remap(Invert)" : " - Remap";

            taskContextHUD.SetCondition(condText);
        }

        if (logDebug)
        {
            Debug.Log($"[Cond] Apply #{_condIndex + 1}/{conditions.Count} '{c.label}' tech={c.technique} loc={c.handLocation} remap={remapOn} invert={c.invertSideToFront}");
        }

        if (workflow != null)
        {
            StartConditionEntryAndRestart();
        }
    }

    private void StartConditionEntryAndRestart()
    {
        _conditionEntryToken++;
        int token = _conditionEntryToken;

        if (phoneTechniqueGate != null)
            phoneTechniqueGate.SetInputFrozen(true);
        if (phoneRouter != null)
            phoneRouter.SetInputSuppressed(true);

        if (_conditionEntryCoroutine != null)
        {
            StopCoroutine(_conditionEntryCoroutine);
            _conditionEntryCoroutine = null;
            if (instructionHUD != null)
                instructionHUD.HideImmediate();
        }

        _conditionEntryCoroutine = StartCoroutine(ConditionEntryThenRestart(token));
    }

    private IEnumerator ConditionEntryThenRestart(int token)
    {
        bool useGate = useConditionEntryPostureGate && phoneMacroPoseDriver != null;
        bool isFirstConditionEntry = !_hasCompletedConditionEntryOnce;
        if (useGate)
        {
            if (logDebug)
                Debug.Log($"[Cond] Condition-entry posture gate start. cond={GetConditionLabel()} mode=phone-connection-wait stableDelay={Mathf.Max(0f, phoneConnectionReadyDelaySeconds):F2}s");

            ResolveInstructionHudIfNeeded();

            if (phoneTechniqueGate != null)
                phoneTechniqueGate.SetInputFrozen(true);

            if (phonePoseReceiver == null)
                phonePoseReceiver = FindFirstObjectByType<PhonePoseStreamReceiver>();
            if (phonePoseReceiver != null)
                phonePoseReceiver.DisarmPhoneQrDetectedAnnouncement();

            if (grabber != null)
                grabber.ForceRelease();

            if (!isFirstConditionEntry)
            {
                if (instructionHUD != null)
                {
                    instructionHUD.HideExample();
                    if (!string.IsNullOrWhiteSpace(conditionBreakText))
                        instructionHUD.Show(conditionBreakText, float.PositiveInfinity);
                }

                if (instructionHUD != null && !string.IsNullOrWhiteSpace(conditionBreakText))
                {
                    float breakWait = GetConditionBreakWaitSeconds();
                    yield return instructionHUD.WaitForTaskGate(conditionBreakText, breakWait);
                }
                else
                {
                    float breakWait = GetConditionBreakWaitSeconds();
                    if (breakWait > 0f)
                        yield return new WaitForSeconds(breakWait);
                }

                float postReadWait = Mathf.Max(0f, conditionBreakPostReadSeconds);
                if (postReadWait > 0f)
                    yield return new WaitForSeconds(postReadWait);

                yield return WaitForConditionBreakReadyTripleTap(token);
                if (token != _conditionEntryToken)
                    yield break;

                if (instructionHUD != null)
                    instructionHUD.HideImmediate();
            }

            yield return ShowConditionExplanationGate(token);
            if (token != _conditionEntryToken)
                yield break;

            ApplyConditionEntryStartPose();

            if (phonePoseReceiver == null)
                phonePoseReceiver = FindFirstObjectByType<PhonePoseStreamReceiver>();
            if (phonePoseReceiver != null)
            {
                _conditionEntryPhoneQrBaselineKey = phonePoseReceiver.LatestPhoneQrDetectionKey;
                _conditionEntryPhoneQrBaselineSet = true;
                phonePoseReceiver.ArmPhoneQrDetectedAnnouncement(requireNewDetection: true);
            }
            else
            {
                _conditionEntryPhoneQrBaselineKey = "";
                _conditionEntryPhoneQrBaselineSet = false;
            }

            Sprite exampleSprite = GetConditionEntryExampleSprite();
            bool showingExample = showConditionEntryExampleImage && exampleSprite != null;
            string posturePrompt = isFirstConditionEntry ? firstConditionEntryPostureText : repeatConditionEntryPostureText;

            if (instructionHUD != null)
            {
                if (showingExample)
                    instructionHUD.ShowExample(exampleSprite);
                else
                    instructionHUD.HideExample();

                if (!string.IsNullOrWhiteSpace(posturePrompt))
                    instructionHUD.Show(posturePrompt, float.PositiveInfinity);
            }

            yield return WaitForReadyTripleTapThenPhoneConnection(token);
            yield return WaitForPhoneStillnessBeforeConditionBaseline(token);

            if (instructionHUD != null)
            {
                instructionHUD.HideExample();
                instructionHUD.HideImmediate();
            }

            if (rebaselineAfterConditionEntry)
            {
                phoneMacroPoseDriver.RebaselineKeepWorldPose();
                // Finalize grip neutralization at gate end so current comfortable phone pose
                // becomes the explicit input-neutral reference.
                phoneMacroPoseDriver.CompleteGripNeutralizationNow();
            }

            if (phoneTechniqueGate != null)
            {
                phoneTechniqueGate.SetInputFrozen(false, recenterOnUnfreeze: false);
                phoneTechniqueGate.RecenterCurrentModeBaselines();
            }
            if (phoneRouter != null)
                phoneRouter.SetInputSuppressed(false);

            if (logDebug)
                Debug.Log("[Cond] Condition-entry posture gate done. Input unfrozen.");
        }

        if (!useGate)
            yield return WaitForPhoneConnectionIfNeeded(token);

        if (token != _conditionEntryToken)
        {
            if (instructionHUD != null)
                instructionHUD.HideImmediate();
            _conditionEntryCoroutine = null;
            yield break;
        }

        if (workflow != null)
        {
            if (basketResetter != null)
                basketResetter.ResetAllToolsToBasket();
            workflow.RestartFromBeginning();
        }

        if (!useGate)
        {
            if (phoneTechniqueGate != null)
            {
                phoneTechniqueGate.SetInputFrozen(false, recenterOnUnfreeze: false);
                phoneTechniqueGate.RecenterCurrentModeBaselines();
            }

            if (phoneRouter != null)
                phoneRouter.SetInputSuppressed(false);
        }

        if (_awaitingInitialConditionReady)
        {
            _awaitingInitialConditionReady = false;
            _initialConditionReady = true;
            if (logDebug)
                Debug.Log("[Cond] Initial condition gate is ready.");
        }

        _hasCompletedConditionEntryOnce = true;

        _conditionEntryCoroutine = null;
    }

    private float GetPhoneAppReadyWaitSeconds()
    {
        if (useFastTestTiming)
            return Mathf.Max(0f, fastPhoneAppReadyMessageSeconds);

        return Mathf.Max(0f, phoneAppReadyMessageSeconds);
    }

    private float GetConditionBreakWaitSeconds()
    {
        if (useFastTestTiming)
            return Mathf.Max(0f, fastConditionBreakSeconds);

        return Mathf.Max(0f, conditionBreakSeconds);
    }

    private IEnumerator WaitForConditionBreakReadyTripleTap(int token)
    {
        if (!requireTripleTapToLeaveConditionBreak)
            yield break;

        yield return WaitForFreshPhoneTripleTap(token, showReadyAcknowledgement: false);
    }

    private IEnumerator ShowConditionExplanationGate(int token)
    {
        if (!showConditionExplanationGate)
            yield break;

        ResolveInstructionHudIfNeeded();

        if (instructionHUD != null)
        {
            instructionHUD.HideExample();
            instructionHUD.Show(BuildConditionExplanationText(), float.PositiveInfinity);
        }

        yield return WaitForFreshPhoneTripleTap(token, showReadyAcknowledgement: false);

        if (token == _conditionEntryToken && instructionHUD != null)
            instructionHUD.HideImmediate();
    }

    private IEnumerator WaitForFreshPhoneTripleTap(int token, bool showReadyAcknowledgement)
    {
        if (phonePoseReceiver == null)
            phonePoseReceiver = FindFirstObjectByType<PhonePoseStreamReceiver>();

        bool baselineSet = false;
        int baselineTripleTapId = 0;

        while (token == _conditionEntryToken)
        {
            if (phonePoseReceiver == null)
            {
                phonePoseReceiver = FindFirstObjectByType<PhonePoseStreamReceiver>();
                yield return null;
                continue;
            }

            if (!phonePoseReceiver.HasPhonePose || !HasFreshPhoneConnection())
            {
                baselineSet = false;
                yield return null;
                continue;
            }

            int latestTripleTapId = phonePoseReceiver.LatestTripleTapId;

            if (!baselineSet)
            {
                baselineTripleTapId = latestTripleTapId;
                baselineSet = true;
            }
            else if (latestTripleTapId < baselineTripleTapId)
            {
                baselineTripleTapId = latestTripleTapId;
            }
            else if (latestTripleTapId > baselineTripleTapId)
            {
                if (showReadyAcknowledgement &&
                    instructionHUD != null &&
                    !string.IsNullOrWhiteSpace(readyTripleTapAcknowledgementText))
                {
                    instructionHUD.HideExample();
                    instructionHUD.Show(readyTripleTapAcknowledgementText, float.PositiveInfinity);
                }

                yield break;
            }

            yield return null;
        }
    }

    private IEnumerator WaitForReadyTripleTapThenPhoneConnection(int token)
    {
        yield return WaitForFreshPhoneQrIfNeeded(token);

        if (token != _conditionEntryToken)
            yield break;

        yield return WaitForFreshPhoneTripleTap(token, showReadyAcknowledgement: true);

        if (token != _conditionEntryToken)
            yield break;

        yield return WaitForPhoneConnectionIfNeeded(token);
    }

    private IEnumerator WaitForFreshPhoneQrIfNeeded(int token)
    {
        if (!requireFreshPhoneQrBeforeConditionReady)
            yield break;

        if (phonePoseReceiver == null)
            phonePoseReceiver = FindFirstObjectByType<PhonePoseStreamReceiver>();

        float reminderSeconds = Mathf.Max(0.25f, waitingForPhoneQrReminderSeconds);
        float nextReminderAt = 0f;

        while (token == _conditionEntryToken)
        {
            if (phonePoseReceiver == null)
            {
                phonePoseReceiver = FindFirstObjectByType<PhonePoseStreamReceiver>();
                yield return null;
                continue;
            }

            if (HasFreshPhoneQrLock())
                yield break;

            if (instructionHUD != null &&
                !string.IsNullOrWhiteSpace(waitingForPhoneQrText) &&
                Time.unscaledTime >= nextReminderAt)
            {
                instructionHUD.ShowOverlay(waitingForPhoneQrText, reminderSeconds, speak: false);
                nextReminderAt = Time.unscaledTime + reminderSeconds;
            }

            yield return null;
        }
    }

    private bool HasFreshPhoneQrLock()
    {
        if (phonePoseReceiver == null)
            return false;

        bool hasQrPose = phonePoseReceiver.HasQrDeltaPose || phonePoseReceiver.HasQrRelativePhonePose;
        if (!hasQrPose)
            return false;

        string key = phonePoseReceiver.LatestPhoneQrDetectionKey;
        if (string.IsNullOrWhiteSpace(key))
            return false;

        if (!_conditionEntryPhoneQrBaselineSet)
            return true;

        if (string.IsNullOrWhiteSpace(_conditionEntryPhoneQrBaselineKey))
            return true;

        return !string.Equals(key, _conditionEntryPhoneQrBaselineKey, StringComparison.Ordinal);
    }

    private string BuildConditionExplanationText()
    {
        string template = string.IsNullOrWhiteSpace(conditionExplanationTextTemplate)
            ? "Condition {condition_index}\n{condition_name}\nThis condition will be explained.\nAfter the explanation is finished, triple tap on the phone to continue."
            : conditionExplanationTextTemplate;

        int index = CurrentConditionIndex1Based;
        int total = ConditionCount;
        string conditionName = GetResearchConditionName();

        return template
            .Replace("{condition_index}", index > 0 ? index.ToString() : "-")
            .Replace("{condition_total}", total > 0 ? total.ToString() : "-")
            .Replace("{condition_name}", conditionName)
            .Replace("{condition_label}", GetConditionLabel());
    }

    private string GetResearchConditionName()
    {
        if (_currentCondition == null)
            return "Large Motion / Near Head";

        string motion = _currentCondition.technique == Technique.Micro ? "Small Motion" : "Large Motion";
        string location = _currentCondition.handLocation == HandLocation.Side ? "Side of Body" : "Near Head";
        return $"{motion} / {location}";
    }

    private IEnumerator WaitForPhoneConnectionIfNeeded(int token)
    {
        if (!requirePhoneConnectionBeforeTaskStart)
            yield break;

        if (phonePoseReceiver == null)
            phonePoseReceiver = FindFirstObjectByType<PhonePoseStreamReceiver>();

        if (phonePoseReceiver == null)
            yield break;

        float readyDelay = Mathf.Max(0f, phoneConnectionReadyDelaySeconds);
        float connectedStableSec = 0f;

        while (token == _conditionEntryToken)
        {
            if (HasFreshPhoneConnection())
            {
                connectedStableSec += Time.unscaledDeltaTime;
                if (connectedStableSec >= readyDelay)
                    yield break;
            }
            else
            {
                connectedStableSec = 0f;
            }

            yield return null;
        }
    }

    private IEnumerator WaitForPhoneStillnessBeforeConditionBaseline(int token)
    {
        if (!requirePhoneStillnessBeforeConditionBaseline)
            yield break;

        if (phonePoseReceiver == null)
            phonePoseReceiver = FindFirstObjectByType<PhonePoseStreamReceiver>();

        if (phonePoseReceiver == null)
            yield break;

        float stableHold = Mathf.Max(0.05f, conditionEntryStableHoldSeconds);
        float timeout = Mathf.Max(stableHold, conditionEntryStableTimeoutSeconds);
        float stableAccum = 0f;
        float startedAt = Time.unscaledTime;

        while (token == _conditionEntryToken)
        {
            if (!phonePoseReceiver.TryGetPhoneMotionEstimate(
                    out _,
                    out Vector3 linearVelocityMetersPerSec,
                    out Vector3 angularVelocityDegPerSec,
                    out float ageSec))
            {
                stableAccum = 0f;
                yield return null;
                continue;
            }

            float freshSeconds = Mathf.Max(0.05f, phoneConnectionFreshSeconds);
            bool fresh = ageSec <= freshSeconds;
            bool linearStable = linearVelocityMetersPerSec.magnitude <= Mathf.Max(0f, conditionEntryStableLinearSpeedMetersPerSec);
            bool angularStable = angularVelocityDegPerSec.magnitude <= Mathf.Max(0f, conditionEntryStableAngularSpeedDegPerSec);

            if (fresh && linearStable && angularStable)
            {
                stableAccum += Time.unscaledDeltaTime;
                if (stableAccum >= stableHold)
                    yield break;
            }
            else
            {
                stableAccum = 0f;
            }

            if (timeout > 0f && (Time.unscaledTime - startedAt) >= timeout)
            {
                if (logConditionEntryStillness)
                {
                    Debug.LogWarning(
                        $"[Cond] Condition-entry stillness gate timed out. Proceeding. " +
                        $"fresh={fresh} lin={linearVelocityMetersPerSec.magnitude:F3}m/s ang={angularVelocityDegPerSec.magnitude:F1}deg/s");
                }
                yield break;
            }

            yield return null;
        }
    }

    private bool HasFreshPhoneConnection()
    {
        if (phonePoseReceiver == null)
            return false;

        if (!phonePoseReceiver.HasPhonePose)
            return false;

        float freshSeconds = Mathf.Max(0.05f, phoneConnectionFreshSeconds);
        return phonePoseReceiver.SecondsSinceLastRx <= freshSeconds;
    }

    private float ComputeSpeechAwareWaitSeconds(string text, float requestedSeconds, float shownSeconds)
    {
        float requested = Mathf.Max(0f, requestedSeconds);
        float shown = Mathf.Max(0f, shownSeconds);
        float baseWait = Mathf.Max(requested, shown);

        if (baseWait <= 0f)
            return 0f;

        if (!useSpeechAwareGateTiming || string.IsNullOrWhiteSpace(text))
            return baseWait;

        float estimatedSpeech = EstimateSpeechDurationSeconds(text);
        if (estimatedSpeech <= 0f)
            return baseWait;

        // Prevent cut-off: keep message until at least the estimated speech time.
        return Mathf.Max(baseWait, estimatedSpeech);
    }

    private float EstimateSpeechDurationSeconds(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return 0f;

        int charCount = 0;
        int wordCount = 0;
        bool inWord = false;

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (!char.IsWhiteSpace(c))
                charCount++;

            bool isWordChar = char.IsLetterOrDigit(c);
            if (isWordChar && !inWord)
            {
                inWord = true;
                wordCount++;
            }
            else if (!isWordChar)
            {
                inWord = false;
            }
        }

        float cps = Mathf.Max(1f, gateSpeechCharsPerSecond);
        float wps = Mathf.Max(0.5f, gateSpeechWordsPerSecond);

        float byChars = charCount / cps;
        float byWords = wordCount / wps;
        float estimated = Mathf.Max(byChars, byWords);
        return Mathf.Max(0f, Mathf.Max(gateSpeechMinSeconds, estimated));
    }

    private void ResolveInstructionHudIfNeeded()
    {
        if (instructionHUD != null)
            return;

        instructionHUD = FindFirstObjectByType<InstructionHUD>();
        if (instructionHUD != null)
            return;

        InstructionHUD[] all = Resources.FindObjectsOfTypeAll<InstructionHUD>();
        for (int i = 0; i < all.Length; i++)
        {
            InstructionHUD hud = all[i];
            if (hud == null)
                continue;
            if (!hud.gameObject.scene.IsValid())
                continue;

            instructionHUD = hud;
            break;
        }

        if (instructionHUD == null && !_warnedInstructionHudMissing)
        {
            _warnedInstructionHudMissing = true;
            Debug.LogWarning("[Cond] InstructionHUD reference is missing. Condition posture gate text will be hidden.");
        }
    }

    private void CaptureInitialStartRotationIfNeeded()
    {
        if (_initialStartRotationCaptured || phoneMacroPoseDriver == null)
            return;

        if (phoneMacroPoseDriver.TryGetNearStartPose(out _, out Quaternion nearStartRot))
        {
            _initialStartRotation = nearStartRot;
            _initialStartRotationCaptured = true;
            return;
        }

        Transform hand = phoneMacroPoseDriver.HandRootTransform;
        if (hand != null)
        {
            _initialStartRotation = hand.rotation;
            _initialStartRotationCaptured = true;
        }
    }

    private void ApplyConditionEntryStartPose()
    {
        if (phoneMacroPoseDriver == null)
            return;

        if (!TryGetStartWorldPositionForCurrentCondition(out Vector3 worldPos))
            return;

        Quaternion targetRot = Quaternion.identity;
        bool hasRot = false;

        if (restoreInitialRotationOnConditionEntry)
        {
            if (!_initialStartRotationCaptured)
                CaptureInitialStartRotationIfNeeded();

            if (_initialStartRotationCaptured)
            {
                targetRot = _initialStartRotation;
                hasRot = true;
            }
        }

        if (!hasRot)
        {
            Transform hand = phoneMacroPoseDriver.HandRootTransform;
            if (hand != null)
            {
                targetRot = hand.rotation;
                hasRot = true;
            }
        }

        if (!hasRot)
            return;

        phoneMacroPoseDriver.SnapToWorldPose(worldPos, targetRot, rebaseline: false);
    }

    private Sprite GetConditionEntryExampleSprite()
    {
        if (_currentCondition == null)
            return null;

        bool isMicro = _currentCondition.technique == Technique.Micro;
        bool isSide = _currentCondition.handLocation == HandLocation.Side;

        if (isMicro)
            return isSide ? microSideOfBodyExampleImage : microNearHeadExampleImage;

        return isSide ? macroSideOfBodyExampleImage : macroNearHeadExampleImage;
    }

    private void ApplyBalancedLatinSquareOrderIfEnabled()
    {
        int n = conditions != null ? conditions.Count : 0;
        _selectedSequenceCount = GetBalancedLatinSequenceCount(n);
        _selectedSequenceIndex1Based = 1;

        if (!useBalancedLatinSquare || n <= 1)
            return;

        int participantNumber = ResolveParticipantSequenceNumber();
        int seqIndex0Based = Mathf.Abs(participantNumber - 1) % Mathf.Max(1, _selectedSequenceCount);
        int[] order = BuildBalancedLatinOrder(n, seqIndex0Based);
        if (order == null || order.Length != n)
            return;

        var source = new List<Condition>(conditions);
        var reordered = new List<Condition>(n);
        for (int i = 0; i < order.Length; i++)
        {
            int idx = Mathf.Clamp(order[i], 0, source.Count - 1);
            reordered.Add(source[idx]);
        }

        conditions.Clear();
        conditions.AddRange(reordered);
        _selectedSequenceIndex1Based = seqIndex0Based + 1;

        if (logDebug)
            Debug.Log($"[Cond] Balanced Latin Square applied. participant={participantNumber}, sequence={_selectedSequenceIndex1Based}/{_selectedSequenceCount}, order={GetConditionOrderLabel()}");
    }

    private int ResolveParticipantSequenceNumber()
    {
        if (useParticipantIdForSequence)
        {
            string key = string.IsNullOrWhiteSpace(participantIdPrefsKey) ? "participant_id" : participantIdPrefsKey;
            string pid = PlayerPrefs.GetString(key, string.Empty);
            int parsed = ExtractFirstPositiveInteger(pid);
            if (parsed > 0)
                return parsed;
        }

        return Mathf.Max(1, manualSequenceNumber);
    }

    private static int ExtractFirstPositiveInteger(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return -1;

        int value = 0;
        bool found = false;
        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            if (!char.IsDigit(ch))
                continue;

            found = true;
            int digit = ch - '0';
            if (value > (int.MaxValue - digit) / 10)
                return int.MaxValue;
            value = value * 10 + digit;
        }

        if (!found || value <= 0)
            return -1;

        return value;
    }

    private static int GetBalancedLatinSequenceCount(int n)
    {
        if (n <= 0) return 0;
        if (n == 1) return 1;
        return (n % 2 == 0) ? n : (n * 2);
    }

    private static int[] BuildBalancedLatinOrder(int n, int sequenceIndex0Based)
    {
        if (n <= 0)
            return Array.Empty<int>();

        int rowCount = GetBalancedLatinSequenceCount(n);
        int rowIndex = ((sequenceIndex0Based % rowCount) + rowCount) % rowCount;

        int baseRow = rowIndex % n;
        bool reverse = (n % 2 != 0) && (rowIndex >= n);
        int[] row = new int[n];

        for (int col = 0; col < n; col++)
        {
            int v;
            if ((col % 2) == 0)
            {
                v = (baseRow + (col / 2)) % n;
            }
            else
            {
                v = (baseRow - ((col + 1) / 2)) % n;
                if (v < 0) v += n;
            }

            row[col] = v;
        }

        if (reverse)
            Array.Reverse(row);

        return row;
    }

    private bool TrySnapProxyHandToBasketStart(Vector3 localOffset, string modeLabel, bool keepCurrentRotation)
    {
        if (phoneMacroPoseDriver == null)
            return false;

        Transform reference = ResolveSideStartReference();
        if (reference == null)
            return false;

        Vector3 clampedOffset = localOffset;
        float maxAbs = Mathf.Max(0.05f, maxAbsSideStartOffsetMeters);
        clampedOffset.x = Mathf.Clamp(clampedOffset.x, -maxAbs, maxAbs);
        clampedOffset.y = Mathf.Clamp(clampedOffset.y, -maxAbs, maxAbs);
        clampedOffset.z = Mathf.Clamp(clampedOffset.z, -maxAbs, maxAbs);
        if (clampedOffset != localOffset && logDebug)
            Debug.LogWarning($"[Cond] {modeLabel}StartLocalOffset was clamped from {localOffset} to {clampedOffset}.");

        if (grabber != null)
            grabber.ForceRelease();

        Vector3 worldPos = reference.TransformPoint(clampedOffset);
        if (keepCurrentRotation)
        {
            phoneMacroPoseDriver.SnapToWorldPosition(worldPos, rebaseline: false);
        }
        else
        {
            Quaternion targetRot = reference.rotation;
            if (phoneMacroPoseDriver.TryGetNearStartPose(out _, out Quaternion nearStartRot))
                targetRot = nearStartRot;

            phoneMacroPoseDriver.SnapToWorldPose(worldPos, targetRot, rebaseline: false);
        }

        if (logDebug)
            Debug.Log($"[Cond] Proxy hand {modeLabel}-start snapped from basket reference '{reference.name}' at {worldPos:F3}");

        return true;
    }

    private Transform ResolveSideStartReference()
    {
        if (sideStartReference != null)
            return sideStartReference;

        if (basketResetter != null && basketResetter.MacroSideStartReferenceTransform != null)
            return basketResetter.MacroSideStartReferenceTransform;

        return null;
    }

    private void SubscribeTrialStartEvents()
    {
        if (placementTask != null) placementTask.OnTrialChanged += HandleAnyTrialChanged;
        if (rotationTask != null) rotationTask.OnTrialChanged += HandleAnyTrialChanged;
        if (scalingTask != null) scalingTask.OnTrialChanged += HandleAnyTrialChanged;
    }

    private void UnsubscribeTrialStartEvents()
    {
        if (placementTask != null) placementTask.OnTrialChanged -= HandleAnyTrialChanged;
        if (rotationTask != null) rotationTask.OnTrialChanged -= HandleAnyTrialChanged;
        if (scalingTask != null) scalingTask.OnTrialChanged -= HandleAnyTrialChanged;
    }

    private void HandleAnyTrialChanged(int current1Based, int total)
    {
        if (!snapProxyHandPositionOnEachTrial)
            return;
        if (phoneMacroPoseDriver == null || _currentCondition == null)
            return;

        bool isMacro = (_currentCondition.technique == Technique.Macro);
        if (isMacro && !snapDuringMacroTrials)
            return; // default: macro uses condition-start snap only
        if (!isMacro && !snapDuringMicroTrials)
            return;

        if (!TryGetStartWorldPositionForCurrentCondition(out Vector3 worldPos))
            return;

        if (grabber != null)
            grabber.ForceRelease();

        // Position-only snap on trial start. Rotation is intentionally preserved.
        phoneMacroPoseDriver.SnapToWorldPosition(worldPos, rebaseline: false);
        if (isMacro)
            phoneMacroPoseDriver.RebaselineKeepWorldPose();

        if (logDebug)
            Debug.Log($"[Cond] Trial-start position snap ({_currentCondition.technique}/{_currentCondition.handLocation}) -> {worldPos:F3}");
    }

    private bool TryGetStartWorldPositionForCurrentCondition(out Vector3 worldPos)
    {
        Transform reference = ResolveSideStartReference();
        if (reference == null || _currentCondition == null)
        {
            worldPos = Vector3.zero;
            return false;
        }

        bool isSide = (_currentCondition.handLocation == HandLocation.Side);
        Vector3 localOffset = isSide ? sideStartLocalOffset : nearStartLocalOffset;
        float maxAbs = Mathf.Max(0.05f, maxAbsSideStartOffsetMeters);

        localOffset.x = Mathf.Clamp(localOffset.x, -maxAbs, maxAbs);
        localOffset.y = Mathf.Clamp(localOffset.y, -maxAbs, maxAbs);
        localOffset.z = Mathf.Clamp(localOffset.z, -maxAbs, maxAbs);

        worldPos = reference.TransformPoint(localOffset);
        return true;
    }
}
