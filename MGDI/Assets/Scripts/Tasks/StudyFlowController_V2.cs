using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

public class StudyFlowController_V2 : MonoBehaviour
{
    [Header("Workflow")]
    [SerializeField] private WorkflowProgressionController workflow;

    [Header("Tasks")]
    [SerializeField] private ToolPlacementTaskManager placementTask;
    [SerializeField] private ToolRotationTaskManager rotationTask;
    [SerializeField] private ToolScalingTaskManager scalingTask;

    [Header("Ghost Targets")]
    [SerializeField] private Transform slotsTargetsRoot;

    [Header("Grabber (optional)")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Header("Grabber rotation policy per phase")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode placementGrabberMode = ProxyHandGrabber.HeldRotationMode.LockAtGrab;
    [SerializeField] private ProxyHandGrabber.HeldRotationMode rotationGrabberMode  = ProxyHandGrabber.HeldRotationMode.ExternalControl;
    [SerializeField] private ProxyHandGrabber.HeldRotationMode scalingGrabberMode   = ProxyHandGrabber.HeldRotationMode.LockAtGrab;

    [SerializeField] private PhoneTechniqueGate phoneTechniqueGate;
    [SerializeField] private PhoneInputRouter phoneRouter;
    [SerializeField] private PhoneProxyHandRootDriver phoneMacroPoseDriver;
    [SerializeField] private ConditionBlockController conditionBlockController;

    [SerializeField] private TaskContextHUD taskContextHUD;
    [SerializeField] private InstructionHUD instructionHUD;
    [SerializeField] private QRWorkspaceLock_OpenXR qrLock;
    [SerializeField] private StudyLogger logger;

    [Header("Block Intro / Countdown")]
    [SerializeField] private float instructionSeconds = 10f;
    [SerializeField] private float workspaceReadyDelaySeconds = 0.75f;
    [SerializeField] private float goShowSeconds = 0.25f;
    [SerializeField] private bool showMainTaskCountdown = false;

    [Header("Pre-Block Grip Calibration")]
    [SerializeField] private bool requireGripNeutralizationBeforeBlockStart = true;
    [TextArea(2, 4)]
    [SerializeField] private string gripCalibratingText = "Adjust phone to a comfortable grip.\nHold still while calibrating.";
    [SerializeField] private float gripCalibratingMessageSeconds = 1.2f;
    [SerializeField] private int gripCalibratingCountdownSeconds = 5;
    [SerializeField] private bool forceCompleteGripNeutralizationAfterCountdown = true;
    [Tooltip("<=0 means no timeout.")]
    [SerializeField] private float gripGateMaxWaitSeconds = 0f;
    [SerializeField] private bool logGripCalibrationGate = false;

    [Header("Practice Trials")]
    [SerializeField] private int practiceTrialsPerBlock = 2;
    [SerializeField] private string practiceToolId = "hammer";
    [TextArea(2, 5)]
    [SerializeField] private string practiceIntroTextTemplate = "Practice ({count} successful trials)\nTool: {tool}";
    [SerializeField] private float practiceIntroSeconds = 2f;
    [TextArea(2, 6)]
    [SerializeField] private string practiceDetailTextTemplate = "Practice: {task}\n{technique}\n{controlInstruction}";
    [SerializeField] private float practiceDetailSeconds = 2.5f;
    [SerializeField] private float practiceIntroPanelGapSeconds = 0.08f;
    [SerializeField] private float minPracticeIntroRepeatInterval = 1.0f;
    [SerializeField] private bool showPracticeCompleteMessage = false;
    [SerializeField] private string practiceCompleteMessage = "Main task starts now";
    [SerializeField] private float practiceCompleteSeconds = 1f;

    private Action _onPlacementFinished;
    private Action _onRotationFinished;
    private Action _onScalingFinished;

    private readonly Dictionary<string, Transform> ghostById = new Dictionary<string, Transform>();
    private readonly Dictionary<string, Transform> ghostVisualById = new Dictionary<string, Transform>();
    private readonly Dictionary<string, Quaternion> goalVisualLocalRotById = new Dictionary<string, Quaternion>();

    [SerializeField] private bool showDebugHUD = true;
    [SerializeField] private float hudUpdateHz = 10f; // 초당 10번
    [SerializeField] private bool logTaskContextUpdates = false;
    private float _nextHudTime = 0f;
    private string _lastToolText = null;
    private string _lastHintText = null;

    private bool _studyStarted = false;
    private bool _hasPendingStep = false;
    private WorkflowProgressionController.Phase _pendingPhase;
    private int _pendingToolIndex;
    private GameObject _pendingTool;

    private Coroutine _workspaceGateCoroutine;
    private Coroutine _blockIntroCoroutine;
    private Coroutine _practiceCoroutine;
    private Coroutine _phaseEntryGateCoroutine;
    private int _phaseEntryGateToken = 0;
    private bool _gripCalibrationCompletedInSession = false;

    private static readonly float WaitingInstructionDurationSec = float.PositiveInfinity;

    private bool inPractice = false;
    private int practiceSuccessCount = 0;
    private int practiceTargetCount = 0;
    private readonly HashSet<string> _practiceDoneKeys = new HashSet<string>();
    private WorkflowProgressionController.Phase _practicePhase;
    private bool _savedPlacementResetTools;
    private bool _savedRotationResetTool;
    private bool _savedScalingResetScale;
    private bool _practiceResetPolicyApplied;
    private string _conditionStateKey = null;
    private bool _carryRotationPoseIntoScaling = false;
    private bool _practiceIntroShownThisSession = false;
    private float _lastPracticeIntroShownAt = -999f;
    private string _lastPracticeIntroShownText = null;
    private bool _hasLastRealIntroContext = false;
    private WorkflowProgressionController.Phase _lastRealIntroPhase;
    private string _lastRealIntroConditionKey = null;
    private MicroPlacementAnalogController _microPlacementAnalogController;
    private MicroRotationAnalogController _microRotationAnalogController;

    void Awake() => Debug.Log("[SFC_V2] Awake");

    void Update()
    {
        UpdateTaskContextHUD();

        if (!showDebugHUD) return;
        if (Time.unscaledTime < _nextHudTime) return;
        _nextHudTime = Time.unscaledTime + (1f / Mathf.Max(1f, hudUpdateHz));

        string phase = workflow != null ? workflow.CurrentPhase.ToString() : "N/A";
        int idx = workflow != null ? workflow.CurrentToolIndex : -1;

        GameObject tool = workflow != null ? workflow.CurrentTool : null;
        string id = "N/A";
        if (tool != null)
        {
            var tid = tool.GetComponent<ToolId>();
            if (tid != null && !string.IsNullOrEmpty(tid.id)) id = tid.id;
        }

        bool holding = grabber != null && grabber.IsHolding;
        string heldName = (grabber != null && grabber.HeldBody != null) ? grabber.HeldBody.name : "none";

        DebugHUD.Log($"WF   phase={phase} toolIndex={idx} id={id}");
        DebugHUD.Log($"GRAB holding={holding} held={heldName}");

        // task running flags (있으면 더 좋음)
        DebugHUD.Log($"TASK place={(placementTask!=null && placementTask.IsTrialRunning)} rot={(rotationTask!=null && rotationTask.IsTrialRunning)} scale={(scalingTask!=null && scalingTask.IsTrialRunning)}");
    }

    void Start()
    {
        RebuildGhostRegistry();
        CacheAuthoredGoalRotations();

        if (workflow == null)
            workflow = FindFirstObjectByType<WorkflowProgressionController>();
        if (conditionBlockController == null)
            conditionBlockController = FindFirstObjectByType<ConditionBlockController>();
        if (logger == null)
            logger = FindFirstObjectByType<StudyLogger>();
        if (phoneMacroPoseDriver == null)
            phoneMacroPoseDriver = FindFirstObjectByType<PhoneProxyHandRootDriver>();

        if (qrLock == null)
            qrLock = FindFirstObjectByType<QRWorkspaceLock_OpenXR>();

        if (workflow == null)
        {
            Debug.LogError("[SFC_V2] WorkflowProgressionController not found.");
            return;
        }

        _conditionStateKey = GetConditionStateKey();

        if (taskContextHUD == null) taskContextHUD = FindFirstObjectByType<TaskContextHUD>();
        if (instructionHUD == null) instructionHUD = FindFirstObjectByType<InstructionHUD>();
        if (taskContextHUD != null) { taskContextHUD.SetVisible(false); taskContextHUD.Clear(); taskContextHUD.SetPracticeText(""); }
        if (instructionHUD != null) instructionHUD.Show("Scan the QR code to begin.", WaitingInstructionDurationSec);

        DisableAllTasks();

        workflow.OnStepChanged += OnWorkflowStepChanged;

        _onPlacementFinished = HandlePlacementBlockFinished;
        _onRotationFinished  = HandleRotationBlockFinished;
        _onScalingFinished   = HandleScalingBlockFinished;

        HookTaskFinishedEvents();

        // Capture current workflow step, but defer task start until workspace lock is ready.
        OnWorkflowStepChanged(workflow.CurrentPhase, workflow.CurrentToolIndex, workflow.CurrentTool);

        _workspaceGateCoroutine = StartCoroutine(WaitForWorkspaceThenBeginStudy());
        Debug.Log("[SFC_V2] Bound to workflow. Waiting for workspace lock.");
    }

    public string GetConditionLabel()
    {
        if (conditionBlockController != null && conditionBlockController.HasCurrentCondition)
            return conditionBlockController.GetConditionLabel();

        string technique = "Macro";
        if (phoneRouter != null && phoneRouter.CurrentMode == PhoneInputRouter.Mode.Micro)
            technique = "Micro";

        return $"{technique} - Near Head";
    }

    public int GetConditionIndex1Based()
    {
        if (conditionBlockController != null && conditionBlockController.HasCurrentCondition)
            return conditionBlockController.CurrentConditionIndex1Based;
        return -1;
    }

    public int GetConditionCount()
    {
        if (conditionBlockController != null)
            return conditionBlockController.ConditionCount;
        return 0;
    }

    public string GetConditionOrderLabel()
    {
        if (conditionBlockController != null)
            return conditionBlockController.GetConditionOrderLabel();
        return string.Empty;
    }

    public int GetConditionSequenceIndex1Based()
    {
        if (conditionBlockController != null)
            return conditionBlockController.SelectedSequenceIndex1Based;
        return -1;
    }

    public int GetConditionSequenceCount()
    {
        if (conditionBlockController != null)
            return conditionBlockController.SelectedSequenceCount;
        return 0;
    }

    private void UpdateTaskContextHUD()
    {
        if (taskContextHUD == null) return;

        string toolLabel = GetActiveToolLabel();
        string toolText = BuildTaskContextActionText(toolLabel);
        if (_lastToolText != toolText)
        {
            taskContextHUD.SetTrialText(toolText);
            _lastToolText = toolText;
            if (logTaskContextUpdates) Debug.Log("[SFC_V2] HUD Tool=" + toolText);
        }

        string hintText = BuildTaskContextHintText();
        if (_lastHintText != hintText)
        {
            taskContextHUD.SetHintText(hintText);
            _lastHintText = hintText;
            if (logTaskContextUpdates) Debug.Log("[SFC_V2] HUD Hint=" + hintText);
        }
    }

    private string GetActiveToolLabel()
    {
        if (inPractice)
        {
            if (placementTask != null && placementTask.IsTrialRunning)
                return ResolveToolLabel(placementTask.ActiveToolTransform, placementTask.ActiveToolId);

            if (rotationTask != null && rotationTask.IsTrialRunning)
                return ResolveToolLabel(rotationTask.ActiveToolTransform, rotationTask.ActiveToolId);

            if (scalingTask != null && scalingTask.IsTrialRunning)
                return ResolveToolLabel(scalingTask.ActiveToolTransform, scalingTask.ActiveId);

            string fallbackPracticeId = GetPracticeToolIdNormalized();
            return string.IsNullOrEmpty(fallbackPracticeId) ? "-" : fallbackPracticeId;
        }

        if (placementTask != null && placementTask.IsTrialRunning)
            return ResolveToolLabel(placementTask.ActiveToolTransform, placementTask.ActiveToolId);

        if (rotationTask != null && rotationTask.IsTrialRunning)
            return ResolveToolLabel(rotationTask.ActiveToolTransform, rotationTask.ActiveToolId);

        if (scalingTask != null && scalingTask.IsTrialRunning)
            return ResolveToolLabel(scalingTask.ActiveToolTransform, scalingTask.ActiveId);

        if (workflow != null && workflow.CurrentTool != null)
        {
            var tid = workflow.CurrentTool.GetComponent<ToolId>();
            if (tid != null && !string.IsNullOrEmpty(tid.id))
                return tid.id;
            return workflow.CurrentTool.name;
        }

        return "-";
    }

    private string ResolveToolLabel(Transform toolTransform, string fallbackId)
    {
        if (!string.IsNullOrEmpty(fallbackId))
            return fallbackId;

        if (toolTransform == null)
            return "-";

        var tid = toolTransform.GetComponent<ToolId>();
        if (tid != null && !string.IsNullOrEmpty(tid.id))
            return tid.id;

        return toolTransform.name;
    }

    private string BuildTaskContextActionText(string toolLabel)
    {
        string displayTool = FormatTaskContextToolName(toolLabel);
        string highlightedTool = $"<color=#FFD166>{displayTool}</color>";

        if (placementTask != null && placementTask.IsTrialRunning)
            return $"Place {highlightedTool}";

        if (rotationTask != null && rotationTask.IsTrialRunning)
            return $"Rotate {highlightedTool}";

        if (scalingTask != null && scalingTask.IsTrialRunning)
            return $"Scale {highlightedTool}";

        return string.Empty;
    }

    private string BuildTaskContextHintText()
    {
        if (!IsCurrentTechniqueMicro())
            return string.Empty;

        switch (GetActivePhaseForContextHint())
        {
            case WorkflowProgressionController.Phase.Placement:
                return BuildMicroPlacementHint();
            case WorkflowProgressionController.Phase.Rotation:
                return BuildMicroRotationHint();
            case WorkflowProgressionController.Phase.Scaling:
                return "Swipe up: smaller\nSwipe down: bigger";
            default:
                return string.Empty;
        }
    }

    private WorkflowProgressionController.Phase GetActivePhaseForContextHint()
    {
        if (placementTask != null && placementTask.IsTrialRunning)
            return WorkflowProgressionController.Phase.Placement;

        if (rotationTask != null && rotationTask.IsTrialRunning)
            return WorkflowProgressionController.Phase.Rotation;

        if (scalingTask != null && scalingTask.IsTrialRunning)
            return WorkflowProgressionController.Phase.Scaling;

        if (inPractice)
            return _practicePhase;

        return workflow != null ? workflow.CurrentPhase : WorkflowProgressionController.Phase.Placement;
    }

    private string BuildMicroPlacementHint()
    {
        if (_microPlacementAnalogController == null)
            _microPlacementAnalogController = FindFirstObjectByType<MicroPlacementAnalogController>();

        bool depthMode = _microPlacementAnalogController != null && _microPlacementAnalogController.IsDepthMode;
        if (depthMode)
            return "Tap: grab / tap again: release\nSwipe L/R: left-right\nSwipe U/D: front-back\nDouble tap: switch U/D back to up-down";

        return "Tap: grab / tap again: release\nSwipe L/R: left-right\nSwipe U/D: up-down\nDouble tap: switch U/D to front-back";
    }

    private string BuildMicroRotationHint()
    {
        if (_microRotationAnalogController == null)
            _microRotationAnalogController = FindFirstObjectByType<MicroRotationAnalogController>();

        bool pitchMode = _microRotationAnalogController != null && _microRotationAnalogController.IsPitchMode;
        if (pitchMode)
            return "Swipe L/R: yaw\nSwipe U/D: pitch\nDouble tap: switch U/D back to roll";

        return "Swipe L/R: yaw\nSwipe U/D: roll\nDouble tap: switch U/D to pitch";
    }

    private static string FormatTaskContextToolName(string toolLabel)
    {
        if (string.IsNullOrWhiteSpace(toolLabel) || toolLabel == "-")
            return "tool";

        string value = toolLabel.Trim();
        value = value.Replace("_", " ");
        value = System.Text.RegularExpressions.Regex.Replace(value, "([a-z])([A-Z])", "$1 $2");
        value = System.Text.RegularExpressions.Regex.Replace(value, "\\s+", " ");
        return value.ToLowerInvariant();
    }

    private void OnDestroy()
    {
        if (_workspaceGateCoroutine != null) StopCoroutine(_workspaceGateCoroutine);
        if (_blockIntroCoroutine != null) StopCoroutine(_blockIntroCoroutine);
        if (_practiceCoroutine != null) StopCoroutine(_practiceCoroutine);
        if (_phaseEntryGateCoroutine != null) StopCoroutine(_phaseEntryGateCoroutine);

        if (workflow != null)
            workflow.OnStepChanged -= OnWorkflowStepChanged;

        EndPracticeSubscriptions();
        RestoreCurrentWorkflowToolForcedId();
        RestorePracticeResetPolicy(_practicePhase);
        RestorePracticeGhostRandomizationPolicy(_practicePhase);
        if (logger != null) logger.LoggingEnabled = true;
        if (taskContextHUD != null) taskContextHUD.SetPracticeText("");

        UnhookTaskFinishedEvents();
    }

    private void HookTaskFinishedEvents()
    {
        if (placementTask != null) placementTask.OnBlockFinished += _onPlacementFinished;
        if (rotationTask != null)  rotationTask.OnBlockFinished  += _onRotationFinished;
        if (scalingTask != null)   scalingTask.OnBlockFinished   += _onScalingFinished;
    }

    private void UnhookTaskFinishedEvents()
    {
        if (placementTask != null) placementTask.OnBlockFinished -= _onPlacementFinished;
        if (rotationTask != null)  rotationTask.OnBlockFinished  -= _onRotationFinished;
        if (scalingTask != null)   scalingTask.OnBlockFinished   -= _onScalingFinished;
    }

    private void HandlePlacementBlockFinished()
    {
        if (inPractice && _practicePhase == WorkflowProgressionController.Phase.Placement)
        {
            EnsurePracticeContinuesAfterBlockFinish();
            return;
        }

        if (workflow != null)
            workflow.Advance();
    }

    private void HandleRotationBlockFinished()
    {
        if (inPractice && _practicePhase == WorkflowProgressionController.Phase.Rotation)
        {
            EnsurePracticeContinuesAfterBlockFinish();
            return;
        }

        if (rotationTask != null)
            rotationTask.CaptureActivePoseForCarry();
        _carryRotationPoseIntoScaling = true;

        if (workflow != null)
            workflow.Advance();
    }

    private void HandleScalingBlockFinished()
    {
        if (inPractice && _practicePhase == WorkflowProgressionController.Phase.Scaling)
        {
            EnsurePracticeContinuesAfterBlockFinish();
            return;
        }

        if (workflow != null)
            workflow.Advance();
    }

    private void EnsurePracticeContinuesAfterBlockFinish()
    {
        if (!inPractice)
            return;
        if (practiceSuccessCount >= practiceTargetCount)
            return;
        if (_practiceCoroutine != null)
            return;

        ApplyForcedIdForPhase(_practicePhase, GetPracticeToolIdNormalized());
        _practiceCoroutine = StartCoroutine(WaitAndStartNextPracticeTrial());
    }

    private void ApplyPracticeResetPolicy(WorkflowProgressionController.Phase phase)
    {
        _practiceResetPolicyApplied = true;
        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                if (placementTask != null)
                {
                    _savedPlacementResetTools = placementTask.ResetToolsToStartAfterTrial;
                    placementTask.ResetToolsToStartAfterTrial = true;
                }
                break;
            case WorkflowProgressionController.Phase.Rotation:
                if (rotationTask != null)
                {
                    _savedRotationResetTool = rotationTask.ResetToolToStartAfterTrial;
                    rotationTask.ResetToolToStartAfterTrial = true;
                }
                break;
            case WorkflowProgressionController.Phase.Scaling:
                if (scalingTask != null)
                {
                    _savedScalingResetScale = scalingTask.ResetScaleAfterTrial;
                    scalingTask.ResetScaleAfterTrial = true;
                }
                break;
        }
    }

    private void ApplyPracticeGhostRandomizationPolicy(WorkflowProgressionController.Phase phase)
    {
        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                if (placementTask != null) placementTask.SetPracticeGhostRandomization(true);
                break;
            case WorkflowProgressionController.Phase.Rotation:
                if (rotationTask != null) rotationTask.SetPracticeGhostRandomization(true);
                break;
            case WorkflowProgressionController.Phase.Scaling:
                if (scalingTask != null) scalingTask.SetPracticeGhostRandomization(true);
                break;
        }
    }

    private void RestorePracticeResetPolicy(WorkflowProgressionController.Phase phase)
    {
        if (!_practiceResetPolicyApplied)
            return;

        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                if (placementTask != null)
                    placementTask.ResetToolsToStartAfterTrial = _savedPlacementResetTools;
                break;
            case WorkflowProgressionController.Phase.Rotation:
                if (rotationTask != null)
                    rotationTask.ResetToolToStartAfterTrial = _savedRotationResetTool;
                break;
            case WorkflowProgressionController.Phase.Scaling:
                if (scalingTask != null)
                    scalingTask.ResetScaleAfterTrial = _savedScalingResetScale;
                break;
        }

        _practiceResetPolicyApplied = false;
    }

    private void RestorePracticeGhostRandomizationPolicy(WorkflowProgressionController.Phase phase)
    {
        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                if (placementTask != null) placementTask.SetPracticeGhostRandomization(false);
                break;
            case WorkflowProgressionController.Phase.Rotation:
                if (rotationTask != null) rotationTask.SetPracticeGhostRandomization(false);
                break;
            case WorkflowProgressionController.Phase.Scaling:
                if (scalingTask != null) scalingTask.SetPracticeGhostRandomization(false);
                break;
        }
    }

    private void OnWorkflowStepChanged(WorkflowProgressionController.Phase phase, int toolIndex, GameObject tool)
    {
        _pendingPhase = phase;
        _pendingToolIndex = toolIndex;
        _pendingTool = tool;
        _hasPendingStep = true;

        if (!_studyStarted)
        {
            DisableAllTasks();
            return;
        }

        ApplyWorkflowStep(phase, toolIndex, tool);
    }

    private IEnumerator WaitForWorkspaceThenBeginStudy()
    {
        while (true)
        {
            if (qrLock == null)
                qrLock = FindFirstObjectByType<QRWorkspaceLock_OpenXR>();

            if (qrLock != null && qrLock.IsWorkspaceReady)
                break;

            yield return null;
        }

        while (conditionBlockController != null && !conditionBlockController.IsInitialConditionReady)
            yield return null;

        yield return new WaitForSeconds(Mathf.Max(0f, workspaceReadyDelaySeconds));

        _studyStarted = true;
        if (instructionHUD != null) instructionHUD.HideImmediate();

        if (_hasPendingStep)
            ApplyWorkflowStep(_pendingPhase, _pendingToolIndex, _pendingTool);
        else if (workflow != null)
            ApplyWorkflowStep(workflow.CurrentPhase, workflow.CurrentToolIndex, workflow.CurrentTool);
    }

    private void ApplyWorkflowStep(WorkflowProgressionController.Phase phase, int toolIndex, GameObject tool)
    {
        HandleConditionStateChangeIfNeeded();

        if (inPractice && phase != _practicePhase)
            AbortPracticeState();

        // ToolId from tool root (your current structure)
        if (tool == null)
        {
            if (!inPractice && taskContextHUD != null)
                taskContextHUD.SetPracticeText("");
            DisableAllTasks();
            return;
        }

        var tid = tool.GetComponent<ToolId>();
        string id = (tid != null) ? tid.id : null;

        if (inPractice && phase == _practicePhase)
        {
            string practiceId = GetPracticeToolIdNormalized();
            if (!string.IsNullOrEmpty(practiceId))
                id = practiceId;
        }

        if (string.IsNullOrEmpty(id))
        {
            Debug.LogError($"[SFC_V2] ToolId missing on active tool '{tool.name}'.");
            DisableAllTasks();
            return;
        }

        // Always release before switching phase to avoid parent/rotation mode leakage
        if (grabber != null)
            grabber.ForceRelease();

        // Default: do not freeze held object (applies to Placement/Rotation)
        if (grabber != null)
            grabber.SetFreezeHeld(false, false, false);

        // Select exactly one task
        DisableAllTasks();

        if (_blockIntroCoroutine != null)
        {
            StopCoroutine(_blockIntroCoroutine);
            _blockIntroCoroutine = null;
        }

        if (!inPractice && taskContextHUD != null)
            taskContextHUD.SetPracticeText("");

        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                // Micro: select micro controller for placement
                /* if (phoneRouter != null && phoneRouter.CurrentMode == PhoneInputRouter.Mode.Micro && phoneTechniqueGate != null) */
                if (phoneTechniqueGate != null) phoneTechniqueGate.SetMicroTaskPlacement();

                if (taskContextHUD != null) taskContextHUD.SetTaskLabel("Placement");

                ApplyGrabberMode(placementGrabberMode);
                ApplyForcedIdToTasks(id);
                StartPhaseEntryWithGripGate(phase);
                break;

            case WorkflowProgressionController.Phase.Rotation:
                // Micro: select micro controller for rotation
                /* if (phoneRouter != null && phoneRouter.CurrentMode == PhoneInputRouter.Mode.Micro && phoneTechniqueGate != null) */
                if (phoneTechniqueGate != null) phoneTechniqueGate.SetMicroTaskRotation();

                if (taskContextHUD != null) taskContextHUD.SetTaskLabel("Rotation");
                ApplyGrabberMode(rotationGrabberMode);
                ApplyForcedIdToTasks(id);
                StartPhaseEntryWithGripGate(phase);
                break;

            case WorkflowProgressionController.Phase.Scaling:
                // Micro: select micro controller for scaling
                /* if (phoneRouter != null && phoneRouter.CurrentMode == PhoneInputRouter.Mode.Micro && phoneTechniqueGate != null) */
                if (phoneTechniqueGate != null) phoneTechniqueGate.SetMicroTaskScaling();

                if (taskContextHUD != null) taskContextHUD.SetTaskLabel("Scaling");

                ApplyGrabberMode(scalingGrabberMode);

                if (grabber != null)
                    grabber.SetFreezeHeld(true, true, false);

                ApplyForcedIdToTasks(id);
                StartPhaseEntryWithGripGate(phase);
                break;
        }
    }

    private void StartPhaseEntryWithGripGate(WorkflowProgressionController.Phase phase)
    {
        _phaseEntryGateToken++;
        int token = _phaseEntryGateToken;

        if (_phaseEntryGateCoroutine != null)
        {
            StopCoroutine(_phaseEntryGateCoroutine);
            _phaseEntryGateCoroutine = null;
        }

        if (!inPractice && taskContextHUD != null)
            taskContextHUD.SetPracticeText("");

        _phaseEntryGateCoroutine = StartCoroutine(WaitGripThenStartPhaseEntry(phase, token));
    }

    private IEnumerator WaitGripThenStartPhaseEntry(WorkflowProgressionController.Phase phase, int token)
    {
        bool runGate = requireGripNeutralizationBeforeBlockStart && phoneMacroPoseDriver != null && !_gripCalibrationCompletedInSession;
        float startedAt = Time.unscaledTime;

        if (runGate)
        {
            phoneMacroPoseDriver.BeginTaskStartNeutralization();

            if (instructionHUD != null && !string.IsNullOrWhiteSpace(gripCalibratingText))
            {
                float msgSeconds = Mathf.Max(0f, gripCalibratingMessageSeconds);
                float shown = instructionHUD.Show(gripCalibratingText, msgSeconds);
                if (shown > 0f)
                    yield return new WaitForSeconds(shown);

                int countdown = Mathf.Max(0, gripCalibratingCountdownSeconds);
                for (int sec = countdown; sec >= 1; sec--)
                {
                    if (token != _phaseEntryGateToken)
                        yield break;
                    float shownCount = instructionHUD.Show(sec.ToString(), 1f);
                    yield return new WaitForSeconds(Mathf.Max(0.05f, shownCount > 0f ? shownCount : 1f));
                }
            }
            else
            {
                float wait = Mathf.Max(0f, gripCalibratingMessageSeconds) + Mathf.Max(0, gripCalibratingCountdownSeconds);
                if (wait > 0f)
                    yield return new WaitForSeconds(wait);
            }

            if (forceCompleteGripNeutralizationAfterCountdown && !phoneMacroPoseDriver.IsGripNeutralizationReady)
                phoneMacroPoseDriver.CompleteGripNeutralizationNow();

            while (token == _phaseEntryGateToken && !phoneMacroPoseDriver.IsGripNeutralizationReady)
            {
                if (gripGateMaxWaitSeconds > 0f &&
                    (Time.unscaledTime - startedAt) >= gripGateMaxWaitSeconds)
                {
                    if (logGripCalibrationGate)
                        Debug.LogWarning("[SFC_V2] Grip calibration gate timed out. Proceeding.");
                    phoneMacroPoseDriver.CompleteGripNeutralizationNow();
                    break;
                }
                yield return null;
            }

            if (instructionHUD != null)
                instructionHUD.HideImmediate();

            _gripCalibrationCompletedInSession = true;
        }

        if (token != _phaseEntryGateToken)
            yield break;

        StartPhaseEntryWithPracticeAndIntro(phase);
        _phaseEntryGateCoroutine = null;
    }

    private void StartPhaseEntryWithPracticeAndIntro(WorkflowProgressionController.Phase phase)
    {
        if (inPractice)
            return;

        if (NeedsPracticeForPhase(phase))
        {
            BeginPracticeForBlock(phase);
            return;
        }

        StartRealBlockWithIntroGate(phase);
    }

    private void StartRealBlockWithIntroGate(WorkflowProgressionController.Phase phase)
    {
        if (_blockIntroCoroutine != null)
        {
            StopCoroutine(_blockIntroCoroutine);
            _blockIntroCoroutine = null;
        }

        if (instructionHUD != null)
            instructionHUD.HideImmediate();

        StartRealBlockForPhase(phase);
    }

    private IEnumerator ShowInstructionThenCountdownThenStartBlock(WorkflowProgressionController.Phase blockType)
    {
        string text = GetBlockInstructionText(blockType);
        float introSec = Mathf.Max(0f, instructionSeconds);

        if (!showMainTaskCountdown)
        {
            yield return ShowInstructionAndWaitForTaskGate(text, introSec);
        }
        else
        {
            float waitBeforeCountdown = Mathf.Max(0f, introSec - 3f);
            yield return ShowInstructionAndWaitForTaskGate(text, waitBeforeCountdown);

            if (instructionHUD != null)
            {
                instructionHUD.Show("3", 1f);
                yield return new WaitForSeconds(1f);
                instructionHUD.Show("2", 1f);
                yield return new WaitForSeconds(1f);
                instructionHUD.Show("1", 1f);
                yield return new WaitForSeconds(1f);
                instructionHUD.Show("Go", Mathf.Max(0.05f, goShowSeconds));
            }
        }

        StartRealBlockForPhase(blockType);

        _blockIntroCoroutine = null;
    }

    private IEnumerator ShowInstructionAndWaitForTaskGate(string text, float fallbackSeconds)
    {
        float wait = Mathf.Max(0f, fallbackSeconds);

        if (instructionHUD == null)
        {
            if (wait > 0f)
                yield return new WaitForSeconds(wait);
            yield break;
        }

        instructionHUD.Show(text, wait);
        yield return instructionHUD.WaitForTaskGate(text, wait);
        instructionHUD.HideImmediate();
    }

    private string GetBlockInstructionText(WorkflowProgressionController.Phase blockType)
    {
        bool isMicro = IsCurrentTechniqueMicro();
        string taskInstruction = BuildTaskInstruction(blockType);
        if (string.IsNullOrEmpty(taskInstruction))
            return "";

        if (!isMicro)
            return $"{taskInstruction}\nTriple tap to submit\nyour current attempt.";

        return $"{taskInstruction}\n{BuildMicroControlHint(blockType)}\nTriple tap to submit\nyour current attempt.";
    }

    private void AbortPracticeState()
    {
        if (!inPractice)
            return;

        if (_practiceCoroutine != null)
        {
            StopCoroutine(_practiceCoroutine);
            _practiceCoroutine = null;
        }

        EndPracticeSubscriptions();
        ClearForcedIdForPhase(_practicePhase);
        RestorePracticeResetPolicy(_practicePhase);
        RestorePracticeGhostRandomizationPolicy(_practicePhase);
        inPractice = false;
        practiceSuccessCount = 0;
        practiceTargetCount = 0;

        if (logger != null)
            logger.LoggingEnabled = true;
        if (taskContextHUD != null)
            taskContextHUD.SetPracticeText("");
    }

    private bool NeedsPracticeForPhase(WorkflowProgressionController.Phase phase)
    {
        if (practiceTrialsPerBlock <= 0)
            return false;

        string key = BuildPracticePhaseKey(phase);
        return !_practiceDoneKeys.Contains(key);
    }

    private void MarkPracticeDone(WorkflowProgressionController.Phase phase)
    {
        _practiceDoneKeys.Add(BuildPracticePhaseKey(phase));
    }

    private void BeginPracticeForBlock(WorkflowProgressionController.Phase phase)
    {
        if (inPractice)
            return;

        int target = Mathf.Max(0, practiceTrialsPerBlock);
        if (target == 0)
        {
            MarkPracticeDone(phase);
            StartRealBlockForPhase(phase);
            return;
        }

        inPractice = true;
        _practicePhase = phase;
        practiceSuccessCount = 0;
        practiceTargetCount = target;
        _practiceIntroShownThisSession = false;
        ApplyPracticeResetPolicy(phase);
        ApplyPracticeGhostRandomizationPolicy(phase);
        ApplyForcedIdForPhase(phase, GetPracticeToolIdNormalized());

        if (logger != null)
            logger.LoggingEnabled = false;

        EndPracticeSubscriptions();
        SubscribePracticeTrialEnded(phase);
        UpdatePracticeText();

        if (_practiceCoroutine != null)
            StopCoroutine(_practiceCoroutine);
        _practiceCoroutine = StartCoroutine(BeginPracticeAfterIntro());
    }

    private IEnumerator BeginPracticeAfterIntro()
    {
        yield return StartCoroutine(ShowPracticeIntroIfNeeded());

        if (!inPractice)
        {
            _practiceCoroutine = null;
            yield break;
        }

        StartPracticeTrialForCurrentPhase();
        _practiceCoroutine = null;
    }

    private void StartPracticeTrialForCurrentPhase()
    {
        if (!inPractice)
            return;

        if (grabber != null)
            grabber.ForceRelease();

        ApplyForcedIdForPhase(_practicePhase, GetPracticeToolIdNormalized());
        StartRealBlockForPhase(_practicePhase);
    }

    private void OnPracticeTrialEnded(bool success, bool timedOut)
    {
        _ = timedOut;

        if (!inPractice)
            return;

        if (success)
            practiceSuccessCount++;

        if (practiceSuccessCount >= practiceTargetCount)
        {
            UpdatePracticeText();
            if (_practiceCoroutine != null)
                StopCoroutine(_practiceCoroutine);
            _practiceCoroutine = StartCoroutine(EndPracticeAndStartRealBlock());
            return;
        }

        UpdatePracticeText();

        if (success)
        {
            if (_practiceCoroutine != null)
                StopCoroutine(_practiceCoroutine);
            _practiceCoroutine = StartCoroutine(WaitAndStartNextPracticeTrial());
        }
    }

    private IEnumerator WaitAndStartNextPracticeTrial()
    {
        yield return null;

        if (!inPractice)
        {
            _practiceCoroutine = null;
            yield break;
        }

        while (IsPracticeTaskRunning())
            yield return null;

        if (inPractice && practiceSuccessCount < practiceTargetCount)
            StartPracticeTrialForCurrentPhase();

        _practiceCoroutine = null;
    }

    private IEnumerator EndPracticeAndStartRealBlock()
    {
        WorkflowProgressionController.Phase finishedPhase = _practicePhase;

        yield return null;

        EndPracticeSubscriptions();
        ClearForcedIdForPhase(finishedPhase);
        RestorePracticeResetPolicy(finishedPhase);
        RestorePracticeGhostRandomizationPolicy(finishedPhase);
        MarkPracticeDone(finishedPhase);
        inPractice = false;
        practiceSuccessCount = 0;

        if (logger != null)
            logger.LoggingEnabled = true;

        if (taskContextHUD != null)
        {
            taskContextHUD.SetPracticeText("");
            taskContextHUD.SetVisible(false);
        }

        float wait = showPracticeCompleteMessage ? Mathf.Max(0f, practiceCompleteSeconds) : 0f;
        if (showPracticeCompleteMessage &&
            instructionHUD != null &&
            !string.IsNullOrWhiteSpace(practiceCompleteMessage))
        {
            float gateWait = wait > 0f ? wait : 0.05f;
            instructionHUD.Show(practiceCompleteMessage, gateWait);
            yield return instructionHUD.WaitForTaskGate(practiceCompleteMessage, gateWait);
            instructionHUD.HideImmediate();
        }
        else if (wait > 0f)
        {
            yield return new WaitForSeconds(wait);
        }

        RestoreCurrentWorkflowToolForcedId();
        StartRealBlockWithIntroGate(finishedPhase);
        _practiceCoroutine = null;
    }

    private string BuildPracticeIntroText()
    {
        return "This is practice time.\nTry the task freely\nbefore the main trials.";
    }

    private string BuildPracticeDetailText(WorkflowProgressionController.Phase phase)
    {
        bool isMicro = IsCurrentTechniqueMicro();
        return BuildPracticeControlInstruction(phase, isMicro);
    }

    private static string GetPracticeTaskLabel(WorkflowProgressionController.Phase phase)
    {
        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                return "Placement";
            case WorkflowProgressionController.Phase.Rotation:
                return "Rotation";
            case WorkflowProgressionController.Phase.Scaling:
                return "Scaling";
            default:
                return "Task";
        }
    }

    private bool IsCurrentTechniqueMicro()
    {
        if (conditionBlockController != null && conditionBlockController.HasCurrentCondition)
            return conditionBlockController.CurrentTechnique == ConditionBlockController.Technique.Micro;

        return phoneRouter != null && phoneRouter.CurrentMode == PhoneInputRouter.Mode.Micro;
    }

    private static string BuildPracticeControlInstruction(WorkflowProgressionController.Phase phase, bool isMicro)
    {
        string taskInstruction = BuildTaskInstruction(phase);
        if (string.IsNullOrEmpty(taskInstruction))
            taskInstruction = "Practice the task.";

        if (isMicro)
            return $"{taskInstruction}\n{BuildMicroControlHint(phase)}\nWhen you're done with the current\npractice round, triple tap.";

        return $"{taskInstruction}\nMove the arm\nto practice.\nWhen you're done with the current\npractice round, triple tap.";
    }

    private static string BuildTaskInstruction(WorkflowProgressionController.Phase phase)
    {
        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                return "Place the tool as close to\nthe target as possible.";
            case WorkflowProgressionController.Phase.Rotation:
                return "Match the tool to the\ntarget rotation as closely\nas possible.";
            case WorkflowProgressionController.Phase.Scaling:
                return "Match the tool to the\ntarget size as closely\nas possible.";
            default:
                return "";
        }
    }

    private static string BuildMicroControlHint(WorkflowProgressionController.Phase phase)
    {
        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                return "Keep your arm as still\nas possible.\nUse phone swipes to adjust\nthe proxy hand.";
            case WorkflowProgressionController.Phase.Rotation:
            case WorkflowProgressionController.Phase.Scaling:
                return "Keep your arm as still\nas possible.\nUse phone swipes to adjust\nthe tool.";
            default:
                return "Keep your arm as still\nas possible.\nUse phone swipes to practice.";
        }
    }

    private IEnumerator ShowPracticeIntroIfNeeded()
    {
        if (instructionHUD == null)
            yield break;

        if (_practiceIntroShownThisSession)
            yield break;

        string text = BuildPracticeIntroText();
        string detailText = BuildPracticeDetailText(_practicePhase);
        string sequenceKey = text + "\n---\n" + detailText;
        float now = Time.unscaledTime;
        float minInterval = Mathf.Max(0f, minPracticeIntroRepeatInterval);

        bool isRapidDuplicate =
            !string.IsNullOrEmpty(_lastPracticeIntroShownText) &&
            string.Equals(_lastPracticeIntroShownText, sequenceKey, StringComparison.Ordinal) &&
            (now - _lastPracticeIntroShownAt) < minInterval;

        if (isRapidDuplicate)
            yield break;

        float firstDuration = instructionHUD.Show(text, Mathf.Max(0f, practiceIntroSeconds));
        _practiceIntroShownThisSession = true;
        _lastPracticeIntroShownText = sequenceKey;
        _lastPracticeIntroShownAt = now;

        if (firstDuration > 0f)
            yield return new WaitForSeconds(firstDuration);

        if (!inPractice)
            yield break;

        float panelGap = Mathf.Max(0f, practiceIntroPanelGapSeconds);
        if (panelGap > 0f)
            yield return new WaitForSeconds(panelGap);

        if (!inPractice)
            yield break;

        float secondDuration = instructionHUD.Show(detailText, Mathf.Max(0f, practiceDetailSeconds));
        if (secondDuration > 0f)
            yield return new WaitForSeconds(secondDuration);
    }

    private void UpdatePracticeText()
    {
        if (taskContextHUD == null)
            return;

        if (!inPractice)
        {
            taskContextHUD.SetPracticeText("");
            return;
        }

        int target = Mathf.Max(1, practiceTargetCount);
        int displayIndex = Mathf.Clamp(practiceSuccessCount + 1, 1, target);
        taskContextHUD.SetPracticeText($"PRACTICE {displayIndex}/{target}");
    }

    private void EndPracticeSubscriptions()
    {
        if (placementTask != null) placementTask.OnTrialEnded -= OnPracticeTrialEnded;
        if (rotationTask != null) rotationTask.OnTrialEnded -= OnPracticeTrialEnded;
        if (scalingTask != null) scalingTask.OnTrialEnded -= OnPracticeTrialEnded;
    }

    private void SubscribePracticeTrialEnded(WorkflowProgressionController.Phase phase)
    {
        EndPracticeSubscriptions();
        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                if (placementTask != null) placementTask.OnTrialEnded += OnPracticeTrialEnded;
                break;
            case WorkflowProgressionController.Phase.Rotation:
                if (rotationTask != null) rotationTask.OnTrialEnded += OnPracticeTrialEnded;
                break;
            case WorkflowProgressionController.Phase.Scaling:
                if (scalingTask != null) scalingTask.OnTrialEnded += OnPracticeTrialEnded;
                break;
        }
    }

    private bool IsPracticeTaskRunning()
    {
        switch (_practicePhase)
        {
            case WorkflowProgressionController.Phase.Placement:
                return placementTask != null && placementTask.IsTrialRunning;
            case WorkflowProgressionController.Phase.Rotation:
                return rotationTask != null && rotationTask.IsTrialRunning;
            case WorkflowProgressionController.Phase.Scaling:
                return scalingTask != null && scalingTask.IsTrialRunning;
            default:
                return false;
        }
    }

    private void ApplyForcedIdForPhase(WorkflowProgressionController.Phase phase, string id)
    {
        string forcedId = NormalizeToolId(id);
        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                if (placementTask != null) placementTask.SetForcedActiveId(forcedId);
                break;
            case WorkflowProgressionController.Phase.Rotation:
                if (rotationTask != null) rotationTask.SetForcedActiveId(forcedId);
                break;
            case WorkflowProgressionController.Phase.Scaling:
                if (scalingTask != null) scalingTask.SetForcedActiveId(forcedId);
                break;
        }
    }

    private void ClearForcedIdForPhase(WorkflowProgressionController.Phase phase)
    {
        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                if (placementTask != null) placementTask.ClearForcedActiveId();
                break;
            case WorkflowProgressionController.Phase.Rotation:
                if (rotationTask != null) rotationTask.ClearForcedActiveId();
                break;
            case WorkflowProgressionController.Phase.Scaling:
                if (scalingTask != null) scalingTask.ClearForcedActiveId();
                break;
        }
    }

    private void RestoreCurrentWorkflowToolForcedId()
    {
        if (workflow == null || workflow.CurrentTool == null)
            return;

        var tid = workflow.CurrentTool.GetComponent<ToolId>();
        string id = (tid != null) ? tid.id : null;
        if (string.IsNullOrEmpty(id))
            return;

        ApplyForcedIdForPhase(workflow.CurrentPhase, id);
    }

    private void StartRealBlockForPhase(WorkflowProgressionController.Phase phase)
    {
        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                StartPlacement();
                break;
            case WorkflowProgressionController.Phase.Rotation:
                StartRotation();
                break;
            case WorkflowProgressionController.Phase.Scaling:
                StartScaling();
                break;
        }
    }

    private void ApplyForcedIdToTasks(string id)
    {
        string forcedId = NormalizeToolId(id);
        if (placementTask != null) placementTask.SetForcedActiveId(forcedId);
        if (rotationTask != null)  rotationTask.SetForcedActiveId(forcedId);
        if (scalingTask != null)   scalingTask.SetForcedActiveId(forcedId);
    }

    private void DisableAllTasks()
    {
        if (placementTask != null) placementTask.enabled = false;
        if (rotationTask != null)  rotationTask.enabled  = false;
        if (scalingTask != null)   scalingTask.enabled   = false;
        if (taskContextHUD != null) taskContextHUD.SetVisible(false);
    }

    private void StartPlacement()
    {
        if (placementTask == null) { Debug.LogError("[SFC_V2] placementTask is NULL"); return; }
        if (taskContextHUD != null) taskContextHUD.SetVisible(true);
        if (phoneTechniqueGate != null)
            phoneTechniqueGate.SetInputFrozen(false, recenterOnUnfreeze: false);
        if (phoneRouter != null)
            phoneRouter.SetInputSuppressed(false);
        if (inPractice && _practicePhase == WorkflowProgressionController.Phase.Placement)
            ApplyForcedIdForPhase(WorkflowProgressionController.Phase.Placement, GetPracticeToolIdNormalized());
        else
            ApplyWorkflowForcedIdForPhase(WorkflowProgressionController.Phase.Placement);
        _carryRotationPoseIntoScaling = false;
        if (rotationTask != null)
            rotationTask.ClearCapturedCarryPose();
        if (!inPractice && taskContextHUD != null) taskContextHUD.SetPracticeText("");
        if (grabber != null)
            grabber.ForceRelease();
        Debug.Log($"[SFC_V2] StartPlacement calling StartBlock() enabled={placementTask.enabled} activeInHierarchy={placementTask.gameObject.activeInHierarchy}");
        ApplyGhostRotations_BaselineForPlacement();
        placementTask.enabled = true;
        placementTask.StartBlock();
    }

    private void StartRotation()
    {
        if (rotationTask == null) return;
        if (taskContextHUD != null) taskContextHUD.SetVisible(true);
        if (phoneTechniqueGate != null)
            phoneTechniqueGate.SetInputFrozen(false, recenterOnUnfreeze: false);
        if (phoneRouter != null)
            phoneRouter.SetInputSuppressed(false);
        if (inPractice && _practicePhase == WorkflowProgressionController.Phase.Rotation)
            ApplyForcedIdForPhase(WorkflowProgressionController.Phase.Rotation, GetPracticeToolIdNormalized());
        else
            ApplyWorkflowForcedIdForPhase(WorkflowProgressionController.Phase.Rotation);
        _carryRotationPoseIntoScaling = false;
        rotationTask.PreserveSolvedPoseForNextPhase = !inPractice;
        if (!inPractice && taskContextHUD != null) taskContextHUD.SetPracticeText("");
        if (grabber != null)
            grabber.ForceRelease();
        ApplyGhostRotations_GoalForRotation();
        rotationTask.enabled = true;
        rotationTask.StartBlock();
    }

    private void StartScaling()
    {
        if (scalingTask == null) return;
        if (taskContextHUD != null) taskContextHUD.SetVisible(true);
        if (phoneTechniqueGate != null)
            phoneTechniqueGate.SetInputFrozen(false, recenterOnUnfreeze: false);
        if (phoneRouter != null)
            phoneRouter.SetInputSuppressed(false);
        if (inPractice && _practicePhase == WorkflowProgressionController.Phase.Scaling)
            ApplyForcedIdForPhase(WorkflowProgressionController.Phase.Scaling, GetPracticeToolIdNormalized());
        else
            ApplyWorkflowForcedIdForPhase(WorkflowProgressionController.Phase.Scaling);
        if (!inPractice && taskContextHUD != null) taskContextHUD.SetPracticeText("");
        bool keepRotPose = !inPractice && _carryRotationPoseIntoScaling;
        if (!keepRotPose)
            ApplyGhostRotations_GoalForRotation();
        else if (rotationTask != null && workflow != null && workflow.CurrentTool != null)
        {
            var tid = workflow.CurrentTool.GetComponent<ToolId>();
            if (tid != null && !string.IsNullOrEmpty(tid.id))
                rotationTask.ApplyCapturedPoseForId(tid.id);
        }
        if (grabber != null)
            grabber.ForceRelease();
        scalingTask.enabled = true;
        scalingTask.StartBlock();
    }

    private void HandleConditionStateChangeIfNeeded()
    {
        string now = GetConditionStateKey();
        if (string.IsNullOrEmpty(_conditionStateKey))
        {
            _conditionStateKey = now;
            return;
        }

        if (string.Equals(_conditionStateKey, now, StringComparison.Ordinal))
            return;

        _conditionStateKey = now;
        _carryRotationPoseIntoScaling = false;
        _hasLastRealIntroContext = false;
        _lastRealIntroConditionKey = null;

        if (rotationTask != null)
        {
            rotationTask.PreserveSolvedPoseForNextPhase = false;
            rotationTask.ClearCapturedCarryPose();
            rotationTask.ResetAllTargetsToSceneBaseline();
        }
    }

    private string BuildPracticePhaseKey(WorkflowProgressionController.Phase phase)
    {
        return $"{GetPracticeProfileKey(phase)}|{phase}";
    }

    private string GetPracticeProfileKey(WorkflowProgressionController.Phase phase)
    {
        if (conditionBlockController != null && conditionBlockController.HasCurrentCondition)
        {
            if (conditionBlockController.CurrentTechnique == ConditionBlockController.Technique.Micro)
                return "micro";

            if (phase == WorkflowProgressionController.Phase.Placement)
            {
                return conditionBlockController.CurrentHandLocation == ConditionBlockController.HandLocation.Side
                    ? "macro_side"
                    : "macro_near";
            }

            return "macro_shared";
        }

        bool isMicro = phoneRouter != null && phoneRouter.CurrentMode == PhoneInputRouter.Mode.Micro;
        if (isMicro)
            return "micro";

        return phase == WorkflowProgressionController.Phase.Placement
            ? "macro_near"
            : "macro_shared";
    }

    private string GetConditionStateKey()
    {
        if (conditionBlockController != null && conditionBlockController.HasCurrentCondition)
            return $"{conditionBlockController.CurrentTechnique}|{conditionBlockController.CurrentHandLocation}";

        bool isMicro = phoneRouter != null && phoneRouter.CurrentMode == PhoneInputRouter.Mode.Micro;
        return isMicro ? "Micro|NearHead" : "Macro|NearHead";
    }

    private string GetPracticeToolIdNormalized()
    {
        return NormalizeToolId(practiceToolId);
    }

    private static string NormalizeToolId(string id)
    {
        return string.IsNullOrWhiteSpace(id) ? null : id.Trim();
    }

    private void ApplyWorkflowForcedIdForPhase(WorkflowProgressionController.Phase phase)
    {
        string workflowId = GetWorkflowCurrentToolId();
        if (string.IsNullOrEmpty(workflowId))
            return;

        ApplyForcedIdForPhase(phase, workflowId);
    }

    private string GetWorkflowCurrentToolId()
    {
        if (workflow == null || workflow.CurrentTool == null)
            return null;

        ToolId tid = workflow.CurrentTool.GetComponent<ToolId>();
        if (tid == null)
            return null;

        return NormalizeToolId(tid.id);
    }

    private void ApplyGrabberMode(ProxyHandGrabber.HeldRotationMode mode)
    {
        if (grabber == null) return;
        grabber.SetHeldRotationMode(mode);
    }

    private void RebuildGhostRegistry()
    {
        ghostById.Clear();
        ghostVisualById.Clear();

        Transform root = slotsTargetsRoot;
        if (root == null)
        {
            var go = GameObject.Find("Slots_Targets");
            if (go != null) root = go.transform;
            slotsTargetsRoot = root;
        }

        if (root == null) return;

        var toolIds = root.GetComponentsInChildren<ToolId>(true);
        for (int i = 0; i < toolIds.Length; i++)
        {
            var tid = toolIds[i];
            if (tid == null || string.IsNullOrEmpty(tid.id)) continue;
            Transform ghostRoot = tid.transform;
            Transform visualRoot = ResolveGhostVisualRoot(ghostRoot, tid.id);
            ghostById[tid.id] = ghostRoot;
            ghostVisualById[tid.id] = visualRoot;
        }
    }

    private void CacheAuthoredGoalRotations()
    {
        if (ghostVisualById.Count == 0) RebuildGhostRegistry();

        goalVisualLocalRotById.Clear();
        foreach (var kv in ghostVisualById)
        {
            if (kv.Value == null) continue;
            goalVisualLocalRotById[kv.Key] = kv.Value.localRotation;
        }
    }

    private void CaptureGoalRotationsFromCurrentGhosts()
    {
        if (ghostVisualById.Count == 0) RebuildGhostRegistry();

        foreach (var kv in ghostVisualById)
        {
            if (kv.Value == null) continue;
            goalVisualLocalRotById[kv.Key] = kv.Value.localRotation;
        }
    }

    private void ApplyGhostRotations_BaselineForPlacement()
    {
        if (ghostVisualById.Count == 0) RebuildGhostRegistry();

        foreach (var kv in ghostVisualById)
        {
            if (kv.Value == null) continue;
            kv.Value.localRotation = Quaternion.identity;
        }
    }

    private void ApplyGhostRotations_GoalForRotation()
    {
        if (ghostVisualById.Count == 0) RebuildGhostRegistry();

        foreach (var kv in ghostVisualById)
        {
            if (kv.Value == null) continue;
            Quaternion goalRot;
            if (goalVisualLocalRotById.TryGetValue(kv.Key, out goalRot))
                kv.Value.localRotation = goalRot;
        }
    }

    private Transform ResolveGhostVisualRoot(Transform ghostRoot, string id)
    {
        if (ghostRoot == null) return null;

        var resolver = ghostRoot.GetComponent("GhostVisualResolver");
        if (resolver != null)
        {
            var prop = resolver.GetType().GetProperty("VisualRoot", BindingFlags.Instance | BindingFlags.Public);
            if (prop != null)
            {
                var resolved = prop.GetValue(resolver, null) as Transform;
                if (resolved != null) return resolved;
            }
        }

        var child = ghostRoot.Find("GhostVisual");
        if (child != null)
            return child;

        Debug.LogWarning($"[SFC_V2] GhostVisual not found for id '{id}'. Falling back to ghost root transform.");
        return ghostRoot;
    }
}
