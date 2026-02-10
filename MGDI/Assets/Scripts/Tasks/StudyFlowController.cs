using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Windows.Speech;

public class StudyFlowController : MonoBehaviour
{
    public enum TaskType { Placement, Rotation, Scaling, NextTaskPlaceholder }
    public enum Technique { Macro, Micro }

    [Header("References")]
    [Tooltip("Workspace controller for CONTENT (moves tools/targets/overlays as a group).")]
    public WorkspaceAnchorController contentWorkspaceController;

    [Tooltip("Workspace controller for HAND (optional). Keeps your existing hand profiles system, but NOT injected to RemoteHandRuntime anymore.")]
    public WorkspaceAnchorController handWorkspaceController;

    [Tooltip("Placement task (tools).")]
    public ToolPlacementTaskManager placementTask;

    [Tooltip("Rotation task (tools).")]
    public ToolRotationTaskManager rotationTask;

    [Tooltip("Scaling task (overlay).")]
    public ToolScalingTaskManager_OverlayCore scalingTask;

    [Tooltip("ProxyHandGrabber instance (recommended). Used for force release and rotation-mode policy at task boundaries.")]
    public ProxyHandGrabber grabber;

    [Header("Phone Input (Macro/Micro routing)")]
    [Tooltip("Phone input router (Macro: hold, Micro: toggle/axis).")]
    [SerializeField] private PhoneInputRouter phoneRouter;

    [Tooltip("Gates macro pose driver vs micro controllers. Also selects which micro controller is active per task.")]
    [SerializeField] private PhoneTechniqueGate phoneTechniqueGate;

    [Header("HUD (Task Context)")]
    public TaskContextHUD taskContextHUD;

    [Header("HUD (Instruction)")]
    public InstructionHUD instructionHUD;

    [Header("Technique Controllers (optional)")]
    public GameObject macroControllerRoot;
    public GameObject microControllerRoot;

    [SerializeField] private MicroHandAutoPlacer microHandAutoPlacer;

    [Header("Current State")]
    public TaskType currentTask = TaskType.Placement;
    public Technique currentTechnique = Technique.Macro;
    public WorkspaceAnchorController.HandLocation currentHandLocation = WorkspaceAnchorController.HandLocation.NearHead;

    [Header("Macro Phone Pose Driver (for Side→Front remap)")]
    [SerializeField] private PhoneProxyHandRootDriver phoneMacroPoseDriver;

    [Header("Side→Front remap (Phone translation)")]
    [SerializeField] private bool enablePhoneSideToFrontRemap = true;

    // 방향이 마음에 안 들면 여기서 바꿈 (Side L/R 별로 다르게 줄 수도 있음)
    [SerializeField] private bool invertRemapSideLeft = false;
    [SerializeField] private bool invertRemapSideRight = false;

    // remap 상태 dedupe
    private bool _lastPhoneRemapEnabled = false;
    private bool _lastPhoneRemapInvert = false;

    // =======================
    // CONTENT PROFILES
    // =======================
    [Header("Workspace Profiles (CONTENT) - Task × Technique × Location")]
    public WorkspaceAnchorController.WorkspaceProfile placement_macro_near;
    public WorkspaceAnchorController.WorkspaceProfile placement_macro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile placement_macro_sideRight;
    public WorkspaceAnchorController.WorkspaceProfile placement_micro_near;
    public WorkspaceAnchorController.WorkspaceProfile placement_micro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile placement_micro_sideRight;

    public WorkspaceAnchorController.WorkspaceProfile rotation_macro_near;
    public WorkspaceAnchorController.WorkspaceProfile rotation_macro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile rotation_macro_sideRight;
    public WorkspaceAnchorController.WorkspaceProfile rotation_micro_near;
    public WorkspaceAnchorController.WorkspaceProfile rotation_micro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile rotation_micro_sideRight;

    public WorkspaceAnchorController.WorkspaceProfile scaling_macro_near;
    public WorkspaceAnchorController.WorkspaceProfile scaling_macro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile scaling_macro_sideRight;
    public WorkspaceAnchorController.WorkspaceProfile scaling_micro_near;
    public WorkspaceAnchorController.WorkspaceProfile scaling_micro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile scaling_micro_sideRight;

    // =======================
    // HAND PROFILES
    // =======================
    [Header("Workspace Profiles (HAND) - Task × Technique × Location (optional)")]
    public WorkspaceAnchorController.WorkspaceProfile hand_placement_macro_near;
    public WorkspaceAnchorController.WorkspaceProfile hand_placement_macro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile hand_placement_macro_sideRight;
    public WorkspaceAnchorController.WorkspaceProfile hand_placement_micro_near;
    public WorkspaceAnchorController.WorkspaceProfile hand_placement_micro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile hand_placement_micro_sideRight;

    public WorkspaceAnchorController.WorkspaceProfile hand_rotation_macro_near;
    public WorkspaceAnchorController.WorkspaceProfile hand_rotation_macro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile hand_rotation_macro_sideRight;
    public WorkspaceAnchorController.WorkspaceProfile hand_rotation_micro_near;
    public WorkspaceAnchorController.WorkspaceProfile hand_rotation_micro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile hand_rotation_micro_sideRight;

    public WorkspaceAnchorController.WorkspaceProfile hand_scaling_macro_near;
    public WorkspaceAnchorController.WorkspaceProfile hand_scaling_macro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile hand_scaling_macro_sideRight;
    public WorkspaceAnchorController.WorkspaceProfile hand_scaling_micro_near;
    public WorkspaceAnchorController.WorkspaceProfile hand_scaling_micro_sideLeft;
    public WorkspaceAnchorController.WorkspaceProfile hand_scaling_micro_sideRight;

    [Header("Grabber rotation policy per task")]
    public ProxyHandGrabber.HeldRotationMode placementGrabberMode = ProxyHandGrabber.HeldRotationMode.LockAtGrab;
    public ProxyHandGrabber.HeldRotationMode rotationGrabberMode = ProxyHandGrabber.HeldRotationMode.ExternalControl;
    public ProxyHandGrabber.HeldRotationMode scalingGrabberMode = ProxyHandGrabber.HeldRotationMode.LockAtGrab;

    [Header("Voice Commands")]
    public bool enableVoice = true;

    [Header("Debug")]
    [SerializeField] private bool showHUDInEditor = true;

    [SerializeField] private MicroThumbIndexSliderInput microSliderInput;
    [SerializeField] private float microCalibWindowSec = 0.20f;

    // Commands
    public string cmdStartPlacement = "start placement";
    public string cmdStartRotation = "start rotation";
    public string cmdStartScaling = "start scaling";

    public string cmdStartMicroPlacement = "start micro placement";
    public string cmdStartMicroRotation = "start micro rotation";
    public string cmdStartMicroScaling = "start micro scaling";

    public string cmdStartMacroPlacement = "start macro placement";
    public string cmdStartMacroRotation = "start macro rotation";
    public string cmdStartMacroScaling = "start macro scaling";

    public string cmdRestart = "restart";
    public string cmdNext = "next";

    public string cmdMacro = "macro";
    public string cmdMicro = "micro";

    public string cmdNear = "near";
    public string cmdSideLeft = "side left";
    public string cmdSideRight = "side right";

    private KeywordRecognizer recognizer;
    private Dictionary<string, Action> actions;

    float _nextCountdownUpdateTime = 0f;
    Coroutine _startTaskCo;

    // Remap toggle dedupe + recenter logic
    private bool _lastRemapEnabled = false;
    private WorkspaceAnchorController.HandLocation _lastRemapLoc;

    // Optional: avoid spamming warnings
    private bool _warnedHandWorkspaceMissing = false;
    private bool _warnedPhoneIntegrationMissing = false;

    private void Start()
    {
        _lastRemapLoc = currentHandLocation;

        if (taskContextHUD != null)
        {
            taskContextHUD.Clear();
            taskContextHUD.SetVisible(Application.isEditor && showHUDInEditor);
        }

        if (instructionHUD != null)
            instructionHUD.HideImmediate();

        EnsurePhoneIntegrationRefs();
        ApplyPhoneInputRouting(); // ensure Router/Gate reflect initial state

        if (!enableVoice) return;

        actions = new Dictionary<string, Action>
        {
            { cmdStartPlacement.ToLower(), () => StartTask(TaskType.Placement) },
            { cmdStartRotation.ToLower(),  () => StartTask(TaskType.Rotation) },
            { cmdStartScaling.ToLower(),   () => StartTask(TaskType.Scaling) },

            { cmdStartMicroPlacement.ToLower(), () => { BeginMicroCalibration(); StartTechniqueAndTask(Technique.Micro, TaskType.Placement); } },
            { cmdStartMicroRotation.ToLower(),  () => { BeginMicroCalibration(); StartTechniqueAndTask(Technique.Micro, TaskType.Rotation); } },
            { cmdStartMicroScaling.ToLower(),   () => { BeginMicroCalibration(); StartTechniqueAndTask(Technique.Micro, TaskType.Scaling); } },

            { cmdStartMacroPlacement.ToLower(), () => StartTechniqueAndTask(Technique.Macro, TaskType.Placement) },
            { cmdStartMacroRotation.ToLower(),  () => StartTechniqueAndTask(Technique.Macro, TaskType.Rotation) },
            { cmdStartMacroScaling.ToLower(),   () => StartTechniqueAndTask(Technique.Macro, TaskType.Scaling) },

            { cmdRestart.ToLower(), RestartCurrent },
            { cmdNext.ToLower(),    NextConditionAndRestart },

            { cmdMacro.ToLower(), () => SetTechnique(Technique.Macro, restart:true) },
            { cmdMicro.ToLower(), () => { BeginMicroCalibration(); SetTechnique(Technique.Micro, restart:true); } },

            { cmdNear.ToLower(),      () => SetHandLocation(WorkspaceAnchorController.HandLocation.NearHead, restart:false) },
            { cmdSideLeft.ToLower(),  () => SetHandLocation(WorkspaceAnchorController.HandLocation.SideOfBodyLeft, restart:false) },
            { cmdSideRight.ToLower(), () => SetHandLocation(WorkspaceAnchorController.HandLocation.SideOfBodyRight, restart:false) },
        };

        recognizer = new KeywordRecognizer(actions.Keys.ToArray());
        recognizer.OnPhraseRecognized += (args) =>
        {
            DebugHUD.Log($"[SFC] Recognized: '{args.text}'");
            string k = args.text.ToLower();
            if (actions.TryGetValue(k, out var a)) a.Invoke();
        };
        recognizer.Start();
    }

    private void EnsurePhoneIntegrationRefs()
    {
        if (phoneRouter == null)
            phoneRouter = FindFirstObjectByType<PhoneInputRouter>();

        if (phoneTechniqueGate == null)
            phoneTechniqueGate = FindFirstObjectByType<PhoneTechniqueGate>();

        if (microHandAutoPlacer == null)
            microHandAutoPlacer = FindFirstObjectByType<MicroHandAutoPlacer>();

        if (phoneMacroPoseDriver == null)
            phoneMacroPoseDriver = FindFirstObjectByType<PhoneProxyHandRootDriver>();
    }

    private void ApplyPhoneInputRouting()
    {
        EnsurePhoneIntegrationRefs();

        // Router: Macro uses hold, Micro uses toggle/axis
        if (phoneRouter != null)
        {
            if (currentTechnique == Technique.Micro) phoneRouter.SetModeMicro();
            else phoneRouter.SetModeMacro();
        }

        // Gate: selects which micro controller is active per task (only matters in Micro)
        if (phoneTechniqueGate != null)
        {
            if (currentTask == TaskType.Placement) phoneTechniqueGate.SetMicroTaskPlacement();
            else if (currentTask == TaskType.Rotation) phoneTechniqueGate.SetMicroTaskRotation();
            else if (currentTask == TaskType.Scaling) phoneTechniqueGate.SetMicroTaskScaling();
        }

        if ((phoneRouter == null || phoneTechniqueGate == null) && !_warnedPhoneIntegrationMissing)
        {
            Debug.LogWarning("[StudyFlowController] Phone integration missing (PhoneInputRouter and/or PhoneTechniqueGate not assigned/found).");
            _warnedPhoneIntegrationMissing = true;
        }
    }

    private void TryAutoPlaceHandNearActiveTool()
    {
        if (currentTechnique != Technique.Micro) return;
        if (microHandAutoPlacer == null) return;

        if (currentTask == TaskType.Rotation && rotationTask != null)
        {
            var t = rotationTask.ActiveToolTransform;
            if (t != null) microHandAutoPlacer.PlaceHandNear(t);
            return;
        }

        if (currentTask == TaskType.Scaling && scalingTask != null)
        {
            var t = scalingTask.ActiveToolTransform; 
            if (t != null) microHandAutoPlacer.PlaceHandNear(t);
            return;
        }
    }

    private void BeginMicroCalibration()
    {
        microSliderInput?.BeginCalibration(microCalibWindowSec);
    }

    private void StartTechniqueAndTask(Technique tech, TaskType task)
    {
        currentTechnique = tech;
        StartTask(task);
    }

    private void Update()
    {
        if (taskContextHUD == null) return;
        if (Time.unscaledTime < _nextCountdownUpdateTime) return;
        _nextCountdownUpdateTime = Time.unscaledTime + 0.10f;

        if (currentTask == TaskType.Placement && placementTask != null && placementTask.IsTrialRunning)
        {
            taskContextHUD.SetTrialWithCountdown(
                placementTask.CurrentTrialIndex1Based,
                placementTask.TotalTrials,
                placementTask.TrialTimeRemainingSec
            );
        }
        else if (currentTask == TaskType.Rotation && rotationTask != null && rotationTask.IsTrialRunning)
        {
            taskContextHUD.SetTrialWithCountdown(
                rotationTask.CurrentTrialIndex1Based,
                rotationTask.TotalTrials,
                rotationTask.TrialTimeRemainingSec
            );
        }
        else if (currentTask == TaskType.Scaling && scalingTask != null && scalingTask.IsTrialRunning)
        {
            taskContextHUD.SetTrialWithCountdown(
                scalingTask.CurrentTrialIndex1Based,
                scalingTask.TotalTrials,
                scalingTask.TrialTimeRemainingSec
            );
        }
    }

    private void OnDisable()
    {
        UnhookTaskFinishedEvents();
        UnhookTrialChangedEvents();

        if (_startTaskCo != null) StopCoroutine(_startTaskCo);
        _startTaskCo = null;

        if (recognizer != null)
        {
            recognizer.Stop();
            recognizer.Dispose();
            recognizer = null;
        }
    }

    // ---------------- Flow ----------------
    public void StartTask(TaskType t)
    {
        if (_startTaskCo != null) StopCoroutine(_startTaskCo);
        _startTaskCo = null;

        currentTask = t;

        if (contentWorkspaceController == null)
        {
            Debug.LogError("[StudyFlowController] contentWorkspaceController missing.");
            return;
        }

        if (handWorkspaceController == null && !_warnedHandWorkspaceMissing)
        {
            Debug.LogWarning("[StudyFlowController] handWorkspaceController is null. HAND profiles will be skipped.");
            _warnedHandWorkspaceMissing = true;
        }

        SetOnlyTaskActive(currentTask);
        ForceReleaseIfPossible();

        ApplyTechnique();

        // Phone Router/Gate (Technique + Task)
        ApplyPhoneInputRouting();

        // Apply workspace profiles (CONTENT + HAND optional)
        ApplyWorkspaceProfiles_ContentAndHand();

        // Side-to-front remap (Macro + Side). Force recenter at task start for clean baseline.
        ApplySideToFrontRemap(currentHandLocation, forceRecenter: false);

        // Grabber mode per task
        ApplyGrabberModeForCurrentTask();

        HookTaskFinishedEvents(currentTask);
        HookTrialChangedEvents(currentTask);

        if (taskContextHUD != null)
        {
            taskContextHUD.Clear();
            taskContextHUD.SetVisible(true);
            UpdateHUDStatic();
        }

        float waitSec = ShowInstructionForCurrentState_ReturnSeconds();
        _startTaskCo = StartCoroutine(StartTaskAfterDelay(waitSec));
    }

    IEnumerator StartTaskAfterDelay(float waitSec)
    {
        if (waitSec > 0f)
            yield return new WaitForSeconds(waitSec);

        switch (currentTask)
        {
            case TaskType.Placement:
                if (placementTask == null) { Debug.LogError("[StudyFlowController] placementTask missing."); yield break; }
                placementTask.StartBlock();
                break;

            case TaskType.Rotation:
                if (rotationTask == null) { Debug.LogError("[StudyFlowController] rotationTask missing."); yield break; }
                rotationTask.StartBlock();
                break;

            case TaskType.Scaling:
                if (scalingTask == null) { Debug.LogError("[StudyFlowController] scalingTask missing."); yield break; }
                scalingTask.StartBlock();
                break;

            case TaskType.NextTaskPlaceholder:
                yield break;
        }
    }

    public void RestartCurrent()
    {
        StartTask(currentTask);
    }

    public void NextConditionAndRestart()
    {
        if (currentHandLocation == WorkspaceAnchorController.HandLocation.NearHead)
            currentHandLocation = WorkspaceAnchorController.HandLocation.SideOfBodyLeft;
        else if (currentHandLocation == WorkspaceAnchorController.HandLocation.SideOfBodyLeft)
            currentHandLocation = WorkspaceAnchorController.HandLocation.SideOfBodyRight;
        else
        {
            currentHandLocation = WorkspaceAnchorController.HandLocation.NearHead;
            currentTechnique = (currentTechnique == Technique.Macro) ? Technique.Micro : Technique.Macro;
        }

        RestartCurrent();
    }

    public void SetTechnique(Technique tech, bool restart)
    {
        currentTechnique = tech;

        // Phone Router (Technique)
        ApplyPhoneInputRouting();

        ApplyTechnique();
        UpdateHUDStatic();

        // Remap enable depends on technique; don't force recenter here (task start will recenter).
        ApplySideToFrontRemap(currentHandLocation, forceRecenter: false);

        if (restart) RestartCurrent();
        else ShowInstructionForCurrentState_ReturnSeconds();
    }

    public void SetHandLocation(WorkspaceAnchorController.HandLocation loc, bool restart)
    {
        currentHandLocation = loc;

        // Apply BOTH profiles (content + hand optional) when location changes
        ApplyWorkspaceProfiles_ContentAndHand();

        // Location change is the classic case where enable stays true (Side L<->R),
        // so force recenter for neutral-based remap.
        ApplySideToFrontRemap(currentHandLocation, forceRecenter: true);

        if (restart) RestartCurrent();
        else
        {
            UpdateHUDStatic();
            ShowInstructionForCurrentState_ReturnSeconds();
        }
    }

    // ---------------- Apply workspace (CONTENT + HAND) ----------------
    private void ApplyWorkspaceProfiles_ContentAndHand()
    {
        // CONTENT
        if (contentWorkspaceController != null)
        {
            contentWorkspaceController.handLocation = currentHandLocation;
            contentWorkspaceController.ApplyProfile(GetContentProfile(currentTask, currentTechnique, currentHandLocation));
        }

        // HAND (optional)
        if (handWorkspaceController != null)
        {
            handWorkspaceController.handLocation = currentHandLocation;
            handWorkspaceController.ApplyProfile(GetHandProfile(currentTask, currentTechnique, currentHandLocation));
        }
    }

    private WorkspaceAnchorController.WorkspaceProfile GetContentProfile(TaskType task, Technique tech, WorkspaceAnchorController.HandLocation loc)
    {
        if (task == TaskType.Placement)
        {
            if (tech == Technique.Macro)
            {
                if (loc == WorkspaceAnchorController.HandLocation.NearHead) return placement_macro_near;
                if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) return placement_macro_sideLeft;
                return placement_macro_sideRight;
            }
            else
            {
                if (loc == WorkspaceAnchorController.HandLocation.NearHead) return placement_micro_near;
                if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) return placement_micro_sideLeft;
                return placement_micro_sideRight;
            }
        }

        if (task == TaskType.Rotation)
        {
            if (tech == Technique.Macro)
            {
                if (loc == WorkspaceAnchorController.HandLocation.NearHead) return rotation_macro_near;
                if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) return rotation_macro_sideLeft;
                return rotation_macro_sideRight;
            }
            else
            {
                if (loc == WorkspaceAnchorController.HandLocation.NearHead) return rotation_micro_near;
                if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) return rotation_micro_sideLeft;
                return rotation_micro_sideRight;
            }
        }

        // Scaling
        if (tech == Technique.Macro)
        {
            if (loc == WorkspaceAnchorController.HandLocation.NearHead) return scaling_macro_near;
            if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) return scaling_macro_sideLeft;
            return scaling_macro_sideRight;
        }
        else
        {
            if (loc == WorkspaceAnchorController.HandLocation.NearHead) return scaling_micro_near;
            if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) return scaling_micro_sideLeft;
            return scaling_micro_sideRight;
        }
    }

    private WorkspaceAnchorController.WorkspaceProfile GetHandProfile(TaskType task, Technique tech, WorkspaceAnchorController.HandLocation loc)
    {
        // If hand profiles are not assigned, fallback to content profiles to avoid nulls.
        WorkspaceAnchorController.WorkspaceProfile p = null;

        if (task == TaskType.Placement)
        {
            if (tech == Technique.Macro)
            {
                if (loc == WorkspaceAnchorController.HandLocation.NearHead) p = hand_placement_macro_near;
                else if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) p = hand_placement_macro_sideLeft;
                else p = hand_placement_macro_sideRight;
            }
            else
            {
                if (loc == WorkspaceAnchorController.HandLocation.NearHead) p = hand_placement_micro_near;
                else if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) p = hand_placement_micro_sideLeft;
                else p = hand_placement_micro_sideRight;
            }
        }
        else if (task == TaskType.Rotation)
        {
            if (tech == Technique.Macro)
            {
                if (loc == WorkspaceAnchorController.HandLocation.NearHead) p = hand_rotation_macro_near;
                else if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) p = hand_rotation_macro_sideLeft;
                else p = hand_rotation_macro_sideRight;
            }
            else
            {
                if (loc == WorkspaceAnchorController.HandLocation.NearHead) p = hand_rotation_micro_near;
                else if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) p = hand_rotation_micro_sideLeft;
                else p = hand_rotation_micro_sideRight;
            }
        }
        else // Scaling
        {
            if (tech == Technique.Macro)
            {
                if (loc == WorkspaceAnchorController.HandLocation.NearHead) p = hand_scaling_macro_near;
                else if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) p = hand_scaling_macro_sideLeft;
                else p = hand_scaling_macro_sideRight;
            }
            else
            {
                if (loc == WorkspaceAnchorController.HandLocation.NearHead) p = hand_scaling_micro_near;
                else if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) p = hand_scaling_micro_sideLeft;
                else p = hand_scaling_micro_sideRight;
            }
        }

        return p != null ? p : GetContentProfile(task, tech, loc);
    }

    // ---------------- Side-to-front remap: Macro + Side only ----------------
    private void ApplySideToFrontRemap(WorkspaceAnchorController.HandLocation loc, bool forceRecenter)
    {
        // Phone pose driver가 없으면 아무것도 하지 않음
        if (phoneMacroPoseDriver == null) return;

        bool isSide =
            (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) ||
            (loc == WorkspaceAnchorController.HandLocation.SideOfBodyRight);

        bool enable = enablePhoneSideToFrontRemap && isSide && (currentTechnique == Technique.Macro);

        bool invert = false;
        if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) invert = invertRemapSideLeft;
        else if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyRight) invert = invertRemapSideRight;

        // enable/invert 변화가 있으면 SetSideToFrontRemap로 토글 + (필요 시) recenter
        bool changed = (enable != _lastPhoneRemapEnabled) || (invert != _lastPhoneRemapInvert);
        DebugHUD.Log($"[SFC] Remap enable={enable} invert={invert} loc={loc} tech={currentTechnique} driver='{phoneMacroPoseDriver?.name}' enabled={phoneMacroPoseDriver!=null && phoneMacroPoseDriver.enabled}");
        if (changed)
        {
            _lastPhoneRemapEnabled = enable;
            _lastPhoneRemapInvert = invert;
            _lastRemapLoc = loc;

            phoneMacroPoseDriver.SetSideToFrontRemap(enable, invert, forceRecenter: true);
            DebugHUD.Log($"[SFC] PhoneRemap toggle enable={enable} invert={invert} loc={loc} tech={currentTechnique}");
            return;
        }

        // enable 유지 중이고, 조건이 바뀌거나 강제 recenter가 필요하면 baseline 재캡처
        if (enable && (forceRecenter || loc != _lastRemapLoc))
        {
            phoneMacroPoseDriver.Recenter();
            DebugHUD.Log($"[SFC] PhoneRemap recenter loc={loc} tech={currentTechnique}");
        }

        _lastRemapLoc = loc;
    }

    // ---------------- Technique toggles ----------------
    private void ApplyTechnique()
    {
        if (macroControllerRoot != null)
            macroControllerRoot.SetActive(currentTechnique == Technique.Macro);

        if (microControllerRoot != null)
            microControllerRoot.SetActive(currentTechnique == Technique.Micro);
    }

    // ---------------- Grabber policy at task boundaries ----------------
    private void ApplyGrabberModeForCurrentTask()
    {
        if (grabber == null) return;

        if (currentTask == TaskType.Placement)
            grabber.SetHeldRotationMode(placementGrabberMode);
        else if (currentTask == TaskType.Rotation)
            grabber.SetHeldRotationMode(rotationGrabberMode);
        else if (currentTask == TaskType.Scaling)
            grabber.SetHeldRotationMode(scalingGrabberMode);
        else
            grabber.SetHeldRotationMode(placementGrabberMode);
    }

    private void ForceReleaseIfPossible()
    {
        if (grabber != null)
            grabber.ForceRelease();
    }

    // ---------------- HUD helpers ----------------
    private void UpdateHUDStatic()
    {
        if (taskContextHUD == null) return;

        string taskName =
            (currentTask == TaskType.Placement) ? "Tool Placement" :
            (currentTask == TaskType.Rotation) ? "Tool Rotation" :
            (currentTask == TaskType.Scaling) ? "Overlay Scaling" : "";

        string cond = $"{currentTechnique} · {ShortHandLocation(currentHandLocation)}";
        taskContextHUD.SetTaskLabel(taskName);
        taskContextHUD.SetCondition(cond);
    }

    private string ShortHandLocation(WorkspaceAnchorController.HandLocation loc)
    {
        switch (loc)
        {
            case WorkspaceAnchorController.HandLocation.NearHead: return "Near";
            case WorkspaceAnchorController.HandLocation.SideOfBodyLeft: return "Side (L)";
            case WorkspaceAnchorController.HandLocation.SideOfBodyRight: return "Side (R)";
            default: return loc.ToString();
        }
    }

    private void OnTrialChanged(int current1Based, int total)
    {
        if (taskContextHUD == null) return;
        taskContextHUD.SetTrialWithCountdown(current1Based, total, 0f);

        TryAutoPlaceHandNearActiveTool(); // NEW
    }

    private void HookTrialChangedEvents(TaskType task)
    {
        UnhookTrialChangedEvents();

        if (task == TaskType.Placement && placementTask != null)
            placementTask.OnTrialChanged += OnTrialChanged;

        if (task == TaskType.Rotation && rotationTask != null)
            rotationTask.OnTrialChanged += OnTrialChanged;

        if (task == TaskType.Scaling && scalingTask != null)
            scalingTask.OnTrialChanged += OnTrialChanged;
    }

    private void UnhookTrialChangedEvents()
    {
        if (placementTask != null) placementTask.OnTrialChanged -= OnTrialChanged;
        if (rotationTask != null) rotationTask.OnTrialChanged -= OnTrialChanged;
        if (scalingTask != null) scalingTask.OnTrialChanged -= OnTrialChanged;
    }

    // ---------------- Instruction helpers ----------------
    private float ShowInstructionForCurrentState_ReturnSeconds()
    {
        if (instructionHUD == null) return 0f;

        if (currentTask == TaskType.Placement)
            return instructionHUD.Show("Place each tool as close as possible to its highlighted silhouette. Release to evaluate.");

        if (currentTask == TaskType.Rotation)
            return instructionHUD.Show("Rotate the highlighted tool to match the target orientation. Stop input to evaluate.");

        if (currentTask == TaskType.Scaling)
            return instructionHUD.Show("Resize the overlay to match the target size. Release to evaluate.");

        instructionHUD.HideImmediate();
        return 0f;
    }

    // ---------------- Task finished wiring ----------------
    private void HookTaskFinishedEvents(TaskType task)
    {
        UnhookTaskFinishedEvents();

        if (task == TaskType.Placement && placementTask != null)
            placementTask.OnBlockFinished += OnActiveTaskFinished;

        if (task == TaskType.Rotation && rotationTask != null)
            rotationTask.OnBlockFinished += OnActiveTaskFinished;

        if (task == TaskType.Scaling && scalingTask != null)
            scalingTask.OnBlockFinished += OnActiveTaskFinished;
    }

    private void UnhookTaskFinishedEvents()
    {
        if (placementTask != null) placementTask.OnBlockFinished -= OnActiveTaskFinished;
        if (rotationTask != null) rotationTask.OnBlockFinished -= OnActiveTaskFinished;
        if (scalingTask != null) scalingTask.OnBlockFinished -= OnActiveTaskFinished;
    }

    private void OnActiveTaskFinished()
    {
        UnhookTaskFinishedEvents();
        UnhookTrialChangedEvents();

        if (_startTaskCo != null) StopCoroutine(_startTaskCo);
        _startTaskCo = null;

        if (taskContextHUD != null)
        {
            taskContextHUD.Clear();
            taskContextHUD.SetVisible(false);
        }

        if (instructionHUD != null)
            instructionHUD.HideImmediate();

        currentTask = TaskType.NextTaskPlaceholder;
        SetOnlyTaskActive(TaskType.NextTaskPlaceholder);

        // Optional: keep phone routing consistent with "no active task"
        ApplyPhoneInputRouting();
    }

    // ---------------- Hard gating: only one task can run ----------------
    private void SetOnlyTaskActive(TaskType taskToRun)
    {
        if (placementTask != null)
        {
            placementTask.enabled = false;
            placementTask.gameObject.SetActive(false);
        }

        if (rotationTask != null)
        {
            rotationTask.enabled = false;
            rotationTask.gameObject.SetActive(false);
        }

        if (scalingTask != null)
        {
            scalingTask.enabled = false;
            scalingTask.gameObject.SetActive(false);
        }

        switch (taskToRun)
        {
            case TaskType.Placement:
                if (placementTask != null)
                {
                    placementTask.gameObject.SetActive(true);
                    placementTask.enabled = true;
                }
                break;

            case TaskType.Rotation:
                if (rotationTask != null)
                {
                    rotationTask.gameObject.SetActive(true);
                    rotationTask.enabled = true;
                }
                break;

            case TaskType.Scaling:
                if (scalingTask != null)
                {
                    scalingTask.gameObject.SetActive(true);
                    scalingTask.enabled = true;
                }
                break;

            case TaskType.NextTaskPlaceholder:
                // nothing
                break;
        }
    }
}
