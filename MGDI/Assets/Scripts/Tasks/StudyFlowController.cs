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
    public WorkspaceAnchorController workspaceController;

    [Tooltip("Placement task (tools).")]
    public ToolPlacementTaskManager placementTask;

    [Tooltip("Rotation task (tools).")]
    public ToolRotationTaskManager rotationTask;

    [Tooltip("Scaling task (overlay).")]
    public ToolScalingTaskManager_OverlayCore scalingTask;

    [Tooltip("ProxyHandGrabber instance (recommended). Used for force release and rotation-mode policy at task boundaries.")]
    public ProxyHandGrabber grabber;

    [Tooltip("RemoteHandRuntime that provides side->front remap toggle.")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("HUD (Task Context)")]
    public TaskContextHUD taskContextHUD;

    [Header("HUD (Instruction)")]
    public InstructionHUD instructionHUD;

    [Header("Technique Controllers (optional)")]
    public GameObject macroControllerRoot;
    public GameObject microControllerRoot;

    [Header("Current State")]
    public TaskType currentTask = TaskType.Placement;
    public Technique currentTechnique = Technique.Macro;
    public WorkspaceAnchorController.HandLocation currentHandLocation = WorkspaceAnchorController.HandLocation.NearHead;

    [Header("Workspace Profiles (Task × Technique × Location)")]
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

    // Basic task commands (start with current technique/location)
    public string cmdStartPlacement = "start placement";
    public string cmdStartRotation  = "start rotation";
    public string cmdStartScaling   = "start scaling";

    // One-shot micro starts (tech + task)
    public string cmdStartMicroPlacement = "start micro placement";
    public string cmdStartMicroRotation  = "start micro rotation";
    public string cmdStartMicroScaling   = "start micro scaling";

    // One-shot macro starts (tech + task)
    public string cmdStartMacroPlacement = "start macro placement";
    public string cmdStartMacroRotation  = "start macro rotation";
    public string cmdStartMacroScaling   = "start macro scaling";

    // NEW: one-shot location + task (keeps current technique)
    public string cmdStartPlacementNear      = "start placement near";
    public string cmdStartPlacementSideLeft  = "start placement side left";
    public string cmdStartPlacementSideRight = "start placement side right";

    public string cmdStartRotationNear      = "start rotation near";
    public string cmdStartRotationSideLeft  = "start rotation side left";
    public string cmdStartRotationSideRight = "start rotation side right";

    public string cmdStartScalingNear      = "start scaling near";
    public string cmdStartScalingSideLeft  = "start scaling side left";
    public string cmdStartScalingSideRight = "start scaling side right";

    // (Optional) one-shot technique + location + task (most explicit; avoids ambiguity)
    public string cmdStartMicroPlacementNear      = "start micro placement near";
    public string cmdStartMicroPlacementSideLeft  = "start micro placement side left";
    public string cmdStartMicroPlacementSideRight = "start micro placement side right";

    public string cmdStartMicroRotationNear      = "start micro rotation near";
    public string cmdStartMicroRotationSideLeft  = "start micro rotation side left";
    public string cmdStartMicroRotationSideRight = "start micro rotation side right";

    public string cmdStartMicroScalingNear      = "start micro scaling near";
    public string cmdStartMicroScalingSideLeft  = "start micro scaling side left";
    public string cmdStartMicroScalingSideRight = "start micro scaling side right";

    public string cmdStartMacroPlacementNear      = "start macro placement near";
    public string cmdStartMacroPlacementSideLeft  = "start macro placement side left";
    public string cmdStartMacroPlacementSideRight = "start macro placement side right";

    public string cmdStartMacroRotationNear      = "start macro rotation near";
    public string cmdStartMacroRotationSideLeft  = "start macro rotation side left";
    public string cmdStartMacroRotationSideRight = "start macro rotation side right";

    public string cmdStartMacroScalingNear      = "start macro scaling near";
    public string cmdStartMacroScalingSideLeft  = "start macro scaling side left";
    public string cmdStartMacroScalingSideRight = "start macro scaling side right";

    // General
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

    private void Start()
    {
        if (taskContextHUD != null)
        {
            taskContextHUD.Clear();
            taskContextHUD.SetVisible(Application.isEditor && showHUDInEditor);
        }

        if (instructionHUD != null)
            instructionHUD.HideImmediate();

        // Make remap consistent at boot
        ApplySideToFrontRemap(currentHandLocation);

        if (!enableVoice) return;

        actions = new Dictionary<string, Action>
        {
            // Basic: start with current technique/location
            { cmdStartPlacement.ToLower(), () => StartTask(TaskType.Placement) },
            { cmdStartRotation.ToLower(),  () => StartTask(TaskType.Rotation) },
            { cmdStartScaling.ToLower(),   () => StartTask(TaskType.Scaling) },

            // Tech + task one-shots (use current location)
            { cmdStartMicroPlacement.ToLower(), () => { BeginMicroCalibration(); StartTechniqueAndTask(Technique.Micro, TaskType.Placement); } },
            { cmdStartMicroRotation.ToLower(),  () => { BeginMicroCalibration(); StartTechniqueAndTask(Technique.Micro, TaskType.Rotation); } },
            { cmdStartMicroScaling.ToLower(),   () => { BeginMicroCalibration(); StartTechniqueAndTask(Technique.Micro, TaskType.Scaling); } },

            { cmdStartMacroPlacement.ToLower(), () => StartTechniqueAndTask(Technique.Macro, TaskType.Placement) },
            { cmdStartMacroRotation.ToLower(),  () => StartTechniqueAndTask(Technique.Macro, TaskType.Rotation) },
            { cmdStartMacroScaling.ToLower(),   () => StartTechniqueAndTask(Technique.Macro, TaskType.Scaling) },

            // Location + task one-shots (keep current technique)
            { cmdStartPlacementNear.ToLower(),      () => StartLocationAndTask(currentTechnique, WorkspaceAnchorController.HandLocation.NearHead, TaskType.Placement) },
            { cmdStartPlacementSideLeft.ToLower(),  () => StartLocationAndTask(currentTechnique, WorkspaceAnchorController.HandLocation.SideOfBodyLeft, TaskType.Placement) },
            { cmdStartPlacementSideRight.ToLower(), () => StartLocationAndTask(currentTechnique, WorkspaceAnchorController.HandLocation.SideOfBodyRight, TaskType.Placement) },

            { cmdStartRotationNear.ToLower(),      () => StartLocationAndTask(currentTechnique, WorkspaceAnchorController.HandLocation.NearHead, TaskType.Rotation) },
            { cmdStartRotationSideLeft.ToLower(),  () => StartLocationAndTask(currentTechnique, WorkspaceAnchorController.HandLocation.SideOfBodyLeft, TaskType.Rotation) },
            { cmdStartRotationSideRight.ToLower(), () => StartLocationAndTask(currentTechnique, WorkspaceAnchorController.HandLocation.SideOfBodyRight, TaskType.Rotation) },

            { cmdStartScalingNear.ToLower(),      () => StartLocationAndTask(currentTechnique, WorkspaceAnchorController.HandLocation.NearHead, TaskType.Scaling) },
            { cmdStartScalingSideLeft.ToLower(),  () => StartLocationAndTask(currentTechnique, WorkspaceAnchorController.HandLocation.SideOfBodyLeft, TaskType.Scaling) },
            { cmdStartScalingSideRight.ToLower(), () => StartLocationAndTask(currentTechnique, WorkspaceAnchorController.HandLocation.SideOfBodyRight, TaskType.Scaling) },

            // Tech + location + task one-shots (most explicit; recommended for experiments)
            { cmdStartMicroPlacementNear.ToLower(),      () => { BeginMicroCalibration(); StartLocationAndTask(Technique.Micro, WorkspaceAnchorController.HandLocation.NearHead, TaskType.Placement); } },
            { cmdStartMicroPlacementSideLeft.ToLower(),  () => { BeginMicroCalibration(); StartLocationAndTask(Technique.Micro, WorkspaceAnchorController.HandLocation.SideOfBodyLeft, TaskType.Placement); } },
            { cmdStartMicroPlacementSideRight.ToLower(), () => { BeginMicroCalibration(); StartLocationAndTask(Technique.Micro, WorkspaceAnchorController.HandLocation.SideOfBodyRight, TaskType.Placement); } },

            { cmdStartMicroRotationNear.ToLower(),      () => { BeginMicroCalibration(); StartLocationAndTask(Technique.Micro, WorkspaceAnchorController.HandLocation.NearHead, TaskType.Rotation); } },
            { cmdStartMicroRotationSideLeft.ToLower(),  () => { BeginMicroCalibration(); StartLocationAndTask(Technique.Micro, WorkspaceAnchorController.HandLocation.SideOfBodyLeft, TaskType.Rotation); } },
            { cmdStartMicroRotationSideRight.ToLower(), () => { BeginMicroCalibration(); StartLocationAndTask(Technique.Micro, WorkspaceAnchorController.HandLocation.SideOfBodyRight, TaskType.Rotation); } },

            { cmdStartMicroScalingNear.ToLower(),      () => { BeginMicroCalibration(); StartLocationAndTask(Technique.Micro, WorkspaceAnchorController.HandLocation.NearHead, TaskType.Scaling); } },
            { cmdStartMicroScalingSideLeft.ToLower(),  () => { BeginMicroCalibration(); StartLocationAndTask(Technique.Micro, WorkspaceAnchorController.HandLocation.SideOfBodyLeft, TaskType.Scaling); } },
            { cmdStartMicroScalingSideRight.ToLower(), () => { BeginMicroCalibration(); StartLocationAndTask(Technique.Micro, WorkspaceAnchorController.HandLocation.SideOfBodyRight, TaskType.Scaling); } },

            { cmdStartMacroPlacementNear.ToLower(),      () => StartLocationAndTask(Technique.Macro, WorkspaceAnchorController.HandLocation.NearHead, TaskType.Placement) },
            { cmdStartMacroPlacementSideLeft.ToLower(),  () => StartLocationAndTask(Technique.Macro, WorkspaceAnchorController.HandLocation.SideOfBodyLeft, TaskType.Placement) },
            { cmdStartMacroPlacementSideRight.ToLower(), () => StartLocationAndTask(Technique.Macro, WorkspaceAnchorController.HandLocation.SideOfBodyRight, TaskType.Placement) },

            { cmdStartMacroRotationNear.ToLower(),      () => StartLocationAndTask(Technique.Macro, WorkspaceAnchorController.HandLocation.NearHead, TaskType.Rotation) },
            { cmdStartMacroRotationSideLeft.ToLower(),  () => StartLocationAndTask(Technique.Macro, WorkspaceAnchorController.HandLocation.SideOfBodyLeft, TaskType.Rotation) },
            { cmdStartMacroRotationSideRight.ToLower(), () => StartLocationAndTask(Technique.Macro, WorkspaceAnchorController.HandLocation.SideOfBodyRight, TaskType.Rotation) },

            { cmdStartMacroScalingNear.ToLower(),      () => StartLocationAndTask(Technique.Macro, WorkspaceAnchorController.HandLocation.NearHead, TaskType.Scaling) },
            { cmdStartMacroScalingSideLeft.ToLower(),  () => StartLocationAndTask(Technique.Macro, WorkspaceAnchorController.HandLocation.SideOfBodyLeft, TaskType.Scaling) },
            { cmdStartMacroScalingSideRight.ToLower(), () => StartLocationAndTask(Technique.Macro, WorkspaceAnchorController.HandLocation.SideOfBodyRight, TaskType.Scaling) },

            // Utility
            { cmdRestart.ToLower(), RestartCurrent },
            { cmdNext.ToLower(),    NextConditionAndRestart },

            { cmdMacro.ToLower(), () => SetTechnique(Technique.Macro, restart:true) },
            { cmdMicro.ToLower(), () => { BeginMicroCalibration(); SetTechnique(Technique.Micro, restart:true); } },

            // Location-only toggles (do not start task)
            { cmdNear.ToLower(),      () => SetHandLocation(WorkspaceAnchorController.HandLocation.NearHead, restart:false) },
            { cmdSideLeft.ToLower(),  () => SetHandLocation(WorkspaceAnchorController.HandLocation.SideOfBodyLeft, restart:false) },
            { cmdSideRight.ToLower(), () => SetHandLocation(WorkspaceAnchorController.HandLocation.SideOfBodyRight, restart:false) },
        };

        recognizer = new KeywordRecognizer(actions.Keys.ToArray());
        recognizer.OnPhraseRecognized += (args) =>
        {
            string k = args.text.ToLower();
            if (actions.TryGetValue(k, out var a)) a.Invoke();
        };
        recognizer.Start();

        if (remoteHand != null && workspaceController != null && workspaceController.workspaceAnchor != null)
        {
            remoteHand.SetWorkspaceOffsetAnchor(workspaceController.workspaceAnchor);
            DebugHUD.Log("[SFC] Injected workspaceAnchor to RemoteHandRuntime at Start: " + workspaceController.workspaceAnchor.name);
        }
        else
        {
            DebugHUD.Log("[SFC] Cannot inject workspaceAnchor at Start (remoteHand/workspaceController/workspaceAnchor is null).");
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

    private void StartLocationAndTask(Technique tech, WorkspaceAnchorController.HandLocation loc, TaskType task)
    {
        currentTechnique = tech;
        currentHandLocation = loc;
        ApplySideToFrontRemap(currentHandLocation);
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

    private void SyncRemoteHandWorkspaceOffset()
    {
        if (remoteHand == null || workspaceController == null) return;
        if (workspaceController.workspaceAnchor == null) return;

        remoteHand.SetWorkspaceOffsetAnchor(workspaceController.workspaceAnchor);

        // baseline을 "near일 때만 다시 잡고 싶다"면 아래 한 줄만 유지(선택)
        if (currentHandLocation == WorkspaceAnchorController.HandLocation.NearHead)
            remoteHand.CaptureWorkspaceBaseFromCurrentAnchor();
    }

    // ---------------- Flow ----------------
    public void StartTask(TaskType t)
    {
        if (_startTaskCo != null) StopCoroutine(_startTaskCo);
        _startTaskCo = null;

        currentTask = t;

        if (workspaceController == null)
        {
            Debug.LogError("[StudyFlowController] workspaceController missing.");
            return;
        }

        SetOnlyTaskActive(currentTask);

        ForceReleaseIfPossible();

        ApplyTechnique();
        ApplyWorkspaceProfile(); // will place workspace per task/tech/location
        remoteHand?.SetWorkspaceOffsetAnchor(workspaceController.workspaceAnchor);
        SyncRemoteHandWorkspaceOffset();
        ApplySideToFrontRemap(currentHandLocation); // keep hand input consistent with location
        ApplyGrabberModeForCurrentTask();

        HookTaskFinishedEvents(currentTask);
        HookTrialChangedEvents(currentTask);

        if (taskContextHUD != null)
        {
            taskContextHUD.Clear();
            taskContextHUD.SetVisible(currentTask == TaskType.Placement || currentTask == TaskType.Rotation || currentTask == TaskType.Scaling);
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

        ApplySideToFrontRemap(currentHandLocation);
        RestartCurrent();
    }

    public void SetTechnique(Technique tech, bool restart)
    {
        currentTechnique = tech;
        ApplyTechnique();
        UpdateHUDStatic();

        if (restart) RestartCurrent();
        else ShowInstructionForCurrentState_ReturnSeconds();
    }

    public void SetHandLocation(WorkspaceAnchorController.HandLocation loc, bool restart)
    {
        currentHandLocation = loc;
        ApplySideToFrontRemap(currentHandLocation);

        if (restart) RestartCurrent();
        else
        {
            ApplyWorkspaceProfile();
            SyncRemoteHandWorkspaceOffset();
            UpdateHUDStatic();
            ShowInstructionForCurrentState_ReturnSeconds();
        }
    }

    // ---------------- Side→Front remap toggle ----------------
    private void ApplySideToFrontRemap(WorkspaceAnchorController.HandLocation loc)
    {
        if (remoteHand == null) return;

        bool isSide =
            (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) ||
            (loc == WorkspaceAnchorController.HandLocation.SideOfBodyRight);

        // near/front => false, side => true
        remoteHand.SetSideToFrontRemap(isSide);
    }

    // ---------------- Apply workspace ----------------
    private void ApplyWorkspaceProfile()
    {
        var profile = GetProfile(currentTask, currentTechnique, currentHandLocation);
        if (profile == null) return;

        workspaceController.handLocation = currentHandLocation;
        workspaceController.ApplyProfile(profile);
    }

    private WorkspaceAnchorController.WorkspaceProfile GetProfile(TaskType task, Technique tech, WorkspaceAnchorController.HandLocation loc)
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

        if (task == TaskType.Scaling)
        {
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

        return null;
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
