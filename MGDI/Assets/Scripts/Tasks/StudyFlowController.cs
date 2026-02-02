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

    [Tooltip("Workspace controller for HAND output offset (drives RemoteHandRuntime output frame).")]
    public WorkspaceAnchorController handWorkspaceController;

    [Tooltip("Placement task (tools).")]
    public ToolPlacementTaskManager placementTask;

    [Tooltip("Rotation task (tools).")]
    public ToolRotationTaskManager rotationTask;

    [Tooltip("Scaling task (overlay).")]
    public ToolScalingTaskManager_OverlayCore scalingTask;

    [Tooltip("ProxyHandGrabber instance (recommended). Used for force release and rotation-mode policy at task boundaries.")]
    public ProxyHandGrabber grabber;

    [Tooltip("RemoteHandRuntime for side-to-front remap and workspace hand-offset injection.")]
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
    [Header("Workspace Profiles (HAND) - Task × Technique × Location")]
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
    public string cmdStartRotation  = "start rotation";
    public string cmdStartScaling   = "start scaling";

    public string cmdStartMicroPlacement = "start micro placement";
    public string cmdStartMicroRotation  = "start micro rotation";
    public string cmdStartMicroScaling   = "start micro scaling";

    public string cmdStartMacroPlacement = "start macro placement";
    public string cmdStartMacroRotation  = "start macro rotation";
    public string cmdStartMacroScaling   = "start macro scaling";

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

    // Remap toggle dedupe
    private bool _lastRemapEnabled = false;

    private void Start()
    {
        if (taskContextHUD != null)
        {
            taskContextHUD.Clear();
            taskContextHUD.SetVisible(Application.isEditor && showHUDInEditor);
        }

        if (instructionHUD != null)
            instructionHUD.HideImmediate();

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
            string k = args.text.ToLower();
            if (actions.TryGetValue(k, out var a)) a.Invoke();
        };
        recognizer.Start();
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

        if (contentWorkspaceController == null || handWorkspaceController == null)
        {
            Debug.LogError("[StudyFlowController] contentWorkspaceController / handWorkspaceController missing.");
            return;
        }

        SetOnlyTaskActive(currentTask);
        ForceReleaseIfPossible();

        ApplyTechnique();

        // Apply BOTH workspace profiles
        ApplyWorkspaceProfiles_ContentAndHand();

        // Side-to-front remap (optional: side + macro only)
        ApplySideToFrontRemap(currentHandLocation);

        // Grabber mode per task
        ApplyGrabberModeForCurrentTask();

        // Inject HAND anchor into RemoteHandRuntime (critical)
        SyncRemoteHandToHandAnchor();

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
        ApplyTechnique();
        UpdateHUDStatic();

        // remap condition depends on technique
        ApplySideToFrontRemap(currentHandLocation);

        if (restart) RestartCurrent();
        else ShowInstructionForCurrentState_ReturnSeconds();
    }

    public void SetHandLocation(WorkspaceAnchorController.HandLocation loc, bool restart)
    {
        currentHandLocation = loc;

        // Apply BOTH profiles (content + hand) when location changes
        ApplyWorkspaceProfiles_ContentAndHand();

        ApplySideToFrontRemap(currentHandLocation);

        // RemoteHand should follow HAND anchor changes too
        SyncRemoteHandToHandAnchor();

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
        contentWorkspaceController.handLocation = currentHandLocation;
        contentWorkspaceController.ApplyProfile(GetContentProfile(currentTask, currentTechnique, currentHandLocation));

        // HAND
        handWorkspaceController.handLocation = currentHandLocation;
        handWorkspaceController.ApplyProfile(GetHandProfile(currentTask, currentTechnique, currentHandLocation));
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
        // If hand profiles are not assigned yet, fallback to content profiles to avoid nulls.
        // (You can tune hand_* profiles later.)
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

    // ---------------- RemoteHand integration ----------------
    private void SyncRemoteHandToHandAnchor()
    {
        if (remoteHand == null) return;
        if (handWorkspaceController == null || handWorkspaceController.workspaceAnchor == null) return;

        remoteHand.SetWorkspaceOffsetAnchor(handWorkspaceController.workspaceAnchor);
    }

    // side-to-front remap: side + macro only
    private void ApplySideToFrontRemap(WorkspaceAnchorController.HandLocation loc)
    {
        if (remoteHand == null) return;

        bool isSide =
            (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft) ||
            (loc == WorkspaceAnchorController.HandLocation.SideOfBodyRight);

        bool enable = isSide && (currentTechnique == Technique.Macro);

        if (enable == _lastRemapEnabled) return;
        _lastRemapEnabled = enable;

        remoteHand.SetSideToFrontRemap(enable);
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
