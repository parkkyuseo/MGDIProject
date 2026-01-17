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
    public LegoPlacementTaskManager placementTask;
    public LegoRotationTaskManager rotationTask;
    public LegoScalingTaskManager scalingTask;

    [Tooltip("Future task reference (optional). Keep null until the task is implemented.")]
    public MonoBehaviour nextTaskPlaceholder;

    [Tooltip("ProxyHandGrabber instance (recommended). Used for force release and rotation-mode policy at task boundaries.")]
    public ProxyHandGrabber grabber;

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

    public string cmdStartPlacement = "start placement";
    public string cmdStartRotation = "start rotation";
    public string cmdStartScaling = "start scaling";
    public string cmdRestart = "restart";
    public string cmdNext = "next";

    public string cmdMacro = "macro";
    public string cmdMicro = "micro";

    public string cmdNear = "near";
    public string cmdSideLeft = "side left";
    public string cmdSideRight = "side right";

    private KeywordRecognizer recognizer;
    private Dictionary<string, System.Action> actions;

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

        if (!enableVoice) return;

        actions = new Dictionary<string, System.Action>
        {
            { cmdStartPlacement.ToLower(), () => StartTask(TaskType.Placement) },
            { cmdStartRotation.ToLower(),  () => StartTask(TaskType.Rotation) },
            { cmdStartScaling.ToLower(),   () => StartTask(TaskType.Scaling) },

            { cmdRestart.ToLower(), RestartCurrent },
            { cmdNext.ToLower(),    NextConditionAndRestart },

            { cmdMacro.ToLower(), () => SetTechnique(Technique.Macro, restart:true) },
            { cmdMicro.ToLower(), () => SetTechnique(Technique.Micro, restart:true) },

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

    private void Update()
    {
        if (taskContextHUD == null) return;
        if (Time.unscaledTime < _nextCountdownUpdateTime) return;
        _nextCountdownUpdateTime = Time.unscaledTime + 0.10f;

        if (currentTask == TaskType.Rotation && rotationTask != null && rotationTask.IsTrialRunning)
        {
            taskContextHUD.SetTrialWithCountdown(
                rotationTask.CurrentTrialIndex1Based,
                rotationTask.TotalTrials,
                rotationTask.TrialTimeRemainingSec
            );
        }
        else if (currentTask == TaskType.Placement && placementTask != null && placementTask.IsTrialRunning)
        {
            taskContextHUD.SetTrialWithCountdown(
                placementTask.CurrentTrialIndex1Based,
                placementTask.TotalTrials,
                placementTask.TrialTimeRemainingSec
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

        if (workspaceController == null)
        {
            Debug.LogError("[StudyFlowController] workspaceController missing.");
            return;
        }

        SetOnlyTaskActive(currentTask);

        ForceReleaseIfPossible();

        ApplyTechnique();
        ApplyWorkspaceProfile();
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
        if (restart) RestartCurrent();
        else
        {
            ApplyWorkspaceProfile();
            UpdateHUDStatic();
            ShowInstructionForCurrentState_ReturnSeconds();
        }
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
        if (currentTask != TaskType.Placement && currentTask != TaskType.Rotation && currentTask != TaskType.Scaling) return;

        string taskName =
            (currentTask == TaskType.Placement) ? "Placement Task" :
            (currentTask == TaskType.Rotation) ? "Rotation Task" :
            (currentTask == TaskType.Scaling) ? "Scaling Task" : "";

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

        if (task == TaskType.Rotation && rotationTask != null)
            rotationTask.OnTrialChanged += OnTrialChanged;

        if (task == TaskType.Placement && placementTask != null)
            placementTask.OnTrialChanged += OnTrialChanged;

        if (task == TaskType.Scaling && scalingTask != null)
            scalingTask.OnTrialChanged += OnTrialChanged;
    }

    private void UnhookTrialChangedEvents()
    {
        if (rotationTask != null) rotationTask.OnTrialChanged -= OnTrialChanged;
        if (placementTask != null) placementTask.OnTrialChanged -= OnTrialChanged;
        if (scalingTask != null) scalingTask.OnTrialChanged -= OnTrialChanged;
    }

    // ---------------- Instruction helpers ----------------
    private float ShowInstructionForCurrentState_ReturnSeconds()
    {
        if (instructionHUD == null) return 0f;

        if (currentTask != TaskType.Placement && currentTask != TaskType.Rotation && currentTask != TaskType.Scaling)
        {
            instructionHUD.HideImmediate();
            return 0f;
        }

        if (currentTask == TaskType.Placement)
            return instructionHUD.Show("Move the block into the highlighted slot.");

        if (currentTask == TaskType.Rotation)
            return instructionHUD.Show("Rotate the block to match the target.");

        if (currentTask == TaskType.Scaling)
            return instructionHUD.Show("Scale the block to match the target.");

        return 0f;
    }

    // ---------------- Task finished wiring ----------------
    private void HookTaskFinishedEvents(TaskType task)
    {
        UnhookTaskFinishedEvents();

        if (task == TaskType.Rotation && rotationTask != null)
            rotationTask.OnBlockFinished += OnActiveTaskFinished;

        if (task == TaskType.Placement && placementTask != null)
            placementTask.OnBlockFinished += OnActiveTaskFinished;

        if (task == TaskType.Scaling && scalingTask != null)
            scalingTask.OnBlockFinished += OnActiveTaskFinished;
    }

    private void UnhookTaskFinishedEvents()
    {
        if (rotationTask != null) rotationTask.OnBlockFinished -= OnActiveTaskFinished;
        if (placementTask != null) placementTask.OnBlockFinished -= OnActiveTaskFinished;
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

        if (nextTaskPlaceholder != null)
        {
            nextTaskPlaceholder.enabled = false;
            nextTaskPlaceholder.gameObject.SetActive(false);
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
                if (nextTaskPlaceholder != null)
                {
                    nextTaskPlaceholder.gameObject.SetActive(true);
                    nextTaskPlaceholder.enabled = true;
                }
                break;
        }
    }
}
