using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Windows.Speech;

public class StudyFlowController : MonoBehaviour
{
    public enum TaskType { Placement, Rotation }
    public enum Technique { Macro, Micro }

    [Header("References")]
    public WorkspaceAnchorController workspaceController;

    public LegoPlacementTaskManager placementTask;
    public LegoRotationTaskManager rotationTask;

    [Tooltip("ProxyHandGrabber instance (recommended). Used for force release and rotation-mode policy at task boundaries.")]
    public ProxyHandGrabber grabber;

    [Header("Technique Controllers (optional)")]
    public GameObject macroControllerRoot; // e.g., ProxyHandR
    public GameObject microControllerRoot; // later

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

    [Header("Grabber rotation policy per task")]
    [Tooltip("Recommended: Placement=LockAtGrab (translation-only), Rotation=ExternalControl (RotationTM controls yaw).")]
    public ProxyHandGrabber.HeldRotationMode placementGrabberMode = ProxyHandGrabber.HeldRotationMode.LockAtGrab;

    [Tooltip("Recommended: Rotation=ExternalControl (RotationTM controls yaw).")]
    public ProxyHandGrabber.HeldRotationMode rotationGrabberMode = ProxyHandGrabber.HeldRotationMode.ExternalControl;

    [Header("Voice Commands")]
    public bool enableVoice = true;

    public string cmdStartPlacement = "start placement";
    public string cmdStartRotation = "start rotation";
    public string cmdRestart = "restart";
    public string cmdNext = "next";

    public string cmdMacro = "macro";
    public string cmdMicro = "micro";

    public string cmdNear = "near";
    public string cmdSideLeft = "side left";
    public string cmdSideRight = "side right";

    private KeywordRecognizer recognizer;
    private Dictionary<string, System.Action> actions;

    private void Start()
    {
        if (!enableVoice) return;

        // Register commands in lowercase to match args.text.ToLower()
        actions = new Dictionary<string, System.Action>
        {
            { cmdStartPlacement.ToLower(), () => StartTask(TaskType.Placement) },
            { cmdStartRotation.ToLower(),  () => StartTask(TaskType.Rotation) },

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

        Debug.Log($"[StudyFlowController] Voice enabled: {string.Join(", ", actions.Keys)}");
    }

    private void OnDisable()
    {
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
        currentTask = t;

        if (workspaceController == null)
        {
            Debug.LogError("[StudyFlowController] workspaceController missing.");
            return;
        }

        // Always release at task boundary to avoid carrying held state across tasks.
        ForceReleaseIfPossible();

        // Switch/activate the relevant task object first
        StopAllTasks();

        // Apply technique + workspace after the correct task is active
        ApplyTechnique();
        ApplyWorkspaceProfile();

        // Set grabber rotation policy per task (central, explicit)
        ApplyGrabberModeForCurrentTask();

        // Start the selected task
        switch (currentTask)
        {
            case TaskType.Placement:
                if (placementTask == null) { Debug.LogError("[StudyFlowController] placementTask missing."); return; }
                placementTask.StartBlock();
                break;

            case TaskType.Rotation:
                if (rotationTask == null) { Debug.LogError("[StudyFlowController] rotationTask missing."); return; }
                rotationTask.StartBlock();
                break;
        }

        Debug.Log($"[StudyFlowController] Started {currentTask} / {currentTechnique} / {currentHandLocation}");
    }

    public void RestartCurrent()
    {
        StartTask(currentTask);
    }

    public void NextConditionAndRestart()
    {
        // Cycle location first, then flip technique
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
        if (restart) RestartCurrent();
    }

    public void SetHandLocation(WorkspaceAnchorController.HandLocation loc, bool restart)
    {
        currentHandLocation = loc;
        if (restart) RestartCurrent();
        else ApplyWorkspaceProfile();
    }

    // ---------------- Apply workspace ----------------
    private void ApplyWorkspaceProfile()
    {
        var profile = GetProfile(currentTask, currentTechnique, currentHandLocation);
        if (profile == null)
        {
            Debug.LogWarning("[StudyFlowController] Workspace profile is null. No workspace movement will be applied.");
            return;
        }

        workspaceController.handLocation = currentHandLocation;
        workspaceController.ApplyProfile(profile);
    }

    private WorkspaceAnchorController.WorkspaceProfile GetProfile(TaskType task, Technique tech, WorkspaceAnchorController.HandLocation loc)
    {
        // Placement
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

        // Rotation
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
    }

    private void ForceReleaseIfPossible()
    {
        if (grabber != null)
        {
            grabber.ForceRelease();
        }
    }

    private void StopAllTasks()
    {
        // Disable both, then enable the current one (simple but effective)
        if (placementTask != null) placementTask.gameObject.SetActive(false);
        if (rotationTask != null) rotationTask.gameObject.SetActive(false);

        if (currentTask == TaskType.Placement && placementTask != null) placementTask.gameObject.SetActive(true);
        if (currentTask == TaskType.Rotation && rotationTask != null) rotationTask.gameObject.SetActive(true);
    }
}
