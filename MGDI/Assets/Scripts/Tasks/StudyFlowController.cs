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

    [Tooltip("Placement task manager.")]
    public LegoPlacementTaskManager placementTask;

    [Tooltip("Rotation task manager.")]
    public LegoRotationTaskManager rotationTask;

    [Header("Technique Controllers (optional)")]
    [Tooltip("Enable this GameObject for Macro technique (e.g., macro grab controller).")]
    public GameObject macroControllerRoot;

    [Tooltip("Enable this GameObject for Micro technique (to be added later).")]
    public GameObject microControllerRoot;

    [Header("Current State")]
    public TaskType currentTask = TaskType.Placement;
    public Technique currentTechnique = Technique.Macro;

    [Header("Flow Settings")]
    public bool autoApplyWorkspaceOnStart = true;

    [Header("Voice Commands")]
    public bool enableVoice = true;

    public string cmdStartPlacement = "start placement";
    public string cmdStartRotation  = "start rotation";
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

        actions = new Dictionary<string, System.Action>
        {
            { cmdStartPlacement, () => StartTask(TaskType.Placement) },
            { cmdStartRotation,  () => StartTask(TaskType.Rotation) },
            { cmdRestart,        RestartCurrent },

            { cmdNext,           NextConditionAndRestart },

            { cmdMacro,          () => SetTechnique(Technique.Macro, restart:true) },
            { cmdMicro,          () => SetTechnique(Technique.Micro, restart:true) },

            { cmdNear,           () => SetHandLocation(WorkspaceAnchorController.HandLocation.NearHead, restart:false) },
            { cmdSideLeft,       () => SetHandLocation(WorkspaceAnchorController.HandLocation.SideOfBodyLeft, restart:false) },
            { cmdSideRight,      () => SetHandLocation(WorkspaceAnchorController.HandLocation.SideOfBodyRight, restart:false) },
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

    // ---------------- Core actions ----------------
    public void StartTask(TaskType t)
    {
        currentTask = t;

        if (workspaceController == null)
        {
            Debug.LogError("[StudyFlowController] workspaceController is missing.");
            return;
        }

        ApplyTechnique();
        if (autoApplyWorkspaceOnStart)
            workspaceController.ApplyWorkspace();

        // Stop other tasks to avoid both running
        StopAllTasks();

        // Start selected task
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

        Debug.Log($"[StudyFlowController] Started {currentTask} / {currentTechnique} / {workspaceController.handLocation}");
    }

    public void RestartCurrent()
    {
        StartTask(currentTask);
    }

    // 2×2×(location 3-state) 중에서: 우선 Technique×Location을 순환 (Near, SideLeft, SideRight)
    // Macro/Micro를 먼저 바꾸고, 그 안에서 Near/Left/Right를 돌리는 식으로 설계
    public void NextConditionAndRestart()
    {
        if (workspaceController == null) return;

        // Cycle location first
        var loc = workspaceController.handLocation;
        if (loc == WorkspaceAnchorController.HandLocation.NearHead)
            loc = WorkspaceAnchorController.HandLocation.SideOfBodyLeft;
        else if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft)
            loc = WorkspaceAnchorController.HandLocation.SideOfBodyRight;
        else
        {
            // if we completed location cycle, flip technique
            loc = WorkspaceAnchorController.HandLocation.NearHead;
            currentTechnique = (currentTechnique == Technique.Macro) ? Technique.Micro : Technique.Macro;
        }

        workspaceController.handLocation = loc;

        RestartCurrent();
    }

    public void SetTechnique(Technique tech, bool restart)
    {
        currentTechnique = tech;
        ApplyTechnique();

        if (restart)
            RestartCurrent();
        else
            Debug.Log($"[StudyFlowController] Technique set to {currentTechnique}");
    }

    public void SetHandLocation(WorkspaceAnchorController.HandLocation loc, bool restart)
    {
        if (workspaceController == null) return;

        workspaceController.handLocation = loc;
        workspaceController.ApplyWorkspace();

        if (restart)
            RestartCurrent();
        else
            Debug.Log($"[StudyFlowController] HandLocation set to {loc}");
    }

    // ---------------- Helpers ----------------
    private void ApplyTechnique()
    {
        // Simple enable/disable toggles. Later, connect micro controller here.
        if (macroControllerRoot != null)
            macroControllerRoot.SetActive(currentTechnique == Technique.Macro);

        if (microControllerRoot != null)
            microControllerRoot.SetActive(currentTechnique == Technique.Micro);
    }

    private void StopAllTasks()
    {
        // We don't have a Stop API, so the simplest safe approach is:
        // disable the GameObject containing the manager, then re-enable when starting.
        // If you prefer, we can add StopBlock() to each manager later.

        if (placementTask != null && placementTask.gameObject.activeSelf)
            placementTask.gameObject.SetActive(false);

        if (rotationTask != null && rotationTask.gameObject.activeSelf)
            rotationTask.gameObject.SetActive(false);

        // Re-enable the one we actually start in StartTask()
        if (currentTask == TaskType.Placement && placementTask != null)
            placementTask.gameObject.SetActive(true);

        if (currentTask == TaskType.Rotation && rotationTask != null)
            rotationTask.gameObject.SetActive(true);
    }
}
