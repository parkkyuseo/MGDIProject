using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Windows.Speech;

public class StudyFlowController : MonoBehaviour
{
    [Header("References")]
    public WorkspaceAnchorController workspaceController;

    [Tooltip("Placement task manager (current).")]
    public LegoPlacementTaskManager placementTask;

    [Header("Flow Settings")]
    public bool autoApplyWorkspaceOnStart = true;

    [Header("Voice Commands")]
    public bool enableVoice = true;
    public string cmdStart = "start";
    public string cmdNext = "next";
    public string cmdNear = "near";
    public string cmdSideLeft = "side left";
    public string cmdSideRight = "side right";
    public string cmdRestart = "restart";

    private KeywordRecognizer recognizer;
    private Dictionary<string, System.Action> actions;

    private void Start()
    {
        if (!enableVoice) return;

        actions = new Dictionary<string, System.Action>
        {
            { cmdStart, OnStart },
            { cmdRestart, OnStart },

            { cmdNext, OnNextCondition },

            // direct set
            { cmdNear, () => SetHandLocation(WorkspaceAnchorController.HandLocation.NearHead) },
            { cmdSideLeft, () => SetHandLocation(WorkspaceAnchorController.HandLocation.SideOfBodyLeft) },
            { cmdSideRight, () => SetHandLocation(WorkspaceAnchorController.HandLocation.SideOfBodyRight) }
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

    private void OnStart()
    {
        if (workspaceController == null || placementTask == null)
        {
            Debug.LogError("[StudyFlowController] Missing references.");
            return;
        }

        if (autoApplyWorkspaceOnStart)
            workspaceController.ApplyWorkspace();

        placementTask.StartBlock();
        Debug.Log($"[StudyFlowController] Started placement with handLocation={workspaceController.handLocation}");
    }

    private void OnNextCondition()
    {
        if (workspaceController == null) return;

        // cycle: Near -> Left -> Right -> Near ...
        var loc = workspaceController.handLocation;
        if (loc == WorkspaceAnchorController.HandLocation.NearHead)
            loc = WorkspaceAnchorController.HandLocation.SideOfBodyLeft;
        else if (loc == WorkspaceAnchorController.HandLocation.SideOfBodyLeft)
            loc = WorkspaceAnchorController.HandLocation.SideOfBodyRight;
        else
            loc = WorkspaceAnchorController.HandLocation.NearHead;

        workspaceController.handLocation = loc;
        workspaceController.ApplyWorkspace();

        if (placementTask != null)
            placementTask.StartBlock();

        Debug.Log($"[StudyFlowController] Next condition handLocation={loc}");
    }

    private void SetHandLocation(WorkspaceAnchorController.HandLocation loc)
    {
        if (workspaceController == null) return;

        workspaceController.handLocation = loc;
        workspaceController.ApplyWorkspace();

        Debug.Log($"[StudyFlowController] Set handLocation={loc}");
    }
}
