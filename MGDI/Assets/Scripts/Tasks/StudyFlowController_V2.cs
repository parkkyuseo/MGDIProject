using System;
using UnityEngine;

public class StudyFlowController_V2 : MonoBehaviour
{
    [Header("Workflow")]
    [SerializeField] private WorkflowProgressionController workflow;

    [Header("Tasks")]
    [SerializeField] private ToolPlacementTaskManager placementTask;
    [SerializeField] private ToolRotationTaskManager rotationTask;
    [SerializeField] private ToolScalingTaskManager_OverlayCore scalingTask;

    [Header("Grabber (optional)")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Header("Grabber rotation policy per phase")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode placementGrabberMode = ProxyHandGrabber.HeldRotationMode.LockAtGrab;
    [SerializeField] private ProxyHandGrabber.HeldRotationMode rotationGrabberMode  = ProxyHandGrabber.HeldRotationMode.ExternalControl;
    [SerializeField] private ProxyHandGrabber.HeldRotationMode scalingGrabberMode   = ProxyHandGrabber.HeldRotationMode.LockAtGrab;

    private Action _onPlacementFinished;
    private Action _onRotationFinished;
    private Action _onScalingFinished;

    [SerializeField] private bool showDebugHUD = true;
    [SerializeField] private float hudUpdateHz = 10f; // 초당 10번
    private float _nextHudTime = 0f;

    void Update()
    {
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
        if (workflow == null)
            workflow = FindFirstObjectByType<WorkflowProgressionController>();

        if (workflow != null)
            workflow.OnStepChanged += OnWorkflowStepChanged;

        _onPlacementFinished = () => workflow?.Advance();
        _onRotationFinished  = () => workflow?.Advance();
        _onScalingFinished   = () => workflow?.Advance();

        HookTaskFinishedEvents();
    }

    private void OnDestroy()
    {
        if (workflow != null)
            workflow.OnStepChanged -= OnWorkflowStepChanged;

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

    private void OnWorkflowStepChanged(WorkflowProgressionController.Phase phase, int toolIndex, GameObject tool)
    {
        if (tool == null)
        {
            DisableAllTasks();
            return;
        }

        // ToolId from tool root (your current structure)
        var tid = tool.GetComponent<ToolId>();
        string id = (tid != null) ? tid.id : null;

        if (string.IsNullOrEmpty(id))
        {
            Debug.LogError($"[SFC_V2] ToolId missing on active tool '{tool.name}'.");
            DisableAllTasks();
            return;
        }

        // Always release before switching phase to avoid parent/rotation mode leakage
        if (grabber != null)
            grabber.ForceRelease();

        // Select exactly one task
        DisableAllTasks();

        switch (phase)
        {
            case WorkflowProgressionController.Phase.Placement:
                ApplyGrabberMode(placementGrabberMode);
                ApplyForcedIdToTasks(id);
                StartPlacement();
                break;

            case WorkflowProgressionController.Phase.Rotation:
                ApplyGrabberMode(rotationGrabberMode);
                ApplyForcedIdToTasks(id);
                StartRotation();
                break;

            case WorkflowProgressionController.Phase.Scaling:
                ApplyGrabberMode(scalingGrabberMode);
                ApplyForcedIdToTasks(id);
                StartScaling();
                break;
        }
    }

    private void ApplyForcedIdToTasks(string id)
    {
        // These methods exist after patches above
        if (placementTask != null) placementTask.SetForcedActiveId(id);
        if (rotationTask != null)  rotationTask.SetForcedActiveId(id);
        if (scalingTask != null)   scalingTask.SetForcedActiveId(id);
    }

    private void DisableAllTasks()
    {
        if (placementTask != null) placementTask.enabled = false;
        if (rotationTask != null)  rotationTask.enabled  = false;
        if (scalingTask != null)   scalingTask.enabled   = false;
    }

    private void StartPlacement()
    {
        if (placementTask == null) { Debug.LogError("[SFC_V2] placementTask is NULL"); return; }
        Debug.Log($"[SFC_V2] StartPlacement calling StartBlock() enabled={placementTask.enabled} activeInHierarchy={placementTask.gameObject.activeInHierarchy}");
        placementTask.enabled = true;
        placementTask.StartBlock();
    }

    private void StartRotation()
    {
        if (rotationTask == null) return;
        rotationTask.enabled = true;
        rotationTask.StartBlock();
    }

    private void StartScaling()
    {
        if (scalingTask == null) return;
        scalingTask.enabled = true;
        scalingTask.StartBlock();
    }

    private void ApplyGrabberMode(ProxyHandGrabber.HeldRotationMode mode)
    {
        if (grabber == null) return;
        grabber.SetHeldRotationMode(mode);
    }
}
