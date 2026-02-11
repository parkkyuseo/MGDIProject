using System;
using System.Collections.Generic;
using UnityEngine;

public class WorkflowProgressionController : MonoBehaviour
{
    public enum ProgressionMode { ToolByTool, PhaseByPhase }
    public enum Phase { Placement = 0, Rotation = 1, Scaling = 2 }

    [Header("Mode")]
    [SerializeField] private ProgressionMode mode = ProgressionMode.ToolByTool;

    [Header("Tools (ordered)")]
    [SerializeField] private List<GameObject> tools = new();

    [Header("Layers")]
    [SerializeField] private string activeLayerName = "Grabbable";
    [SerializeField] private string inactiveLayerName = "ToolInactive";

    [Header("Activation")]
    [SerializeField] private bool toggleColliders = true;

    [Header("Debug")]
    [SerializeField] private bool debugLog = true;
    [SerializeField] private bool autoStart = true;

    public int CurrentToolIndex { get; private set; } = -1;
    public Phase CurrentPhase { get; private set; } = Phase.Placement;
    public GameObject CurrentTool => (CurrentToolIndex >= 0 && CurrentToolIndex < tools.Count) ? tools[CurrentToolIndex] : null;

    public event Action<Phase, int, GameObject> OnStepChanged;
    public event Action OnAllCompleted;

    private int _activeLayer;
    private int _inactiveLayer;

    void Awake()
    {
        _activeLayer = LayerMask.NameToLayer(activeLayerName);
        _inactiveLayer = LayerMask.NameToLayer(inactiveLayerName);
    }

    void Start()
    {
        DeactivateAll();

        if (autoStart && tools.Count > 0)
        {
            CurrentPhase = Phase.Placement;
            CurrentToolIndex = 0;
            ApplyActiveTool();
            EmitStepChanged("Start");
        }
    }

    public void Advance()
    {
        if (tools == null || tools.Count == 0) return;
        if (CurrentToolIndex < 0) CurrentToolIndex = 0;

        if (mode == ProgressionMode.ToolByTool)
        {
            if (CurrentPhase != Phase.Scaling)
            {
                CurrentPhase = (Phase)((int)CurrentPhase + 1);
            }
            else
            {
                CurrentToolIndex++;
                CurrentPhase = Phase.Placement;
            }
        }
        else // PhaseByPhase
        {
            CurrentToolIndex++;
            if (CurrentToolIndex >= tools.Count)
            {
                CurrentToolIndex = 0;
                if (CurrentPhase != Phase.Scaling)
                    CurrentPhase = (Phase)((int)CurrentPhase + 1);
            }
        }

        // End condition
        if (mode == ProgressionMode.ToolByTool && CurrentToolIndex >= tools.Count)
        {
            DeactivateAll();
            if (debugLog) Debug.Log("[Workflow] Completed all tools.");
            OnAllCompleted?.Invoke();
            return;
        }

        ApplyActiveTool();
        EmitStepChanged("Advance");
    }

    public void SetMode(ProgressionMode newMode)
    {
        mode = newMode;
        ApplyActiveTool();
        EmitStepChanged("ModeChanged");
    }

    private void ApplyActiveTool()
    {
        DeactivateAll();

        int idx = Mathf.Clamp(CurrentToolIndex, 0, tools.Count - 1);
        SetToolActive(tools[idx], true);
    }

    private void DeactivateAll()
    {
        for (int i = 0; i < tools.Count; i++)
            SetToolActive(tools[i], false);
    }

    private void SetToolActive(GameObject tool, bool active)
    {
        if (tool == null) return;

        int layer = active ? _activeLayer : _inactiveLayer;
        SetLayerRecursively(tool, layer);

        if (toggleColliders)
        {
            Collider[] cols = tool.GetComponentsInChildren<Collider>(true);
            for (int i = 0; i < cols.Length; i++)
                cols[i].enabled = active;
        }
    }

    private void SetLayerRecursively(GameObject obj, int layer)
    {
        obj.layer = layer;
        foreach (Transform child in obj.transform)
            SetLayerRecursively(child.gameObject, layer);
    }

    private void EmitStepChanged(string prefix)
    {
        if (debugLog)
        {
            string toolName = CurrentTool != null ? CurrentTool.name : "(none)";
            Debug.Log($"[Workflow] {prefix} | Mode={mode} | Phase={CurrentPhase} | ToolIndex={CurrentToolIndex} | Tool={toolName}");
        }

        OnStepChanged?.Invoke(CurrentPhase, CurrentToolIndex, CurrentTool);
    }

    public event Action OnAllCompleted;

    public void RestartFromBeginning()
    {
        // reset state and re-apply first step
        CurrentToolIndex = 0;
        CurrentPhase = Phase.Placement;
        ApplyActiveTool();
        EmitStepChanged("RestartFromBeginning");
    }
}
