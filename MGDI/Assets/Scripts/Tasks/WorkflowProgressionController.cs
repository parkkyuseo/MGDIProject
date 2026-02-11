using System.Collections.Generic;
using UnityEngine;

public class WorkflowProgressionController : MonoBehaviour
{
    public enum ProgressionMode
    {
        ToolByTool,   // A: Tool1 P->R->S, Tool2 P->R->S ...
        PhaseByPhase  // B: All tools P, then all tools R, then all tools S
    }

    public enum Phase
    {
        Placement = 0,
        Rotation = 1,
        Scaling = 2
    }

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

    // Current state
    public int CurrentToolIndex { get; private set; } = -1;
    public Phase CurrentPhase { get; private set; } = Phase.Placement;

    private int _activeLayer;
    private int _inactiveLayer;

    void Awake()
    {
        _activeLayer = LayerMask.NameToLayer(activeLayerName);
        _inactiveLayer = LayerMask.NameToLayer(inactiveLayerName);
    }

    void Start()
    {
        // Start with everything inactive
        for (int i = 0; i < tools.Count; i++)
            SetToolActive(tools[i], false);

        if (autoStart)
        {
            // Initialize to first step depending on mode
            CurrentPhase = Phase.Placement;
            CurrentToolIndex = 0;
            ApplyActiveTool();
            LogState("Start");
        }
    }

    /// <summary>
    /// Call this when the current step is completed (e.g., placement success, rotation success, scaling success).
    /// </summary>
    public void Advance()
    {
        if (tools == null || tools.Count == 0) return;
        if (CurrentToolIndex < 0) CurrentToolIndex = 0;

        if (mode == ProgressionMode.ToolByTool)
        {
            // Tool1 P->R->S then next tool
            if (CurrentPhase != Phase.Scaling)
            {
                CurrentPhase = (Phase)((int)CurrentPhase + 1);
            }
            else
            {
                // move to next tool, reset phase
                CurrentToolIndex++;
                CurrentPhase = Phase.Placement;
            }
        }
        else // PhaseByPhase
        {
            // All tools in Placement, then all in Rotation, then all in Scaling
            CurrentToolIndex++;
            if (CurrentToolIndex >= tools.Count)
            {
                CurrentToolIndex = 0;
                if (CurrentPhase != Phase.Scaling)
                    CurrentPhase = (Phase)((int)CurrentPhase + 1);
                else
                    CurrentPhase = Phase.Scaling; // end reached (could set a Completed flag)
            }
        }

        // End condition
        if (mode == ProgressionMode.ToolByTool && CurrentToolIndex >= tools.Count)
        {
            // Completed all tools
            DeactivateAll();
            LogState("Completed");
            return;
        }

        ApplyActiveTool();
        LogState("Advance");
    }

    public void SetMode(ProgressionMode newMode)
    {
        mode = newMode;

        // Re-apply current active tool after mode change
        if (CurrentToolIndex < 0) CurrentToolIndex = 0;
        ApplyActiveTool();
        LogState("ModeChanged");
    }

    private void ApplyActiveTool()
    {
        // Deactivate all, then activate only current
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
            var cols = tool.GetComponentsInChildren<Collider>(true);
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

    private void LogState(string prefix)
    {
        if (!debugLog) return;
        string toolName = (CurrentToolIndex >= 0 && CurrentToolIndex < tools.Count && tools[CurrentToolIndex] != null)
            ? tools[CurrentToolIndex].name
            : "(none)";

        Debug.Log($"[Workflow] {prefix} | Mode={mode} | Phase={CurrentPhase} | ToolIndex={CurrentToolIndex} | Tool={toolName}");
    }
}
