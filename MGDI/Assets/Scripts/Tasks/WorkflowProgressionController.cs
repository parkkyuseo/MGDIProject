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

    [Header("Demo / Block control")]
    [SerializeField] private int toolsPerBlock = 1; // 0 = use all tools

    [Header("Debug")]
    [SerializeField] private bool debugLog = true;
    [SerializeField] private bool autoStart = true;

    public int CurrentToolIndex { get; private set; } = -1;
    public Phase CurrentPhase { get; private set; } = Phase.Placement;
    public GameObject CurrentTool => (CurrentToolIndex >= 0 && CurrentToolIndex < tools.Count) ? tools[CurrentToolIndex] : null;
    public int ToolCount => tools != null ? tools.Count : 0;

    public event Action<Phase, int, GameObject> OnStepChanged;
    public event Action OnAllCompleted;

    private int _activeLayer;
    private int _inactiveLayer;
    private int _skipCompensation = 0;

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

        int blockToolCount = GetBlockToolCount();

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

            if (CurrentToolIndex >= blockToolCount)
            {
                CompleteBlock(blockToolCount);
                return;
            }
        }
        else // PhaseByPhase
        {
            CurrentToolIndex++;
            if (CurrentToolIndex >= blockToolCount)
            {
                CurrentToolIndex = 0;
                if (CurrentPhase != Phase.Scaling)
                {
                    CurrentPhase = (Phase)((int)CurrentPhase + 1);
                }
                else
                {
                    CompleteBlock(blockToolCount);
                    return;
                }
            }
        }

        ApplyActiveTool();
        EmitStepChanged("Advance");
    }

    private void CompleteBlock(int blockToolCount)
    {
        CurrentToolIndex = 0;
        CurrentPhase = Phase.Placement;
        _skipCompensation = 0;

        DeactivateAll();
        if (debugLog)
            Debug.Log($"[Workflow] Block completed (mode={mode}, toolsPerBlock={toolsPerBlock}, blockToolCount={blockToolCount}).");
        OnAllCompleted?.Invoke();
    }

    public void SetMode(ProgressionMode newMode)
    {
        mode = newMode;
        ApplyActiveTool();
        EmitStepChanged("ModeChanged");
    }

    public string GetToolIdAtIndex(int index)
    {
        if (tools == null || index < 0 || index >= tools.Count || tools[index] == null)
            return null;

        var tid = tools[index].GetComponent<ToolId>();
        return tid != null ? NormalizeToolId(tid.id) : null;
    }

    public bool ToolAtIndexMatchesId(int index, string id)
    {
        string normalized = NormalizeToolId(id);
        if (string.IsNullOrEmpty(normalized))
            return false;

        return string.Equals(GetToolIdAtIndex(index), normalized, StringComparison.OrdinalIgnoreCase);
    }

    public bool CurrentToolMatchesId(string id)
    {
        return ToolAtIndexMatchesId(CurrentToolIndex, id);
    }

    public bool SkipCurrentToolIfId(string id, bool emitStepChanged = true)
    {
        string normalized = NormalizeToolId(id);
        if (string.IsNullOrEmpty(normalized) || !CurrentToolMatchesId(normalized))
            return false;

        int nextIndex = CurrentToolIndex + 1;
        if (nextIndex >= tools.Count)
        {
            Debug.LogWarning($"[Workflow] Cannot skip current tool '{normalized}' because no later tool exists.");
            return false;
        }

        CurrentToolIndex = nextIndex;
        if (toolsPerBlock > 0)
            _skipCompensation = Mathf.Max(_skipCompensation, 1);

        ApplyActiveTool();

        if (emitStepChanged)
            EmitStepChanged("SkipCurrentToolIfId");
        else if (debugLog)
            Debug.Log($"[Workflow] SkipCurrentToolIfId | Phase={CurrentPhase} | ToolIndex={CurrentToolIndex} | Tool={CurrentTool?.name}");

        return true;
    }

    public bool ActivateToolByIdWithoutChangingStep(string id)
    {
        string normalized = NormalizeToolId(id);
        if (string.IsNullOrEmpty(normalized) || tools == null)
            return false;

        for (int i = 0; i < tools.Count; i++)
        {
            if (!ToolAtIndexMatchesId(i, normalized))
                continue;

            DeactivateAll();
            SetToolActive(tools[i], true);

            if (debugLog)
                Debug.Log($"[Workflow] Temporarily activated tool '{normalized}' without changing workflow step.");

            return true;
        }

        Debug.LogWarning($"[Workflow] Could not temporarily activate tool '{normalized}' because it is not in the workflow list.");
        return false;
    }

    public void ReapplyCurrentActiveTool()
    {
        if (tools == null || tools.Count == 0 || CurrentToolIndex < 0)
            return;

        ApplyActiveTool();
    }

    private int GetBlockToolCount()
    {
        if (tools == null || tools.Count == 0)
            return 0;

        if (toolsPerBlock <= 0)
            return tools.Count;

        int compensated = toolsPerBlock + Mathf.Max(0, _skipCompensation);
        return Mathf.Clamp(compensated, 1, tools.Count);
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

    public void RestartFromBeginning()
    {
        // reset state and re-apply first step
        CurrentToolIndex = 0;
        CurrentPhase = Phase.Placement;
        _skipCompensation = 0;
        ApplyActiveTool();
        EmitStepChanged("RestartFromBeginning");
    }

    private static string NormalizeToolId(string id)
    {
        return string.IsNullOrWhiteSpace(id) ? null : id.Trim();
    }
}
