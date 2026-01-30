using System.Collections.Generic;
using UnityEngine;

public class ToolTargetRegistry : MonoBehaviour
{
    [Header("Roots")]
    public Transform toolsDynamicRoot;  // ContentRoot/Tools_Dynamic
    public Transform slotsTargetsRoot;  // ContentRoot/Slots_Targets

    public Dictionary<string, Transform> ToolById { get; private set; } = new();
    public Dictionary<string, Transform> TargetById { get; private set; } = new();

    [ContextMenu("Rebuild Registry")]
    public void Rebuild()
    {
        ToolById.Clear();
        TargetById.Clear();

        if (toolsDynamicRoot != null)
        {
            foreach (var tid in toolsDynamicRoot.GetComponentsInChildren<ToolId>(true))
            {
                if (tid == null || string.IsNullOrEmpty(tid.id)) continue;
                ToolById[tid.id] = tid.transform; // ToolId가 붙은 transform을 기준으로
            }
        }

        if (slotsTargetsRoot != null)
        {
            foreach (var tid in slotsTargetsRoot.GetComponentsInChildren<ToolId>(true))
            {
                if (tid == null || string.IsNullOrEmpty(tid.id)) continue;
                TargetById[tid.id] = tid.transform; // ghost의 ToolId transform
            }
        }

        Debug.Log($"[ToolTargetRegistry] tools={ToolById.Count}, targets={TargetById.Count}");
    }

    void Awake()
    {
        Rebuild();
    }
}
