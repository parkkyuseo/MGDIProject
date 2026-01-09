using UnityEngine;

public class TargetPivotAligner : MonoBehaviour
{
    [Header("Assign these")]
    public Transform targetSlotRoot;    // TargetSlotRoot (Empty)
    public Transform targetSlotVisual;  // TargetSlotVisual (Empty)
    public Renderer targetRenderer;     // Renderer under the visual (transparent lego)

    private Renderer ResolveRenderer()
    {
        if (targetRenderer != null) return targetRenderer;
        if (targetSlotVisual == null) return null;
        return targetSlotVisual.GetComponentInChildren<Renderer>(true);
    }

    [ContextMenu("Align Root To Visual Center (MOVE ROOT ONLY)")]
    public void AlignRoot_MoveRootOnly()
    {
        if (targetSlotRoot == null)
        {
            Debug.LogError("[TargetPivotAligner] targetSlotRoot is null.");
            return;
        }

        var r = ResolveRenderer();
        if (r == null)
        {
            Debug.LogError("[TargetPivotAligner] No Renderer found under targetSlotVisual.");
            return;
        }

        Vector3 center = r.bounds.center;
        targetSlotRoot.position = center;

        Debug.Log("[TargetPivotAligner] Root moved to visual center (move root only).");
    }

    [ContextMenu("Align Root To Visual Center (KEEP VISUAL WORLD POSE)")]
    public void AlignRoot_KeepVisualWorldPose()
    {
        if (targetSlotRoot == null || targetSlotVisual == null)
        {
            Debug.LogError("[TargetPivotAligner] Missing targetSlotRoot or targetSlotVisual.");
            return;
        }

        var r = ResolveRenderer();
        if (r == null)
        {
            Debug.LogError("[TargetPivotAligner] No Renderer found under targetSlotVisual.");
            return;
        }

        Vector3 visualCenterWorld = r.bounds.center;

        Vector3 rootPosBefore = targetSlotRoot.position;
        Vector3 visualPosBefore = targetSlotVisual.position;

        Vector3 delta = visualCenterWorld - rootPosBefore;

        targetSlotRoot.position = rootPosBefore + delta;
        targetSlotVisual.position = visualPosBefore - delta;

        Debug.Log($"[TargetPivotAligner] Root aligned while keeping visual pose. delta={delta}");
    }
}
