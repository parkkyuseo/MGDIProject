using UnityEngine;

public class TargetPivotAligner : MonoBehaviour
{
    [Header("Assign these")]
    [Tooltip("Empty root pivot to be aligned (TargetSlotRoot).")]
    public Transform targetSlotRoot;

    [Tooltip("Visual parent under the root (TargetSlotVisual).")]
    public Transform targetSlotVisual;

    [Tooltip("Renderer inside the visual (the transparent lego). If null, auto-finds under targetSlotVisual.")]
    public Renderer targetRenderer;

    [ContextMenu("Align TargetSlotRoot To Visual Center (Keep Visual World Pose)")]
    public void AlignRootToVisualCenter()
    {
        if (targetSlotRoot == null || targetSlotVisual == null)
        {
            Debug.LogError("[TargetPivotAligner] Missing targetSlotRoot or targetSlotVisual.");
            return;
        }

        if (targetRenderer == null)
            targetRenderer = targetSlotVisual.GetComponentInChildren<Renderer>(true);

        if (targetRenderer == null)
        {
            Debug.LogError("[TargetPivotAligner] No Renderer found under targetSlotVisual.");
            return;
        }

        // Current world center of the visible target mesh
        Vector3 visualCenterWorld = targetRenderer.bounds.center;

        // Keep visual world pose while moving the root pivot to the visual center
        Vector3 rootPosBefore = targetSlotRoot.position;
        Vector3 visualPosBefore = targetSlotVisual.position;

        Vector3 delta = visualCenterWorld - rootPosBefore;

        // Move root to the visual center
        targetSlotRoot.position = rootPosBefore + delta;

        // Move visual back so its world position doesn't change (keep appearance stable)
        targetSlotVisual.position = visualPosBefore - delta;

        Debug.Log($"[TargetPivotAligner] Aligned root by delta={delta}.");
    }
}
