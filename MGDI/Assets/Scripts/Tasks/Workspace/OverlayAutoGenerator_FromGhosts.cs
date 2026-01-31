using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Creates scaling overlays automatically from existing slot ghosts.
/// - Input: Slots_Targets contains SlotGhost_* objects with ToolId and renderers.
/// - Output:
///   ContentRoot/Overlays_Current  (user-controlled overlay; slightly stronger alpha)
///   ContentRoot/Overlays_Target   (goal overlay; lighter alpha)
/// 
/// This does NOT modify the source ghosts. It instantiates copies.
/// </summary>
public class OverlayAutoGenerator_FromGhosts : MonoBehaviour
{
    [Header("Source")]
    [Tooltip("Root that contains your placement target ghosts (Slots_Targets).")]
    public Transform slotsTargetsRoot;

    [Header("Destination")]
    [Tooltip("Parent under which Overlays_Current/Overlays_Target will be created (usually ContentRoot).")]
    public Transform contentRoot;

    [Tooltip("Name for current overlay root.")]
    public string overlaysCurrentName = "Overlays_Current";

    [Tooltip("Name for target overlay root.")]
    public string overlaysTargetName = "Overlays_Target";

    [Header("Filtering")]
    [Tooltip("If true, only clone objects whose name starts with this prefix (e.g., SlotGhost_). Leave empty to clone all ToolId objects.")]
    public string namePrefixFilter = "SlotGhost_";

    [Header("Appearance")]
    [Range(0.01f, 1f)] public float currentAlpha = 0.45f;
    [Range(0.01f, 1f)] public float targetAlpha = 0.20f;

    [Tooltip("If true, disables all colliders on generated overlays.")]
    public bool disableColliders = true;

    [Tooltip("If true, removes all rigidbodies on generated overlays.")]
    public bool removeRigidbodies = true;

    [Header("Regenerate")]
    [Tooltip("If true, deletes existing Overlays_Current/Overlays_Target children before generating.")]
    public bool clearExisting = true;

    [ContextMenu("Generate Overlays From Ghosts")]
    public void Generate()
    {
        if (slotsTargetsRoot == null)
        {
            Debug.LogError("[OverlayAutoGen] slotsTargetsRoot missing.");
            return;
        }

        if (contentRoot == null)
        {
            Debug.LogError("[OverlayAutoGen] contentRoot missing.");
            return;
        }

        Transform curRoot = EnsureChild(contentRoot, overlaysCurrentName);
        Transform tgtRoot = EnsureChild(contentRoot, overlaysTargetName);

        if (clearExisting)
        {
            ClearChildren(curRoot);
            ClearChildren(tgtRoot);
        }

        // Collect unique ToolId transforms under slotsTargetsRoot.
        // We clone the ToolId transform's root object (the GameObject that carries ToolId),
        // not necessarily the renderer root, because ToolId is how tasks match them.
        var toolIds = slotsTargetsRoot.GetComponentsInChildren<ToolId>(true);

        int created = 0;
        var seenIds = new HashSet<string>();

        foreach (var tid in toolIds)
        {
            if (tid == null) continue;
            if (string.IsNullOrEmpty(tid.id)) continue;

            // Optional name prefix filter (commonly SlotGhost_*)
            if (!string.IsNullOrEmpty(namePrefixFilter))
            {
                if (tid.gameObject.name == null || !tid.gameObject.name.StartsWith(namePrefixFilter))
                    continue;
            }

            // Avoid cloning multiple ToolId components for the same id (if duplicates exist)
            if (!seenIds.Add(tid.id))
                continue;

            // Create current overlay (user-controlled)
            GameObject cur = Instantiate(tid.gameObject, curRoot);
            cur.name = $"OverlayCurrent_{tid.id}";
            PrepareOverlay(cur, currentAlpha);

            // Create target overlay (goal)
            GameObject tgt = Instantiate(tid.gameObject, tgtRoot);
            tgt.name = $"OverlayTarget_{tid.id}";
            PrepareOverlay(tgt, targetAlpha);

            created++;
        }

        Debug.Log($"[OverlayAutoGen] Generated overlays for {created} tool ids.");
    }

    private void PrepareOverlay(GameObject go, float alpha)
    {
        if (go == null) return;

        if (disableColliders)
        {
            foreach (var c in go.GetComponentsInChildren<Collider>(true))
                c.enabled = false;
        }

        if (removeRigidbodies)
        {
            foreach (var rb in go.GetComponentsInChildren<Rigidbody>(true))
            {
                if (Application.isPlaying) Destroy(rb);
                else DestroyImmediate(rb);
            }
        }

        // Make materials translucent by adjusting _Color alpha when available.
        // This assumes your ghost materials were already set up for transparency.
        var renderers = go.GetComponentsInChildren<Renderer>(true);
        foreach (var r in renderers)
        {
            if (r == null) continue;

            // Use instance materials so we don't touch shared assets.
            var mats = r.materials;
            for (int i = 0; i < mats.Length; i++)
            {
                var m = mats[i];
                if (m == null) continue;

                if (m.HasProperty("_Color"))
                {
                    var c = m.color;
                    c.a = alpha;
                    m.color = c;
                }
            }
            r.materials = mats;
        }
    }

    private Transform EnsureChild(Transform parent, string name)
    {
        var t = parent.Find(name);
        if (t != null) return t;

        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = Vector3.zero;
        go.transform.localRotation = Quaternion.identity;
        go.transform.localScale = Vector3.one;
        return go.transform;
    }

    private void ClearChildren(Transform root)
    {
        for (int i = root.childCount - 1; i >= 0; i--)
        {
            var ch = root.GetChild(i);
            if (Application.isPlaying) Destroy(ch.gameObject);
            else DestroyImmediate(ch.gameObject);
        }
    }
}
