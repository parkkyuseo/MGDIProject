using System.Collections.Generic;
using UnityEngine;

public class GhostHighlightController : MonoBehaviour
{
    [Header("Roots")]
    public Transform slotsTargetsRoot;      // ContentRoot/Slots_Targets
    public ProxyHandGrabber grabber;        // 프록시 핸드 Grabber

    [Header("Alpha")]
    [Range(0.01f, 1f)] public float normalAlpha = 0.25f;
    [Range(0.01f, 1f)] public float highlightAlpha = 0.60f;

    private readonly Dictionary<string, List<Renderer>> ghostMap = new();
    private string currentId;

    void Awake()
    {
        RebuildGhostMap();

        if (grabber != null)
        {
            grabber.OnGrabbed += HandleGrabbed;
            grabber.OnReleased += HandleReleased;
        }

        SetAllGhostAlpha(normalAlpha);
    }

    void OnDestroy()
    {
        if (grabber != null)
        {
            grabber.OnGrabbed -= HandleGrabbed;
            grabber.OnReleased -= HandleReleased;
        }
    }

    [ContextMenu("Rebuild Ghost Map")]
    public void RebuildGhostMap()
    {
        ghostMap.Clear();
        if (slotsTargetsRoot == null) return;

        // Slots_Targets 아래의 모든 ToolId(ghost들)를 모아서 id->renderers 매핑
        var ids = slotsTargetsRoot.GetComponentsInChildren<ToolId>(true);
        foreach (var tid in ids)
        {
            if (tid == null || string.IsNullOrEmpty(tid.id)) continue;

            var rs = tid.GetComponentsInChildren<Renderer>(true);
            if (rs == null || rs.Length == 0) continue;

            if (!ghostMap.TryGetValue(tid.id, out var list))
            {
                list = new List<Renderer>();
                ghostMap.Add(tid.id, list);
            }

            for (int i = 0; i < rs.Length; i++) list.Add(rs[i]);
        }
    }

    private void HandleGrabbed(Rigidbody heldBody)
    {
        if (heldBody == null) return;

        // 1) 자식에서 찾기
        var tid = heldBody.GetComponentInChildren<ToolId>(true);

        // 2) 안 나오면 부모에서 찾기 (이게 핵심)
        if (tid == null)
            tid = heldBody.GetComponentInParent<ToolId>();

        if (tid == null || string.IsNullOrEmpty(tid.id))
        {
            Debug.Log("[GhostHL] ToolId not found on held body: " + heldBody.name);
            return;
        }

        Debug.Log("[GhostHL] grabbed id=" + tid.id);
        HighlightOnly(tid.id);
    }

    private void HandleReleased(Rigidbody releasedBody)
    {
        ClearHighlight();
    }

    private void HighlightOnly(string id)
    {
        if (currentId == id) return;

        SetAllGhostAlpha(normalAlpha);
        currentId = id;

        if (ghostMap.TryGetValue(id, out var rs))
            SetRenderersAlpha(rs, highlightAlpha);
    }

    private void ClearHighlight()
    {
        currentId = null;
        SetAllGhostAlpha(normalAlpha);
    }

    private void SetAllGhostAlpha(float a)
    {
        foreach (var kv in ghostMap)
            SetRenderersAlpha(kv.Value, a);
    }

    private void SetRenderersAlpha(List<Renderer> renderers, float a)
    {
        if (renderers == null) return;

        for (int i = 0; i < renderers.Count; i++)
        {
            var r = renderers[i];
            if (r == null) continue;

            // 중요: sharedMaterial 말고 materials (인스턴스)만 만진다
            var mats = r.materials;
            for (int m = 0; m < mats.Length; m++)
            {
                var mat = mats[m];
                if (mat == null) continue;
                if (!mat.HasProperty("_Color")) continue;

                var c = mat.color;
                c.a = a;
                mat.color = c;
            }
            r.materials = mats;
        }
    }
}
