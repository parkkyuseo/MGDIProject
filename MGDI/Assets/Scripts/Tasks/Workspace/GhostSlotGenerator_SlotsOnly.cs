using System.Collections.Generic;
using UnityEngine;

public class GhostSlotGenerator_SlotsOnly : MonoBehaviour
{
    [Header("Roots")]
    public Transform toolsDynamicRoot;   // ContentRoot/Tools_Dynamic (또는 ToolsRuntime)
    public Transform slotsTargetsRoot;   // ContentRoot/Slots_Targets

    [Header("Layout (local to slotsTargetsRoot parent)")]
    public Vector3 slotsStartLocalPos = new Vector3(0.12f, 0.00f, 0.10f);
    public int columns = 3;
    public float spacingX = 0.10f;
    public float spacingZ = 0.10f;
    public float yLiftSlots = 0.002f;

    [Header("Ghost appearance")]
    [Range(0.05f, 1f)] public float ghostAlpha = 0.25f;
    public bool ghostUnlit = false;
    public Color ghostTint = Color.white;

    [Header("Regeneration")]
    public bool clearExistingSlots = true;

    [ContextMenu("Generate Ghost Slots Only")]
    public void GenerateSlotsOnly()
    {
        if (toolsDynamicRoot == null || slotsTargetsRoot == null)
        {
            Debug.LogError("[GhostSlotGenerator_SlotsOnly] Missing roots.");
            return;
        }

        if (clearExistingSlots)
            ClearChildren(slotsTargetsRoot);

        // toolsDynamicRoot 아래 ToolId들을 기준으로 슬롯 생성
        var toolIds = toolsDynamicRoot.GetComponentsInChildren<ToolId>(true);
        var list = new List<ToolId>();
        foreach (var t in toolIds)
        {
            if (t == null || string.IsNullOrEmpty(t.id)) continue;
            list.Add(t);
        }

        // 안정적인 순서(id)
        list.Sort((a, b) => string.CompareOrdinal(a.id, b.id));

        for (int i = 0; i < list.Count; i++)
        {
            ToolId srcId = list[i];
            if (srcId == null) continue;

            // "툴의 시각 프리팹을 그대로 복제"하려면 srcId.gameObject를 복제하면 되는데,
            // GrabRoot/Rigidbody/Collider가 섞이면 복잡해짐.
            // 가장 안전한 방식: "렌더러만 있는 복제본"을 만들기 위해 원본을 instantiate한 뒤, 상호작용 컴포넌트를 제거.
            var ghost = Instantiate(srcId.gameObject, slotsTargetsRoot);
            ghost.name = $"SlotGhost_{srcId.id}";

            // 위치 배치
            Vector3 slotPos = slotsStartLocalPos + GridOffset(i);
            slotPos.y += yLiftSlots;
            ghost.transform.localPosition = slotPos;
            ghost.transform.localRotation = Quaternion.identity;

            // ToolId 보장
            var ghostId = ghost.GetComponent<ToolId>();
            if (ghostId == null) ghostId = ghost.AddComponent<ToolId>();
            ghostId.id = srcId.id;

            MakeGhost(ghost);
        }

        Debug.Log($"[GhostSlotGenerator_SlotsOnly] Generated {list.Count} ghost slots.");
    }

    private Vector3 GridOffset(int idx)
    {
        int c = Mathf.Max(columns, 1);
        int col = idx % c;
        int row = idx / c;
        return new Vector3(col * spacingX, 0f, row * spacingZ);
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

    private void MakeGhost(GameObject go)
    {
        foreach (var col in go.GetComponentsInChildren<Collider>(true))
            col.enabled = false;

        foreach (var rb in go.GetComponentsInChildren<Rigidbody>(true))
        {
            if (Application.isPlaying) Destroy(rb);
            else DestroyImmediate(rb);
        }

        var renderers = go.GetComponentsInChildren<Renderer>(true);
        foreach (var r in renderers)
        {
            var mats = r.materials;
            for (int i = 0; i < mats.Length; i++)
                mats[i] = ConvertToGhostMaterial(mats[i]);
            r.materials = mats;
        }
    }

    private Material ConvertToGhostMaterial(Material src)
    {
        if (src == null) return null;

        if (ghostUnlit)
        {
            var shader = Shader.Find("Unlit/Color");
            if (shader == null) return src;

            var m = new Material(shader);
            var c = ghostTint; c.a = ghostAlpha;
            m.color = c;
            return m;
        }

        var std = Shader.Find("Standard");
        if (std == null) return src;

        var mat = new Material(std);

        if (src.HasProperty("_MainTex"))
            mat.SetTexture("_MainTex", src.GetTexture("_MainTex"));

        Color c2 = ghostTint; c2.a = ghostAlpha;
        if (src.HasProperty("_Color"))
        {
            var sc = src.GetColor("_Color");
            c2.r *= sc.r; c2.g *= sc.g; c2.b *= sc.b;
        }
        mat.SetColor("_Color", c2);

        // Standard Transparent
        mat.SetFloat("_Mode", 3f);
        mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        mat.SetInt("_ZWrite", 0);
        mat.DisableKeyword("_ALPHATEST_ON");
        mat.EnableKeyword("_ALPHABLEND_ON");
        mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;

        if (mat.HasProperty("_Metallic")) mat.SetFloat("_Metallic", 0f);
        if (mat.HasProperty("_Glossiness")) mat.SetFloat("_Glossiness", 0f);

        return mat;
    }
}
