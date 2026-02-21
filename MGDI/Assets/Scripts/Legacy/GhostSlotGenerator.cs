using System.Collections.Generic;
using UnityEngine;

public class GhostSlotGenerator : MonoBehaviour
{
    [Header("ContentRoot refs")]
    public Transform toolsDynamicRoot;   // ContentRoot/Tools_Dynamic
    public Transform slotsTargetsRoot;   // ContentRoot/Slots_Targets

    [Header("Tool Prefabs")]
    public List<GameObject> toolPrefabs = new List<GameObject>();

    [Header("Layout (meters, local to ContentRoot)")]
    public Vector3 toolsStartLocalPos = new Vector3(-0.12f, 0.00f, 0.10f);
    public Vector3 slotsStartLocalPos = new Vector3( 0.12f, 0.00f, 0.10f);

    public int columns = 3;
    public float spacingX = 0.10f;
    public float spacingZ = 0.10f;

    [Header("Placement tweaks")]
    public float yLiftTools = 0.00f;     // 실제 도구를 살짝 띄우고 싶으면
    public float yLiftSlots = 0.002f;    // 슬롯은 z-fighting 방지용으로 살짝 띄움
    public bool faceForward = true;      // +Z 방향을 "앞"으로 두고 정렬

    [Header("Ghost appearance (Built-in Standard shader)")]
    [Range(0.05f, 1f)] public float ghostAlpha = 0.25f;
    public bool ghostUnlit = false;      // true면 Unlit/Color로 단색 유령
    public Color ghostTint = Color.white;

    [Header("Naming")]
    public string toolsPrefix = "Tool_";
    public string slotPrefix = "SlotGhost_";

    [Header("Regeneration")]
    public bool clearExistingBeforeGenerate = true;

    [ContextMenu("Generate Tools + Ghost Slots")]
    public void Generate()
    {
        if (toolsDynamicRoot == null || slotsTargetsRoot == null)
        {
            Debug.LogError("[GhostSlotGenerator] toolsDynamicRoot / slotsTargetsRoot not assigned.");
            return;
        }

        if (toolPrefabs == null || toolPrefabs.Count == 0)
        {
            Debug.LogWarning("[GhostSlotGenerator] No tool prefabs assigned.");
            return;
        }

        if (clearExistingBeforeGenerate)
        {
            ClearChildren(toolsDynamicRoot);
            ClearChildren(slotsTargetsRoot);
        }

        Quaternion rot = faceForward ? Quaternion.identity : Quaternion.Euler(0, 180f, 0);

        for (int i = 0; i < toolPrefabs.Count; i++)
        {
            var prefab = toolPrefabs[i];
            if (prefab == null) continue;

            // 1) 실제 도구 배치
            Vector3 toolPos = toolsStartLocalPos + GridOffset(i);
            toolPos.y += yLiftTools;

            var tool = Instantiate(prefab, toolsDynamicRoot);
            tool.name = toolsPrefix + prefab.name;
            tool.transform.localPosition = toolPos;
            tool.transform.localRotation = rot;

            // 2) ghost 슬롯 배치 (같은 프리팹을 복제해서 반투명 처리)
            Vector3 slotPos = slotsStartLocalPos + GridOffset(i);
            slotPos.y += yLiftSlots;

            var ghost = Instantiate(prefab, slotsTargetsRoot);
            ghost.name = slotPrefix + prefab.name;
            ghost.transform.localPosition = slotPos;
            ghost.transform.localRotation = rot;

            MakeGhost(ghost);
        }

        Debug.Log($"[GhostSlotGenerator] Generated {toolPrefabs.Count} tools and ghost slots.");
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
        // editor/runtime 모두 안전하게
        for (int i = root.childCount - 1; i >= 0; i--)
        {
            var ch = root.GetChild(i);
            if (Application.isPlaying) Destroy(ch.gameObject);
            else DestroyImmediate(ch.gameObject);
        }
    }

    private void MakeGhost(GameObject go)
    {
        // collider/rigidbody 등 상호작용 요소는 ghost에선 제거(원하면)
        foreach (var col in go.GetComponentsInChildren<Collider>(true))
            col.enabled = false;

        foreach (var rb in go.GetComponentsInChildren<Rigidbody>(true))
            DestroyComponentSafe(rb);

        // 렌더러 머티리얼을 "인스턴스"로 만들어서 투명 설정
        var renderers = go.GetComponentsInChildren<Renderer>(true);
        foreach (var r in renderers)
        {
            // sharedMaterials를 건드리면 원본 프리팹까지 바뀔 수 있으니 금지
            var mats = r.materials; // 인스턴스 생성됨
            for (int i = 0; i < mats.Length; i++)
            {
                mats[i] = ConvertToGhostMaterial(mats[i]);
            }
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
            var c = ghostTint;
            c.a = ghostAlpha;
            m.color = c;
            return m;
        }
        else
        {
            // Built-in Standard transparent 설정
            var shader = Shader.Find("Standard");
            if (shader == null) return src;

            var m = new Material(shader);

            // 텍스처가 있으면 유지 (팔레트 텍스처)
            if (src.HasProperty("_MainTex"))
                m.SetTexture("_MainTex", src.GetTexture("_MainTex"));

            // 컬러 곱
            Color baseCol = ghostTint;
            baseCol.a = ghostAlpha;
            if (src.HasProperty("_Color"))
            {
                // 원본 틴트도 섞고 싶으면 곱하기
                var sc = src.GetColor("_Color");
                baseCol.r *= sc.r;
                baseCol.g *= sc.g;
                baseCol.b *= sc.b;
            }
            m.SetColor("_Color", baseCol);

            // 투명 모드 설정(표준 방식)
            SetupStandardTransparent(m);

            // 반사 줄이기
            if (m.HasProperty("_Metallic")) m.SetFloat("_Metallic", 0f);
            if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", 0f);

            return m;
        }
    }

    private void SetupStandardTransparent(Material m)
    {
        // Standard shader: Rendering Mode = Transparent 세팅
        // (Unity 내부 구현을 그대로 따라감)
        m.SetFloat("_Mode", 3f); // Transparent
        m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        m.SetInt("_ZWrite", 0);
        m.DisableKeyword("_ALPHATEST_ON");
        m.EnableKeyword("_ALPHABLEND_ON");
        m.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
    }

    private void DestroyComponentSafe(Component c)
    {
        if (c == null) return;
        if (Application.isPlaying) Destroy(c);
        else DestroyImmediate(c);
    }
}
