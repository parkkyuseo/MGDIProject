using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class GrabCandidateOutline : MonoBehaviour
{
    [SerializeField] private Color outlineColor = new Color(1f, 0.88f, 0.10f, 1f);
    [SerializeField] private float outlineScale = 1.01f;
    [SerializeField] private string outlineRootName = "__GrabOutline";

    private readonly List<GameObject> _outlineObjects = new List<GameObject>(16);
    private Material _outlineMaterial;
    private bool _built;

    public void SetVisible(bool visible)
    {
        if (!_built)
            BuildIfNeeded();

        for (int i = 0; i < _outlineObjects.Count; i++)
        {
            GameObject go = _outlineObjects[i];
            if (go != null && go.activeSelf != visible)
                go.SetActive(visible);
        }
    }

    private void OnDisable()
    {
        SetVisible(false);
    }

    private void OnDestroy()
    {
        if (_outlineMaterial != null)
            Destroy(_outlineMaterial);
    }

    private void BuildIfNeeded()
    {
        if (_built)
            return;

        _built = true;
        _outlineMaterial = CreateOutlineMaterial();
        if (_outlineMaterial == null)
            return;

        MeshFilter[] meshFilters = GetComponentsInChildren<MeshFilter>(true);
        for (int i = 0; i < meshFilters.Length; i++)
        {
            MeshFilter mf = meshFilters[i];
            if (mf == null || mf.sharedMesh == null)
                continue;

            Transform sourceTf = mf.transform;
            if (sourceTf.name.StartsWith(outlineRootName))
                continue;

            MeshRenderer sourceRenderer = sourceTf.GetComponent<MeshRenderer>();
            if (sourceRenderer == null || !sourceRenderer.enabled)
                continue;

            GameObject clone = new GameObject(outlineRootName);
            clone.transform.SetParent(sourceTf, false);
            clone.transform.localPosition = Vector3.zero;
            clone.transform.localRotation = Quaternion.identity;
            clone.transform.localScale = Vector3.one * Mathf.Max(1.001f, outlineScale);
            clone.layer = sourceTf.gameObject.layer;

            MeshFilter cloneMf = clone.AddComponent<MeshFilter>();
            cloneMf.sharedMesh = mf.sharedMesh;

            MeshRenderer cloneMr = clone.AddComponent<MeshRenderer>();
            cloneMr.sharedMaterial = _outlineMaterial;
            cloneMr.shadowCastingMode = ShadowCastingMode.Off;
            cloneMr.receiveShadows = false;
            cloneMr.lightProbeUsage = LightProbeUsage.Off;
            cloneMr.reflectionProbeUsage = ReflectionProbeUsage.Off;
            cloneMr.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
            cloneMr.allowOcclusionWhenDynamic = false;
            cloneMr.enabled = true;

            clone.SetActive(false);
            _outlineObjects.Add(clone);
        }
    }

    private Material CreateOutlineMaterial()
    {
        Shader shader =
            Shader.Find("Hidden/Internal-Colored") ??
            Shader.Find("Unlit/Color") ??
            Shader.Find("Sprites/Default") ??
            Shader.Find("Universal Render Pipeline/Unlit") ??
            Shader.Find("Standard");

        if (shader == null)
            return null;

        Material mat = new Material(shader);
        mat.name = "GrabCandidateOutline_Mat";

        if (mat.HasProperty("_Color"))
            mat.SetColor("_Color", outlineColor);
        if (mat.HasProperty("_BaseColor"))
            mat.SetColor("_BaseColor", outlineColor);
        if (mat.HasProperty("_EmissionColor"))
            mat.SetColor("_EmissionColor", outlineColor * 0.3f);

        if (mat.HasProperty("_Cull"))
            mat.SetInt("_Cull", (int)CullMode.Front);
        if (mat.HasProperty("_ZTest"))
            mat.SetInt("_ZTest", (int)CompareFunction.LessEqual);
        if (mat.HasProperty("_ZWrite"))
            mat.SetInt("_ZWrite", 1);
        if (mat.HasProperty("_Glossiness"))
            mat.SetFloat("_Glossiness", 0f);
        if (mat.HasProperty("_Smoothness"))
            mat.SetFloat("_Smoothness", 0f);
        if (mat.HasProperty("_Metallic"))
            mat.SetFloat("_Metallic", 0f);

        mat.renderQueue = (int)RenderQueue.Geometry + 1;

        return mat;
    }
}
