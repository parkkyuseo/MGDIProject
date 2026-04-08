using UnityEngine;
using UnityEngine.SceneManagement;

[DefaultExecutionOrder(205)]
public class ProxyHandLocatorBeacon : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private PhoneProxyHandRootDriver phoneMacroPoseDriver;
    [SerializeField] private PhoneInputRouter phoneRouter;
    [SerializeField] private ToolPlacementTaskManager placementTask;
    [SerializeField] private ToolRotationTaskManager rotationTask;
    [SerializeField] private ToolScalingTaskManager scalingTask;

    [Header("Behavior")]
    [SerializeField] private bool showOnlyDuringActiveTrials = true;
    [SerializeField] private float onScreenViewportPadding = 0.05f;

    [Header("Tether Line")]
    [SerializeField] private Transform tetherRoot;
    [SerializeField] private float centerAnchorDistance = 0.42f;
    [SerializeField] private float edgeAnchorDistance = 0.80f;
    [SerializeField] private Vector2 edgeViewportMargins = new Vector2(0.18f, 0.20f);
    [SerializeField] private Vector2 edgeViewportOffset = new Vector2(0f, 0.08f);
    [SerializeField] private float minLineWidth = 0.002f;
    [SerializeField] private float maxLineWidth = 0.007f;
    [SerializeField] private bool showTip = false;
    [SerializeField] private float tipMinScale = 0.008f;
    [SerializeField] private float tipMaxScale = 0.020f;
    [SerializeField] private float intensityOverflowForMax = 1.0f;
    [SerializeField] private float pulseFrequency = 1.7f;
    [SerializeField] private float pulseAmplitude = 0.08f;
    [SerializeField] private Color tetherColor = new Color(0.20f, 0.92f, 1.0f, 1.0f);
    [SerializeField] private float minEmission = 0.20f;
    [SerializeField] private float maxEmission = 1.30f;

    private LineRenderer _lineRenderer;
    private Transform _tipTransform;
    private Renderer _tipRenderer;
    private static bool _sceneHookInstalled;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void InstallSceneHook()
    {
        if (_sceneHookInstalled)
            return;

        SceneManager.sceneLoaded += HandleSceneLoaded;
        _sceneHookInstalled = true;
    }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void EnsureAfterInitialSceneLoad()
    {
        EnsureExistsInCurrentScene();
    }

    private static void HandleSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        _ = scene;
        _ = mode;
        EnsureExistsInCurrentScene();
    }

    private static void EnsureExistsInCurrentScene()
    {
        if (FindFirstObjectByType<ProxyHandLocatorBeacon>() != null)
            return;

        if (FindFirstObjectByType<PhoneProxyHandRootDriver>() == null)
            return;

        GameObject go = new GameObject("ProxyHandLocatorBeacon");
        go.AddComponent<ProxyHandLocatorBeacon>();
    }

    private void Awake()
    {
        ResolveReferences();
        EnsureTether();
        SetTetherVisible(false);
    }

    private void OnEnable()
    {
        ResolveReferences();
        EnsureTether();
        SetTetherVisible(false);
    }

    private void Update()
    {
        ResolveReferences();

        Camera cam = Camera.main;
        Transform handRoot = phoneMacroPoseDriver != null ? phoneMacroPoseDriver.HandRootTransform : null;
        if (cam == null || handRoot == null)
        {
            SetTetherVisible(false);
            return;
        }

        if (showOnlyDuringActiveTrials && !IsAnyTrialRunning())
        {
            SetTetherVisible(false);
            return;
        }

        if (ShouldSuppressForCurrentTechniqueTask())
        {
            SetTetherVisible(false);
            return;
        }

        if (IsTransformOnScreen(cam, handRoot, onScreenViewportPadding))
        {
            SetTetherVisible(false);
            return;
        }

        if (!TryGetEdgeViewport(cam, handRoot, edgeViewportMargins, out Vector3 edgeViewport, out float overflow))
        {
            SetTetherVisible(false);
            return;
        }

        EnsureTether();
        if (_lineRenderer == null || (showTip && _tipTransform == null))
        {
            SetTetherVisible(false);
            return;
        }

        edgeViewport.x = Mathf.Clamp01(edgeViewport.x + edgeViewportOffset.x);
        edgeViewport.y = Mathf.Clamp01(edgeViewport.y + edgeViewportOffset.y);

        Vector3 handWorld = handRoot.position;
        Vector3 centerWorld = cam.ViewportToWorldPoint(new Vector3(0.5f, 0.5f, centerAnchorDistance));

        float intensity = Mathf.Clamp01(Mathf.Max(0f, overflow) / Mathf.Max(0.01f, intensityOverflowForMax));
        intensity = Mathf.Max(0.18f, intensity);

        float pulse = 1f + Mathf.Sin(Time.time * pulseFrequency * Mathf.PI * 2f) * pulseAmplitude * intensity;
        float lineWidth = Mathf.Lerp(minLineWidth, maxLineWidth, intensity) * pulse;
        float tipScale = Mathf.Lerp(tipMinScale, tipMaxScale, intensity) * pulse;

        _lineRenderer.positionCount = 2;
        _lineRenderer.SetPosition(0, handWorld);
        _lineRenderer.SetPosition(1, centerWorld);
        _lineRenderer.startWidth = lineWidth * 0.55f;
        _lineRenderer.endWidth = lineWidth * 0.85f;

        if (_tipTransform != null)
        {
            _tipTransform.gameObject.SetActive(showTip);
            if (showTip)
            {
                _tipTransform.position = centerWorld;
                _tipTransform.rotation = cam.transform.rotation;
                _tipTransform.localScale = Vector3.one * tipScale;
            }
        }

        ApplyTetherVisual(intensity);
        SetTetherVisible(true);
    }

    private void ResolveReferences()
    {
        if (phoneMacroPoseDriver == null)
            phoneMacroPoseDriver = FindFirstObjectByType<PhoneProxyHandRootDriver>();
        if (phoneRouter == null)
            phoneRouter = FindFirstObjectByType<PhoneInputRouter>();
        if (placementTask == null)
            placementTask = FindFirstObjectByType<ToolPlacementTaskManager>();
        if (rotationTask == null)
            rotationTask = FindFirstObjectByType<ToolRotationTaskManager>();
        if (scalingTask == null)
            scalingTask = FindFirstObjectByType<ToolScalingTaskManager>();
    }

    private bool IsAnyTrialRunning()
    {
        return (placementTask != null && placementTask.IsTrialRunning) ||
               (rotationTask != null && rotationTask.IsTrialRunning) ||
               (scalingTask != null && scalingTask.IsTrialRunning);
    }

    private bool ShouldSuppressForCurrentTechniqueTask()
    {
        if (phoneRouter == null || phoneRouter.CurrentMode != PhoneInputRouter.Mode.Micro)
            return false;

        bool microRotationRunning = rotationTask != null && rotationTask.IsTrialRunning;
        bool microScalingRunning = scalingTask != null && scalingTask.IsTrialRunning;
        return microRotationRunning || microScalingRunning;
    }

    private void EnsureTether()
    {
        if (tetherRoot == null)
            tetherRoot = BuildDefaultTether("TetherLine");

        if (tetherRoot == null)
            return;

        if (_lineRenderer == null)
            _lineRenderer = tetherRoot.GetComponentInChildren<LineRenderer>(true);

        if (_tipTransform == null)
        {
            Transform tip = tetherRoot.Find("Tip");
            if (tip != null)
                _tipTransform = tip;
        }

        if (_tipRenderer == null && _tipTransform != null)
            _tipRenderer = _tipTransform.GetComponent<Renderer>();

        ApplyTetherVisual(0f);
    }

    private Transform BuildDefaultTether(string rootName)
    {
        GameObject root = new GameObject(rootName);
        root.transform.SetParent(transform, false);

        GameObject lineGo = new GameObject("Line");
        lineGo.transform.SetParent(root.transform, false);

        LineRenderer lr = lineGo.AddComponent<LineRenderer>();
        lr.useWorldSpace = true;
        lr.alignment = LineAlignment.View;
        lr.textureMode = LineTextureMode.Stretch;
        lr.positionCount = 2;
        lr.numCapVertices = 6;
        lr.numCornerVertices = 4;
        lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        lr.receiveShadows = false;

        Shader lineShader = Shader.Find("Sprites/Default");
        if (lineShader != null)
            lr.material = new Material(lineShader);

        GameObject tip = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        tip.name = "Tip";
        tip.transform.SetParent(root.transform, false);
        tip.transform.localScale = Vector3.one * tipMinScale;

        Collider[] colliders = root.GetComponentsInChildren<Collider>(true);
        for (int i = 0; i < colliders.Length; i++)
        {
            if (colliders[i] != null)
                Destroy(colliders[i]);
        }

        return root.transform;
    }

    private void ApplyTetherVisual(float intensity)
    {
        Color litColor = Color.Lerp(tetherColor * 0.55f, tetherColor, intensity);
        float emission = Mathf.Lerp(minEmission, maxEmission, intensity);

        if (_lineRenderer != null)
        {
            Color lineStart = litColor * 0.65f;
            lineStart.a = 0.75f;
            Color lineEnd = litColor;
            lineEnd.a = 0.95f;
            _lineRenderer.startColor = lineStart;
            _lineRenderer.endColor = lineEnd;

            if (_lineRenderer.material != null)
            {
                _lineRenderer.material.color = lineEnd;
                if (_lineRenderer.material.HasProperty("_EmissionColor"))
                {
                    _lineRenderer.material.EnableKeyword("_EMISSION");
                    _lineRenderer.material.SetColor("_EmissionColor", lineEnd * emission * 0.6f);
                }
            }
        }

        if (_tipRenderer != null)
        {
            _tipRenderer.gameObject.SetActive(showTip);
            if (!showTip)
                return;

            Material mat = _tipRenderer.material;
            if (mat != null)
            {
                mat.color = litColor;
                if (mat.HasProperty("_EmissionColor"))
                {
                    mat.EnableKeyword("_EMISSION");
                    mat.SetColor("_EmissionColor", litColor * emission);
                }
            }

            _tipRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            _tipRenderer.receiveShadows = false;
        }
    }

    private void SetTetherVisible(bool visible)
    {
        if (tetherRoot == null)
            return;

        if (tetherRoot.gameObject.activeSelf != visible)
            tetherRoot.gameObject.SetActive(visible);
    }

    private bool IsTransformOnScreen(Camera cam, Transform target, float viewportPadding)
    {
        Vector3 viewport = cam.WorldToViewportPoint(target.position);
        return viewport.z > 0f &&
               viewport.x >= viewportPadding &&
               viewport.x <= 1f - viewportPadding &&
               viewport.y >= viewportPadding &&
               viewport.y <= 1f - viewportPadding;
    }

    private bool TryGetEdgeViewport(Camera cam, Transform target, Vector2 margins, out Vector3 edgeViewport, out float overflow)
    {
        Vector3 viewport = cam.WorldToViewportPoint(target.position);
        Vector2 fromCenter = new Vector2(viewport.x - 0.5f, viewport.y - 0.5f);
        if (viewport.z < 0f)
            fromCenter = -fromCenter;

        if (fromCenter.sqrMagnitude < 1e-6f)
        {
            edgeViewport = default;
            overflow = 0f;
            return false;
        }

        float halfW = Mathf.Max(0.01f, 0.5f - margins.x);
        float halfH = Mathf.Max(0.01f, 0.5f - margins.y);
        float tx = Mathf.Abs(fromCenter.x) / halfW;
        float ty = Mathf.Abs(fromCenter.y) / halfH;
        float t = Mathf.Max(tx, ty);
        overflow = Mathf.Max(0f, t - 1f);
        t = Mathf.Max(1f, t);

        edgeViewport = new Vector3(
            0.5f + fromCenter.x / t,
            0.5f + fromCenter.y / t,
            0f);

        edgeViewport.x = Mathf.Clamp(edgeViewport.x, margins.x, 1f - margins.x);
        edgeViewport.y = Mathf.Clamp(edgeViewport.y, margins.y, 1f - margins.y);
        return true;
    }
}
