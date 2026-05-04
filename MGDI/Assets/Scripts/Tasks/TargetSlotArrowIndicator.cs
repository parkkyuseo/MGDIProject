using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

[DefaultExecutionOrder(200)]
public class TargetSlotArrowIndicator : MonoBehaviour
{
    [Header("Task Managers (optional auto-find)")]
    [SerializeField] private ToolPlacementTaskManager placementTask;
    [SerializeField] private ToolRotationTaskManager rotationTask;
    [SerializeField] private ToolScalingTaskManager scalingTask;
    [SerializeField] private PhoneProxyHandRootDriver phoneMacroPoseDriver;
    [SerializeField] private ProxyHandGrabber grabber;

    [Header("World Marker")]
    [SerializeField] private Transform markerRoot;
    [SerializeField] private float verticalOffset = 0.10f;
    [SerializeField] private float bobAmplitude = 0.015f;
    [SerializeField] private float bobFrequency = 2.0f;
    [SerializeField] private float spinDegPerSec = 60f;
    [SerializeField] private float baseWorldScale = 1.0f;

    [Header("Edge Indicator")]
    [SerializeField] private Transform edgeMarkerRoot;
    [SerializeField] private float edgeMarkerDistance = 0.85f;
    [SerializeField] private float edgeMarkerScale = 0.40f;
    [SerializeField] private Vector2 edgeViewportMargins = new Vector2(0.12f, 0.16f);
    [SerializeField] private float edgeTopViewportInset = 0.10f;
    [SerializeField] private float onScreenViewportPadding = 0.04f;

    [Header("Tool Edge Indicator")]
    [SerializeField] private bool showToolWorldIndicator = true;
    [SerializeField] private Transform toolWorldMarkerRoot;
    [SerializeField] private float toolWorldIndicatorVisibleSeconds = 2.0f;
    [SerializeField] private bool showToolEdgeIndicator = true;
    [SerializeField] private Transform toolEdgeMarkerRoot;
    [SerializeField] private float toolEdgeMarkerDistance = 0.78f;
    [SerializeField] private float toolEdgeMarkerScale = 0.34f;
    [SerializeField] private Vector2 toolEdgeViewportMargins = new Vector2(0.18f, 0.22f);
    [SerializeField] private Vector2 toolEdgeViewportOffset = new Vector2(0f, -0.08f);
    [SerializeField] private Vector3 toolPreviewLocalPosition = new Vector3(0.15f, 0f, 0f);
    [SerializeField] private float toolPreviewMaxSize = 0.14f;
    [SerializeField] private bool useToolPreviewSprites = true;
    [SerializeField] private string toolPreviewResourcesPath = "Image";
    [SerializeField] private Color toolPreviewSpriteTint = Color.white;

    [Header("Visibility")]
    [SerializeField] private bool hideTargetIndicatorsUntilActiveToolHeld = true;
    [SerializeField] private bool hideToolIndicatorsAfterActiveToolHeld = true;
    [SerializeField] private bool hideWhenToolIsNearTarget = true;
    [SerializeField] private float hideWhenToolWithinMeters = 0.18f;
    [SerializeField] private bool hideToolWorldIndicatorWhenHandNear = true;
    [SerializeField] private float hideToolWorldIndicatorWhenHandWithinMeters = 0.18f;
    [SerializeField] private Color markerColor = new Color(1.0f, 0.78f, 0.12f, 1.0f);
    [SerializeField] private Color toolMarkerColor = new Color(0.20f, 0.85f, 1.0f, 1.0f);

    private Renderer[] _worldMarkerRenderers;
    private Renderer[] _edgeMarkerRenderers;
    private Renderer[] _toolWorldMarkerRenderers;
    private Renderer[] _toolEdgeMarkerRenderers;
    private bool _hasWorldMarker;
    private bool _hasEdgeMarker;
    private bool _hasToolWorldMarker;
    private bool _hasToolEdgeMarker;
    private bool _toolWorldIndicatorWindowActive;
    private float _toolWorldIndicatorHideAt = -1f;
    private Transform _toolWorldIndicatorActiveTool;
    private Transform _toolPreviewHolder;
    private Transform _toolPreviewRoot;
    private TextMesh _toolPreviewLabel;
    private SpriteRenderer _toolPreviewSpriteRenderer;
    private string _toolPreviewKey;
    private bool _toolPreviewSpritesLoaded;
    private readonly Dictionary<string, Sprite> _toolPreviewSpriteById = new Dictionary<string, Sprite>(StringComparer.OrdinalIgnoreCase);
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
        if (FindFirstObjectByType<TargetSlotArrowIndicator>() != null)
            return;

        bool hasTaskManagers =
            FindFirstObjectByType<ToolPlacementTaskManager>() != null ||
            FindFirstObjectByType<ToolRotationTaskManager>() != null ||
            FindFirstObjectByType<ToolScalingTaskManager>() != null;

        if (!hasTaskManagers)
            return;

        GameObject go = new GameObject("TargetSlotArrowIndicator");
        go.AddComponent<TargetSlotArrowIndicator>();
    }

    private void Awake()
    {
        ResolveReferences();
        EnsureMarkers();
        SetWorldMarkerVisible(false);
        SetEdgeMarkerVisible(false);
        SetToolWorldMarkerVisible(false);
        SetToolEdgeMarkerVisible(false);
        ResetToolWorldIndicatorWindow();
    }

    private void OnEnable()
    {
        ResolveReferences();
        EnsureMarkers();
        SetWorldMarkerVisible(false);
        SetEdgeMarkerVisible(false);
        SetToolWorldMarkerVisible(false);
        SetToolEdgeMarkerVisible(false);
        ResetToolWorldIndicatorWindow();
    }

    private void Update()
    {
        ResolveReferences();

        Transform activeTool = GetActiveToolTransform();
        string activeToolId = GetActiveToolId();
        bool activeToolHeld = IsActiveToolHeld(activeTool);
        bool showTargetIndicators = ShouldShowTargetIndicators();
        Transform activeTarget = showTargetIndicators ? GetActiveTargetTransform() : null;

        Camera cam = Camera.main;
        if (cam == null)
        {
            SetWorldMarkerVisible(false);
            SetEdgeMarkerVisible(false);
            SetToolWorldMarkerVisible(false);
            SetToolEdgeMarkerVisible(false);
            ResetToolWorldIndicatorWindow();
            return;
        }

        if (activeTarget == null || (hideTargetIndicatorsUntilActiveToolHeld && !activeToolHeld))
        {
            SetWorldMarkerVisible(false);
            SetEdgeMarkerVisible(false);
        }
        else if (hideWhenToolIsNearTarget && IsToolNearTarget(activeTool, activeTarget))
        {
            SetWorldMarkerVisible(false);
            SetEdgeMarkerVisible(false);
        }
        else
        {
            EnsureMarkers();
            bool isOnScreen = IsTargetOnScreen(cam, activeTarget);
            if (isOnScreen)
            {
                if (_hasWorldMarker)
                {
                    SetWorldMarkerVisible(true);
                    UpdateWorldMarkerPose(activeTarget);
                }

                SetEdgeMarkerVisible(false);
            }
            else
            {
                SetWorldMarkerVisible(false);
                if (_hasEdgeMarker)
                {
                    SetEdgeMarkerVisible(true);
                    UpdateEdgeMarkerPose(cam, activeTarget);
                }
            }
        }

        if ((!showToolWorldIndicator && !showToolEdgeIndicator) ||
            activeTool == null ||
            (hideToolIndicatorsAfterActiveToolHeld && activeToolHeld))
        {
            SetToolWorldMarkerVisible(false);
            SetToolEdgeMarkerVisible(false);
            ResetToolWorldIndicatorWindow();
            return;
        }

        EnsureMarkers();
        bool toolOnScreen = IsTransformOnScreen(cam, activeTool, onScreenViewportPadding);
        if (toolOnScreen)
        {
            bool toolWorldIndicatorEligible =
                _hasToolWorldMarker &&
                showToolWorldIndicator &&
                !IsHandNearTool(activeTool);

            if (toolWorldIndicatorEligible)
            {
                UpdateToolWorldIndicatorWindow(activeTool);
                if (ShouldShowToolWorldIndicatorNow(activeTool))
                {
                    SetToolWorldMarkerVisible(true);
                    UpdateToolWorldMarkerPose(activeTool);
                }
                else
                {
                    SetToolWorldMarkerVisible(false);
                }
            }
            else
            {
                SetToolWorldMarkerVisible(false);
                ResetToolWorldIndicatorWindow();
            }

            SetToolEdgeMarkerVisible(false);
            return;
        }

        SetToolWorldMarkerVisible(false);
        ResetToolWorldIndicatorWindow();
        if (_hasToolEdgeMarker && showToolEdgeIndicator)
        {
            EnsureToolPreview(activeTool, activeToolId);
            SetToolEdgeMarkerVisible(true);
            UpdateToolEdgeMarkerPose(cam, activeTool);
        }
        else
        {
            SetToolEdgeMarkerVisible(false);
        }
    }

    private void ResolveReferences()
    {
        if (placementTask == null)
            placementTask = FindFirstObjectByType<ToolPlacementTaskManager>();
        if (rotationTask == null)
            rotationTask = FindFirstObjectByType<ToolRotationTaskManager>();
        if (scalingTask == null)
            scalingTask = FindFirstObjectByType<ToolScalingTaskManager>();
        if (phoneMacroPoseDriver == null)
            phoneMacroPoseDriver = FindFirstObjectByType<PhoneProxyHandRootDriver>();
        if (grabber == null)
            grabber = FindFirstObjectByType<ProxyHandGrabber>();
    }

    private bool ShouldShowTargetIndicators()
    {
        return placementTask != null && placementTask.IsTrialRunning;
    }

    private Transform GetActiveTargetTransform()
    {
        if (placementTask != null && placementTask.IsTrialRunning)
            return placementTask.ActiveTargetTransform;

        if (rotationTask != null && rotationTask.IsTrialRunning)
            return rotationTask.ActiveTargetTransform;

        if (scalingTask != null && scalingTask.IsTrialRunning)
            return scalingTask.ActiveTargetTransform;

        return null;
    }

    private Transform GetActiveToolTransform()
    {
        if (placementTask != null && placementTask.IsTrialRunning)
            return placementTask.ActiveToolTransform;

        if (rotationTask != null && rotationTask.IsTrialRunning)
            return rotationTask.ActiveToolTransform;

        if (scalingTask != null && scalingTask.IsTrialRunning)
            return scalingTask.ActiveToolTransform;

        return null;
    }

    private string GetActiveToolId()
    {
        if (placementTask != null && placementTask.IsTrialRunning)
            return placementTask.ActiveToolId;

        if (rotationTask != null && rotationTask.IsTrialRunning)
            return rotationTask.ActiveToolId;

        if (scalingTask != null && scalingTask.IsTrialRunning)
            return scalingTask.ActiveId;

        return null;
    }

    private void UpdateWorldMarkerPose(Transform activeTarget)
    {
        Bounds bounds;
        bool hasBounds = TryGetRenderableBounds(activeTarget, out bounds);
        Vector3 targetCenter = hasBounds
            ? bounds.center
            : activeTarget.position;

        float height = hasBounds
            ? Mathf.Max(bounds.extents.y * 2f, 0.02f)
            : 0.02f;

        float bob = bobAmplitude * Mathf.Sin(Time.time * bobFrequency * Mathf.PI * 2f);
        Vector3 worldPos = targetCenter + Vector3.up * (height * 0.5f + verticalOffset + bob);

        markerRoot.position = worldPos;
        markerRoot.rotation = Quaternion.Euler(0f, Time.time * spinDegPerSec, 0f);
        markerRoot.localScale = Vector3.one * Mathf.Max(0.001f, baseWorldScale);
    }

    private void UpdateToolWorldMarkerPose(Transform activeTool)
    {
        Bounds bounds;
        bool hasBounds = TryGetRenderableBounds(activeTool, out bounds);
        Vector3 toolCenter = hasBounds
            ? bounds.center
            : activeTool.position;

        float height = hasBounds
            ? Mathf.Max(bounds.extents.y * 2f, 0.02f)
            : 0.02f;

        float bob = bobAmplitude * Mathf.Sin(Time.time * bobFrequency * Mathf.PI * 2f);
        Vector3 worldPos = toolCenter + Vector3.up * (height * 0.5f + verticalOffset + bob);

        toolWorldMarkerRoot.position = worldPos;
        toolWorldMarkerRoot.rotation = Quaternion.Euler(0f, Time.time * spinDegPerSec, 0f);
        toolWorldMarkerRoot.localScale = Vector3.one * Mathf.Max(0.001f, baseWorldScale);
    }

    private void UpdateEdgeMarkerPose(Camera cam, Transform activeTarget)
    {
        Bounds bounds;
        Vector3 targetCenter = TryGetRenderableBounds(activeTarget, out bounds)
            ? bounds.center
            : activeTarget.position;

        Vector3 viewport = cam.WorldToViewportPoint(targetCenter);
        Vector2 fromCenter = new Vector2(viewport.x - 0.5f, viewport.y - 0.5f);
        if (viewport.z < 0f)
            fromCenter = -fromCenter;

        if (fromCenter.sqrMagnitude < 1e-6f)
            fromCenter = Vector2.up;

        float halfW = Mathf.Max(0.01f, 0.5f - edgeViewportMargins.x);
        float halfH = Mathf.Max(0.01f, 0.5f - edgeViewportMargins.y);
        float t = Mathf.Max(
            Mathf.Abs(fromCenter.x) / halfW,
            Mathf.Abs(fromCenter.y) / halfH);
        t = Mathf.Max(1f, t);

        Vector2 edgeViewport = new Vector2(
            0.5f + fromCenter.x / t,
            0.5f + fromCenter.y / t);

        edgeViewport.x = Mathf.Clamp(edgeViewport.x, edgeViewportMargins.x, 1f - edgeViewportMargins.x);
        edgeViewport.y = Mathf.Clamp(edgeViewport.y, edgeViewportMargins.y, 1f - edgeViewportMargins.y);

        float topEdgeY = 1f - edgeViewportMargins.y;
        if (edgeViewport.y >= topEdgeY - 0.001f)
            edgeViewport.y = Mathf.Max(edgeViewportMargins.y, edgeViewport.y - Mathf.Max(0f, edgeTopViewportInset));

        Vector3 worldPos = cam.ViewportToWorldPoint(new Vector3(edgeViewport.x, edgeViewport.y, edgeMarkerDistance));
        edgeMarkerRoot.position = worldPos;

        Vector2 dir = fromCenter.normalized;
        float zRot = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg + 90f;
        edgeMarkerRoot.rotation = cam.transform.rotation * Quaternion.Euler(0f, 0f, zRot);
        edgeMarkerRoot.localScale = Vector3.one * Mathf.Max(0.001f, edgeMarkerScale);
    }

    private void UpdateToolEdgeMarkerPose(Camera cam, Transform activeTool)
    {
        Vector3 edgeViewport;
        float zRot;
        if (!TryGetEdgeViewportAndRotation(cam, activeTool, toolEdgeViewportMargins, out edgeViewport, out zRot))
        {
            SetToolEdgeMarkerVisible(false);
            return;
        }

        edgeViewport.x = Mathf.Clamp01(edgeViewport.x + toolEdgeViewportOffset.x);
        edgeViewport.y = Mathf.Clamp01(edgeViewport.y + toolEdgeViewportOffset.y);

        Vector3 worldPos = cam.ViewportToWorldPoint(new Vector3(edgeViewport.x, edgeViewport.y, toolEdgeMarkerDistance));
        toolEdgeMarkerRoot.position = worldPos;
        toolEdgeMarkerRoot.rotation = cam.transform.rotation * Quaternion.Euler(0f, 0f, zRot);
        toolEdgeMarkerRoot.localScale = Vector3.one * Mathf.Max(0.001f, toolEdgeMarkerScale);

        if (_toolPreviewHolder != null)
        {
            _toolPreviewHolder.localPosition = toolPreviewLocalPosition;
            _toolPreviewHolder.localRotation = Quaternion.Euler(0f, 0f, -zRot);
        }
    }

    private bool TryGetRenderableBounds(Transform target, out Bounds bounds)
    {
        Renderer[] renderers = target.GetComponentsInChildren<Renderer>(true);
        if (renderers == null || renderers.Length == 0)
        {
            bounds = default;
            return false;
        }

        bool found = false;
        bounds = default;

        for (int i = 0; i < renderers.Length; i++)
        {
            Renderer r = renderers[i];
            if (r == null || !r.enabled)
                continue;

            if (!found)
            {
                bounds = r.bounds;
                found = true;
            }
            else
            {
                bounds.Encapsulate(r.bounds);
            }
        }

        if (!found)
        {
            bounds = default;
            return false;
        }

        return true;
    }

    private void EnsureMarkers()
    {
        if (markerRoot == null)
            markerRoot = BuildDefaultMarker("WorldArrow");

        if (edgeMarkerRoot == null)
            edgeMarkerRoot = BuildDefaultMarker("EdgeArrow");

        if (toolWorldMarkerRoot == null)
            toolWorldMarkerRoot = BuildDefaultMarker("ToolWorldArrow");

        if (toolEdgeMarkerRoot == null)
            toolEdgeMarkerRoot = BuildDefaultMarker("ToolEdgeArrow");

        EnsureToolPreviewInfrastructure();

        _hasWorldMarker = markerRoot != null;
        _hasEdgeMarker = edgeMarkerRoot != null;
        _hasToolWorldMarker = toolWorldMarkerRoot != null;
        _hasToolEdgeMarker = toolEdgeMarkerRoot != null;
        _worldMarkerRenderers = _hasWorldMarker ? markerRoot.GetComponentsInChildren<Renderer>(true) : null;
        _edgeMarkerRenderers = _hasEdgeMarker ? edgeMarkerRoot.GetComponentsInChildren<Renderer>(true) : null;
        _toolWorldMarkerRenderers = _hasToolWorldMarker ? toolWorldMarkerRoot.GetComponentsInChildren<Renderer>(true) : null;
        _toolEdgeMarkerRenderers = _hasToolEdgeMarker ? toolEdgeMarkerRoot.GetComponentsInChildren<Renderer>(true) : null;
        ApplyMarkerColor();
    }

    private Transform BuildDefaultMarker(string rootName)
    {
        GameObject root = new GameObject(rootName);
        root.transform.SetParent(transform, false);

        GameObject shaft = GameObject.CreatePrimitive(PrimitiveType.Cube);
        shaft.name = "Shaft";
        shaft.transform.SetParent(root.transform, false);
        shaft.transform.localScale = new Vector3(0.022f, 0.140f, 0.022f);
        shaft.transform.localPosition = new Vector3(0f, -0.010f, 0f);

        CreateArrowHead(root.transform, "HeadX_Pos", new Vector3(0.030f, -0.075f, 0f), new Vector3(0f, 0f, -45f));
        CreateArrowHead(root.transform, "HeadX_Neg", new Vector3(-0.030f, -0.075f, 0f), new Vector3(0f, 0f, 45f));
        CreateArrowHead(root.transform, "HeadZ_Pos", new Vector3(0f, -0.075f, 0.030f), new Vector3(45f, 0f, 0f));
        CreateArrowHead(root.transform, "HeadZ_Neg", new Vector3(0f, -0.075f, -0.030f), new Vector3(-45f, 0f, 0f));

        Collider[] colliders = root.GetComponentsInChildren<Collider>(true);
        for (int i = 0; i < colliders.Length; i++)
        {
            if (colliders[i] != null)
                Destroy(colliders[i]);
        }

        return root.transform;
    }

    private void CreateArrowHead(Transform parent, string name, Vector3 localPosition, Vector3 localEuler)
    {
        GameObject part = GameObject.CreatePrimitive(PrimitiveType.Cube);
        part.name = name;
        part.transform.SetParent(parent, false);
        part.transform.localScale = new Vector3(0.018f, 0.060f, 0.018f);
        part.transform.localPosition = localPosition;
        part.transform.localRotation = Quaternion.Euler(localEuler);
    }

    private void ApplyMarkerColor()
    {
        ApplyColorToRenderers(_worldMarkerRenderers, markerColor);
        ApplyColorToRenderers(_edgeMarkerRenderers, markerColor);
        ApplyColorToRenderers(_toolWorldMarkerRenderers, toolMarkerColor);
        ApplyColorToRenderers(_toolEdgeMarkerRenderers, toolMarkerColor, _toolPreviewHolder);
    }

    private void ApplyColorToRenderers(Renderer[] renderers, Color color, Transform excludeRoot = null)
    {
        if (renderers == null)
            return;

        for (int i = 0; i < renderers.Length; i++)
        {
            Renderer r = renderers[i];
            if (r == null)
                continue;
            if (excludeRoot != null && r.transform.IsChildOf(excludeRoot))
                continue;

            Material mat = r.material;
            if (mat == null)
                continue;

            mat.color = color;
            if (mat.HasProperty("_EmissionColor"))
            {
                mat.EnableKeyword("_EMISSION");
                mat.SetColor("_EmissionColor", color * 0.4f);
            }

            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            r.receiveShadows = false;
        }
    }

    private bool IsTargetOnScreen(Camera cam, Transform activeTarget)
    {
        return IsTransformOnScreen(cam, activeTarget, onScreenViewportPadding);
    }

    private bool IsTransformOnScreen(Camera cam, Transform target, float viewportPadding)
    {
        Bounds bounds;
        Vector3 targetCenter = TryGetRenderableBounds(target, out bounds)
            ? bounds.center
            : target.position;

        Vector3 viewport = cam.WorldToViewportPoint(targetCenter);
        return viewport.z > 0f &&
               viewport.x >= viewportPadding &&
               viewport.x <= 1f - viewportPadding &&
               viewport.y >= viewportPadding &&
               viewport.y <= 1f - viewportPadding;
    }

    private bool IsToolNearTarget(Transform tool, Transform target)
    {
        if (tool == null || target == null)
            return false;

        Bounds toolBounds;
        Bounds targetBounds;
        Vector3 toolCenter = TryGetRenderableBounds(tool, out toolBounds) ? toolBounds.center : tool.position;
        Vector3 targetCenter = TryGetRenderableBounds(target, out targetBounds) ? targetBounds.center : target.position;
        return Vector3.Distance(toolCenter, targetCenter) <= hideWhenToolWithinMeters;
    }

    private bool IsHandNearTool(Transform tool)
    {
        if (!hideToolWorldIndicatorWhenHandNear || tool == null)
            return false;

        Transform handReference = GetHandReferenceTransform();
        if (handReference == null)
            return false;

        Bounds toolBounds;
        Vector3 toolCenter = TryGetRenderableBounds(tool, out toolBounds) ? toolBounds.center : tool.position;
        return Vector3.Distance(handReference.position, toolCenter) <= hideToolWorldIndicatorWhenHandWithinMeters;
    }

    private bool IsActiveToolHeld(Transform activeTool)
    {
        if (activeTool == null || grabber == null || !grabber.IsHolding || grabber.HeldBody == null)
            return false;

        Transform heldTransform = grabber.HeldBody.transform;
        return heldTransform == activeTool ||
               heldTransform.IsChildOf(activeTool) ||
               activeTool.IsChildOf(heldTransform);
    }

    private Transform GetHandReferenceTransform()
    {
        if (grabber != null && grabber.grabAnchor != null)
            return grabber.grabAnchor;

        if (phoneMacroPoseDriver != null)
            return phoneMacroPoseDriver.HandRootTransform;

        return null;
    }

    private void SetWorldMarkerVisible(bool visible)
    {
        if (!_hasWorldMarker || markerRoot == null)
            return;

        if (markerRoot.gameObject.activeSelf != visible)
            markerRoot.gameObject.SetActive(visible);
    }

    private void SetEdgeMarkerVisible(bool visible)
    {
        if (!_hasEdgeMarker || edgeMarkerRoot == null)
            return;

        if (edgeMarkerRoot.gameObject.activeSelf != visible)
            edgeMarkerRoot.gameObject.SetActive(visible);
    }

    private void SetToolWorldMarkerVisible(bool visible)
    {
        if (!_hasToolWorldMarker || toolWorldMarkerRoot == null)
            return;

        if (toolWorldMarkerRoot.gameObject.activeSelf != visible)
            toolWorldMarkerRoot.gameObject.SetActive(visible);
    }

    private void UpdateToolWorldIndicatorWindow(Transform activeTool)
    {
        if (activeTool == null)
        {
            ResetToolWorldIndicatorWindow();
            return;
        }

        if (!_toolWorldIndicatorWindowActive || _toolWorldIndicatorActiveTool != activeTool)
        {
            _toolWorldIndicatorWindowActive = true;
            _toolWorldIndicatorActiveTool = activeTool;
            _toolWorldIndicatorHideAt = Time.unscaledTime + Mathf.Max(0f, toolWorldIndicatorVisibleSeconds);
        }
    }

    private bool ShouldShowToolWorldIndicatorNow(Transform activeTool)
    {
        if (!_toolWorldIndicatorWindowActive || activeTool == null)
            return false;

        if (_toolWorldIndicatorActiveTool != activeTool)
            return false;

        return Time.unscaledTime <= _toolWorldIndicatorHideAt;
    }

    private void ResetToolWorldIndicatorWindow()
    {
        _toolWorldIndicatorWindowActive = false;
        _toolWorldIndicatorHideAt = -1f;
        _toolWorldIndicatorActiveTool = null;
    }

    private void SetToolEdgeMarkerVisible(bool visible)
    {
        if (!_hasToolEdgeMarker || toolEdgeMarkerRoot == null)
            return;

        if (toolEdgeMarkerRoot.gameObject.activeSelf != visible)
            toolEdgeMarkerRoot.gameObject.SetActive(visible);
    }

    private bool TryGetEdgeViewportAndRotation(
        Camera cam,
        Transform target,
        Vector2 margins,
        out Vector3 edgeViewport,
        out float zRot)
    {
        Bounds bounds;
        Vector3 targetCenter = TryGetRenderableBounds(target, out bounds)
            ? bounds.center
            : target.position;

        Vector3 viewport = cam.WorldToViewportPoint(targetCenter);
        Vector2 fromCenter = new Vector2(viewport.x - 0.5f, viewport.y - 0.5f);
        if (viewport.z < 0f)
            fromCenter = -fromCenter;

        if (fromCenter.sqrMagnitude < 1e-6f)
        {
            edgeViewport = default;
            zRot = 0f;
            return false;
        }

        float halfW = Mathf.Max(0.01f, 0.5f - margins.x);
        float halfH = Mathf.Max(0.01f, 0.5f - margins.y);
        float t = Mathf.Max(
            Mathf.Abs(fromCenter.x) / halfW,
            Mathf.Abs(fromCenter.y) / halfH);
        t = Mathf.Max(1f, t);

        edgeViewport = new Vector3(
            0.5f + fromCenter.x / t,
            0.5f + fromCenter.y / t,
            0f);

        edgeViewport.x = Mathf.Clamp(edgeViewport.x, margins.x, 1f - margins.x);
        edgeViewport.y = Mathf.Clamp(edgeViewport.y, margins.y, 1f - margins.y);

        Vector2 dir = fromCenter.normalized;
        zRot = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg + 90f;
        return true;
    }

    private void EnsureToolPreviewInfrastructure()
    {
        if (toolEdgeMarkerRoot == null)
            return;

        if (_toolPreviewHolder == null)
        {
            Transform existing = toolEdgeMarkerRoot.Find("ToolPreviewHolder");
            if (existing != null)
            {
                _toolPreviewHolder = existing;
            }
            else
            {
                GameObject go = new GameObject("ToolPreviewHolder");
                _toolPreviewHolder = go.transform;
                _toolPreviewHolder.SetParent(toolEdgeMarkerRoot, false);
            }
        }

        if (_toolPreviewRoot == null)
        {
            Transform existing = _toolPreviewHolder.Find("ToolPreviewRoot");
            if (existing != null)
            {
                _toolPreviewRoot = existing;
            }
            else
            {
                GameObject go = new GameObject("ToolPreviewRoot");
                _toolPreviewRoot = go.transform;
                _toolPreviewRoot.SetParent(_toolPreviewHolder, false);
            }
        }

        if (_toolPreviewLabel == null)
        {
            Transform existing = _toolPreviewHolder.Find("ToolPreviewLabel");
            if (existing != null)
            {
                _toolPreviewLabel = existing.GetComponent<TextMesh>();
            }

            if (_toolPreviewLabel == null)
            {
                GameObject go = new GameObject("ToolPreviewLabel");
                go.transform.SetParent(_toolPreviewHolder, false);
                _toolPreviewLabel = go.AddComponent<TextMesh>();
                _toolPreviewLabel.anchor = TextAnchor.MiddleLeft;
                _toolPreviewLabel.alignment = TextAlignment.Left;
                _toolPreviewLabel.characterSize = 0.08f;
                _toolPreviewLabel.fontSize = 56;
                _toolPreviewLabel.color = Color.white;
            }
        }

        if (_toolPreviewSpriteRenderer == null)
        {
            Transform existing = _toolPreviewHolder.Find("ToolPreviewSprite");
            if (existing != null)
                _toolPreviewSpriteRenderer = existing.GetComponent<SpriteRenderer>();

            if (_toolPreviewSpriteRenderer == null)
            {
                GameObject go = new GameObject("ToolPreviewSprite");
                go.transform.SetParent(_toolPreviewHolder, false);
                _toolPreviewSpriteRenderer = go.AddComponent<SpriteRenderer>();
                _toolPreviewSpriteRenderer.color = toolPreviewSpriteTint;
                _toolPreviewSpriteRenderer.enabled = false;
            }
        }
    }

    private void EnsureToolPreview(Transform activeTool, string toolId)
    {
        if (_toolPreviewRoot == null || _toolPreviewHolder == null)
            return;

        string key = !string.IsNullOrWhiteSpace(toolId) ? toolId.Trim() : activeTool.GetInstanceID().ToString();
        if (string.Equals(_toolPreviewKey, key, System.StringComparison.Ordinal) && _toolPreviewRoot.childCount > 0)
            return;

        ClearToolPreview();

        Sprite previewSprite = FindToolPreviewSprite(toolId, activeTool);
        if (previewSprite != null && _toolPreviewSpriteRenderer != null)
        {
            _toolPreviewSpriteRenderer.sprite = previewSprite;
            _toolPreviewSpriteRenderer.color = toolPreviewSpriteTint;
            _toolPreviewSpriteRenderer.enabled = true;
            ScaleToolPreviewSprite(previewSprite);

            if (_toolPreviewLabel != null)
                _toolPreviewLabel.gameObject.SetActive(false);

            _toolPreviewKey = key;
            return;
        }

        bool hasPreview = CopyRenderableHierarchy(activeTool, _toolPreviewRoot);
        if (hasPreview)
        {
            Bounds sourceBounds;
            if (TryGetRenderableBounds(activeTool, out sourceBounds))
            {
                float maxDim = Mathf.Max(sourceBounds.size.x, Mathf.Max(sourceBounds.size.y, sourceBounds.size.z));
                float scale = toolPreviewMaxSize / Mathf.Max(0.001f, maxDim);
                Vector3 localCenter = activeTool.InverseTransformPoint(sourceBounds.center);
                _toolPreviewRoot.localRotation = Quaternion.identity;
                _toolPreviewRoot.localScale = Vector3.one * scale;
                _toolPreviewRoot.localPosition = -localCenter * scale;
            }

            if (_toolPreviewLabel != null)
                _toolPreviewLabel.gameObject.SetActive(false);
        }
        else if (_toolPreviewLabel != null)
        {
            _toolPreviewLabel.text = GetToolLabel(toolId, activeTool);
            _toolPreviewLabel.transform.localPosition = Vector3.zero;
            _toolPreviewLabel.transform.localRotation = Quaternion.identity;
            _toolPreviewLabel.gameObject.SetActive(true);
        }

        _toolPreviewKey = key;
    }

    private void ClearToolPreview()
    {
        if (_toolPreviewSpriteRenderer != null)
        {
            _toolPreviewSpriteRenderer.sprite = null;
            _toolPreviewSpriteRenderer.enabled = false;
            _toolPreviewSpriteRenderer.transform.localPosition = Vector3.zero;
            _toolPreviewSpriteRenderer.transform.localRotation = Quaternion.identity;
            _toolPreviewSpriteRenderer.transform.localScale = Vector3.one;
        }

        if (_toolPreviewRoot != null)
        {
            for (int i = _toolPreviewRoot.childCount - 1; i >= 0; i--)
                Destroy(_toolPreviewRoot.GetChild(i).gameObject);
        }

        if (_toolPreviewLabel != null)
            _toolPreviewLabel.gameObject.SetActive(false);
    }

    private bool CopyRenderableHierarchy(Transform source, Transform destinationParent)
    {
        bool copiedAny = false;
        CopyRenderableHierarchyRecursive(source, destinationParent, ref copiedAny);
        return copiedAny;
    }

    private void CopyRenderableHierarchyRecursive(Transform source, Transform destinationParent, ref bool copiedAny)
    {
        GameObject clone = new GameObject(source.name);
        Transform cloneTf = clone.transform;
        cloneTf.SetParent(destinationParent, false);
        cloneTf.localPosition = source.localPosition;
        cloneTf.localRotation = source.localRotation;
        cloneTf.localScale = source.localScale;

        MeshFilter sourceFilter = source.GetComponent<MeshFilter>();
        MeshRenderer sourceRenderer = source.GetComponent<MeshRenderer>();
        if (sourceFilter != null && sourceRenderer != null && sourceFilter.sharedMesh != null)
        {
            MeshFilter cloneFilter = clone.AddComponent<MeshFilter>();
            MeshRenderer cloneRenderer = clone.AddComponent<MeshRenderer>();
            cloneFilter.sharedMesh = sourceFilter.sharedMesh;
            cloneRenderer.sharedMaterials = sourceRenderer.sharedMaterials;
            cloneRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            cloneRenderer.receiveShadows = false;
            copiedAny = true;
        }

        for (int i = 0; i < source.childCount; i++)
            CopyRenderableHierarchyRecursive(source.GetChild(i), cloneTf, ref copiedAny);
    }

    private string GetToolLabel(string toolId, Transform activeTool)
    {
        if (!string.IsNullOrWhiteSpace(toolId))
            return toolId.Trim();
        return activeTool != null ? activeTool.name : "Tool";
    }

    private Sprite FindToolPreviewSprite(string toolId, Transform activeTool)
    {
        if (!useToolPreviewSprites)
            return null;

        EnsureToolPreviewSpriteCache();

        if (!string.IsNullOrWhiteSpace(toolId))
        {
            Sprite sprite;
            if (_toolPreviewSpriteById.TryGetValue(NormalizeToolPreviewKey(toolId), out sprite))
                return sprite;
        }

        if (activeTool != null)
        {
            Sprite sprite;
            if (_toolPreviewSpriteById.TryGetValue(NormalizeToolPreviewKey(activeTool.name), out sprite))
                return sprite;
        }

        return null;
    }

    private void EnsureToolPreviewSpriteCache()
    {
        if (_toolPreviewSpritesLoaded)
            return;

        _toolPreviewSpritesLoaded = true;
        _toolPreviewSpriteById.Clear();

        string path = string.IsNullOrWhiteSpace(toolPreviewResourcesPath) ? "Image" : toolPreviewResourcesPath.Trim();
        Sprite[] sprites = Resources.LoadAll<Sprite>(path);
        if (sprites == null)
            return;

        for (int i = 0; i < sprites.Length; i++)
        {
            Sprite sprite = sprites[i];
            if (sprite == null)
                continue;

            string key = NormalizeToolPreviewKey(sprite.name);
            if (string.IsNullOrEmpty(key))
                continue;

            _toolPreviewSpriteById[key] = sprite;
        }
    }

    private void ScaleToolPreviewSprite(Sprite sprite)
    {
        if (_toolPreviewSpriteRenderer == null || sprite == null)
            return;

        Bounds bounds = sprite.bounds;
        float maxDim = Mathf.Max(bounds.size.x, bounds.size.y);
        float scale = toolPreviewMaxSize / Mathf.Max(0.001f, maxDim);
        Transform tf = _toolPreviewSpriteRenderer.transform;
        tf.localPosition = Vector3.zero;
        tf.localRotation = Quaternion.identity;
        tf.localScale = Vector3.one * scale;
    }

    private static string NormalizeToolPreviewKey(string key)
    {
        return string.IsNullOrWhiteSpace(key) ? null : key.Trim().ToLowerInvariant();
    }
}
