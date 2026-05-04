using System;
using System.Collections.Generic;
using UnityEngine;

public class ProxyHandGrabber : MonoBehaviour
{
    public enum HeldRotationMode
    {
        FollowAnchor,
        LockAtGrab,
        ExternalControl
    }

    [Header("Input (Router)")]
    [Tooltip("Grab signal is driven by PhoneInputRouter.Grab. Macro: hold, Micro: toggle.")]
    [SerializeField] private PhoneInputRouter router;

    [Header("Grab anchor")]
    public Transform grabAnchor;

    [Header("Grab settings")]
    public float grabRadius = 0.08f;
    public LayerMask grabbableLayers;
    public string requiredTag = "";

    [Tooltip("Must be within this distance to actually attach (meters). Should be <= grabRadius.")]
    public float attachDistance = 0.05f;

    [Header("Re-grab cooldown")]
    public float regrabCooldownSec = 0.10f;

    [Header("Collision")]
    public bool disableHeldColliders = true;

    [Header("Proxy hand visibility")]
    [Tooltip("If true, cached proxy-hand renderers are hidden while an object is held and restored on release.")]
    [SerializeField] private bool hideProxyHandWhileHolding = true;
    [Tooltip("Optional explicit visual root for the proxy hand. If empty, a hand visual root is auto-resolved from the phone hand driver.")]
    [SerializeField] private Transform proxyHandVisualRoot;

    [Header("Held object follow filter")]
    public bool filterHeldObject = true;
    public float heldPosDeadZoneMeters = 0.006f;
    public float heldMaxStepMeters = 0.06f;
    public float heldTauSec = 0.08f;

    [Header("Held rotation behavior")]
    [Tooltip("FollowAnchor: follow grabAnchor rotation. LockAtGrab: keep rotation fixed. ExternalControl: grabber does not touch rotation.")]
    public HeldRotationMode heldRotationMode = HeldRotationMode.FollowAnchor;

    [Tooltip("If true, rotation smoothing is applied when HeldRotationMode is FollowAnchor.")]
    public bool filterHeldRotation = false;
    public float heldMaxDegPerSec = 360f;
    public float heldRotTauSec = 0.10f;

    [Header("Grab offset behavior")]
    [Tooltip("If true, keep the initial relative pose between hand and object (no snap).")]
    public bool keepInitialOffset = true;

    [Header("Freeze (optional constraints)")]
    [Tooltip("If true, held object position is frozen at the grab moment (world).")]
    public bool freezeHeldPosition = false;

    [Tooltip("If true, held object rotation is frozen at the grab moment (world).")]
    public bool freezeHeldRotation = false;

    [Tooltip("If true, held object localScale is frozen at the grab moment.")]
    public bool freezeHeldScale = false;

    [Header("Debug")]
    public bool logDebug = true;

    public event Action<Rigidbody> OnGrabbed;
    public event Action<Rigidbody> OnReleased;

    public bool IsHolding => _heldBody != null;
    public Rigidbody HeldBody => _heldBody;
    public Rigidbody HoverCandidateBody => _hoverBody;
    public bool HasAttachCandidateNow
    {
        get
        {
            if (_heldBody != null)
                return false;

            Collider bestCol;
            float bestDist;
            return TryFindBestCandidate(out bestCol, out bestDist, requireAttachDistance: true, emitDebugLogs: false) &&
                   bestCol != null;
        }
    }
    public bool IsHoldingOrHasAttachCandidateNow => _heldBody != null || HasAttachCandidateNow;

    private readonly Collider[] _overlapBuffer = new Collider[32];

    private Rigidbody _heldBody;
    private Transform _heldOriginalParent;
    private Collider[] _heldColliders;

    // initial offset captured at grab time (relative to anchor)
    private Vector3 _heldLocalPosToAnchor;
    private Quaternion _heldLocalRotToAnchor = Quaternion.identity;

    // rotation captured at grab time (world space)
    private Quaternion _heldRotFixedWorld = Quaternion.identity;

    // freeze snapshots (world/world/local)
    private Vector3 _heldPosFixedWorld = Vector3.zero;
    private Quaternion _heldRotFixedWorld2 = Quaternion.identity;
    private Vector3 _heldScaleFixedLocal = Vector3.one;

    // held follow filter state
    private Vector3 _heldPosSm;
    private Quaternion _heldRotSm = Quaternion.identity;
    private bool _heldFollowInit = false;

    // grab signal state
    private bool _lastGrabSignal = false;

    // release time
    private float _lastReleaseTime = -999f;
    private Rigidbody _hoverBody;
    private GrabCandidateOutline _hoverOutline;
    private Renderer[] _proxyHandRenderers;
    private bool[] _proxyHandRendererWasEnabled;
    private bool _proxyHandVisualsCached;
    private bool _forceHideProxyHandVisuals;

    void Start()
    {
        if (router == null)
            router = FindFirstObjectByType<PhoneInputRouter>();

        if (router != null)
            _lastGrabSignal = router.Grab;

        CacheProxyHandVisualRenderers();
    }

    private void Update()
    {
        if (router == null) return;

        if (!_proxyHandVisualsCached && _heldBody == null)
            CacheProxyHandVisualRenderers();

        bool grabSignal = router.Grab;

        if (grabSignal != _lastGrabSignal)
        {
            _lastGrabSignal = grabSignal;
            if (logDebug) DebugHUD.Log("[Grabber] GrabSignal=" + grabSignal);
        }

        if (_heldBody == null)
        {
            UpdateHoverCandidate();

            if (_lastGrabSignal)
            {
                if ((Time.unscaledTime - _lastReleaseTime) >= regrabCooldownSec)
                    TryGrab();
            }
        }
        else
        {
            ClearHoverCandidate();

            if (!_lastGrabSignal)
                TryRelease();
        }
    }

    private void LateUpdate()
    {
        if (_heldBody == null) return;
        if (grabAnchor == null) return;

        Transform t = _heldBody.transform;

        Vector3 targetPos;
        Quaternion targetRot;

        if (keepInitialOffset)
        {
            targetPos = grabAnchor.TransformPoint(_heldLocalPosToAnchor);
            targetRot = grabAnchor.rotation * _heldLocalRotToAnchor;
        }
        else
        {
            targetPos = grabAnchor.position;
            targetRot = grabAnchor.rotation;
        }

        // Rotation mode policy (applies before freeze override)
        switch (heldRotationMode)
        {
            case HeldRotationMode.FollowAnchor:
                break;

            case HeldRotationMode.LockAtGrab:
                targetRot = _heldRotFixedWorld;
                break;

            case HeldRotationMode.ExternalControl:
                targetRot = t.rotation;
                break;
        }

        // Freeze overrides (final authority)
        if (freezeHeldPosition)
            targetPos = _heldPosFixedWorld;

        if (freezeHeldRotation)
            targetRot = _heldRotFixedWorld2;

        if (!filterHeldObject)
        {
            if (heldRotationMode == HeldRotationMode.ExternalControl && !freezeHeldRotation)
            {
                t.position = targetPos;
            }
            else
            {
                t.SetPositionAndRotation(targetPos, targetRot);
            }

            if (freezeHeldScale)
                t.localScale = _heldScaleFixedLocal;

            return;
        }

        if (!_heldFollowInit)
        {
            _heldPosSm = t.position;
            _heldRotSm = t.rotation;
            _heldFollowInit = true;
        }

        // Position filter: dead-zone + step clamp + LPF
        Vector3 dp = targetPos - _heldPosSm;
        float dz = Mathf.Max(0f, heldPosDeadZoneMeters);

        if (dz > 0f && dp.sqrMagnitude < dz * dz)
        {
            targetPos = _heldPosSm;
        }
        else
        {
            float maxStep = Mathf.Max(0f, heldMaxStepMeters);
            float mag = dp.magnitude;
            if (maxStep > 0f && mag > maxStep)
                targetPos = _heldPosSm + dp / Mathf.Max(mag, 1e-6f) * maxStep;

            float dt = Mathf.Max(Time.unscaledDeltaTime, 1f / 120f);
            float tau = Mathf.Max(1e-4f, heldTauSec);
            float a = 1f - Mathf.Exp(-dt / tau);

            _heldPosSm = Vector3.Lerp(_heldPosSm, targetPos, a);
        }

        // Rotation handling
        if (freezeHeldRotation)
        {
            _heldRotSm = _heldRotFixedWorld2;
        }
        else if (heldRotationMode == HeldRotationMode.ExternalControl)
        {
            _heldRotSm = t.rotation;
        }
        else if (heldRotationMode == HeldRotationMode.LockAtGrab)
        {
            _heldRotSm = _heldRotFixedWorld;
        }
        else // FollowAnchor
        {
            if (filterHeldRotation)
            {
                float dt = Mathf.Max(Time.unscaledDeltaTime, 1f / 120f);

                float ang = Quaternion.Angle(_heldRotSm, targetRot);
                float maxStepDeg = Mathf.Max(1f, heldMaxDegPerSec) * dt;
                if (ang > maxStepDeg && ang > 1e-3f)
                    targetRot = Quaternion.Slerp(_heldRotSm, targetRot, maxStepDeg / ang);

                float tau = Mathf.Max(1e-4f, heldRotTauSec);
                float a = 1f - Mathf.Exp(-dt / tau);

                _heldRotSm = Quaternion.Slerp(_heldRotSm, targetRot, a);
            }
            else
            {
                _heldRotSm = targetRot;
            }
        }

        if (freezeHeldPosition)
            _heldPosSm = _heldPosFixedWorld;

        if (heldRotationMode == HeldRotationMode.ExternalControl && !freezeHeldRotation)
        {
            t.position = _heldPosSm;
        }
        else
        {
            t.SetPositionAndRotation(_heldPosSm, _heldRotSm);
        }

        if (freezeHeldScale)
            t.localScale = _heldScaleFixedLocal;
    }

    private void TryGrab()
    {
        if (grabAnchor == null)
        {
            if (logDebug) DebugHUD.Log("[Grabber] No grabAnchor.");
            return;
        }

        Collider bestCol;
        float bestDist;
        if (!TryFindBestCandidate(out bestCol, out bestDist, requireAttachDistance: true, emitDebugLogs: true))
            return;

        if (bestCol == null)
        {
            if (logDebug) DebugHUD.Log("[Grabber] No valid rigidbody candidate.");
            return;
        }

        Rigidbody body = bestCol.attachedRigidbody;
        if (body == null) return;

        ClearHoverCandidate();

        _heldBody = body;
        _heldOriginalParent = body.transform.parent;

        _heldBody.isKinematic = true;
        _heldBody.velocity = Vector3.zero;
        _heldBody.angularVelocity = Vector3.zero;

        body.transform.SetParent(grabAnchor, true);

        _heldLocalPosToAnchor = grabAnchor.InverseTransformPoint(body.transform.position);
        _heldLocalRotToAnchor = Quaternion.Inverse(grabAnchor.rotation) * body.transform.rotation;

        _heldRotFixedWorld = body.transform.rotation;

        // Freeze snapshots captured at grab moment
        _heldPosFixedWorld = body.transform.position;
        _heldRotFixedWorld2 = body.transform.rotation;
        _heldScaleFixedLocal = body.transform.localScale;

        _heldPosSm = body.transform.position;
        _heldRotSm = body.transform.rotation;
        _heldFollowInit = true;

        if (disableHeldColliders)
        {
            _heldColliders = body.GetComponentsInChildren<Collider>(true);
            for (int i = 0; i < _heldColliders.Length; i++)
            {
                if (_heldColliders[i] != null)
                    _heldColliders[i].enabled = false;
            }
        }

        if (logDebug) DebugHUD.Log("[Grabber] Grabbed " + body.name);
        SetProxyHandVisualVisible(false);
        OnGrabbed?.Invoke(_heldBody);
    }

    private void TryRelease()
    {
        if (_heldBody == null) return;

        var releasedBody = _heldBody;

        if (logDebug) DebugHUD.Log("[Grabber] Released " + _heldBody.name);

        if (disableHeldColliders && _heldColliders != null)
        {
            for (int i = 0; i < _heldColliders.Length; i++)
            {
                if (_heldColliders[i] != null)
                    _heldColliders[i].enabled = true;
            }
        }

        Transform t = _heldBody.transform;
        t.SetParent(_heldOriginalParent, true);

        _heldBody.isKinematic = false;

        _heldBody = null;
        _heldOriginalParent = null;
        _heldColliders = null;

        _heldFollowInit = false;
        _lastReleaseTime = Time.unscaledTime;

        SetProxyHandVisualVisible(true);
        OnReleased?.Invoke(releasedBody);
    }

    public void ForceRelease()
    {
        TryRelease();
    }

    private void OnDisable()
    {
        SetProxyHandVisualVisible(true);
    }

    public void SetProxyHandVisualForceHidden(bool forceHidden)
    {
        _forceHideProxyHandVisuals = forceHidden;

        if (!_proxyHandVisualsCached)
            CacheProxyHandVisualRenderers();

        bool shouldShow = _heldBody == null && !_forceHideProxyHandVisuals;
        SetProxyHandVisualVisible(shouldShow);
    }

    private void UpdateHoverCandidate()
    {
        Collider bestCol;
        float bestDist;
        if (!TryFindBestCandidate(out bestCol, out bestDist, requireAttachDistance: true, emitDebugLogs: false))
        {
            ClearHoverCandidate();
            return;
        }

        Rigidbody body = bestCol != null ? bestCol.attachedRigidbody : null;
        if (body == null)
        {
            ClearHoverCandidate();
            return;
        }

        if (_hoverBody == body && _hoverOutline != null)
            return;

        ClearHoverCandidate();

        _hoverBody = body;
        _hoverOutline = body.GetComponent<GrabCandidateOutline>();
        if (_hoverOutline == null)
            _hoverOutline = body.gameObject.AddComponent<GrabCandidateOutline>();

        _hoverOutline.SetVisible(true);
    }

    private void ClearHoverCandidate()
    {
        if (_hoverOutline != null)
            _hoverOutline.SetVisible(false);

        _hoverBody = null;
        _hoverOutline = null;
    }

    private bool TryFindBestCandidate(out Collider bestCol, out float bestDist, bool requireAttachDistance, bool emitDebugLogs)
    {
        bestCol = null;
        bestDist = float.MaxValue;

        if (grabAnchor == null)
            return false;

        int count = Physics.OverlapSphereNonAlloc(
            grabAnchor.position,
            grabRadius,
            _overlapBuffer,
            grabbableLayers,
            QueryTriggerInteraction.Collide
        );

        if (count <= 0)
        {
            if (emitDebugLogs && logDebug) DebugHUD.Log("[Grabber] No candidate in range.");
            return false;
        }

        for (int i = 0; i < count; i++)
        {
            Collider col = _overlapBuffer[i];
            if (col == null)
                continue;

            if (!string.IsNullOrEmpty(requiredTag) && !col.CompareTag(requiredTag))
                continue;

            Rigidbody rb = col.attachedRigidbody;
            if (rb == null)
                continue;

            float d2 = (col.ClosestPoint(grabAnchor.position) - grabAnchor.position).sqrMagnitude;
            if (d2 < bestDist)
            {
                bestDist = d2;
                bestCol = col;
            }
        }

        if (bestCol == null)
            return false;

        if (requireAttachDistance)
        {
            float attachD = Mathf.Max(0f, attachDistance);
            if (attachD > 0f && bestDist > attachD * attachD)
            {
                if (emitDebugLogs && logDebug) DebugHUD.Log("[Grabber] Candidate too far for attach.");
                bestCol = null;
                return false;
            }
        }

        return true;
    }

    public void SetHeldRotationMode(HeldRotationMode mode)
    {
        heldRotationMode = mode;

        if (_heldBody == null) return;

        if (heldRotationMode == HeldRotationMode.LockAtGrab)
        {
            _heldRotFixedWorld = _heldBody.transform.rotation;
            _heldRotSm = _heldRotFixedWorld;
        }
        else if (heldRotationMode == HeldRotationMode.ExternalControl)
        {
            _heldRotSm = _heldBody.transform.rotation;
        }
    }

    public void SetFollowHeldRotation(bool value)
    {
        SetHeldRotationMode(value ? HeldRotationMode.FollowAnchor : HeldRotationMode.LockAtGrab);
    }

    public void SetFreezeHeld(bool freezePos, bool freezeRot, bool freezeScaleValue)
    {
        freezeHeldPosition = freezePos;
        freezeHeldRotation = freezeRot;
        freezeHeldScale = freezeScaleValue;

        // If currently holding, refresh snapshots immediately
        if (_heldBody != null)
        {
            _heldPosFixedWorld = _heldBody.transform.position;
            _heldRotFixedWorld2 = _heldBody.transform.rotation;
            _heldScaleFixedLocal = _heldBody.transform.localScale;
        }
    }

    private void OnDrawGizmosSelected()
    {
        if (grabAnchor == null) return;

        Gizmos.color = new Color(0f, 1f, 1f, 0.35f);
        Gizmos.DrawWireSphere(grabAnchor.position, grabRadius);

        Gizmos.color = new Color(1f, 0f, 0f, 0.35f);
        Gizmos.DrawWireSphere(grabAnchor.position, attachDistance);
    }

    private void CacheProxyHandVisualRenderers()
    {
        if (!hideProxyHandWhileHolding)
            return;

        List<Renderer> renderers = ResolveProxyHandVisualRenderers();
        if (renderers == null || renderers.Count == 0)
            return;

        _proxyHandRenderers = renderers.ToArray();
        _proxyHandRendererWasEnabled = new bool[_proxyHandRenderers.Length];
        for (int i = 0; i < _proxyHandRenderers.Length; i++)
            _proxyHandRendererWasEnabled[i] = _proxyHandRenderers[i] != null && _proxyHandRenderers[i].enabled;

        _proxyHandVisualsCached = true;
    }

    private List<Renderer> ResolveProxyHandVisualRenderers()
    {
        List<Renderer> renderers = new List<Renderer>(16);
        HashSet<Renderer> seen = new HashSet<Renderer>();

        if (proxyHandVisualRoot != null)
        {
            AddRenderersFromRoot(proxyHandVisualRoot, renderers, seen);
            return renderers;
        }

        Transform seed = null;

        PhoneProxyHandRootDriver phoneDriver = FindFirstObjectByType<PhoneProxyHandRootDriver>();
        if (phoneDriver != null && phoneDriver.HandRootTransform != null)
            seed = phoneDriver.HandRootTransform;
        else if (grabAnchor != null)
            seed = grabAnchor;

        if (seed != null)
        {
            Transform current = seed;
            while (current != null)
            {
                AddRenderersFromRoot(current, renderers, seen);
                if (renderers.Count > 0)
                {
                    proxyHandVisualRoot = current;
                    return renderers;
                }

                current = current.parent;
            }
        }

        Transform namedRoot = FindNamedProxyHandVisualRoot();
        if (namedRoot != null)
            proxyHandVisualRoot = namedRoot;

        AddNamedProxyHandRenderers(renderers, seen);
        return renderers;
    }

    private void AddNamedProxyHandRenderers(List<Renderer> renderers, HashSet<Renderer> seen)
    {
        Transform[] allTransforms = FindObjectsByType<Transform>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        for (int i = 0; i < allTransforms.Length; i++)
        {
            Transform tf = allTransforms[i];
            if (tf == null)
                continue;

            string name = tf.name;
            if (name != "RightHand" && name != "RemoteHand" && name != "MicroHandVisual")
                continue;

            AddRenderersFromRoot(tf, renderers, seen);
        }
    }

    private Transform FindNamedProxyHandVisualRoot()
    {
        Transform[] allTransforms = FindObjectsByType<Transform>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        for (int i = 0; i < allTransforms.Length; i++)
        {
            Transform tf = allTransforms[i];
            if (tf == null)
                continue;

            string name = tf.name;
            if (name == "RightHand" || name == "RemoteHand" || name == "MicroHandVisual")
                return tf;
        }

        return null;
    }

    private static void AddRenderersFromRoot(Transform root, List<Renderer> renderers, HashSet<Renderer> seen)
    {
        if (root == null)
            return;

        Renderer[] rootRenderers = root.GetComponentsInChildren<Renderer>(true);
        for (int i = 0; i < rootRenderers.Length; i++)
        {
            Renderer renderer = rootRenderers[i];
            if (renderer == null || !seen.Add(renderer))
                continue;

            renderers.Add(renderer);
        }
    }

    private void SetProxyHandVisualVisible(bool visible)
    {
        if (!hideProxyHandWhileHolding)
            return;

        if (!_proxyHandVisualsCached)
            return;

        for (int i = 0; i < _proxyHandRenderers.Length; i++)
        {
            Renderer renderer = _proxyHandRenderers[i];
            if (renderer == null)
                continue;

            bool shouldBeVisible = visible && !_forceHideProxyHandVisuals;
            renderer.enabled = shouldBeVisible ? _proxyHandRendererWasEnabled[i] : false;
        }
    }
}
