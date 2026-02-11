using System;
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

    [Header("Debug")]
    public bool logDebug = true;

    public event Action<Rigidbody> OnGrabbed;
    public event Action<Rigidbody> OnReleased;

    public bool IsHolding => _heldBody != null;
    public Rigidbody HeldBody => _heldBody;

    private readonly Collider[] _overlapBuffer = new Collider[32];

    private Rigidbody _heldBody;
    private Transform _heldOriginalParent;
    private Collider[] _heldColliders;

    // initial offset captured at grab time (relative to anchor)
    private Vector3 _heldLocalPosToAnchor;
    private Quaternion _heldLocalRotToAnchor = Quaternion.identity;

    // rotation captured at grab time (world space)
    private Quaternion _heldRotFixedWorld = Quaternion.identity;

    // held follow filter state
    private Vector3 _heldPosSm;
    private Quaternion _heldRotSm = Quaternion.identity;
    private bool _heldFollowInit = false;

    // grab signal state
    private bool _lastGrabSignal = false;

    // release time
    private float _lastReleaseTime = -999f;

    void Start()
    {
        if (router == null)
            router = FindFirstObjectByType<PhoneInputRouter>();

        if (router != null)
            _lastGrabSignal = router.Grab;
    }

    private void Update()
    {
        if (router == null) return;

        bool grabSignal = router.Grab;

        if (grabSignal != _lastGrabSignal)
        {
            _lastGrabSignal = grabSignal;
            if (logDebug) DebugHUD.Log("[Grabber] GrabSignal=" + grabSignal);
        }

        if (_heldBody == null)
        {
            if (_lastGrabSignal)
            {
                if ((Time.unscaledTime - _lastReleaseTime) >= regrabCooldownSec)
                    TryGrab();
            }
        }
        else
        {
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

        if (!filterHeldObject)
        {
            if (heldRotationMode == HeldRotationMode.ExternalControl)
            {
                t.position = targetPos;
            }
            else
            {
                t.SetPositionAndRotation(targetPos, targetRot);
            }
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
        if (heldRotationMode == HeldRotationMode.ExternalControl)
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

        if (heldRotationMode == HeldRotationMode.ExternalControl)
        {
            t.position = _heldPosSm;
        }
        else
        {
            t.SetPositionAndRotation(_heldPosSm, _heldRotSm);
        }
    }

    private void TryGrab()
    {
        if (grabAnchor == null)
        {
            if (logDebug) DebugHUD.Log("[Grabber] No grabAnchor.");
            return;
        }

        int count = Physics.OverlapSphereNonAlloc(
            grabAnchor.position,
            grabRadius,
            _overlapBuffer,
            grabbableLayers,
            QueryTriggerInteraction.Collide
        );

        if (count <= 0)
        {
            if (logDebug) DebugHUD.Log("[Grabber] No candidate in range.");
            return;
        }

        Collider bestCol = null;
        float bestDist = float.MaxValue;

        for (int i = 0; i < count; i++)
        {
            Collider col = _overlapBuffer[i];
            if (col == null) continue;

            if (!string.IsNullOrEmpty(requiredTag) && !col.CompareTag(requiredTag))
                continue;

            Rigidbody rb = col.attachedRigidbody;
            if (rb == null) continue;

            float d2 = (col.ClosestPoint(grabAnchor.position) - grabAnchor.position).sqrMagnitude;
            if (d2 < bestDist)
            {
                bestDist = d2;
                bestCol = col;
            }
        }

        if (bestCol == null)
        {
            if (logDebug) DebugHUD.Log("[Grabber] No valid rigidbody candidate.");
            return;
        }

        float attachD = Mathf.Max(0f, attachDistance);
        if (attachD > 0f && bestDist > attachD * attachD)
        {
            if (logDebug) DebugHUD.Log("[Grabber] Candidate too far for attach.");
            return;
        }

        Rigidbody body = bestCol.attachedRigidbody;
        if (body == null) return;

        _heldBody = body;
        _heldOriginalParent = body.transform.parent;

        _heldBody.isKinematic = true;
        _heldBody.velocity = Vector3.zero;
        _heldBody.angularVelocity = Vector3.zero;

        body.transform.SetParent(grabAnchor, true);

        _heldLocalPosToAnchor = grabAnchor.InverseTransformPoint(body.transform.position);
        _heldLocalRotToAnchor = Quaternion.Inverse(grabAnchor.rotation) * body.transform.rotation;

        _heldRotFixedWorld = body.transform.rotation;

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

        OnReleased?.Invoke(releasedBody);
    }

    public void ForceRelease()
    {
        TryRelease();
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

    private void OnDrawGizmosSelected()
    {
        if (grabAnchor == null) return;

        Gizmos.color = new Color(0f, 1f, 1f, 0.35f);
        Gizmos.DrawWireSphere(grabAnchor.position, grabRadius);

        Gizmos.color = new Color(1f, 0f, 0f, 0.35f);
        Gizmos.DrawWireSphere(grabAnchor.position, attachDistance);
    }
}
