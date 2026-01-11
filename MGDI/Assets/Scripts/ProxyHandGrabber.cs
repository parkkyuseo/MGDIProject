using System;
using UnityEngine;

public class ProxyHandGrabber : MonoBehaviour
{
    [Header("Grab anchor")]
    public Transform grabAnchor;

    [Header("Grab settings")]
    public float grabRadius = 0.08f;
    public LayerMask grabbableLayers;
    public string requiredTag = "";

    [Tooltip("Must be within this distance to actually attach (meters). Should be <= grabRadius.")]
    public float attachDistance = 0.05f;

    [Header("Grip debounce")]
    public float closeHoldSec = 0.08f;
    public float openHoldSec = 0.12f;
    public float regrabCooldownSec = 0.10f;

    [Header("Collision")]
    public bool disableHeldColliders = true;

    [Header("Held object follow filter")]
    public bool filterHeldObject = true;
    public float heldPosDeadZoneMeters = 0.006f;
    public float heldMaxStepMeters = 0.06f;
    public float heldTauSec = 0.08f;

    [Tooltip("If true, rotation follows grabAnchor (with optional smoothing below). If false, rotation is fixed (kept at grab-start).")]
    public bool followHeldRotation = true;

    public bool filterHeldRotation = false;
    public float heldMaxDegPerSec = 360f;
    public float heldRotTauSec = 0.10f;

    [Header("Grab offset behavior")]
    [Tooltip("If true, keep the initial relative pose between hand and object (no snap).")]
    public bool keepInitialOffset = true;

    [Header("Debug")]
    public bool logDebug = true;

    public bool IsHolding => _heldBody != null;
    public Rigidbody HeldBody => _heldBody;

    Collider[] _overlapBuffer = new Collider[32];

    Rigidbody _heldBody;
    Transform _heldOriginalParent;
    Collider[] _heldColliders;

    // initial offset captured at grab time
    Vector3 _heldLocalPosToAnchor;
    Quaternion _heldLocalRotToAnchor = Quaternion.identity;

    // rotation fixed at grab start (world space)
    Quaternion _heldRotFixedWorld = Quaternion.identity;

    // debounce state
    UdpHandReceiver.GripState _lastGripState = UdpHandReceiver.GripState.Unknown;
    float _stateSinceTime = -1f;
    float _lastReleaseTime = -999f;

    // held follow filter state
    Vector3 _heldPosSm;
    Quaternion _heldRotSm = Quaternion.identity;
    bool _heldFollowInit = false;

    void OnEnable()
    {
        UdpHandReceiver.OnGripStateChanged += OnGripStateChanged;
    }

    void OnDisable()
    {
        UdpHandReceiver.OnGripStateChanged -= OnGripStateChanged;
    }

    void OnGripStateChanged(UdpHandReceiver.GripState state)
    {
        if (state != _lastGripState)
        {
            _lastGripState = state;
            _stateSinceTime = Time.unscaledTime;
            if (logDebug) Debug.Log("[Grabber] GripState=" + state);
        }
    }

    void Update()
    {
        if (_stateSinceTime < 0f) return;

        float heldFor = Time.unscaledTime - _stateSinceTime;

        if (_heldBody == null && _lastGripState == UdpHandReceiver.GripState.Closed)
        {
            if (heldFor >= closeHoldSec && (Time.unscaledTime - _lastReleaseTime) >= regrabCooldownSec)
                TryGrab();
            return;
        }

        if (_heldBody != null && _lastGripState == UdpHandReceiver.GripState.Open)
        {
            if (heldFor >= openHoldSec)
                TryRelease();
            return;
        }
    }

    void LateUpdate()
    {
        if (_heldBody == null) return;
        if (grabAnchor == null) return;

        Transform t = _heldBody.transform;

        // target pose = anchor pose * initial local offset
        Vector3 targetPos;
        Quaternion targetRot;

        if (keepInitialOffset)
        {
            targetPos = grabAnchor.TransformPoint(_heldLocalPosToAnchor);

            // For rotation: compute normally, but may be overridden if followHeldRotation == false
            targetRot = grabAnchor.rotation * _heldLocalRotToAnchor;
        }
        else
        {
            targetPos = grabAnchor.position;
            targetRot = grabAnchor.rotation;
        }

        // If rotation should not follow (e.g., translation-only task), keep a fixed world rotation.
        if (!followHeldRotation)
        {
            targetRot = _heldRotFixedWorld;
        }

        if (!filterHeldObject)
        {
            t.SetPositionAndRotation(targetPos, targetRot);
            return;
        }

        if (!_heldFollowInit)
        {
            _heldPosSm = t.position;
            _heldRotSm = t.rotation;
            _heldFollowInit = true;
        }

        // position filter: dead-zone + step clamp + LPF
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

        // rotation filter:
        // - if followHeldRotation is false, keep fixed rotation (no smoothing needed, but keep _heldRotSm consistent)
        // - else, behave as before (optional smoothing)
        if (!followHeldRotation)
        {
            _heldRotSm = _heldRotFixedWorld;
        }
        else if (filterHeldRotation)
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

        t.SetPositionAndRotation(_heldPosSm, _heldRotSm);
    }

    void TryGrab()
    {
        if (grabAnchor == null)
        {
            if (logDebug) Debug.Log("[Grabber] No grabAnchor.");
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
            if (logDebug) Debug.Log("[Grabber] No candidate in range.");
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
            if (logDebug) Debug.Log("[Grabber] No valid rigidbody candidate.");
            return;
        }

        // distance gate: only attach if already close enough
        float attachD = Mathf.Max(0f, attachDistance);
        if (attachD > 0f && bestDist > attachD * attachD)
        {
            if (logDebug) Debug.Log("[Grabber] Candidate too far for attach.");
            return;
        }

        Rigidbody body = bestCol.attachedRigidbody;
        if (body == null) return;

        _heldBody = body;
        _heldOriginalParent = body.transform.parent;

        // freeze physics
        _heldBody.isKinematic = true;
        _heldBody.velocity = Vector3.zero;
        _heldBody.angularVelocity = Vector3.zero;

        // keep world pose (no snap)
        body.transform.SetParent(grabAnchor, true);

        // capture offset relative to anchor so the object follows without snapping
        _heldLocalPosToAnchor = grabAnchor.InverseTransformPoint(body.transform.position);
        _heldLocalRotToAnchor = Quaternion.Inverse(grabAnchor.rotation) * body.transform.rotation;

        // capture fixed rotation at grab-start (world rotation)
        _heldRotFixedWorld = body.transform.rotation;

        // init filter state from current pose (no jump)
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

        if (logDebug) Debug.Log("[Grabber] Grabbed " + body.name);
    }

    void TryRelease()
    {
        if (_heldBody == null) return;

        if (logDebug) Debug.Log("[Grabber] Released " + _heldBody.name);

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
    }

    // Allows TaskManager or other controllers to force a release at any time.
    public void ForceRelease()
    {
        TryRelease();
    }

    // Allows TaskManager to toggle rotation-follow at runtime (e.g., translation task vs rotation task).
    public void SetFollowHeldRotation(bool value)
    {
        followHeldRotation = value;

        // If rotation-follow is turned off while holding an object, lock rotation to current pose immediately.
        if (_heldBody != null && !followHeldRotation)
        {
            _heldRotFixedWorld = _heldBody.transform.rotation;
            _heldRotSm = _heldRotFixedWorld;
        }
    }

    void OnDrawGizmosSelected()
    {
        if (grabAnchor == null) return;
        Gizmos.DrawWireSphere(grabAnchor.position, grabRadius);
        Gizmos.color = new Color(1f, 0f, 0f, 0.35f);
        Gizmos.DrawWireSphere(grabAnchor.position, attachDistance);
    }
}
