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

    [Header("Grip debounce")]
    public float closeHoldSec = 0.08f;   // require Closed for this long before grabbing
    public float openHoldSec = 0.12f;    // require Open for this long before releasing
    public float regrabCooldownSec = 0.10f; // small cooldown after release

    [Header("Collision")]
    public bool disableHeldColliders = true;

    [Header("Debug")]
    public bool logDebug = true;

    Collider[] _overlapBuffer = new Collider[32];

    Rigidbody _heldBody;
    Transform _heldOriginalParent;
    Collider[] _heldColliders;

    // debounce state
    UdpHandReceiver.GripState _lastGripState = UdpHandReceiver.GripState.Unknown;
    float _stateSinceTime = -1f;
    float _lastReleaseTime = -999f;

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
        // record state change time
        if (state != _lastGripState)
        {
            _lastGripState = state;
            _stateSinceTime = Time.unscaledTime;
            if (logDebug) Debug.Log("[Grabber] GripState=" + state);
        }
    }

    void Update()
    {
        // no state yet
        if (_stateSinceTime < 0f) return;

        float heldFor = Time.unscaledTime - _stateSinceTime;

        // grab on sustained Closed
        if (_heldBody == null && _lastGripState == UdpHandReceiver.GripState.Closed)
        {
            if (heldFor >= closeHoldSec && (Time.unscaledTime - _lastReleaseTime) >= regrabCooldownSec)
            {
                TryGrab();
            }
            return;
        }

        // release on sustained Open
        if (_heldBody != null && _lastGripState == UdpHandReceiver.GripState.Open)
        {
            if (heldFor >= openHoldSec)
            {
                TryRelease();
            }
            return;
        }
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

        Rigidbody body = bestCol.attachedRigidbody;
        if (body == null) return;

        _heldBody = body;
        _heldOriginalParent = body.transform.parent;

        // freeze physics
        _heldBody.isKinematic = true;
        _heldBody.velocity = Vector3.zero;
        _heldBody.angularVelocity = Vector3.zero;

        // attach (snap to anchor)
        body.transform.SetParent(grabAnchor, false);
        body.transform.localPosition = Vector3.zero;
        body.transform.localRotation = Quaternion.identity;

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

        _lastReleaseTime = Time.unscaledTime;
    }

    void OnDrawGizmosSelected()
    {
        if (grabAnchor == null) return;
        Gizmos.DrawWireSphere(grabAnchor.position, grabRadius);
    }
}
