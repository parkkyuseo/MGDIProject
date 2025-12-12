using UnityEngine;

public class ProxyHandGrabber : MonoBehaviour
{
    [Header("Hand input")]
    [Tooltip("Optional reference to the UdpHandReceiver that drives this hand (for clarity).")]
    public UdpHandReceiver handReceiver; // not strictly required if using static event

    [Header("Grab anchor on proxy hand")]
    [Tooltip("Where grabbed objects will be attached (for example a child under the wrist or palm).")]
    public Transform grabAnchor;

    [Header("Grab settings")]
    [Tooltip("Radius around the grabAnchor to search for grabbable objects.")]
    public float grabRadius = 0.08f; // 8 cm

    [Tooltip("Layers that contain grabbable objects.")]
    public LayerMask grabbableLayers;

    [Tooltip("If true, only objects with this tag will be considered grabbable (leave empty to ignore).")]
    public string requiredTag = ""; // e.g. "Grabbable"

    [Tooltip("Log basic grab / release events to the console.")]
    public bool logDebug = true;

    // internal state
    Collider[] _overlapBuffer = new Collider[16];
    Rigidbody _heldBody;
    Transform _heldOriginalParent;
    bool _isGrabbing;


    void OnEnable()
    {
        // subscribe to global grip state events
        UdpHandReceiver.OnGripStateChanged += OnGripStateChanged;
    }

    void OnDisable()
    {
        UdpHandReceiver.OnGripStateChanged -= OnGripStateChanged;
    }

    void OnGripStateChanged(UdpHandReceiver.GripState state)
    {
        // if you later have two hands, you may want to filter here
        if (state == UdpHandReceiver.GripState.Closed)
        {
            TryGrab();
        }
        else if (state == UdpHandReceiver.GripState.Open)
        {
            TryRelease();
        }
    }

    void TryGrab()
    {
        if (_isGrabbing) return;
        if (grabAnchor == null)
        {
            if (logDebug) Debug.Log("[ProxyHandGrabber] No grabAnchor assigned.");
            return;
        }

        // find nearby colliders around the grab anchor
        int count = Physics.OverlapSphereNonAlloc(
            grabAnchor.position,
            grabRadius,
            _overlapBuffer,
            grabbableLayers,
            QueryTriggerInteraction.Collide
        );

        if (count == 0)
        {
            if (logDebug) Debug.Log("[ProxyHandGrabber] No grabbable object in range.");
            return;
        }

        // choose the closest valid collider with a rigidbody
        Collider bestCol = null;
        float bestDist = float.MaxValue;

        for (int i = 0; i < count; i++)
        {
            var col = _overlapBuffer[i];
            if (col == null) continue;

            if (!string.IsNullOrEmpty(requiredTag) && !col.CompareTag(requiredTag))
                continue;

            float dist = Vector3.SqrMagnitude(col.ClosestPoint(grabAnchor.position) - grabAnchor.position);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestCol = col;
            }
        }

        if (bestCol == null)
        {
            if (logDebug) Debug.Log("[ProxyHandGrabber] Found colliders but none matched tag filter.");
            return;
        }

        Rigidbody rb = bestCol.attachedRigidbody;
        if (rb == null)
        {
            if (logDebug) Debug.Log("[ProxyHandGrabber] Best collider has no Rigidbody.");
            return;
        }

        // attach the rigidbody to the hand
        _heldBody = rb;
        _heldOriginalParent = rb.transform.parent;
        _isGrabbing = true;

        // make it kinematic so physics does not fight the hand
        _heldBody.isKinematic = true;
        rb.transform.SetParent(grabAnchor, true); // keep world pose

        if (logDebug) Debug.Log("[ProxyHandGrabber] Grabbed " + rb.name);
    }

    void TryRelease()
    {
        if (!_isGrabbing) return;
        if (_heldBody == null)
        {
            _isGrabbing = false;
            return;
        }

        if (logDebug) Debug.Log("[ProxyHandGrabber] Released " + _heldBody.name);

        // detach from hand and restore parent
        Transform t = _heldBody.transform;
        t.SetParent(_heldOriginalParent, true);
        _heldBody.isKinematic = false;

        _heldBody = null;
        _heldOriginalParent = null;
        _isGrabbing = false;
    }

    void OnDrawGizmosSelected()
    {
        if (grabAnchor == null) return;
        Gizmos.color = new Color(0f, 1f, 0f, 0.25f);
        Gizmos.DrawWireSphere(grabAnchor.position, grabRadius);
    }
}
