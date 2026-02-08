using UnityEngine;

public class MicroRotationSwipeController : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ProxyHandGrabber grabber;
    [SerializeField] private Transform cameraTransform;

    [Header("Rotation Step")]
    [SerializeField] private float yawDegPerSwipe = 5f;
    [SerializeField] private float pitchDegPerSwipe = 5f;
    [SerializeField] private bool useCameraFrame = true;
    [SerializeField] private bool yawOnly = true;

    void Awake()
    {
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
        if (grabber == null) grabber = FindFirstObjectByType<ProxyHandGrabber>();
        if (cameraTransform == null && Camera.main != null) cameraTransform = Camera.main.transform;
    }

    void Update()
    {
        if (router == null || grabber == null) return;
        if (router.CurrentMode != PhoneInputRouter.Mode.Micro) return;
        if (!grabber.IsHolding) return;

        if (!router.TryConsumeSwipe(out int dir)) return;

        Rigidbody rb = grabber.HeldBody;
        if (rb == null) return;

        Transform t = rb.transform;

        float yaw = 0f;
        float pitch = 0f;

        if (dir == 3) yaw = -yawDegPerSwipe;      // left
        else if (dir == 4) yaw = +yawDegPerSwipe; // right
        else if (!yawOnly)
        {
            if (dir == 1) pitch = -pitchDegPerSwipe;     // up
            else if (dir == 2) pitch = +pitchDegPerSwipe; // down
        }

        if (Mathf.Approximately(yaw, 0f) && Mathf.Approximately(pitch, 0f)) return;

        Vector3 yawAxis = Vector3.up;
        Vector3 pitchAxis = Vector3.right;

        if (useCameraFrame && cameraTransform != null)
        {
            yawAxis = cameraTransform.up;
            pitchAxis = cameraTransform.right;
        }

        Quaternion dq = Quaternion.identity;
        if (!Mathf.Approximately(yaw, 0f))
            dq = Quaternion.AngleAxis(yaw, yawAxis) * dq;

        if (!Mathf.Approximately(pitch, 0f))
            dq = Quaternion.AngleAxis(pitch, pitchAxis) * dq;

        t.rotation = dq * t.rotation;
    }
}
