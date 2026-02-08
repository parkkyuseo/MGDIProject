using UnityEngine;

public class MicroRotationAnalogController : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ProxyHandGrabber grabber;
    [SerializeField] private Transform cameraTransform;

    [Header("Settings")]
    [SerializeField] private float yawDegPerSec = 90f;
    [SerializeField] private float pitchDegPerSec = 90f;
    [SerializeField] private bool useCameraFrame = true;
    [SerializeField] private bool yawOnly = true;
    [SerializeField] private float deadzone = 0.08f;

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
        if (!router.AxisActive) return;

        Vector2 a = router.Axis;
        if (a.magnitude < deadzone) return;

        float dt = Mathf.Max(Time.unscaledDeltaTime, 1e-4f);

        float yaw = a.x * yawDegPerSec * dt;
        float pitch = yawOnly ? 0f : (-a.y * pitchDegPerSec * dt);

        Rigidbody rb = grabber.HeldBody;
        if (rb == null) return;

        Transform t = rb.transform;

        Vector3 yawAxis = Vector3.up;
        Vector3 pitchAxis = Vector3.right;

        if (useCameraFrame && cameraTransform != null)
        {
            yawAxis = cameraTransform.up;
            pitchAxis = cameraTransform.right;
        }

        Quaternion dq = Quaternion.AngleAxis(yaw, yawAxis);
        if (!Mathf.Approximately(pitch, 0f))
            dq = Quaternion.AngleAxis(pitch, pitchAxis) * dq;

        t.rotation = dq * t.rotation;
    }
}
