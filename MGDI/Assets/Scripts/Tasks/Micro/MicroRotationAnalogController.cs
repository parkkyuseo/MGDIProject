using UnityEngine;

public class MicroRotationAnalogController : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ProxyHandGrabber grabber;
    [SerializeField] private Transform cameraTransform;

    [Header("Settings")]
    [SerializeField] private float yawDegPerSec = 120f;   // 좌/우
    [SerializeField] private float rollDegPerSec = 120f;  // 위/아래
    [SerializeField] private bool useCameraFrame = true;
    [SerializeField] private float deadzone = 0.08f;

    [Header("Invert")]
    [SerializeField] private bool invertYaw = false;
    [SerializeField] private bool invertRoll = true; // 보통 위로 드래그=롤이 원하는 방향과 반대라 true가 자주 맞음

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

        if (invertYaw) a.x = -a.x;
        if (invertRoll) a.y = -a.y;

        float dt = Mathf.Max(Time.unscaledDeltaTime, 1e-4f);

        float yaw = a.x * yawDegPerSec * dt;
        float roll = a.y * rollDegPerSec * dt;

        Rigidbody rb = grabber.HeldBody;
        if (rb == null) return;

        Transform t = rb.transform;

        Vector3 yawAxis = Vector3.up;
        Vector3 rollAxis = Vector3.forward;

        if (useCameraFrame && cameraTransform != null)
        {
            yawAxis = cameraTransform.up;
            rollAxis = cameraTransform.forward;
        }

        Quaternion dq = Quaternion.identity;

        if (!Mathf.Approximately(yaw, 0f))
            dq = Quaternion.AngleAxis(yaw, yawAxis) * dq;

        if (!Mathf.Approximately(roll, 0f))
            dq = Quaternion.AngleAxis(roll, rollAxis) * dq;

        t.rotation = dq * t.rotation;
    }
}
