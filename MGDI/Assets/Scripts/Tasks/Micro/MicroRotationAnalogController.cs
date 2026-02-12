using UnityEngine;

public class MicroRotationAnalogController : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ProxyHandGrabber grabber;
    [SerializeField] private ToolRotationTaskManager rotationTask;   // NEW (for active tool when not holding)
    [SerializeField] private Transform cameraTransform;

    [Header("Settings")]
    [SerializeField] private float yawDegPerSec = 120f;   // left/right
    [SerializeField] private float rollDegPerSec = 120f;  // up/down
    [SerializeField] private bool useCameraFrame = true;
    [SerializeField] private float deadzone = 0.08f;

    [Header("Invert")]
    [SerializeField] private bool invertYaw = false;
    [SerializeField] private bool invertRoll = true;

    [Header("Micro only policy")]
    [Tooltip("If true, allows rotating the active tool even when not holding (Micro mode only).")]
    [SerializeField] private bool allowWithoutHoldingInMicro = true;

    void Awake()
    {
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
        if (grabber == null) grabber = FindFirstObjectByType<ProxyHandGrabber>();
        if (rotationTask == null) rotationTask = FindFirstObjectByType<ToolRotationTaskManager>();
        if (cameraTransform == null && Camera.main != null) cameraTransform = Camera.main.transform;
    }

    void Update()
    {
        if (router == null) return;
        if (router.CurrentMode != PhoneInputRouter.Mode.Micro) return;

        if (grabber == null) return;
        if (!router.AxisActive) { if (rotationTask != null) rotationTask.SetExternalDriving(false); return; }

        Vector2 a = router.Axis;
        if (a.magnitude < deadzone) { if (rotationTask != null) rotationTask.SetExternalDriving(false); return; }

        if (invertYaw) a.x = -a.x;
        if (invertRoll) a.y = -a.y;

        float dt = Mathf.Max(Time.unscaledDeltaTime, 1e-4f);
        float yaw = a.x * yawDegPerSec * dt;
        float roll = a.y * rollDegPerSec * dt;

        // Signal "driving" to rotation task so evaluation gating works
        if (rotationTask != null) rotationTask.SetExternalDriving(true);

        Transform t = null;

        // Prefer held object if holding
        if (grabber.IsHolding && grabber.HeldBody != null)
        {
            t = grabber.HeldBody.transform;
        }
        else
        {
            // Not holding: allow only in Micro if enabled
            if (!allowWithoutHoldingInMicro) return;
            if (rotationTask == null) return;

            // Requires ToolRotationTaskManager patch that provides this method
            t = rotationTask.GetMicroRotationTargetTransform();
            if (t == null) return;
        }

        Vector3 yawAxis = Vector3.up;
        Vector3 rollAxis = Vector3.forward;

        if (useCameraFrame && cameraTransform != null)
        {
            yawAxis = cameraTransform.up;
            rollAxis = cameraTransform.forward;
        }

        Quaternion dq = Quaternion.identity;
        if (!Mathf.Approximately(yaw, 0f))  dq = Quaternion.AngleAxis(yaw, yawAxis) * dq;
        if (!Mathf.Approximately(roll, 0f)) dq = Quaternion.AngleAxis(roll, rollAxis) * dq;

        t.rotation = dq * t.rotation;
    }

    void OnDisable()
    {
        if (rotationTask != null) rotationTask.SetExternalDriving(false);
    }
}
