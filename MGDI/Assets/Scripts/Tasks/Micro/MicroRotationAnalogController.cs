using UnityEngine;

public class MicroRotationAnalogController : MonoBehaviour
{
    private enum SecondaryAxisMode
    {
        Roll = 0,
        Pitch = 1
    }

    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ProxyHandGrabber grabber;
    [SerializeField] private ToolRotationTaskManager rotationTask;
    [SerializeField] private Transform cameraTransform;

    [Header("Settings")]
    [SerializeField] private SecondaryAxisMode secondaryAxisMode = SecondaryAxisMode.Roll;
    [SerializeField] private float yawDegPerSec = 120f;
    [SerializeField] private float rollDegPerSec = 120f;
    [SerializeField] private float pitchDegPerSec = 120f;
    [SerializeField] private bool useCameraFrame = true;
    [SerializeField] private float deadzone = 0.08f;

    [Header("Invert")]
    [SerializeField] private bool invertYaw = false;
    [SerializeField] private bool invertRoll = true;
    [SerializeField] private bool invertPitch = true;

    [Header("Adaptive Gain (Micro analog)")]
    [SerializeField] private bool useAdaptiveGain = true;
    [SerializeField] private float minGain = 0.35f;
    [SerializeField] private float maxGain = 2.0f;
    [SerializeField] private float gainGamma = 1.4f;
    [SerializeField] private float gainLerp = 12f;

    private float _gain = 1f;

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

        if (rotationTask == null && grabber == null) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (rotationTask != null && !rotationTask.IsTrialRunning)
        {
            UpdateAdaptiveGain(Vector2.zero, false, dt);
            rotationTask.SetExternalDriving(false);
            return;
        }

        if (router.TryConsumeModeToggle())
        {
            secondaryAxisMode = (secondaryAxisMode == SecondaryAxisMode.Roll)
                ? SecondaryAxisMode.Pitch
                : SecondaryAxisMode.Roll;
            _gain = 1f;
        }

        if (!router.AxisActive)
        {
            UpdateAdaptiveGain(Vector2.zero, false, dt);
            if (rotationTask != null) rotationTask.SetExternalDriving(false);
            return;
        }

        Vector2 a = router.Axis;
        UpdateAdaptiveGain(a, true, dt);

        if (a.magnitude < deadzone)
        {
            if (rotationTask != null) rotationTask.SetExternalDriving(false);
            return;
        }

        if (invertYaw) a.x = -a.x;
        if (secondaryAxisMode == SecondaryAxisMode.Roll)
        {
            if (invertRoll) a.y = -a.y;
        }
        else
        {
            if (invertPitch) a.y = -a.y;
        }

        float yaw = a.x * (yawDegPerSec * _gain) * dt;
        float secondaryRate = (secondaryAxisMode == SecondaryAxisMode.Roll) ? rollDegPerSec : pitchDegPerSec;
        float secondary = a.y * (secondaryRate * _gain) * dt;

        Transform t = null;
        if (rotationTask != null)
            t = rotationTask.GetMicroRotationTargetTransform();
        else if (grabber != null && grabber.IsHolding && grabber.HeldBody != null)
            t = grabber.HeldBody.transform;

        if (t == null)
        {
            if (rotationTask != null) rotationTask.SetExternalDriving(false);
            return;
        }

        // Signal "driving" to rotation task so evaluation gating works
        if (rotationTask != null) rotationTask.SetExternalDriving(true);

        Vector3 yawAxis = Vector3.up;
        Vector3 rollAxis = Vector3.forward;
        Vector3 pitchAxis = Vector3.right;

        if (useCameraFrame && cameraTransform != null)
        {
            yawAxis = cameraTransform.up;
            rollAxis = cameraTransform.forward;
            pitchAxis = cameraTransform.right;
        }

        Quaternion dq = Quaternion.identity;
        if (!Mathf.Approximately(yaw, 0f)) dq = Quaternion.AngleAxis(yaw, yawAxis) * dq;
        if (!Mathf.Approximately(secondary, 0f))
        {
            Vector3 secondaryAxis = (secondaryAxisMode == SecondaryAxisMode.Roll) ? rollAxis : pitchAxis;
            dq = Quaternion.AngleAxis(secondary, secondaryAxis) * dq;
        }

        t.rotation = dq * t.rotation;
    }

    void OnDisable()
    {
        if (rotationTask != null) rotationTask.SetExternalDriving(false);
    }

    private void UpdateAdaptiveGain(Vector2 axis, bool axisActive, float dt)
    {
        if (!useAdaptiveGain)
        {
            _gain = 1f;
            return;
        }

        float targetGain = 1f;
        if (axisActive)
        {
            float m = Mathf.Clamp01(axis.magnitude);
            float shaped = Mathf.Pow(m, gainGamma);
            float gainMax = Mathf.Max(minGain, maxGain);
            targetGain = Mathf.Lerp(minGain, gainMax, shaped);
        }

        float t = 1f - Mathf.Exp(-gainLerp * dt);
        _gain = Mathf.Lerp(_gain, targetGain, t);
    }
}
