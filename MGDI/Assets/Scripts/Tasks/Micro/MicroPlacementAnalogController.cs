using UnityEngine;

public class MicroPlacementAnalogController : MonoBehaviour
{
    public enum PlaneMode { XY = 0, XZ = 1 }

    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private Transform target; // Example: Remote_Wrist
    [SerializeField] private Transform cameraTransform;

    [Header("Settings")]
    [SerializeField] private PlaneMode planeMode = PlaneMode.XY;
    [SerializeField] private float speedMetersPerSec = 0.12f; // Continuous speed
    [SerializeField] private bool useCameraFrame = true;
    [SerializeField] private float deadzone = 0.08f;

    [SerializeField] private bool invertAxisY = true;
    [SerializeField] private bool invertAxisX = false;

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
        if (cameraTransform == null && Camera.main != null) cameraTransform = Camera.main.transform;
    }

    void Update()
    {
        if (router == null || target == null) return;
        if (router.CurrentMode != PhoneInputRouter.Mode.Micro) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (router.TryConsumeModeToggle())
            planeMode = (planeMode == PlaneMode.XY) ? PlaneMode.XZ : PlaneMode.XY;

        if (!router.AxisActive)
        {
            UpdateAdaptiveGain(Vector2.zero, false, dt);
            return;
        }

        Vector2 a = router.Axis;
        if (invertAxisX) a.x = -a.x;
        if (invertAxisY) a.y = -a.y;

        UpdateAdaptiveGain(a, true, dt);

        if (a.magnitude < deadzone) return;

        float effectiveSpeed = speedMetersPerSec * _gain;

        Vector3 delta;
        if (useCameraFrame && cameraTransform != null)
        {
            Vector3 right = cameraTransform.right;
            Vector3 up = cameraTransform.up;
            Vector3 fwd = cameraTransform.forward;

            if (planeMode == PlaneMode.XY)
                delta = (right * a.x + up * a.y) * (effectiveSpeed * dt);
            else
                delta = (right * a.x + fwd * a.y) * (effectiveSpeed * dt);
        }
        else
        {
            if (planeMode == PlaneMode.XY)
                delta = new Vector3(a.x, a.y, 0f) * (effectiveSpeed * dt);
            else
                delta = new Vector3(a.x, 0f, a.y) * (effectiveSpeed * dt);
        }

        target.position += delta;
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
