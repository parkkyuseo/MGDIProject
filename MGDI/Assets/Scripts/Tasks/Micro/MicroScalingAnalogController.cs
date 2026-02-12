using UnityEngine;

public class MicroScalingAnalogController : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ProxyHandGrabber grabber;
    [SerializeField] private ToolScalingTaskManager scalingTask; // NEW: drive task manager, not transform scale

    [Header("Settings")]
    [SerializeField] private float scaleRatePerSec = 1.0f; // factor *= exp(rate * axisY * dt)
    [SerializeField] private float deadzone = 0.08f;

    [Header("Factor clamp (relative)")]
    [SerializeField] private float minFactor = 0.60f;
    [SerializeField] private float maxFactor = 1.80f;

    [Header("Micro only policy")]
    [Tooltip("If false, scaling will require holding (Micro mode only). If true, controller can scale without holding, but TaskManager gating may still block depending on its settings.")]
    [SerializeField] private bool allowWithoutHoldingInMicro = true;

    private float _factor = 1f;
    private bool _prevActive = false;

    void Awake()
    {
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
        if (grabber == null) grabber = FindFirstObjectByType<ProxyHandGrabber>();
        if (scalingTask == null) scalingTask = FindFirstObjectByType<ToolScalingTaskManager>();
    }

    void Update()
    {
        if (router == null || scalingTask == null) return;
        if (router.CurrentMode != PhoneInputRouter.Mode.Micro) { scalingTask.SetExternalDriving(false); _prevActive = false; return; }
        if (!scalingTask.IsTrialRunning) { scalingTask.SetExternalDriving(false); _prevActive = false; _factor = 1f; return; }

        if (!router.AxisActive)
        {
            scalingTask.SetExternalDriving(false);
            _prevActive = false;
            return;
        }

        Vector2 a = router.Axis;
        if (Mathf.Abs(a.y) < deadzone)
        {
            scalingTask.SetExternalDriving(false);
            _prevActive = false;
            return;
        }

        // holding gate (Micro only)
        if (!allowWithoutHoldingInMicro)
        {
            if (grabber == null || !grabber.IsHolding)
            {
                scalingTask.SetExternalDriving(false);
                _prevActive = false;
                return;
            }
        }

        float dt = Mathf.Max(Time.unscaledDeltaTime, 1e-4f);

        // On first active frame, start from current cmd (or 1.0). Here we reset to 1 for predictability.
        if (!_prevActive)
        {
            _factor = 1f;
            scalingTask.ApplyScaleFactor(_factor);
            _prevActive = true;
        }

        // factor *= exp(rate * y * dt)
        _factor *= Mathf.Exp(scaleRatePerSec * a.y * dt);
        _factor = Mathf.Clamp(_factor, minFactor, maxFactor);

        scalingTask.SetExternalDriving(true);
        scalingTask.ApplyScaleFactor(_factor);
    }

    void OnDisable()
    {
        if (scalingTask != null) scalingTask.SetExternalDriving(false);
    }
}
