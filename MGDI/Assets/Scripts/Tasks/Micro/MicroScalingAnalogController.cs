using UnityEngine;

public class MicroScalingAnalogController : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ToolScalingTaskManager scalingTask; // Drive task manager, not transform scale

    [Header("Settings")]
    [SerializeField] private float scaleRatePerSec = 1.0f; // factor *= exp(rate * axisY * dt)
    [SerializeField] private float deadzone = 0.08f;

    [Header("Factor clamp (relative)")]
    [SerializeField] private float minFactor = 0.60f;
    [SerializeField] private float maxFactor = 1.80f;

    [Header("Adaptive Gain (Micro analog)")]
    [SerializeField] private bool useAdaptiveGain = true;
    [SerializeField] private float minGain = 0.35f;
    [SerializeField] private float maxGain = 2.0f;
    [SerializeField] private float gainGamma = 1.4f;
    [SerializeField] private float gainLerp = 12f;
    [SerializeField] private float maxGainScaling = 1.6f;

    private float _factor = 1f;
    private bool _prevActive = false;
    private float _gain = 1f;

    void Awake()
    {
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
        if (scalingTask == null) scalingTask = FindFirstObjectByType<ToolScalingTaskManager>();
    }

    void Update()
    {
        if (router == null || scalingTask == null) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (router.CurrentMode != PhoneInputRouter.Mode.Micro)
        {
            scalingTask.SetExternalDriving(false);
            _prevActive = false;
            UpdateAdaptiveGain(Vector2.zero, false, dt);
            return;
        }

        if (!scalingTask.IsTrialRunning)
        {
            scalingTask.SetExternalDriving(false);
            _prevActive = false;
            _factor = 1f;
            UpdateAdaptiveGain(Vector2.zero, false, dt);
            return;
        }

        if (!router.AxisActive)
        {
            scalingTask.SetExternalDriving(false);
            _prevActive = false;
            UpdateAdaptiveGain(Vector2.zero, false, dt);
            return;
        }

        Vector2 a = router.Axis;
        UpdateAdaptiveGain(a, true, dt);

        if (Mathf.Abs(a.y) < deadzone)
        {
            scalingTask.SetExternalDriving(false);
            _prevActive = false;
            return;
        }

        if (!scalingTask.CanDriveNow())
        {
            scalingTask.SetExternalDriving(false);
            _prevActive = false;
            return;
        }

        // On first active frame, start from current cmd (or 1.0). Here it resets to 1 for predictability.
        if (!_prevActive)
        {
            _factor = 1f;
            scalingTask.ApplyScaleFactor(_factor);
            _prevActive = true;
        }

        // factor *= exp((rate * gain) * y * dt)
        _factor *= Mathf.Exp((scaleRatePerSec * _gain) * a.y * dt);
        _factor = Mathf.Clamp(_factor, minFactor, maxFactor);

        scalingTask.SetExternalDriving(true);
        scalingTask.ApplyScaleFactor(_factor);
    }

    void OnDisable()
    {
        if (scalingTask != null) scalingTask.SetExternalDriving(false);
    }

    private void UpdateAdaptiveGain(Vector2 axis, bool axisActive, float dt)
    {
        if (!useAdaptiveGain)
        {
            _gain = 1f;
            return;
        }

        float gainMax = Mathf.Min(maxGain, maxGainScaling);
        gainMax = Mathf.Max(minGain, gainMax);

        float targetGain = 1f;
        if (axisActive)
        {
            float m = Mathf.Clamp01(axis.magnitude);
            float shaped = Mathf.Pow(m, gainGamma);
            targetGain = Mathf.Lerp(minGain, gainMax, shaped);
        }

        float t = 1f - Mathf.Exp(-gainLerp * dt);
        _gain = Mathf.Lerp(_gain, targetGain, t);
    }
}
