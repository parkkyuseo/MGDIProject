using UnityEngine;

public class MicroScalingController_Slider : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private ToolScalingTaskManager_OverlayCore scalingTask;

    [Header("Mapping")]
    [Tooltip("Exponential scale rate. Higher = faster scaling.")]
    [SerializeField] private float scaleGainPerSec = 1.6f;

    [Tooltip("Invert scaling direction (swap grow/shrink for the same slide direction).")]
    [SerializeField] private bool invertScale = false;

    [Header("Precision curve (gamma)")]
    [Tooltip("Gamma for scale precision. >1 compresses small inputs for finer control.")]
    [SerializeField] private float scalePrecisionGamma = 2.0f;

    [Header("Two-stage speed (auto precision)")]
    [Tooltip("If |v| is below this, apply precisionSpeedScale for finer motion.")]
    [SerializeField] private float precisionBandAbs = 0.40f;

    [Tooltip("Speed multiplier when in the precision band.")]
    [Range(0.05f, 1f)]
    [SerializeField] private float precisionSpeedScale = 0.35f;

    [Header("Gate: stop scaling when ThumbOnIndex is OFF")]
    [SerializeField] private bool stopScalingWhenThumbOff = true;

    [Header("Reattach smoothing (OFF -> ON)")]
    [Tooltip("Ignore scaling for this duration after ThumbOnIndex becomes ON (sec).")]
    [SerializeField] private float reattachSuppressSec = 0.08f;

    [Tooltip("After suppressing, ramp scaling strength from 0->1 over this duration (sec).")]
    [SerializeField] private float reattachRampSec = 0.12f;

    [Tooltip("If true, uses a smooth ramp after suppress window.")]
    [SerializeField] private bool useReattachRamp = true;

    [Header("Clamp (factor)")]
    [SerializeField] private float minFactor = 0.60f;
    [SerializeField] private float maxFactor = 1.80f;

    private float _factor = 1f;
    private bool _baselineCaptured = false;

    private bool _prevThumbOn = true;
    private float _suppressUntil = -999f;
    private float _rampStartTime = -999f;

    void Update()
    {
        if (input == null || scalingTask == null) return;

        if (!scalingTask.IsTrialRunning)
        {
            scalingTask.SetExternalDriving(false);
            _baselineCaptured = false;
            _prevThumbOn = true;
            return;
        }

        // Task-level gating (holding / active tool match, etc.)
        if (!scalingTask.CanDriveNow())
        {
            scalingTask.SetExternalDriving(false);
            _baselineCaptured = false;
            return;
        }

        // Capture baseline when entering scaling trial
        if (!_baselineCaptured)
        {
            _factor = 1f;
            _baselineCaptured = true;

            _prevThumbOn = input.Debug_thumbOnIndex;
            _suppressUntil = -999f;
            _rampStartTime = -999f;

            scalingTask.ApplyScaleFactor(_factor);
        }

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        // Gate by ThumbOnIndex stable state
        bool thumbOn = input.Debug_thumbOnIndex;

        if (stopScalingWhenThumbOff)
        {
            if (!thumbOn)
            {
                scalingTask.SetExternalDriving(false);
                _prevThumbOn = false;
                return;
            }

            // OFF -> ON edge: suppress then ramp
            if (!_prevThumbOn && thumbOn)
            {
                _suppressUntil = Time.time + Mathf.Max(0f, reattachSuppressSec);
                _rampStartTime = _suppressUntil;
            }
        }

        _prevThumbOn = thumbOn;

        // Suppress scaling immediately after reattach
        if (stopScalingWhenThumbOff && Time.time < _suppressUntil)
        {
            scalingTask.SetExternalDriving(false);
            return;
        }

        // Slide drives scaling
        float vRaw = input.AxisValue; // [-1..1]
        if (invertScale) vRaw = -vRaw;

        float v = ApplyPrecisionCurve(vRaw, scalePrecisionGamma);

        float gain = scaleGainPerSec;

        // Two-stage gain scaling near center
        if (Mathf.Abs(v) <= Mathf.Max(0f, precisionBandAbs))
            gain *= Mathf.Clamp(precisionSpeedScale, 0.05f, 1f);

        // Reattach ramp
        float reattachFactor = 1f;
        if (stopScalingWhenThumbOff && useReattachRamp && reattachRampSec > 1e-4f)
        {
            float t = (Time.time - _rampStartTime) / reattachRampSec;
            t = Mathf.Clamp01(t);
            reattachFactor = t * t * (3f - 2f * t);
        }

        v *= reattachFactor;
        if (Mathf.Abs(v) < 1e-5f)
        {
            scalingTask.SetExternalDriving(false);
            return;
        }

        scalingTask.SetExternalDriving(true);

        // Exponential integration in factor space
        _factor *= Mathf.Exp(gain * v * dt);
        _factor = Mathf.Clamp(_factor, minFactor, maxFactor);

        scalingTask.ApplyScaleFactor(_factor);
    }

    static float ApplyPrecisionCurve(float v, float gamma)
    {
        float a = Mathf.Abs(v);
        float g = Mathf.Max(1e-3f, gamma);

        if (Mathf.Abs(g - 1f) > 1e-3f)
            a = Mathf.Pow(a, g);

        return Mathf.Sign(v) * a;
    }
}
