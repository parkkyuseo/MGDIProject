using UnityEngine;

public class MicroRotationController_Slider : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private LegoRotationTaskManager rotationTask;
    [SerializeField] private Transform blockRoot;

    [Header("Mapping")]
    [Tooltip("Base yaw speed in deg/sec.")]
    [SerializeField] private float yawSpeedDegPerSec = 120f;

    [Tooltip("Invert yaw direction.")]
    [SerializeField] private bool invertYaw = false;

    [Header("Precision curve (gamma)")]
    [Tooltip("Gamma for yaw precision. >1 compresses small inputs for finer control.")]
    [SerializeField] private float yawPrecisionGamma = 2.0f;

    [Header("Two-stage speed (auto precision)")]
    [Tooltip("If |v| is below this, apply precisionSpeedScale for finer motion.")]
    [SerializeField] private float precisionBandAbs = 0.40f;

    [Tooltip("Speed multiplier when in the precision band.")]
    [Range(0.05f, 1f)]
    [SerializeField] private float precisionSpeedScale = 0.35f;

    [Header("Gate: stop rotation when ThumbOnIndex is OFF")]
    [SerializeField] private bool stopRotationWhenThumbOff = true;

    [Header("Reattach smoothing (OFF -> ON)")]
    [Tooltip("Ignore rotation for this duration after ThumbOnIndex becomes ON (sec).")]
    [SerializeField] private float reattachSuppressSec = 0.08f;

    [Tooltip("After suppressing, ramp rotation strength from 0->1 over this duration (sec).")]
    [SerializeField] private float reattachRampSec = 0.12f;

    [Tooltip("If true, uses a smooth ramp after suppress window.")]
    [SerializeField] private bool useReattachRamp = true;

    // Internal
    private bool _prevThumbOn = true;
    private float _suppressUntil = -999f;
    private float _rampStartTime = -999f;

    void Update()
    {
        if (input == null || rotationTask == null || blockRoot == null) return;
        if (!rotationTask.IsTrialRunning)
        {
            rotationTask.SetExternalDriving(false);
            return;
        }

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        // Gate by ThumbOnIndex stable state
        bool thumbOn = input.Debug_thumbOnIndex;

        if (stopRotationWhenThumbOff)
        {
            if (!thumbOn)
            {
                rotationTask.SetExternalDriving(false);
                _prevThumbOn = false;
                return;
            }

            if (!_prevThumbOn && thumbOn)
            {
                _suppressUntil = Time.time + Mathf.Max(0f, reattachSuppressSec);
                _rampStartTime = _suppressUntil;
            }
        }

        _prevThumbOn = thumbOn;

        if (stopRotationWhenThumbOff && Time.time < _suppressUntil)
        {
            rotationTask.SetExternalDriving(false);
            return;
        }

        // Slide drives yaw
        float vRaw = input.AxisValue; // [-1..1]
        if (invertYaw) vRaw = -vRaw;

        float v = ApplyPrecisionCurve(vRaw, yawPrecisionGamma);

        float speed = yawSpeedDegPerSec;

        // Two-stage speed scaling near center
        if (Mathf.Abs(v) <= Mathf.Max(0f, precisionBandAbs))
            speed *= Mathf.Clamp(precisionSpeedScale, 0.05f, 1f);

        // Reattach ramp
        float reattachFactor = 1f;
        if (stopRotationWhenThumbOff && useReattachRamp && reattachRampSec > 1e-4f)
        {
            float t = (Time.time - _rampStartTime) / reattachRampSec;
            t = Mathf.Clamp01(t);
            reattachFactor = t * t * (3f - 2f * t);
        }

        v *= reattachFactor;

        if (Mathf.Abs(v) < 1e-5f)
        {
            rotationTask.SetExternalDriving(false);
            return;
        }

        rotationTask.SetExternalDriving(true);
        float dyaw = v * speed * dt;

        // Safer than eulerAngles accumulation for wrap cases
        blockRoot.rotation = Quaternion.AngleAxis(dyaw, Vector3.up) * blockRoot.rotation;
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
