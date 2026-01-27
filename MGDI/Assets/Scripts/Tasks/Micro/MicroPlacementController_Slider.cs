using UnityEngine;

public class MicroPlacementController_Slider : MonoBehaviour
{
    public enum ControlledAxis
    {
        X = 0,
        Y = 1,
        Z = 2
    }

    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private LegoPlacementTaskManager placementTask;
    [SerializeField] private Transform blockRoot;

    [Header("Mapping")]
    [SerializeField] private ControlledAxis controlledAxis = ControlledAxis.X;

    [Tooltip("Base speed for slide (X/Z) in meters/sec.")]
    [SerializeField] private float slideSpeedMetersPerSec = 0.10f;

    [Tooltip("Base speed for twist->Y in meters/sec.")]
    [SerializeField] private float ySpeedMetersPerSec = 0.06f;

    [Tooltip("If true, axes are camera-relative. If false, world axes.")]
    [SerializeField] private bool useCameraFrame = true;

    [Header("Auto switch: Slide (X/Z) <-> Twist (Y)")]
    [SerializeField] private bool autoSwitchToYByTwist = true;

    [Tooltip("Enter Y-control when |AxisY| >= this.")]
    [SerializeField] private float yEnterAbs = 0.30f;

    [Tooltip("Exit Y-control when |AxisY| <= this.")]
    [SerializeField] private float yExitAbs = 0.20f;

    [Tooltip("Time (sec) that the condition must hold before switching.")]
    [SerializeField] private float switchConfirmSec = 0.06f;

    [Tooltip("If true, slide direction follows input.Mode (XY->X, Z->Z) when controlling slide.")]
    [SerializeField] private bool followInputModeForSlide = true;

    [Header("Patch 2: Precision curve (gamma)")]
    [Tooltip("Gamma for slide precision. >1 compresses small inputs for finer control.")]
    [SerializeField] private float slidePrecisionGamma = 2.0f;

    [Tooltip("Gamma for Y precision. >1 compresses small inputs for finer control.")]
    [SerializeField] private float yPrecisionGamma = 2.6f;

    [Header("Patch 3: Two-stage speed (auto precision)")]
    [Tooltip("If |v| is below this, apply precisionSpeedScale for finer motion.")]
    [SerializeField] private float precisionBandAbs = 0.40f;

    [Tooltip("Speed multiplier when in the precision band.")]
    [Range(0.05f, 1f)]
    [SerializeField] private float precisionSpeedScale = 0.35f;

    bool _yMode = false;
    float _enterHeld = 0f;
    float _exitHeld = 0f;

    void Update()
    {
        if (input == null || placementTask == null || blockRoot == null) return;
        if (!placementTask.IsTrialRunning) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        float slide = input.AxisValue; // [-1..1]
        float twist = input.AxisY;     // [-1..1]

        if (autoSwitchToYByTwist)
        {
            UpdateYMode(dt, twist);
        }
        else
        {
            _yMode = (controlledAxis == ControlledAxis.Y);
        }

        float vRaw;
        float vShaped;
        float speed;
        Vector3 axis;

        if (_yMode)
        {
            vRaw = twist;
            vShaped = ApplyPrecisionCurve(vRaw, yPrecisionGamma);
            speed = ySpeedMetersPerSec;
            axis = GetAxisY();
        }
        else
        {
            vRaw = slide;
            vShaped = ApplyPrecisionCurve(vRaw, slidePrecisionGamma);
            speed = slideSpeedMetersPerSec;
            axis = GetSlideAxis();
        }

        // Patch 3: auto precision speed scaling near center
        if (Mathf.Abs(vShaped) <= Mathf.Max(0f, precisionBandAbs))
            speed *= Mathf.Clamp(precisionSpeedScale, 0.05f, 1f);

        if (Mathf.Abs(vShaped) < 1e-5f) return;

        blockRoot.position += axis * (vShaped * speed * dt);
    }

    void UpdateYMode(float dt, float twist)
    {
        float a = Mathf.Abs(twist);

        if (!_yMode)
        {
            if (a >= Mathf.Max(0f, yEnterAbs))
            {
                _enterHeld += dt;
                _exitHeld = 0f;
                if (_enterHeld >= Mathf.Max(0.01f, switchConfirmSec))
                {
                    _yMode = true;
                    _enterHeld = 0f;
                }
            }
            else
            {
                _enterHeld = 0f;
            }
        }
        else
        {
            if (a <= Mathf.Max(0f, yExitAbs))
            {
                _exitHeld += dt;
                _enterHeld = 0f;
                if (_exitHeld >= Mathf.Max(0.01f, switchConfirmSec))
                {
                    _yMode = false;
                    _exitHeld = 0f;
                }
            }
            else
            {
                _exitHeld = 0f;
            }
        }
    }

    static float ApplyPrecisionCurve(float v, float gamma)
    {
        float a = Mathf.Abs(v);
        float g = Mathf.Max(1e-3f, gamma);

        if (Mathf.Abs(g - 1f) > 1e-3f)
            a = Mathf.Pow(a, g);

        return Mathf.Sign(v) * a;
    }

    Vector3 GetAxisY()
    {
        if (useCameraFrame && Camera.main != null) return Camera.main.transform.up;
        return Vector3.up;
    }

    Vector3 GetSlideAxis()
    {
        bool useZ;

        if (followInputModeForSlide)
            useZ = (input.Mode == MicroThumbIndexSliderInput.AxisMode.Z);
        else
            useZ = (controlledAxis == ControlledAxis.Z);

        if (useCameraFrame && Camera.main != null)
        {
            var cam = Camera.main.transform;
            return useZ ? cam.forward : cam.right;
        }

        return useZ ? Vector3.forward : Vector3.right;
    }
}
