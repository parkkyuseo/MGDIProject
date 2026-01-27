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
    [SerializeField] private float speedMetersPerSec = 0.18f;

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

    bool _yMode = false;
    float _enterHeld = 0f;
    float _exitHeld = 0f;

    void Update()
    {
        if (input == null || placementTask == null || blockRoot == null) return;
        if (!placementTask.IsTrialRunning) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        float slide = input.AxisValue;
        float twist = input.AxisY;

        if (autoSwitchToYByTwist)
        {
            UpdateYMode(dt, twist);
        }
        else
        {
            _yMode = (controlledAxis == ControlledAxis.Y);
        }

        float v;
        Vector3 axis;

        if (_yMode)
        {
            v = twist;
            axis = GetAxisY();
        }
        else
        {
            v = slide;
            axis = GetSlideAxis();
        }

        if (Mathf.Abs(v) < 1e-5f) return;
        blockRoot.position += axis * (v * speedMetersPerSec * dt);
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

    Vector3 GetAxisY()
    {
        if (useCameraFrame && Camera.main != null) return Camera.main.transform.up;
        return Vector3.up;
    }

    Vector3 GetSlideAxis()
    {
        bool useZ = false;

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
