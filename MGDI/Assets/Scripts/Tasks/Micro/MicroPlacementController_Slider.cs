using UnityEngine;

public class MicroPlacementController_Slider : MonoBehaviour
{
    public enum ControlledAxis
    {
        X = 0, // slide (uses AxisValue); direction can follow input.Mode
        Y = 1, // twist (AxisY)
        Z = 2  // slide (uses AxisValue); direction can follow input.Mode
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

    [Tooltip("If true and ControlledAxis is X or Z, slide direction follows input.Mode (XY->X, Z->Z).")]
    [SerializeField] private bool followInputModeForSlide = true;

    void Update()
    {
        if (input == null || placementTask == null || blockRoot == null) return;
        if (!placementTask.IsTrialRunning) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        // Driving value
        float v;
        if (controlledAxis == ControlledAxis.Y)
        {
            v = input.AxisY;          // twist
        }
        else
        {
            v = input.AxisValue;      // slide (always available regardless of Mode)
        }

        if (Mathf.Abs(v) < 1e-5f) return;

        // Direction axis
        Vector3 axis = GetAxisDirection();

        blockRoot.position += axis * (v * speedMetersPerSec * dt);
    }

    private Vector3 GetAxisDirection()
    {
        bool isSlide = controlledAxis != ControlledAxis.Y;

        if (useCameraFrame && Camera.main != null)
        {
            Transform cam = Camera.main.transform;

            if (isSlide)
            {
                if (followInputModeForSlide)
                    return (input.Mode == MicroThumbIndexSliderInput.AxisMode.XY) ? cam.right : cam.forward;

                return (controlledAxis == ControlledAxis.X) ? cam.right : cam.forward;
            }

            return cam.up;
        }
        else
        {
            if (isSlide)
            {
                if (followInputModeForSlide)
                    return (input.Mode == MicroThumbIndexSliderInput.AxisMode.XY) ? Vector3.right : Vector3.forward;

                return (controlledAxis == ControlledAxis.X) ? Vector3.right : Vector3.forward;
            }

            return Vector3.up;
        }
    }
}
