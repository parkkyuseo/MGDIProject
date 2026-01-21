using UnityEngine;

public class MicroPlacementController_Slider : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private LegoPlacementTaskManager placementTask;
    [SerializeField] private Transform blockRoot;

    [Header("Mapping")]
    [SerializeField] private float speedMetersPerSec = 0.18f;

    [Tooltip("If true, axes are camera-relative. If false, world axes.")]
    [SerializeField] private bool useCameraFrame = true;

    void Update()
    {
        if (input == null || placementTask == null || blockRoot == null) return;
        if (!placementTask.IsTrialRunning) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float v = input.AxisValue; // [-1..1], signed

        if (Mathf.Abs(v) < 1e-5f) return;

        Vector3 axis;

        if (useCameraFrame && Camera.main != null)
        {
            var cam = Camera.main.transform;
            axis = (input.Mode == MicroThumbIndexSliderInput.AxisMode.X) ? cam.right :
                   (input.Mode == MicroThumbIndexSliderInput.AxisMode.Y) ? cam.up :
                   cam.forward;
        }
        else
        {
            axis = (input.Mode == MicroThumbIndexSliderInput.AxisMode.X) ? Vector3.right :
                   (input.Mode == MicroThumbIndexSliderInput.AxisMode.Y) ? Vector3.up :
                   Vector3.forward;
        }

        blockRoot.position += axis * (v * speedMetersPerSec * dt);
    }
}
