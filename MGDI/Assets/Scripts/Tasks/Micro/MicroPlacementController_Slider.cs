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

    void Update()
    {
        if (input == null || placementTask == null || blockRoot == null) return;
        if (!placementTask.IsTrialRunning) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        // Select the driving value from the new API
        float v = 0f;
        switch (controlledAxis)
        {
            case ControlledAxis.X: v = input.AxisX; break;
            case ControlledAxis.Y: v = input.AxisY; break;
            case ControlledAxis.Z: v = input.AxisZ; break;
        }

        if (Mathf.Abs(v) < 1e-5f) return;

        Vector3 axis;

        if (useCameraFrame && Camera.main != null)
        {
            var cam = Camera.main.transform;
            axis = (controlledAxis == ControlledAxis.X) ? cam.right :
                   (controlledAxis == ControlledAxis.Y) ? cam.up :
                   cam.forward;
        }
        else
        {
            axis = (controlledAxis == ControlledAxis.X) ? Vector3.right :
                   (controlledAxis == ControlledAxis.Y) ? Vector3.up :
                   Vector3.forward;
        }

        blockRoot.position += axis * (v * speedMetersPerSec * dt);
    }
}
