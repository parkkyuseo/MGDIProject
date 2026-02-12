using UnityEngine;

public class MicroRotationController : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroInputThumbDpadToggle input;
    [SerializeField] private ToolRotationTaskManager rotationTask;

    [Header("Mapping")]
    [Tooltip("Yaw speed in degrees per second at full Dpad.x = 1.")]
    [SerializeField] private float yawSpeedDegPerSec = 120f;

    void Update()
    {
        if (input == null || rotationTask == null) return;
        if (!rotationTask.IsTrialRunning) return;

        // Driving flag for eval gating
        bool driving = input.IsEngaged;
        rotationTask.SetExternalDriving(driving);

        if (!driving) return;

        Transform tgt = rotationTask.GetMicroRotationTargetTransform();
        if (tgt == null) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float dyaw = input.Dpad.x * yawSpeedDegPerSec * dt;

        // World yaw rotation
        float y = tgt.eulerAngles.y + dyaw;
        tgt.rotation = Quaternion.Euler(0f, y, 0f);
    }
}
