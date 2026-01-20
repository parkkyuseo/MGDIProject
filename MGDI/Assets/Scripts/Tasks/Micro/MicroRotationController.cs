using UnityEngine;

public class MicroRotationController : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroInputThumbDpadToggle input;
    [SerializeField] private LegoRotationTaskManager rotationTask;
    [SerializeField] private Transform blockRoot;

    [Header("Mapping")]
    [Tooltip("Yaw speed in degrees per second at full Dpad.x = 1.")]
    [SerializeField] private float yawSpeedDegPerSec = 120f;

    void Update()
    {
        if (input == null || rotationTask == null || blockRoot == null) return;
        if (!rotationTask.IsTrialRunning) return;
        if (!input.IsEngaged) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float dyaw = input.Dpad.x * yawSpeedDegPerSec * dt;

        float y = blockRoot.eulerAngles.y + dyaw;
        blockRoot.rotation = Quaternion.Euler(0f, y, 0f);
    }
}
