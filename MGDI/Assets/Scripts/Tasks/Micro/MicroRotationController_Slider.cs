using UnityEngine;

public class MicroRotationController_Slider : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private LegoRotationTaskManager rotationTask;
    [SerializeField] private Transform blockRoot;

    [Header("Mapping")]
    [SerializeField] private float yawSpeedDegPerSec = 120f;

    void Update()
    {
        if (input == null || rotationTask == null || blockRoot == null) return;
        if (!rotationTask.IsTrialRunning) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float v = input.AxisValue;

        if (Mathf.Abs(v) < 1e-5f) return;

        float dyaw = v * yawSpeedDegPerSec * dt;
        float y = blockRoot.eulerAngles.y + dyaw;
        blockRoot.rotation = Quaternion.Euler(0f, y, 0f);
    }
}
