using UnityEngine;

public class MicroRotationController : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroInputThumbDpadToggle input;
    [SerializeField] private ToolRotationTaskManager rotationTask;
    [SerializeField] private PhoneInputRouter phoneRouter; // 추가

    [Header("Mapping")]
    [SerializeField] private float yawSpeedDegPerSec = 120f;

    void Awake()
    {
        if (phoneRouter == null) phoneRouter = FindFirstObjectByType<PhoneInputRouter>();
    }

    void Update()
    {
        if (input == null || rotationTask == null) return;
        if (!rotationTask.IsTrialRunning) return;

        // ✅ Micro only gate
        if (phoneRouter != null && phoneRouter.CurrentMode != PhoneInputRouter.Mode.Micro)
        {
            rotationTask.SetExternalDriving(false);
            return;
        }

        bool driving = input.IsEngaged;
        rotationTask.SetExternalDriving(driving);
        if (!driving) return;

        Transform tgt = rotationTask.GetMicroRotationTargetTransform();
        if (tgt == null) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float dyaw = input.Dpad.x * yawSpeedDegPerSec * dt;

        float y = tgt.eulerAngles.y + dyaw;
        tgt.rotation = Quaternion.Euler(0f, y, 0f);
    }
}
