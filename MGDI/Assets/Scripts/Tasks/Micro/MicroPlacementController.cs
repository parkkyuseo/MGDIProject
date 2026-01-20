using UnityEngine;

public class MicroPlacementController : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroInputThumbDpadToggle input;
    [SerializeField] private LegoPlacementTaskManager placementTask;
    [SerializeField] private Transform blockRoot;

    [Header("Mapping")]
    [SerializeField] private float xySpeedMetersPerSec = 0.18f;
    [SerializeField] private float zSpeedMetersPerSec = 0.18f;

    [Tooltip("If true, XY uses camera right/up. If false, uses world right/up.")]
    [SerializeField] private bool xyUseCameraFrame = true;

    [Tooltip("If true, Z uses camera forward. If false, uses world forward.")]
    [SerializeField] private bool zUseCameraForward = true;

    void Update()
    {
        if (input == null || placementTask == null || blockRoot == null) return;
        if (!placementTask.IsTrialRunning) return;
        if (!input.IsEngaged) return;

        Vector2 d = input.Dpad;
        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (!input.ZMode)
        {
            Vector3 right = Vector3.right;
            Vector3 up = Vector3.up;

            if (xyUseCameraFrame && Camera.main != null)
            {
                right = Camera.main.transform.right;
                up = Camera.main.transform.up;
            }

            Vector3 delta = (right * d.x + up * d.y) * (xySpeedMetersPerSec * dt);
            blockRoot.position += delta;
        }
        else
        {
            Vector3 forward = Vector3.forward;
            if (zUseCameraForward && Camera.main != null)
                forward = Camera.main.transform.forward;

            Vector3 delta = forward * (d.y * zSpeedMetersPerSec * dt);
            blockRoot.position += delta;
        }
    }
}
