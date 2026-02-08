using UnityEngine;

public class MicroPlacementAnalogController : MonoBehaviour
{
    public enum PlaneMode { XY = 0, XZ = 1 }

    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private Transform target; // 예: Remote_Wrist
    [SerializeField] private Transform cameraTransform;

    [Header("Settings")]
    [SerializeField] private PlaneMode planeMode = PlaneMode.XY;
    [SerializeField] private float speedMetersPerSec = 0.12f; // 연속 속도
    [SerializeField] private bool useCameraFrame = true;
    [SerializeField] private float deadzone = 0.08f;

    void Awake()
    {
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
        if (cameraTransform == null && Camera.main != null) cameraTransform = Camera.main.transform;
    }

    void Update()
    {
        if (router == null || target == null) return;
        if (router.CurrentMode != PhoneInputRouter.Mode.Micro) return;

        if (router.TryConsumeModeToggle())
            planeMode = (planeMode == PlaneMode.XY) ? PlaneMode.XZ : PlaneMode.XY;

        if (!router.AxisActive) return;

        Vector2 a = router.Axis;
        if (a.magnitude < deadzone) return;

        float dt = Mathf.Max(Time.unscaledDeltaTime, 1e-4f);

        Vector3 delta;
        if (useCameraFrame && cameraTransform != null)
        {
            Vector3 right = cameraTransform.right;
            Vector3 up = cameraTransform.up;
            Vector3 fwd = cameraTransform.forward;

            if (planeMode == PlaneMode.XY)
                delta = (right * a.x + up * a.y) * (speedMetersPerSec * dt);
            else
                delta = (right * a.x + fwd * a.y) * (speedMetersPerSec * dt);
        }
        else
        {
            if (planeMode == PlaneMode.XY)
                delta = new Vector3(a.x, a.y, 0f) * (speedMetersPerSec * dt);
            else
                delta = new Vector3(a.x, 0f, a.y) * (speedMetersPerSec * dt);
        }

        target.position += delta;
    }
}
