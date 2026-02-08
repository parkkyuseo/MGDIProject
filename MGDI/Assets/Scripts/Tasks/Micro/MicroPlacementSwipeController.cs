using UnityEngine;

public class MicroPlacementSwipeController : MonoBehaviour
{
    public enum PlaneMode { XY = 0, XZ = 1 }

    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private Transform target; // 예: Remote_Wrist (권장)
    [SerializeField] private Transform cameraTransform;

    [Header("Settings")]
    [SerializeField] private PlaneMode planeMode = PlaneMode.XY;
    [SerializeField] private float stepMeters = 0.01f; // 1cm per swipe
    [SerializeField] private bool useCameraFrame = true;

    [Header("Debug")]
    [SerializeField] private bool logModeChanges = false;

    void Awake()
    {
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
        if (cameraTransform == null && Camera.main != null) cameraTransform = Camera.main.transform;
    }

    void Update()
    {
        if (router == null || target == null) return;
        if (router.CurrentMode != PhoneInputRouter.Mode.Micro) return;

        // plane toggle (double tap)
        if (router.TryConsumeModeToggle())
        {
            planeMode = (planeMode == PlaneMode.XY) ? PlaneMode.XZ : PlaneMode.XY;
            if (logModeChanges) DebugHUD.Log($"[MicroPlacement] planeMode={planeMode}");
        }

        // swipe move
        if (!router.TryConsumeSwipe(out int dir)) return;

        // 1 up,2 down,3 left,4 right
        Vector2 d2 =
            (dir == 1) ? Vector2.up :
            (dir == 2) ? Vector2.down :
            (dir == 3) ? Vector2.left :
            (dir == 4) ? Vector2.right :
            Vector2.zero;

        if (d2 == Vector2.zero) return;

        Vector3 delta;

        if (useCameraFrame && cameraTransform != null)
        {
            Vector3 right = cameraTransform.right;
            Vector3 up = cameraTransform.up;
            Vector3 fwd = cameraTransform.forward;

            if (planeMode == PlaneMode.XY)
            {
                delta = (right * d2.x + up * d2.y) * stepMeters;
            }
            else // XZ
            {
                delta = (right * d2.x + fwd * d2.y) * stepMeters;
            }
        }
        else
        {
            // world frame fallback: XY uses world right/up, XZ uses world right/forward
            if (planeMode == PlaneMode.XY)
                delta = new Vector3(d2.x, d2.y, 0f) * stepMeters;
            else
                delta = new Vector3(d2.x, 0f, d2.y) * stepMeters;
        }

        target.position += delta;
    }
}
