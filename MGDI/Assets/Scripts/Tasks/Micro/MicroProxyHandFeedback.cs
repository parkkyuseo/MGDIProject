using UnityEngine;

public class MicroProxyHandFeedback : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroInputThumbDpadToggle input;

    [Tooltip("Proxy hand visual root (the transform to offset for feedback).")]
    [SerializeField] private Transform proxyHandVisualRoot;

    [Header("Feedback motion")]
    [Tooltip("Max offset range (meters) for the proxy hand visual.")]
    [SerializeField] private float maxOffsetMeters = 0.03f; // 3 cm

    [Tooltip("How fast the proxy hand follows the target offset (higher = snappier).")]
    [SerializeField] private float followSpeed = 12f;

    [Tooltip("If true, Dpad mapping is in camera frame (right/up). If false, world frame.")]
    [SerializeField] private bool useCameraFrame = true;

    [Tooltip("If true, when ZMode is ON, use Dpad.y as camera-forward offset instead of up.")]
    [SerializeField] private bool showZModeAsDepth = true;

    Vector3 _baseLocalPos;
    bool _baseCaptured = false;

    void Start()
    {
        CaptureBase();
    }

    void OnEnable()
    {
        CaptureBase();
    }

    void CaptureBase()
    {
        if (proxyHandVisualRoot == null) return;
        if (_baseCaptured) return;

        _baseLocalPos = proxyHandVisualRoot.localPosition;
        _baseCaptured = true;
    }

    void Update()
    {
        if (input == null || proxyHandVisualRoot == null) return;
        CaptureBase();

        Vector3 targetOffset = Vector3.zero;

        if (input.IsEngaged)
        {
            Vector2 d = input.Dpad;

            Vector3 right = Vector3.right;
            Vector3 up = Vector3.up;
            Vector3 forward = Vector3.forward;

            if (useCameraFrame && Camera.main != null)
            {
                Transform cam = Camera.main.transform;
                right = cam.right;
                up = cam.up;
                forward = cam.forward;
            }

            if (showZModeAsDepth && input.ZMode)
            {
                // depth feedback: use Dpad.y as forward/back offset
                targetOffset = (right * d.x + forward * d.y) * maxOffsetMeters;
            }
            else
            {
                // planar feedback
                targetOffset = (right * d.x + up * d.y) * maxOffsetMeters;
            }

            // clamp just in case
            if (targetOffset.magnitude > maxOffsetMeters)
                targetOffset = targetOffset.normalized * maxOffsetMeters;
        }

        // Smooth follow in LOCAL space
        Vector3 desiredLocal = _baseLocalPos + proxyHandVisualRoot.parent.InverseTransformVector(targetOffset);

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float k = 1f - Mathf.Exp(-followSpeed * dt);

        proxyHandVisualRoot.localPosition = Vector3.Lerp(proxyHandVisualRoot.localPosition, desiredLocal, k);
    }

    [ContextMenu("Reset proxy hand offset now")]
    public void ResetNow()
    {
        if (proxyHandVisualRoot == null) return;
        CaptureBase();
        proxyHandVisualRoot.localPosition = _baseLocalPos;
    }
}
