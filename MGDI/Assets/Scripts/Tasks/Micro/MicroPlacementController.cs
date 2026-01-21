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

    [Header("Fingerpad tuning (XY)")]
    [Tooltip("Multiplier for Y input to balance X/Y sensitivity.")]
    [SerializeField] private float yGain = 1.4f;

    [Tooltip("If true, suppress diagonal drift by keeping only the dominant axis.")]
    [SerializeField] private bool useDominantAxis = true;

    [Tooltip("Margin for dominant-axis selection. Larger = stronger separation (0.2~0.6).")]
    [SerializeField] private float dominantAxisRatio = 0.35f;

    [Tooltip("If true, keep the previous dominant axis in the tie zone to reduce flicker.")]
    [SerializeField] private bool useAxisHysteresis = true;

    enum AxisMode { None = 0, X = 1, Y = 2 }
    AxisMode _prevAxis = AxisMode.None;

    void Update()
    {
        if (input == null || placementTask == null || blockRoot == null) return;
        if (!placementTask.IsTrialRunning) return;
        if (!input.IsEngaged) { _prevAxis = AxisMode.None; return; }

        Vector2 d = input.Dpad;

        // Balance Y vs X in XY mode only
        if (!input.ZMode)
            d.y *= yGain;

        // Dominant-axis separation (XY mode only)
        if (!input.ZMode && useDominantAxis)
        {
            float ax = Mathf.Abs(d.x);
            float ay = Mathf.Abs(d.y);

            // If almost no input, clear axis memory
            if (ax < 1e-4f && ay < 1e-4f)
            {
                _prevAxis = AxisMode.None;
            }
            else
            {
                bool xWins = ax >= ay * (1f + dominantAxisRatio);
                bool yWins = ay >= ax * (1f + dominantAxisRatio);

                if (xWins)
                {
                    d.y = 0f;
                    _prevAxis = AxisMode.X;
                }
                else if (yWins)
                {
                    d.x = 0f;
                    _prevAxis = AxisMode.Y;
                }
                else
                {
                    // Tie zone: keep both OR lock to previous axis
                    if (useAxisHysteresis)
                    {
                        if (_prevAxis == AxisMode.X) d.y = 0f;
                        else if (_prevAxis == AxisMode.Y) d.x = 0f;
                        // else: no previous axis -> keep both for this frame
                    }
                }
            }
        }

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
