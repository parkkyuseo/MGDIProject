using UnityEngine;

public class MicroPlacementAnalogController : MonoBehaviour
{
    public enum PlaneMode { XY = 0, XZ = 1 }
    private enum DominantAxis { None, X, Y, Mixed }

    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ToolPlacementTaskManager placementTask;
    [SerializeField] private Transform target; // Example: Remote_Wrist
    [SerializeField] private Transform cameraTransform;

    [Header("Settings")]
    [SerializeField] private PlaneMode planeMode = PlaneMode.XY;
    [SerializeField] private float speedMetersPerSec = 0.12f; // Continuous speed
    [Tooltip("Receiver-side sensitivity multiplier for micro placement movement.")]
    [SerializeField] private float speedMultiplier = 1f;
    [SerializeField] private bool useCameraFrame = true;
    [SerializeField] private float deadzone = 0.08f;

    [SerializeField] private bool invertAxisY = true;
    [SerializeField] private bool invertAxisX = false;

    [Header("Adaptive Gain (Micro analog)")]
    [SerializeField] private bool useAdaptiveGain = true;
    [SerializeField] private float minGain = 0.35f;
    [SerializeField] private float maxGain = 2.0f;
    [SerializeField] private float gainGamma = 1.4f;
    [SerializeField] private float gainLerp = 12f;

    [Header("Dominant Axis (Micro analog)")]
    [SerializeField] private bool useDominantAxisLock = true;
    [SerializeField] private float dominanceRatio = 1.35f;
    [SerializeField] private float smallThreshold = 0.05f;
    [SerializeField] private bool softBiasWhenDiagonal = true;
    [SerializeField] private float diagonalBiasStrength = 0.75f;

    [Header("Debug")]
    [SerializeField] private bool logDominantAxis = false;

    private float _gain = 1f;

    public bool IsDepthMode => planeMode == PlaneMode.XZ;

    void Awake()
    {
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
        if (placementTask == null) placementTask = FindFirstObjectByType<ToolPlacementTaskManager>();
        if (cameraTransform == null && Camera.main != null) cameraTransform = Camera.main.transform;
    }

    void Update()
    {
        if (router == null || target == null) return;
        if (router.CurrentMode != PhoneInputRouter.Mode.Micro) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (placementTask != null && !placementTask.IsTrialRunning)
        {
            UpdateAdaptiveGain(Vector2.zero, false, dt);
            return;
        }

        if (router.TryConsumeModeToggle())
        {
            planeMode = (planeMode == PlaneMode.XY) ? PlaneMode.XZ : PlaneMode.XY;
            ResetModeSensitiveState();
        }

        if (!router.AxisActive)
        {
            UpdateAdaptiveGain(Vector2.zero, false, dt);
            return;
        }

        Vector2 a = router.Axis;
        if (invertAxisX) a.x = -a.x;
        if (invertAxisY) a.y = -a.y;

        DominantAxis dominant;
        Vector2 useAxis = ApplyDominantAxisPolicy(a, out dominant);

        UpdateAdaptiveGain(useAxis, true, dt);

        if (useAxis.magnitude < deadzone) return;

        float effectiveSpeed = speedMetersPerSec * Mathf.Max(0.01f, speedMultiplier) * _gain;

        if (logDominantAxis)
        {
            Debug.Log($"[MicroPlacementAnalog] mode={planeMode} raw={a} used={useAxis} dominant={dominant}");
        }

        Vector3 delta;
        if (useCameraFrame && cameraTransform != null)
        {
            Vector3 right = cameraTransform.right;
            Vector3 up = cameraTransform.up;
            Vector3 fwd = cameraTransform.forward;

            if (planeMode == PlaneMode.XY)
                delta = (right * useAxis.x + up * useAxis.y) * (effectiveSpeed * dt);
            else
                delta = (right * useAxis.x + fwd * useAxis.y) * (effectiveSpeed * dt);
        }
        else
        {
            if (planeMode == PlaneMode.XY)
                delta = new Vector3(useAxis.x, useAxis.y, 0f) * (effectiveSpeed * dt);
            else
                delta = new Vector3(useAxis.x, 0f, useAxis.y) * (effectiveSpeed * dt);
        }

        target.position += delta;
    }

    private Vector2 ApplyDominantAxisPolicy(Vector2 axis, out DominantAxis dominant)
    {
        dominant = DominantAxis.Mixed;
        if (!useDominantAxisLock)
            return axis;

        float absX = Mathf.Abs(axis.x);
        float absY = Mathf.Abs(axis.y);

        if (absX + absY < smallThreshold)
        {
            dominant = DominantAxis.None;
            return Vector2.zero;
        }

        float ratio = Mathf.Max(1f, dominanceRatio);
        if (absX >= absY * ratio)
        {
            dominant = DominantAxis.X;
            return new Vector2(axis.x, 0f);
        }

        if (absY >= absX * ratio)
        {
            dominant = DominantAxis.Y;
            return new Vector2(0f, axis.y);
        }

        dominant = DominantAxis.Mixed;
        if (!softBiasWhenDiagonal)
            return axis;

        float major = Mathf.Max(absX, absY);
        float minor = Mathf.Max(1e-5f, Mathf.Min(absX, absY));
        float majorToMinor = major / minor;
        float t = Mathf.InverseLerp(1f, ratio, majorToMinor);
        t = Mathf.Clamp01(t) * Mathf.Clamp01(diagonalBiasStrength);

        Vector2 hardLocked = absX >= absY ? new Vector2(axis.x, 0f) : new Vector2(0f, axis.y);
        return Vector2.Lerp(axis, hardLocked, t);
    }

    private void UpdateAdaptiveGain(Vector2 axis, bool axisActive, float dt)
    {
        if (!useAdaptiveGain)
        {
            _gain = 1f;
            return;
        }

        float targetGain = 1f;
        if (axisActive)
        {
            float m = Mathf.Clamp01(axis.magnitude);
            float shaped = Mathf.Pow(m, gainGamma);
            float gainMax = Mathf.Max(minGain, maxGain);
            targetGain = Mathf.Lerp(minGain, gainMax, shaped);
        }

        float t = 1f - Mathf.Exp(-gainLerp * dt);
        _gain = Mathf.Lerp(_gain, targetGain, t);
    }

    private void ResetModeSensitiveState()
    {
        _gain = 1f;
    }
}
