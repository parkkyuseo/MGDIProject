using UnityEngine;

public class MicroScalingController : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroInputThumbDpadToggle input;
    [SerializeField] private ToolScalingTaskManager scalingTask;
    [SerializeField] private PhoneInputRouter phoneRouter;

    [Header("Mapping")]
    [Tooltip("Scale rate (factor per second) in exponential form. Higher = more sensitive.")]
    [SerializeField] private float scaleGainPerSec = 1.8f;

    [Header("Factor clamp (relative)")]
    [SerializeField] private float minFactor = 0.60f;
    [SerializeField] private float maxFactor = 1.80f;

    private float _factor = 1f;
    private bool _prevEngaged = false;

    void Awake()
    {
        if (phoneRouter == null) phoneRouter = FindFirstObjectByType<PhoneInputRouter>();
    }

    void Update()
    {
        if (input == null || scalingTask == null) return;

        // ✅ Micro only gate
        if (phoneRouter != null && phoneRouter.CurrentMode != PhoneInputRouter.Mode.Micro)
        {
            scalingTask.SetExternalDriving(false);
            _prevEngaged = false;
            _factor = 1f;
            return;
        }

        if (!scalingTask.IsTrialRunning)
        {
            scalingTask.SetExternalDriving(false);
            _prevEngaged = false;
            _factor = 1f;
            return;
        }

        bool engaged = input.IsEngaged;

        // driving flag for eval gating (macro in task should not overwrite when micro is driving)
        scalingTask.SetExternalDriving(engaged);

        // On engage edge, continue from current task scale to avoid jump-to-baseline.
        if (engaged && !_prevEngaged)
        {
            float minClamp = ResolveMinFactor();
            float maxClamp = ResolveMaxFactor(minClamp);
            float current = scalingTask.ActiveCurrentFactor;
            if (float.IsNaN(current) || float.IsInfinity(current))
                current = scalingTask.GetScaleFactorCmd();
            if (float.IsNaN(current) || float.IsInfinity(current))
                current = 1f;
            _factor = Mathf.Clamp(current, minClamp, maxClamp);
        }

        _prevEngaged = engaged;

        if (!engaged) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float y = input.Dpad.y;
        if (Mathf.Abs(y) < 1e-6f) return;

        // Exponential integration: factor *= exp(gain * y * dt)
        _factor *= Mathf.Exp(scaleGainPerSec * y * dt);
        float minF = ResolveMinFactor();
        float maxF = ResolveMaxFactor(minF);
        _factor = Mathf.Clamp(_factor, minF, maxF);

        scalingTask.ApplyScaleFactor(_factor);
    }

    private float ResolveMinFactor()
    {
        if (scalingTask != null)
            return Mathf.Max(0.01f, scalingTask.EffectiveMinScaleFactor);
        return Mathf.Max(0.01f, minFactor);
    }

    private float ResolveMaxFactor(float minResolved)
    {
        if (scalingTask != null)
            return Mathf.Max(minResolved, scalingTask.EffectiveMaxScaleFactor);
        return Mathf.Max(minResolved, maxFactor);
    }
}
