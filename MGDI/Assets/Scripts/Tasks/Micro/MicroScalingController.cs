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

        // rising edge: reset factor to 1
        if (engaged && !_prevEngaged)
        {
            _factor = 1f;
            // start from baseline
            scalingTask.ApplyScaleFactor(_factor);
        }

        _prevEngaged = engaged;

        if (!engaged) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float y = input.Dpad.y;
        if (Mathf.Abs(y) < 1e-6f) return;

        // Exponential integration: factor *= exp(gain * y * dt)
        _factor *= Mathf.Exp(scaleGainPerSec * y * dt);
        _factor = Mathf.Clamp(_factor, minFactor, maxFactor);

        scalingTask.ApplyScaleFactor(_factor);
    }
}
