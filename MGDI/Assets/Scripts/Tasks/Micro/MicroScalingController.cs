using UnityEngine;

public class MicroScalingController : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroInputThumbDpadToggle input;
    [SerializeField] private LegoScalingTaskManager scalingTask;
    [SerializeField] private Transform blockRoot;

    [Header("Mapping")]
    [Tooltip("Scale rate (factor per second) in exponential form. Higher = more sensitive.")]
    [SerializeField] private float scaleGainPerSec = 1.8f;

    [Header("Factor clamp (relative)")]
    [SerializeField] private float minFactor = 0.60f;
    [SerializeField] private float maxFactor = 1.80f;

    Vector3 _baseWorldScale = Vector3.one;
    float _factor = 1f;
    bool _hasBaseline = false;
    bool _prevEngaged = false;

    void Update()
    {
        if (input == null || scalingTask == null || blockRoot == null) return;
        if (!scalingTask.IsTrialRunning) { _hasBaseline = false; _prevEngaged = false; return; }

        // detect engage rising edge
        if (input.IsEngaged && !_prevEngaged)
        {
            _baseWorldScale = blockRoot.lossyScale;
            _factor = 1f;
            _hasBaseline = true;
        }
        else if (!input.IsEngaged)
        {
            _hasBaseline = false;
        }

        _prevEngaged = input.IsEngaged;

        if (!input.IsEngaged) return;
        if (!_hasBaseline) return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float y = input.Dpad.y;

        if (Mathf.Abs(y) < 1e-6f) return;

        // Exponential integration: factor *= exp(gain * y * dt)
        _factor *= Mathf.Exp(scaleGainPerSec * y * dt);
        _factor = Mathf.Clamp(_factor, minFactor, maxFactor);

        SetWorldScale(blockRoot, _baseWorldScale * _factor);
    }

    static void SetWorldScale(Transform t, Vector3 desiredWorldScale)
    {
        if (t == null) return;

        Vector3 parentLossy = Vector3.one;
        if (t.parent != null)
            parentLossy = t.parent.lossyScale;

        float px = Mathf.Abs(parentLossy.x) < 1e-6f ? 1e-6f : parentLossy.x;
        float py = Mathf.Abs(parentLossy.y) < 1e-6f ? 1e-6f : parentLossy.y;
        float pz = Mathf.Abs(parentLossy.z) < 1e-6f ? 1e-6f : parentLossy.z;

        t.localScale = new Vector3(
            desiredWorldScale.x / px,
            desiredWorldScale.y / py,
            desiredWorldScale.z / pz
        );
    }
}
