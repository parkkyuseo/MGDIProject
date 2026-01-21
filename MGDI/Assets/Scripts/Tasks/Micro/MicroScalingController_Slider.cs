using UnityEngine;

public class MicroScalingController_Slider : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private LegoScalingTaskManager scalingTask;
    [SerializeField] private Transform blockRoot;

    [Header("Mapping")]
    [Tooltip("Exponential scale rate. Higher = faster scaling.")]
    [SerializeField] private float scaleGainPerSec = 1.6f;

    [Header("Clamp")]
    [SerializeField] private float minFactor = 0.60f;
    [SerializeField] private float maxFactor = 1.80f;

    Vector3 _baseWorldScale = Vector3.one;
    float _factor = 1f;
    bool _baselineCaptured = false;

    void Update()
    {
        if (input == null || scalingTask == null || blockRoot == null) return;
        if (!scalingTask.IsTrialRunning) { _baselineCaptured = false; return; }

        // Capture baseline when entering Scaling task or after mode switch recenter
        if (!_baselineCaptured)
        {
            _baseWorldScale = blockRoot.lossyScale;
            _factor = 1f;
            _baselineCaptured = true;
        }

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float v = input.AxisValue;

        if (Mathf.Abs(v) < 1e-5f) return;

        _factor *= Mathf.Exp(scaleGainPerSec * v * dt);
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
