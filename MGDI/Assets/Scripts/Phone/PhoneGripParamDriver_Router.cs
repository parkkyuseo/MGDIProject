using UnityEngine;

[RequireComponent(typeof(Animator))]
public class PhoneGripParamDriver_Router : MonoBehaviour
{
    [Header("Input")]
    [SerializeField] private PhoneInputRouter router;

    [Header("Animator")]
    public Animator animator;
    public string paramName = "Grip01";
    public float riseSpeed = 10f;
    public float fallSpeed = 10f;
    public float lossTimeout = 0.35f;

    [Header("Debug (manual override)")]
    public bool useDebugManual = false;
    [Range(0, 1)] public float debugManual = 0f;

    private float _target; // 0..1
    private float _value;  // 0..1
    private float _lastRx;

    void Awake()
    {
        if (!animator) animator = GetComponent<Animator>();
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
    }

    void Update()
    {
        if (useDebugManual)
        {
            _target = Mathf.Clamp01(debugManual);
            _lastRx = Time.unscaledTime;
            Apply();
            return;
        }

        if (router != null)
        {
            _target = router.Grab ? 1f : 0f;
            _lastRx = Time.unscaledTime;
        }

        if (lossTimeout > 0f && (Time.unscaledTime - _lastRx) > lossTimeout)
            _target = 0f;

        Apply();
    }

    private void Apply()
    {
        float sp = (_target > _value) ? riseSpeed : fallSpeed;
        _value = Mathf.MoveTowards(_value, _target, sp * Time.deltaTime);
        animator.SetFloat(paramName, _value);
    }
}
