using UnityEngine;

[RequireComponent(typeof(Animator))]
public class PhoneGripParamDriver : MonoBehaviour
{
    [Header("Input (Phone)")]
    [SerializeField] private PhonePoseStreamReceiver phoneRx; // LatestGrab 사용

    [Header("Animator")]
    public Animator animator;                 // 자동으로 채워집니다
    public string paramName = "Grip01";       // Blend Tree 파라미터명
    public float riseSpeed = 10f;             // 쥘 때 속도(초당 변화량)
    public float fallSpeed = 10f;             // 펼 때 속도

    [Tooltip("If no packets are received for this long, auto-open. Use 0 to disable.")]
    public float lossTimeout = 0.35f;

    [Header("Debug (manual override)")]
    public bool useDebugManual = false;
    [Range(0, 1)] public float debugManual = 0f;

    private float _target;    // 0..1
    private float _value;     // 0..1
    private float _lastRxTime;

    private bool _lastGrab;
    private bool _grabInit;

    void Awake()
    {
        if (!animator) animator = GetComponent<Animator>();
        if (phoneRx == null) phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();
    }

    void Update()
    {
        // 1) Manual debug overrides everything
        if (useDebugManual)
        {
            _target = Mathf.Clamp01(debugManual);
            _lastRxTime = Time.unscaledTime;
            Apply();
            return;
        }

        // 2) Read phone grab toggle (tap-to-toggle)
        if (phoneRx != null)
        {
            bool g = phoneRx.LatestGrab;

            // consider "rx alive" if we have pose packets (grab is part of same packet)
            if (phoneRx.HasPhonePose)
                _lastRxTime = Time.unscaledTime;

            // update target only when grab changes (avoids unnecessary churn)
            if (!_grabInit || g != _lastGrab)
            {
                _grabInit = true;
                _lastGrab = g;
                _target = g ? 1f : 0f;
            }
        }

        // 3) Loss timeout: if packets stop, auto-open
        if (lossTimeout > 0f && (Time.unscaledTime - _lastRxTime) > lossTimeout)
        {
            _target = 0f;
        }

        Apply();
    }

    private void Apply()
    {
        float sp = (_target > _value) ? riseSpeed : fallSpeed;
        _value = Mathf.MoveTowards(_value, _target, sp * Time.deltaTime);
        animator.SetFloat(paramName, _value);
    }

    // Optional API (kept for compatibility)
    public void SetGrip01(float v)
    {
        _target = Mathf.Clamp01(v);
        _lastRxTime = Time.unscaledTime;
    }

    public void SetFist(bool fist)
    {
        SetGrip01(fist ? 1f : 0f);
    }
}
