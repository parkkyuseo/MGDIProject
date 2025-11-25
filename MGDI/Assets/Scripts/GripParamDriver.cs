using UnityEngine;

[RequireComponent(typeof(Animator))]
public class GripParamDriver : MonoBehaviour
{
    public Animator animator;                 // 자동으로 채워집니다
    public string paramName = "Grip01";       // Blend Tree 파라미터명
    public float riseSpeed = 10f;             // 쥘 때 속도(초당 변화량)
    public float fallSpeed = 10f;             // 펼 때 속도
    public float lossTimeout = 0.35f;         // 이 시간 수신 없으면 자동으로 풀기(0)

    // (테스트용) UDP 없이 수동 제어
    public bool useDebugManual = false;
    [Range(0, 1)] public float debugManual = 0f;

    float _target;    // 0..1
    float _value;     // 0..1
    float _lastRx;

    void Awake()
    {
        if (!animator) animator = GetComponent<Animator>();
    }

    void Update()
    {
        // 수동 모드면 디버그 값 사용
        if (useDebugManual) _target = debugManual;

        // 수신 끊김 방지: 일정 시간 패킷 없으면 자동으로 펼치기
        if (!useDebugManual && Time.time - _lastRx > lossTimeout) _target = 0f;

        float sp = (_target > _value) ? riseSpeed : fallSpeed;
        _value = Mathf.MoveTowards(_value, _target, sp * Time.deltaTime);
        animator.SetFloat(paramName, _value);
    }

    // UDP 수신부에서 호출 (0..1, 주먹=1, 평상시=0)
    public void SetGrip01(float v)
    {
        _target = Mathf.Clamp01(v);
        _lastRx = Time.time;
    }
    public void SetFist(bool fist) { SetGrip01(fist ? 1f : 0f); }
}
