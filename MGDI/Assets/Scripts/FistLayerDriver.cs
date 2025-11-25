using UnityEngine;

[DefaultExecutionOrder(10000)] // 다른 처리 끝난 뒤 적용
public class FistLayerDriver : MonoBehaviour
{
    public Animator animator;                // RightHand의 Animator
    public string layerName = "FistLayer";   // 레이어 이름
    [Range(0, 1)] public float weight;        // 현재 Weight(디버그 표시)
    public float riseSpeed = 10f;            // 쥘 때 속도(초당 변화량)
    public float fallSpeed = 10f;            // 펼 때 속도(초당 변화량)
    int layerIndex = -1;
    float target;

    void Awake()
    {
        if (!animator) animator = GetComponent<Animator>();
        layerIndex = animator.GetLayerIndex(layerName);
        if (layerIndex < 0) Debug.LogError($"Layer '{layerName}' not found");
        // 시작은 열림
        animator.SetLayerWeight(layerIndex, 0f);
        weight = 0f;
        target = 0f;
    }

    // UDP/감지 코드에서 0..1 값 넣기 (0은 열림, 1은 완전 주먹)
    public void SetGrip01(float v)
    {
        target = Mathf.Clamp01(v);
    }

    void LateUpdate()
    {
        if (layerIndex < 0) return;
        float speed = (target > weight) ? riseSpeed : fallSpeed;
        weight = Mathf.MoveTowards(weight, target, speed * Time.deltaTime);
        animator.SetLayerWeight(layerIndex, weight);
    }
}
