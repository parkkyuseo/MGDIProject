using UnityEngine;

public class ProxyHandAnchor : MonoBehaviour
{
    [Tooltip("머리(카메라)에서 앞으로 얼마나 떨어뜨릴지 (미터)")]
    public float forwardDistance = 0.6f; // 60cm 정도
    [Tooltip("머리 기준 위/아래 오프셋 (미터, 양수=위, 음수=아래)")]
    public float upOffset = -0.05f;      // 살짝 아래

    [Tooltip("Offset/Clear & Re-arm을 호출할 RemoteHandRuntime")]
    public RemoteHandRuntime handRuntime;

    [Tooltip("XR이 안정될 때까지 기다릴 프레임 수")]
    public int settleFrames = 30;

    bool _anchored = false;
    int _frameCount = 0;

    void LateUpdate()
    {
        if (_anchored) return;
        var cam = Camera.main;
        if (!cam) return;       // 아직 카메라가 안 잡혔으면 기다림

        _frameCount++;
        if (_frameCount < settleFrames) return; // XR 시스템이 카메라 pose 잡을 때까지 잠깐 대기

        // 1) 카메라 앞/위 기준으로 ProxyHandR 위치 재설정
        Transform t = transform;
        t.position = cam.transform.position
                     + cam.transform.forward * forwardDistance
                     + cam.transform.up * upOffset;

        // 2) 이 새 위치에서 다시 오프셋 캡처
        if (handRuntime != null)
        {
            handRuntime.ContextClearAndRearm();
        }

        Debug.Log($"[ProxyHandAnchor] Anchored ProxyHandR at {t.position}");
        _anchored = true;
    }
}
