using UnityEngine;

public class MicroHandFollowBlock : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroInputThumbDpadToggle input;  // engage 상태/모드 참고용
    [SerializeField] private Transform blockRoot;              // 조작 대상
    [SerializeField] private Transform microHandVisualRoot;    // micro 손(복제본)

    [Header("Follow")]
    [SerializeField] private Vector3 offsetCamFrame = new Vector3(0.04f, 0.02f, 0.00f); // (right, up, forward)
    [SerializeField] private float followSpeed = 14f;

    [Header("Optional: Dpad wiggle feedback")]
    [SerializeField] private float dpadMaxOffsetMeters = 0.015f;
    [SerializeField] private bool showZModeAsDepth = true;

    void Update()
    {
        if (input == null || blockRoot == null || microHandVisualRoot == null) return;
        if (!input.IsEngaged) return; // engaged일 때만 "잡은" 느낌으로 붙기 (원하면 false로 바꿔도 됨)

        Transform cam = Camera.main != null ? Camera.main.transform : null;
        Vector3 right = cam != null ? cam.right : Vector3.right;
        Vector3 up = cam != null ? cam.up : Vector3.up;
        Vector3 forward = cam != null ? cam.forward : Vector3.forward;

        Vector3 basePos = blockRoot.position
                        + right   * offsetCamFrame.x
                        + up      * offsetCamFrame.y
                        + forward * offsetCamFrame.z;

        Vector2 d = input.Dpad;
        Vector3 feedback = Vector3.zero;

        if (showZModeAsDepth && input.ZMode)
            feedback = (right * d.x + forward * d.y) * dpadMaxOffsetMeters;
        else
            feedback = (right * d.x + up * d.y) * dpadMaxOffsetMeters;

        Vector3 desired = basePos + feedback;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float k = 1f - Mathf.Exp(-followSpeed * dt);

        microHandVisualRoot.position = Vector3.Lerp(microHandVisualRoot.position, desired, k);
    }
}
