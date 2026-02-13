using UnityEngine;

public class AxisPermFrameLogger : MonoBehaviour
{
    [SerializeField] private Transform axisPermFrame;
    [SerializeField] private int logEveryNFrames = 120;

    static float Wrap180(float a)
    {
        a = Mathf.Repeat(a + 180f, 360f) - 180f;
        return a;
    }

    void Update()
    {
        if (axisPermFrame == null) return;
        if (Time.frameCount % Mathf.Max(1, logEveryNFrames) != 0) return;

        Vector3 e = axisPermFrame.rotation.eulerAngles; // world euler
        float pitch = Wrap180(e.x);
        float yaw   = Wrap180(e.y);
        float roll  = Wrap180(e.z);

        DebugHUD.Log($"[AxisPermFrame] pitch={pitch:F2} yaw={yaw:F2} roll={roll:F2}");
    }
}
