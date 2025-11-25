using UnityEngine;
using UnityEngine.Animations.Rigging;

public class HandRigArmer : MonoBehaviour
{
    [Header("Rigs")]
    public Rig rigWrist;    // Rig_Wrist
    public Rig rigFingers;  // Rig_Fingers

    [Header("Weights")]
    public float armDuration = 0.25f; // 켜질 때 서서히 0→1
    float t = 0f;
    bool armed = false;

    void OnEnable()
    {
        SetWeight(0f);
        t = 0f;
        armed = false;
    }

    void LateUpdate()
    {
        if (!armed) { SetWeight(0f); return; }
        float w = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(t / armDuration));
        SetWeight(w);
        t += Time.deltaTime;
    }

    public void ArmNow()
    {
        if (armed) return;
        armed = true;
        t = 0f;
    }

    void SetWeight(float w)
    {
        if (rigWrist) rigWrist.weight = w;
        if (rigFingers) rigFingers.weight = w;
    }
}
