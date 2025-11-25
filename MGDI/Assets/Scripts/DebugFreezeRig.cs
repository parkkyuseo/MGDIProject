using UnityEngine;
using UnityEngine.Animations.Rigging;

[DefaultExecutionOrder(10000)]
public class DebugFreezeRig : MonoBehaviour
{
    public Rig rigWrist, rigFingers;
    void LateUpdate()
    {
        if (rigWrist) rigWrist.weight = 0f;
        if (rigFingers) rigFingers.weight = 0f;
    }
}
