using UnityEngine;

/// <summary>
/// Prints reference-hand (RefHand_L) wrist/palm status to DebugHUD each frame.
/// Drag the 'refRoot' = RefHand_L (scene instance).
/// If wrist/palm not assigned, it will try to find "Wrist"/"Palm" under refRoot once.
/// </summary>
public class RefHandHUDProbe : MonoBehaviour
{
    public Transform refRoot;     // RefHand_L (scene instance)
    public Transform refWrist;    // (optional) drag "Wrist" under refRoot
    public Transform refPalm;     // (optional) drag "Palm" under refRoot
    public float printEverySec = 0.10f;

    float _next;
    void TryAutoFind()
    {
        if (!refRoot) return;
        if (!refWrist) refWrist = FindDeep(refRoot, "Wrist");
        if (!refPalm) refPalm = FindDeep(refRoot, "Palm");
    }

    void Update()
    {
        if (Time.unscaledTime < _next) return;
        _next = Time.unscaledTime + printEverySec;

        if (!refRoot)
        {
            DebugHUD.Log("[Ref] refRoot = null");
            return;
        }
        if (!refWrist || !refPalm) TryAutoFind();

        if (!refWrist || !refPalm)
        {
            DebugHUD.Log("[Ref] Wrist/Palm not found under refRoot");
            return;
        }

        // world rotations/positions for quick sanity
        var wRot = refWrist.rotation.eulerAngles;
        var pRot = refPalm.rotation.eulerAngles;
        var wPos = refWrist.position;
        var pPos = refPalm.position;

        DebugHUD.Log(
            $"[Ref] Wrist yaw/pitch/roll=({wRot.y:0}/{wRot.x:0}/{wRot.z:0})  " +
            $"Palm yaw/pitch/roll=({pRot.y:0}/{pRot.x:0}/{pRot.z:0})\n" +
            $"[Ref] Wrist pos=({wPos.x:0.00},{wPos.y:0.00},{wPos.z:0.00})  " +
            $"Palm pos=({pPos.x:0.00},{pPos.y:0.00},{pPos.z:0.00})"
        );
    }

    Transform FindDeep(Transform root, string exact)
    {
        foreach (var t in root.GetComponentsInChildren<Transform>(true))
            if (t.name == exact) return t;
        return null;
    }
}
