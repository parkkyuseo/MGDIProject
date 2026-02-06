using UnityEngine;

public class PhoneWorldAlignmentManager : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhonePoseReceiver phoneRx;
    [SerializeField] private HoloQrMarkerPoseProvider_OpenXR holoMarker;
    [SerializeField] private Transform target; // PhoneProxyCube (또는 프록시 손목 루트)

    [Header("Smoothing")]
    [SerializeField] private float posLerp = 18f;
    [SerializeField] private float rotLerp = 18f;

    private bool _hasAlign;
    private Pose _worldH_from_worldP;

    void Update()
    {
        if (!_hasAlign) return;
        if (phoneRx == null || target == null) return;
        if (!phoneRx.HasPhonePose) return;

        Pose worldP_phone = phoneRx.LatestPhonePose;
        Pose worldH_phone = Mul(_worldH_from_worldP, worldP_phone);

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float aPos = 1f - Mathf.Exp(-posLerp * dt);
        float aRot = 1f - Mathf.Exp(-rotLerp * dt);

        target.position = Vector3.Lerp(target.position, worldH_phone.position, aPos);
        target.rotation = Quaternion.Slerp(target.rotation, worldH_phone.rotation, aRot);
    }

    [ContextMenu("Calibrate Now")]
    public void CalibrateNow()
    {
        if (phoneRx == null || holoMarker == null)
        {
            Debug.LogWarning("[Align] Missing refs.");
            return;
        }

        if (!holoMarker.MarkerVisible)
        {
            Debug.LogWarning("[Align] HoloLens marker not visible.");
            return;
        }

        if (!phoneRx.HasPhoneMarker)
        {
            Debug.LogWarning("[Align] Phone marker not visible (mvis=false).");
            return;
        }

        Pose worldH_marker = new Pose(holoMarker.MarkerPosition, holoMarker.MarkerRotation);
        Pose worldP_marker = phoneRx.LatestPhoneMarkerPose;

        _worldH_from_worldP = Mul(worldH_marker, Inv(worldP_marker));
        _hasAlign = true;

        Debug.Log("[Align] Calibrated. Using world alignment now.");
    }

    [ContextMenu("Clear Alignment")]
    public void ClearAlignment()
    {
        _hasAlign = false;
        Debug.Log("[Align] Cleared.");
    }

    // Pose math
    private static Pose Mul(Pose a, Pose b)
    {
        return new Pose(a.position + a.rotation * b.position, a.rotation * b.rotation);
    }

    private static Pose Inv(Pose p)
    {
        Quaternion rInv = Quaternion.Inverse(p.rotation);
        return new Pose(rInv * (-p.position), rInv);
    }
}
