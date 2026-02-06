using UnityEngine;

public class PhoneWorldAlignmentManager : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhonePoseStreamReceiver phoneRx;
    [SerializeField] private HoloQrMarkerPoseProvider_OpenXR holoMarker;
    [SerializeField] private Transform target; // PhoneProxyCube or proxy root

    [Header("Alignment")]
    [SerializeField] private bool yawOnlyAlignment = true;

    [Tooltip("If true, translation uses dominant axis to suppress diagonals.")]
    [SerializeField] private bool dominantAxisLock = true;

    [Tooltip("Minimum translation magnitude (m) before axis lock engages.")]
    [SerializeField] private float axisLockDeadZone = 0.01f;

    [Header("Smoothing")]
    [SerializeField] private float posLerp = 22f;
    [SerializeField] private float rotLerp = 22f;

    private bool _hasAlign;
    private Pose _worldH_from_worldP;

    // axis lock state
    private int _lockedAxis = -1; // 0=x,1=y,2=z
    private float _lockHoldUntil;

    void Update()
    {
        if (!_hasAlign) return;
        if (phoneRx == null || target == null) return;
        if (!phoneRx.HasPhonePose) return;

        Pose worldP_phone = phoneRx.LatestPhonePose;
        Pose worldH_phone = Mul(_worldH_from_worldP, worldP_phone);

        // Apply dominant axis lock on translation to suppress diagonals
        Vector3 desiredPos = worldH_phone.position;
        Quaternion desiredRot = worldH_phone.rotation;

        if (dominantAxisLock)
            desiredPos = ApplyAxisLock(desiredPos);

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float aPos = 1f - Mathf.Exp(-posLerp * dt);
        float aRot = 1f - Mathf.Exp(-rotLerp * dt);

        target.position = Vector3.Lerp(target.position, desiredPos, aPos);
        target.rotation = Quaternion.Slerp(target.rotation, desiredRot, aRot);
    }

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

        Pose worldH_marker = holoMarker.MarkerPose;
        Pose worldP_marker = phoneRx.LatestPhoneMarkerPose;

        Pose worldH_from_worldP = Mul(worldH_marker, Inv(worldP_marker));

        if (yawOnlyAlignment)
            worldH_from_worldP = MakeYawOnly(worldH_from_worldP);

        _worldH_from_worldP = worldH_from_worldP;
        _hasAlign = true;

        _lockedAxis = -1;
        _lockHoldUntil = 0f;

        Debug.Log("[Align] Calibrated. Using world alignment now.");
    }

    public void ClearAlignment()
    {
        _hasAlign = false;
        _lockedAxis = -1;
        Debug.Log("[Align] Cleared.");
    }

    public void SetDominantAxisLock(bool enabled)
    {
        dominantAxisLock = enabled;
        _lockedAxis = -1;
        Debug.Log($"[Align] dominantAxisLock={dominantAxisLock}");
    }

    public void SetYawOnlyAlignment(bool enabled)
    {
        yawOnlyAlignment = enabled;
        Debug.Log($"[Align] yawOnlyAlignment={yawOnlyAlignment}");
    }

    // ---------- helpers ----------

    private Vector3 ApplyAxisLock(Vector3 desiredWorldPos)
    {
        // Lock is applied on delta from current target position (rate-like behavior)
        Vector3 delta = desiredWorldPos - target.position;

        float mag = delta.magnitude;
        if (mag < axisLockDeadZone)
        {
            _lockedAxis = -1;
            return desiredWorldPos;
        }

        float now = Time.unscaledTime;

        // Hold the same axis briefly to prevent rapid flipping
        if (_lockedAxis >= 0 && now < _lockHoldUntil)
        {
            delta = KeepOnlyAxis(delta, _lockedAxis);
            return target.position + delta;
        }

        // Select dominant axis
        float ax = Mathf.Abs(delta.x);
        float ay = Mathf.Abs(delta.y);
        float az = Mathf.Abs(delta.z);

        int axis = 0;
        float best = ax;
        if (ay > best) { best = ay; axis = 1; }
        if (az > best) { best = az; axis = 2; }

        _lockedAxis = axis;
        _lockHoldUntil = now + 0.12f; // short hysteresis

        delta = KeepOnlyAxis(delta, axis);
        return target.position + delta;
    }

    private static Vector3 KeepOnlyAxis(Vector3 v, int axis)
    {
        if (axis == 0) return new Vector3(v.x, 0f, 0f);
        if (axis == 1) return new Vector3(0f, v.y, 0f);
        return new Vector3(0f, 0f, v.z);
    }

    private static Pose MakeYawOnly(Pose p)
    {
        // Keep translation as-is, but clamp rotation to yaw about world up
        Vector3 fwd = p.rotation * Vector3.forward;
        fwd.y = 0f;

        Quaternion yawRot;
        if (fwd.sqrMagnitude < 1e-6f)
        {
            yawRot = Quaternion.identity;
        }
        else
        {
            fwd.Normalize();
            yawRot = Quaternion.LookRotation(fwd, Vector3.up);
        }

        return new Pose(p.position, yawRot);
    }

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
