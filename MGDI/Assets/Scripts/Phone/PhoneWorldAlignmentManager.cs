#if false
using UnityEngine;

public class PhoneWorldAlignmentManager : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhonePoseStreamReceiver phoneRx;
    [SerializeField] private HoloQrMarkerPoseProvider_OpenXR holoMarker;
    [SerializeField] private Transform target; // PhoneProxyCube (or proxy root)

    [Header("Before Calibrate (Preview like Phase 0)")]
    [SerializeField] private bool previewBeforeCalibrate = true;

    [Header("Alignment")]
    [SerializeField] private bool yawOnlyAlignment = true;

    [Header("Diagonal suppression (optional)")]
    [SerializeField] private bool dominantAxisLock = false;
    [SerializeField] private float axisLockDeadZone = 0.01f;

    [Header("Smoothing")]
    [SerializeField] private float posLerp = 22f;
    [SerializeField] private float rotLerp = 22f;

    // Alignment state
    private bool _hasAlign;
    private Quaternion _R_align = Quaternion.identity; // only rotation used (yaw-only recommended)

    // Preview (Phase0-like) origin
    private bool _hasPreviewOrigin;
    private Pose _previewPhone0;
    private Pose _previewTarget0;

    // Calibrated delta origin (prevents teleport)
    private Pose _calibPhone0;
    private Pose _calibTarget0;

    // axis lock
    private int _lockedAxis = -1;
    private float _lockHoldUntil;

    void Update()
    {
        if (phoneRx == null || target == null) return;
        if (!phoneRx.HasPhonePose) return;

        Pose phonePose = phoneRx.LatestPhonePose;

        // ---------- PREVIEW MODE (before calibrate) ----------
        if (!_hasAlign)
        {
            if (!previewBeforeCalibrate) return;

            if (!_hasPreviewOrigin)
            {
                _previewPhone0 = phonePose;
                _previewTarget0 = new Pose(target.position, target.rotation);
                _hasPreviewOrigin = true;
            }

            Vector3 dpP = phonePose.position - _previewPhone0.position;
            Quaternion dqP = phonePose.rotation * Quaternion.Inverse(_previewPhone0.rotation);

            Vector3 desiredPos = _previewTarget0.position + dpP;
            Quaternion desiredRot = dqP * _previewTarget0.rotation;

            ApplySmoothed(desiredPos, desiredRot);
            return;
        }

        // ---------- CALIBRATED MODE (no teleport; delta-only in aligned frame) ----------
        Vector3 dpP_cal = phonePose.position - _calibPhone0.position;
        Quaternion dqP_cal = phonePose.rotation * Quaternion.Inverse(_calibPhone0.rotation);

        // map translation delta to Holo world using alignment rotation
        Vector3 dpH = _R_align * dpP_cal;

        // map rotation delta into Holo basis (conjugation)
        Quaternion dqH = _R_align * dqP_cal * Quaternion.Inverse(_R_align);

        Vector3 desiredPos2 = _calibTarget0.position + dpH;
        Quaternion desiredRot2 = dqH * _calibTarget0.rotation;

        if (dominantAxisLock)
            desiredPos2 = ApplyAxisLock(desiredPos2);

        ApplySmoothed(desiredPos2, desiredRot2);
    }

    public void CalibrateNow()
    {
        if (phoneRx == null || holoMarker == null)
        {
            DebugHUD.Log("[Align] Missing refs.");
            return;
        }

        if (!holoMarker.MarkerVisible)
        {
            DebugHUD.Log("[Align] HoloLens marker not visible.");
            return;
        }

        if (!phoneRx.HasPhoneMarker)
        {
            DebugHUD.Log("[Align] Phone marker not visible (mvis=false).");
            return;
        }

        Pose worldH_marker = holoMarker.MarkerPose;
        Pose worldP_marker = phoneRx.LatestPhoneMarkerPose;

        // full alignment rotation
        Quaternion R = worldH_marker.rotation * Quaternion.Inverse(worldP_marker.rotation);

        if (yawOnlyAlignment)
            R = YawOnly(R);

        _R_align = R;
        _hasAlign = true;

        // IMPORTANT: prevent teleport by anchoring to current target pose
        _calibPhone0 = phoneRx.LatestPhonePose;
        _calibTarget0 = new Pose(target.position, target.rotation);

        // reset preview origin so next time (if cleared) preview restarts cleanly
        _hasPreviewOrigin = false;

        _lockedAxis = -1;
        _lockHoldUntil = 0f;

        DebugHUD.Log("[Align] Calibrated (no-teleport, delta-only).");
    }

    public void ClearAlignment()
    {
        _hasAlign = false;
        _hasPreviewOrigin = false;
        _lockedAxis = -1;
        DebugHUD.Log("[Align] Cleared.");
    }

    // ---------- helpers ----------

    private void ApplySmoothed(Vector3 desiredPos, Quaternion desiredRot)
    {
        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float aPos = 1f - Mathf.Exp(-posLerp * dt);
        float aRot = 1f - Mathf.Exp(-rotLerp * dt);

        target.position = Vector3.Lerp(target.position, desiredPos, aPos);
        target.rotation = Quaternion.Slerp(target.rotation, desiredRot, aRot);
    }

    private static Quaternion YawOnly(Quaternion q)
    {
        Vector3 fwd = q * Vector3.forward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude < 1e-6f) return Quaternion.identity;
        fwd.Normalize();
        return Quaternion.LookRotation(fwd, Vector3.up);
    }

    private Vector3 ApplyAxisLock(Vector3 desiredWorldPos)
    {
        Vector3 delta = desiredWorldPos - target.position;
        float mag = delta.magnitude;

        if (mag < axisLockDeadZone)
        {
            _lockedAxis = -1;
            return desiredWorldPos;
        }

        float now = Time.unscaledTime;

        if (_lockedAxis >= 0 && now < _lockHoldUntil)
        {
            delta = KeepOnlyAxis(delta, _lockedAxis);
            return target.position + delta;
        }

        float ax = Mathf.Abs(delta.x);
        float ay = Mathf.Abs(delta.y);
        float az = Mathf.Abs(delta.z);

        int axis = 0;
        float best = ax;
        if (ay > best) { best = ay; axis = 1; }
        if (az > best) { best = az; axis = 2; }

        _lockedAxis = axis;
        _lockHoldUntil = now + 0.12f;

        delta = KeepOnlyAxis(delta, axis);
        return target.position + delta;
    }

    private static Vector3 KeepOnlyAxis(Vector3 v, int axis)
    {
        if (axis == 0) return new Vector3(v.x, 0f, 0f);
        if (axis == 1) return new Vector3(0f, v.y, 0f);
        return new Vector3(0f, 0f, v.z);
    }
    public void RecenterNow()
    {
        if (phoneRx == null || target == null)
        {
            DebugHUD.Log("[Recenter] Missing refs.");
            return;
        }

        if (!phoneRx.HasPhonePose)
        {
            DebugHUD.Log("[Recenter] No phone pose yet.");
            return;
        }

        Pose phonePose = phoneRx.LatestPhonePose;

        // preview 기준점 재설정 (Phase0 느낌 유지)
        _previewPhone0 = phonePose;
        _previewTarget0 = new Pose(target.position, target.rotation);
        _hasPreviewOrigin = true;

        // 만약 이미 calibrated 모드라면, calibrated 기준점도 같이 재설정
        if (_hasAlign)
        {
            _calibPhone0 = phonePose;
            _calibTarget0 = new Pose(target.position, target.rotation);
        }

        _lockedAxis = -1;
        _lockHoldUntil = 0f;

        DebugHUD.Log("[Recenter] Baseline reset (no QR needed).");
    }
}
#endif