using UnityEngine;

public class PhoneRotationOnlyDriver : MonoBehaviour
{
    [Header("Input")]
    [SerializeField] private PhonePoseStreamReceiver phoneRx;

    [Header("Apply")]
    [SerializeField] private bool yawOnly = false;         // 필요하면 yaw만
    [SerializeField] private Vector3 rotationOffsetEuler = Vector3.zero;

    [Header("Smoothing")]
    [SerializeField] private float rotLerp = 24f;

    private bool _hasBaseline;
    private Quaternion _phoneRot0;
    private Quaternion _targetRot0;

    void Awake()
    {
        if (phoneRx == null) phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();
    }

    void Update()
    {
        if (phoneRx == null || !phoneRx.HasPhonePose) return;

        Quaternion phoneRot = phoneRx.LatestPhonePose.rotation;

        if (!_hasBaseline)
        {
            _phoneRot0 = phoneRot;
            Quaternion off0 = Quaternion.Euler(rotationOffsetEuler);
            _targetRot0 = transform.rotation * Quaternion.Inverse(off0);
            _hasBaseline = true;
        }

        Quaternion dq = phoneRot * Quaternion.Inverse(_phoneRot0);
        if (yawOnly) dq = YawOnly(dq);

        Quaternion desiredRot = dq * _targetRot0;

        Quaternion off = Quaternion.Euler(rotationOffsetEuler);
        desiredRot = desiredRot * off;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float aRot = 1f - Mathf.Exp(-rotLerp * dt);

        transform.rotation = Quaternion.Slerp(transform.rotation, desiredRot, aRot);
    }

    public void RecenterRotation()
    {
        if (phoneRx == null || !phoneRx.HasPhonePose) return;
        _phoneRot0 = phoneRx.LatestPhonePose.rotation;
        Quaternion off = Quaternion.Euler(rotationOffsetEuler);
        _targetRot0 = transform.rotation * Quaternion.Inverse(off);
        _hasBaseline = true;
        DebugHUD.Log("[PhoneRotationOnlyDriver] RecenterRotation");
    }

    private static Quaternion YawOnly(Quaternion q)
    {
        Vector3 fwd = q * Vector3.forward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude < 1e-6f) return Quaternion.identity;
        fwd.Normalize();
        return Quaternion.LookRotation(fwd, Vector3.up);
    }
}
