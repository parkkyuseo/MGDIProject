using UnityEngine;

public class PhoneProxyHandRootDriver : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhonePoseStreamReceiver phoneRx;
    [SerializeField] private Transform handRoot; // PhoneDrivenHandRoot (recommended)

    [Header("Mapping")]
    [SerializeField] private float positionGain = 1.0f;
    [SerializeField] private bool applyRotation = true;

    [Header("Smoothing")]
    [SerializeField] private float posLerp = 24f;
    [SerializeField] private float rotLerp = 24f;

    [Header("Baseline")]
    [SerializeField] private bool autoRecenterOnFirstPose = true;

    private bool _hasBaseline;
    private Pose _phone0;
    private Pose _root0;

    void Start()
    {
        if (phoneRx == null) phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();
    }

    void Update()
    {
        if (phoneRx == null || handRoot == null) return;
        if (!phoneRx.HasPhonePose) return;

        Pose phone = phoneRx.LatestPhonePose;

        if (!_hasBaseline)
        {
            if (!autoRecenterOnFirstPose) return;
            Recenter();
        }

        Vector3 dp = (phone.position - _phone0.position) * positionGain;

        Quaternion dq = Quaternion.identity;
        if (applyRotation)
            dq = phone.rotation * Quaternion.Inverse(_phone0.rotation);

        Vector3 desiredPos = _root0.position + dp;
        Quaternion desiredRot = dq * _root0.rotation;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float aPos = 1f - Mathf.Exp(-posLerp * dt);
        float aRot = 1f - Mathf.Exp(-rotLerp * dt);

        handRoot.position = Vector3.Lerp(handRoot.position, desiredPos, aPos);
        if (applyRotation)
            handRoot.rotation = Quaternion.Slerp(handRoot.rotation, desiredRot, aRot);
    }

    public void Recenter()
    {
        if (phoneRx == null || handRoot == null) return;
        if (!phoneRx.HasPhonePose) return;

        _phone0 = phoneRx.LatestPhonePose;
        _root0 = new Pose(handRoot.position, handRoot.rotation);
        _hasBaseline = true;

        Debug.Log("[PhoneProxyHandRootDriver] Recenter baseline captured.");
    }
}
