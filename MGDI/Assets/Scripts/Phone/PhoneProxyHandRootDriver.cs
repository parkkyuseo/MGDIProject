using UnityEngine;

public class PhoneProxyHandRootDriver : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhonePoseStreamReceiver phoneRx;
    [SerializeField] private Transform handRoot; // Remote_Wrist or HandRoot target
    [SerializeField] private Transform cameraTransform; // default: Camera.main

    [Header("Mapping")]
    [SerializeField] private float positionGain = 1.0f;
    [SerializeField] private bool applyRotation = true;

    [Header("Smoothing")]
    [SerializeField] private float posLerp = 24f;
    [SerializeField] private float rotLerp = 24f;

    [Header("Baseline")]
    [SerializeField] private bool autoRecenterOnFirstPose = true;

    [Header("Position Offset (meters)")]
    [SerializeField] private Vector3 positionOffset = new Vector3(0f, 0f, 0.25f); // 25cm forward (root local)

    [Header("Rotation Offset (fix forward direction)")]
    [SerializeField] private Vector3 rotationOffsetEuler = new Vector3(0f, 180f, 0f);

    [Header("Side→Front Remap (Macro + Side only)")]
    [Tooltip("If true, translation delta is remapped so up/down tends to become forward/back (reduces diagonal feel).")]
    [SerializeField] private bool enableSideToFrontRemap = false;

    [Tooltip("If true, flips the remap direction (use this if forward/back feels inverted).")]
    [SerializeField] private bool invertSideToFront = false;

    [Tooltip("Use camera yaw frame for remap (recommended).")]
    [SerializeField] private bool useCameraYawFrame = true;

    private bool _hasBaseline;
    private Pose _phone0;
    private Pose _root0;

    void Start()
    {
        if (phoneRx == null) phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();
        if (cameraTransform == null && Camera.main != null) cameraTransform = Camera.main.transform;
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

        // Phone translation delta
        Vector3 dp = (phone.position - _phone0.position) * positionGain;

        // Optional: Side->Front remap (translation only)
        if (enableSideToFrontRemap && cameraTransform != null)
        {
            dp = RemapSideToFront(dp, cameraTransform, useCameraYawFrame, invertSideToFront);
        }

        // Phone rotation delta (unchanged)
        Quaternion dq = Quaternion.identity;
        if (applyRotation)
            dq = phone.rotation * Quaternion.Inverse(_phone0.rotation);

        /* Vector3 desiredPos = _root0.position + dp + (_root0.rotation * positionOffset); */
        Quaternion camYaw = Quaternion.identity;
        if (cameraTransform != null)
            camYaw = Quaternion.Euler(0f, cameraTransform.eulerAngles.y, 0f);

        Vector3 offsetWorld =
            camYaw * new Vector3(positionOffset.x, 0f, positionOffset.z) +
            Vector3.up * positionOffset.y;

        Vector3 desiredPos = _root0.position + dp + offsetWorld;

        Quaternion desiredRot = dq * _root0.rotation;
        Quaternion rotOffset = Quaternion.Euler(rotationOffsetEuler);
        desiredRot = desiredRot * rotOffset;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float aPos = 1f - Mathf.Exp(-posLerp * dt);
        float aRot = 1f - Mathf.Exp(-rotLerp * dt);

        handRoot.position = Vector3.Lerp(handRoot.position, desiredPos, aPos);
        if (applyRotation)
            handRoot.rotation = Quaternion.Slerp(handRoot.rotation, desiredRot, aRot);
    }

    /// <summary>
    /// Call this when toggling remap on/off, or when switching Side L/R, to avoid a jump.
    /// </summary>
    public void Recenter()
    {
        if (phoneRx == null || handRoot == null) return;
        if (!phoneRx.HasPhonePose) return;

        _phone0 = phoneRx.LatestPhonePose;
        _root0 = new Pose(handRoot.position, handRoot.rotation);
        _hasBaseline = true;

        Debug.Log("[PhoneProxyHandRootDriver] Recenter baseline captured.");
    }

    public void RecenterInputOnly()
    {
        if (phoneRx == null) return;
        if (!phoneRx.HasPhonePose) return;

        _phone0 = phoneRx.LatestPhonePose;
        _hasBaseline = true;

        Debug.Log("[PhoneProxyHandRootDriver] RecenterInputOnly (dp=0).");
    }

    /// <summary>
    /// StudyFlowController can call this when condition changes.
    /// </summary>
    public void SetSideToFrontRemap(bool enabled, bool invert, bool forceRecenter = false)
    {
        bool changed = (enableSideToFrontRemap != enabled) || (invertSideToFront != invert);
        enableSideToFrontRemap = enabled;
        invertSideToFront = invert;

        if (changed && forceRecenter)
            Recenter();
    }

    private static Vector3 RemapSideToFront(Vector3 dpWorld, Transform cam, bool yawOnly, bool invert)
    {
        // Use camera yaw frame (ignore pitch/roll) for stability
        Quaternion yawRot;
        if (yawOnly)
        {
            float yaw = cam.eulerAngles.y;
            yawRot = Quaternion.Euler(0f, yaw, 0f);
        }
        else
        {
            yawRot = cam.rotation;
        }

        // dp in camera-yaw local
        Vector3 dpCam = Quaternion.Inverse(yawRot) * dpWorld;

        // Rotate around local X so Y becomes Z (Up/Down -> Forward/Back)
        // +90 around X maps (x,y,z) -> (x, -z, y)  => y -> z
        // If direction feels wrong, flip sign with invert.
        float ang = invert ? -90f : 90f;
        Quaternion rotX = Quaternion.AngleAxis(ang, Vector3.right);

        Vector3 dpCam2 = rotX * dpCam;

        // back to world
        return yawRot * dpCam2;
    }
}
