using UnityEngine;

public class BillboardUIFollower : MonoBehaviour
{
    [Header("Reference")]
    [SerializeField] private Transform anchorTransform;
    [SerializeField] private Transform cameraTransform;

    [Header("Placement")]
    [Tooltip("Distance in front of the camera (meters).")]
    [SerializeField] private float distance = 1.2f;

    [Tooltip("Vertical offset relative to camera (meters). Positive is up.")]
    [SerializeField] private float heightOffset = -0.10f;

    [Tooltip("Horizontal offset relative to camera (meters). Positive is right.")]
    [SerializeField] private float rightOffset = 0.0f;

    [Header("Billboard")]
    [Tooltip("If true, fully face the camera. If false, yaw-only (keeps upright).")]
    [SerializeField] private bool yawOnly = true;

    [Tooltip("Smoothing (higher = snappier). 0 for no smoothing.")]
    [SerializeField] private float followLerp = 12f;

    [Tooltip("If true, place once relative to the camera, then stop following.")]
    [SerializeField] private bool lockAfterInitialPlacement = false;

    private bool hasLockedPose = false;

    private void Awake()
    {
        ResolveCameraTransform();
    }

    private void OnEnable()
    {
        hasLockedPose = false;
        ResolveCameraTransform();
    }

    private void LateUpdate()
    {
        if (anchorTransform != null)
        {
            transform.SetPositionAndRotation(anchorTransform.position, anchorTransform.rotation);
            return;
        }

        if (!ResolveCameraTransform()) return;

        Vector3 targetPos =
            cameraTransform.position +
            cameraTransform.forward * distance +
            cameraTransform.up * heightOffset +
            cameraTransform.right * rightOffset;

        Quaternion targetRot = ComputeTargetRotation(targetPos);

        if (lockAfterInitialPlacement)
        {
            if (hasLockedPose) return;

            transform.SetPositionAndRotation(targetPos, targetRot);
            hasLockedPose = true;
            return;
        }

        if (followLerp <= 0f)
            transform.position = targetPos;
        else
        {
            float a = 1f - Mathf.Exp(-followLerp * Time.deltaTime);
            transform.position = Vector3.Lerp(transform.position, targetPos, a);
        }

        transform.rotation = targetRot;
    }

    private bool ResolveCameraTransform()
    {
        if (cameraTransform == null && Camera.main != null)
            cameraTransform = Camera.main.transform;

        return cameraTransform != null;
    }

    private Quaternion ComputeTargetRotation(Vector3 targetPos)
    {
        if (yawOnly)
        {
            Vector3 lookDir = targetPos - cameraTransform.position;
            lookDir.y = 0f;
            if (lookDir.sqrMagnitude > 1e-6f)
                return Quaternion.LookRotation(lookDir.normalized, Vector3.up);
        }
        else
        {
            Vector3 lookDir = targetPos - cameraTransform.position;
            if (lookDir.sqrMagnitude > 1e-6f)
                return Quaternion.LookRotation(lookDir.normalized, cameraTransform.up);
        }

        return transform.rotation;
    }
}
