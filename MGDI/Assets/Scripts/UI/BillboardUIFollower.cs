using UnityEngine;

public class BillboardUIFollower : MonoBehaviour
{
    [Header("Reference")]
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

    private void Awake()
    {
        if (cameraTransform == null && Camera.main != null)
            cameraTransform = Camera.main.transform;
    }

    private void LateUpdate()
    {
        if (cameraTransform == null) return;

        Vector3 targetPos =
            cameraTransform.position +
            cameraTransform.forward * distance +
            cameraTransform.up * heightOffset +
            cameraTransform.right * rightOffset;

        if (followLerp <= 0f)
            transform.position = targetPos;
        else
        {
            float a = 1f - Mathf.Exp(-followLerp * Time.deltaTime);
            transform.position = Vector3.Lerp(transform.position, targetPos, a);
        }

        if (yawOnly)
        {
            Vector3 lookDir = transform.position - cameraTransform.position;
            lookDir.y = 0f;
            if (lookDir.sqrMagnitude > 1e-6f)
                transform.rotation = Quaternion.LookRotation(lookDir.normalized, Vector3.up);
        }
        else
        {
            transform.rotation = Quaternion.LookRotation(transform.position - cameraTransform.position, cameraTransform.up);
        }
    }
}
