using UnityEngine;

public class WorkspaceReadyFeedbackAnchorSnap : MonoBehaviour
{
    [SerializeField] private QRWorkspaceLock_OpenXR qrLock;
    [SerializeField] private Transform anchorTransform;
    [SerializeField] private Transform cameraTransform;

    [Header("Snap Placement")]
    [SerializeField] private float distance = 1.0f;
    [SerializeField] private float heightOffset = -0.10f;
    [SerializeField] private float rightOffset = 0.0f;
    [SerializeField] private bool yawOnly = true;
    [SerializeField] private float yawOffsetDeg = 180f;

    private bool _snapped;

    private void Awake()
    {
        ResolveRefs();
    }

    private void OnEnable()
    {
        ResolveRefs();

        if (qrLock != null)
            qrLock.OnWorkspaceReady += HandleWorkspaceReady;

        if (qrLock != null && qrLock.IsWorkspaceReady)
            HandleWorkspaceReady();
    }

    private void OnDisable()
    {
        if (qrLock != null)
            qrLock.OnWorkspaceReady -= HandleWorkspaceReady;
    }

    private void ResolveRefs()
    {
        if (qrLock == null)
            qrLock = FindFirstObjectByType<QRWorkspaceLock_OpenXR>();

        if (cameraTransform == null && Camera.main != null)
            cameraTransform = Camera.main.transform;
    }

    private void HandleWorkspaceReady()
    {
        if (_snapped) return;
        if (anchorTransform == null) return;

        ResolveRefs();
        if (qrLock == null || qrLock.workshopEnvironment == null || cameraTransform == null)
            return;

        Vector3 horizontalForward = Vector3.ProjectOnPlane(cameraTransform.forward, Vector3.up);
        if (horizontalForward.sqrMagnitude < 1e-6f)
            horizontalForward = Vector3.forward;
        else
            horizontalForward.Normalize();

        Vector3 horizontalRight = Vector3.Cross(Vector3.up, horizontalForward).normalized;

        Vector3 desiredWorldPos =
            cameraTransform.position +
            horizontalForward * distance +
            Vector3.up * heightOffset +
            horizontalRight * rightOffset;

        Quaternion desiredWorldRot = yawOnly
            ? Quaternion.Euler(0f, qrLock.workshopEnvironment.rotation.eulerAngles.y + yawOffsetDeg, 0f)
            : qrLock.workshopEnvironment.rotation * Quaternion.Euler(0f, yawOffsetDeg, 0f);

        Transform parent = anchorTransform.parent;
        if (parent != null)
        {
            anchorTransform.localPosition = parent.InverseTransformPoint(desiredWorldPos);
            anchorTransform.localRotation = Quaternion.Inverse(parent.rotation) * desiredWorldRot;
        }
        else
        {
            anchorTransform.SetPositionAndRotation(desiredWorldPos, desiredWorldRot);
        }

        _snapped = true;
    }
}
