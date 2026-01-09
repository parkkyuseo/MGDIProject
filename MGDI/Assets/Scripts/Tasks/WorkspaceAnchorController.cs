using UnityEngine;

public class WorkspaceAnchorController : MonoBehaviour
{
    public enum FrameBasis
    {
        ViewBased,
        WebcamBased
    }

    public enum HandLocation
    {
        NearHead,
        SideOfBodyLeft,
        SideOfBodyRight
    }

    [System.Serializable]
    public class WorkspaceProfile
    {
        [Tooltip("If false, no position change is applied.")]
        public bool applyPosition = true;

        [Tooltip("If false, no rotation change is applied.")]
        public bool applyRotation = true;

        [Tooltip("Offset in the chosen basis frame (meters). x=right, y=up, z=forward.")]
        public Vector3 offset = new Vector3(0f, 0.05f, 0.60f);

        [Tooltip("Extra yaw added on top of the chosen basis yaw (degrees).")]
        public float yawOffsetDeg = 0f;
    }

    [Header("References")]
    public Transform workspaceAnchor;
    public Transform viewFrame;     // usually Camera.main.transform
    public Transform webcamFrame;   // can be null for now

    [Header("Basis")]
    public FrameBasis basis = FrameBasis.ViewBased;

    [Header("Debug state (set by FlowController)")]
    public HandLocation handLocation = HandLocation.NearHead;

    public void ApplyProfile(WorkspaceProfile profile)
    {
        if (workspaceAnchor == null)
        {
            Debug.LogError("[WorkspaceAnchorController] workspaceAnchor is null.");
            return;
        }

        Transform frame = (basis == FrameBasis.ViewBased) ? viewFrame : webcamFrame;
        if (frame == null)
        {
            Debug.LogError("[WorkspaceAnchorController] basis frame is null (viewFrame/webcamFrame).");
            return;
        }

        // --- Position ---
        if (profile != null && profile.applyPosition)
        {
            Vector3 off = profile.offset;
            workspaceAnchor.position =
                frame.position +
                frame.right * off.x +
                frame.up * off.y +
                frame.forward * off.z;
        }

        // --- Rotation (yaw only) ---
        if (profile != null && profile.applyRotation)
        {
            Vector3 fwd = frame.forward;
            fwd.y = 0f;
            if (fwd.sqrMagnitude > 1e-6f)
            {
                Quaternion baseYaw = Quaternion.LookRotation(fwd.normalized, Vector3.up);
                Quaternion extraYaw = Quaternion.Euler(0f, profile.yawOffsetDeg, 0f);
                workspaceAnchor.rotation = baseYaw * extraYaw;
            }
        }
    }
}
