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

    [Header("References")]
    public Transform workspaceAnchor;
    public Transform viewFrame;    // usually Camera.main
    public Transform webcamFrame;  // WebcamFrame (Empty)

    [Header("Offsets (meters)")]
    public Vector3 nearHeadOffset = new Vector3(0f, 0.05f, 0.6f);
    public Vector3 sideLeftOffset = new Vector3(-0.25f, 0f, 0.6f);
    public Vector3 sideRightOffset = new Vector3(0.25f, 0f, 0.6f);

    [Header("Settings")]
    public FrameBasis basis = FrameBasis.WebcamBased;
    public HandLocation handLocation = HandLocation.NearHead;

    public void ApplyWorkspace()
    {
        Transform frame = (basis == FrameBasis.ViewBased) ? viewFrame : webcamFrame;
        if (frame == null || workspaceAnchor == null) return;

        Vector3 offset = nearHeadOffset;
        switch (handLocation)
        {
            case HandLocation.SideOfBodyLeft:
                offset = sideLeftOffset;
                break;
            case HandLocation.SideOfBodyRight:
                offset = sideRightOffset;
                break;
        }

        workspaceAnchor.position =
            frame.position +
            frame.right * offset.x +
            frame.up * offset.y +
            frame.forward * offset.z;

        // Yaw-only alignment (keep upright)
        Vector3 fwd = frame.forward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude > 1e-6f)
            workspaceAnchor.rotation = Quaternion.LookRotation(fwd.normalized, Vector3.up);
    }
}
