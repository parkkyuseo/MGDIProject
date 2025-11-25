using UnityEngine;

/// <summary>
/// Keeps the attached transform in front of the main camera at a fixed distance,
/// facing the user each frame. Designed for world-space UI debug panels.
/// </summary>
public class FollowHeadHUD : MonoBehaviour
{
    public float distance = 1.0f;
    public Vector3 localOffset = new Vector3(0f, -0.15f, 0f); // slight down offset

    void LateUpdate()
    {
        if (Camera.main == null) return;
        var cam = Camera.main.transform;

        Vector3 targetPos = cam.position + cam.forward * distance + cam.TransformVector(localOffset);
        transform.position = targetPos;

        // Face the user with upright orientation
        var lookRot = Quaternion.LookRotation(cam.forward, Vector3.up);
        transform.rotation = lookRot;
    }
}
