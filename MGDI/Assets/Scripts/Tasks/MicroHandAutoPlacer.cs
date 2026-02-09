using UnityEngine;

public class MicroHandAutoPlacer : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ProxyHandGrabber grabber;

    [Tooltip("This is the rig source that drives the wrist (your Remote_Wrist).")]
    [SerializeField] private Transform remoteWrist;

    [Tooltip("Optional: use head frame for offset direction.")]
    [SerializeField] private Transform cameraTransform;

    [Header("Placement")]
    [SerializeField] private Vector3 localOffsetFromTool = new Vector3(0f, 0f, -0.06f); // 6cm in front of tool (tool-local)
    [SerializeField] private bool offsetInToolSpace = true;

    [Tooltip("If true, will not move hand when already holding something.")]
    [SerializeField] private bool skipIfHolding = true;

    void Awake()
    {
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
        if (grabber == null) grabber = FindFirstObjectByType<ProxyHandGrabber>();
        if (cameraTransform == null && Camera.main != null) cameraTransform = Camera.main.transform;
    }

    public void PlaceHandNear(Transform tool)
    {
        if (tool == null || remoteWrist == null) return;
        if (router != null && router.CurrentMode != PhoneInputRouter.Mode.Micro) return;
        if (skipIfHolding && grabber != null && grabber.IsHolding) return;

        Vector3 worldOffset;
        if (offsetInToolSpace)
            worldOffset = tool.TransformVector(localOffsetFromTool);
        else
            worldOffset = localOffsetFromTool;

        remoteWrist.position = tool.position + worldOffset;

        // Optional: face roughly like the tool (or keep current)
        // remoteWrist.rotation = tool.rotation;

        // If you want, also clear any internal baseline of rotation-only driver here via SendMessage:
        remoteWrist.SendMessage("RecenterRotation", SendMessageOptions.DontRequireReceiver);
    }
}
