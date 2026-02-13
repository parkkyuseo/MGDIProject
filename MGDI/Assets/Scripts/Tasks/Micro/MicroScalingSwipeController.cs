#if false
using UnityEngine;

public class MicroScalingSwipeController : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ProxyHandGrabber grabber;

    [Header("Scaling Step")]
    [SerializeField] private float scaleStep = 0.02f; // 2% per swipe
    [SerializeField] private float minScale = 0.1f;
    [SerializeField] private float maxScale = 3.0f;

    void Awake()
    {
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
        if (grabber == null) grabber = FindFirstObjectByType<ProxyHandGrabber>();
    }

    void Update()
    {
        if (router == null || grabber == null) return;
        if (router.CurrentMode != PhoneInputRouter.Mode.Micro) return;
        if (!grabber.IsHolding) return;

        if (!router.TryConsumeSwipe(out int dir)) return;

        // 1 up,2 down
        float factor = 1f;

        if (dir == 1) factor = 1f + scaleStep;
        else if (dir == 2) factor = 1f - scaleStep;
        else return;

        Rigidbody rb = grabber.HeldBody;
        if (rb == null) return;

        Transform t = rb.transform;

        Vector3 s = t.localScale * factor;

        // clamp uniformly
        float u = Mathf.Clamp(s.x, minScale, maxScale);
        s = new Vector3(u, u, u);

        t.localScale = s;
    }
}
#endif