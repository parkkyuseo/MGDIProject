using UnityEngine;

public class MicroScalingAnalogController : MonoBehaviour
{
    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private ProxyHandGrabber grabber;

    [Header("Settings")]
    [SerializeField] private float scaleRatePerSec = 1.0f; // exp(rate * axisY * dt)
    [SerializeField] private float minScale = 0.1f;
    [SerializeField] private float maxScale = 3.0f;
    [SerializeField] private float deadzone = 0.08f;

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
        if (!router.AxisActive) return;

        Vector2 a = router.Axis;
        if (Mathf.Abs(a.y) < deadzone) return;

        float dt = Mathf.Max(Time.unscaledDeltaTime, 1e-4f);

        Rigidbody rb = grabber.HeldBody;
        if (rb == null) return;

        Transform t = rb.transform;

        float factor = Mathf.Exp(scaleRatePerSec * a.y * dt); // axis.y>0 => grow
        Vector3 s = t.localScale * factor;

        float u = Mathf.Clamp(s.x, minScale, maxScale);
        t.localScale = new Vector3(u, u, u);
    }
}
