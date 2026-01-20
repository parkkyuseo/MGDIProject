using UnityEngine;

public class MicroInputThumbDpadToggle : MonoBehaviour
{
    public enum PinchKind
    {
        None = 0,
        ThumbIndex = 1,
        ThumbMiddle = 2
    }

    [Header("References")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("Pinch thresholds (meters)")]
    [SerializeField] private float pinchOnMeters = 0.030f;
    [SerializeField] private float pinchOffMeters = 0.040f;
    [SerializeField] private float winnerMarginMeters = 0.006f;
    [SerializeField] private float pinchCooldownSec = 0.20f;

    [Header("D-pad (thumb)")]
    [SerializeField] private float dpadDeadZone = 0.25f;
    [SerializeField] private float dpadFullScale = 0.85f;
    [SerializeField] private bool useFourWay = false;
    [SerializeField] private bool requireEngagedForDpad = true;

    // Outputs (read-only in inspector: keep as normal properties)
    public bool IsEngaged { get; private set; } = false;
    public bool ZMode { get; private set; } = false;
    public Vector2 Dpad { get; private set; } = Vector2.zero;

    public bool EngageToggledThisFrame { get; private set; } = false;
    public bool ZModeToggledThisFrame { get; private set; } = false;

    // pinch state machine
    private bool _pinching = false;
    private PinchKind _winner = PinchKind.None;
    private float _cooldownUntil = 0f;

    void ResetFrameOutputs()
    {
        Dpad = Vector2.zero;
        EngageToggledThisFrame = false;
        ZModeToggledThisFrame = false;
    }

    void Update()
    {
        ResetFrameOutputs();

        if (Time.time < _cooldownUntil)
        {
            UpdateDpad();
            return;
        }

        if (remoteHand == null || remoteHand.remoteByIndex == null || remoteHand.remoteByIndex.Length < 18)
            return;

        Transform thumbTip = remoteHand.remoteByIndex[4];
        Transform indexTip = remoteHand.remoteByIndex[8];
        Transform middleTip = remoteHand.remoteByIndex[12];

        if (thumbTip == null || indexTip == null || middleTip == null)
            return;

        float dTI = Vector3.Distance(thumbTip.position, indexTip.position);
        float dTM = Vector3.Distance(thumbTip.position, middleTip.position);

        // 1) pinch toggle (winner/lockout)
        if (!_pinching)
        {
            float dMin = Mathf.Min(dTI, dTM);

            if (dMin <= pinchOnMeters)
            {
                float diff = Mathf.Abs(dTI - dTM);
                if (diff >= winnerMarginMeters)
                {
                    _pinching = true;
                    _winner = (dTI < dTM) ? PinchKind.ThumbIndex : PinchKind.ThumbMiddle;
                }
            }
        }
        else
        {
            float dW = (_winner == PinchKind.ThumbIndex) ? dTI : dTM;

            if (dW >= pinchOffMeters)
            {
                _pinching = false;

                if (_winner == PinchKind.ThumbIndex)
                {
                    IsEngaged = !IsEngaged;
                    EngageToggledThisFrame = true;
                }
                else if (_winner == PinchKind.ThumbMiddle)
                {
                    ZMode = !ZMode;
                    ZModeToggledThisFrame = true;
                }

                _winner = PinchKind.None;
                _cooldownUntil = Time.time + Mathf.Max(0f, pinchCooldownSec);
            }
        }

        // 2) D-pad
        UpdateDpad();
    }

    void UpdateDpad()
    {
        if (remoteHand == null || remoteHand.remoteByIndex == null || remoteHand.remoteByIndex.Length < 18)
            return;

        Transform thumbTip = remoteHand.remoteByIndex[4];
        Transform wrist = remoteHand.remoteByIndex[0];
        Transform indexMcp = remoteHand.remoteByIndex[5];
        Transform middleMcp = remoteHand.remoteByIndex[9];
        Transform ringMcp = remoteHand.remoteByIndex[13];
        Transform pinkyMcp = remoteHand.remoteByIndex[17];

        if (thumbTip == null || wrist == null || indexMcp == null || middleMcp == null || ringMcp == null || pinkyMcp == null)
            return;

        if (requireEngagedForDpad && !IsEngaged)
        {
            Dpad = Vector2.zero;
            return;
        }

        Vector3 center = (indexMcp.position + middleMcp.position + ringMcp.position + pinkyMcp.position) * 0.25f;

        Vector3 uAxis = pinkyMcp.position - indexMcp.position;
        if (uAxis.sqrMagnitude < 1e-8f) return;
        uAxis.Normalize();

        Vector3 vAxis = middleMcp.position - wrist.position;
        vAxis = vAxis - Vector3.Dot(vAxis, uAxis) * uAxis;
        if (vAxis.sqrMagnitude < 1e-8f) return;
        vAxis.Normalize();

        Vector3 rel = thumbTip.position - center;

        float palmWidth = Vector3.Distance(indexMcp.position, pinkyMcp.position);
        palmWidth = Mathf.Max(0.03f, palmWidth);

        float u = Vector3.Dot(rel, uAxis) / palmWidth;
        float v = Vector3.Dot(rel, vAxis) / palmWidth;

        Vector2 raw = new Vector2(u, v);

        float mag = raw.magnitude;
        if (mag < dpadDeadZone)
        {
            Dpad = Vector2.zero;
            return;
        }

        float denom = Mathf.Max(dpadDeadZone + 1e-6f, dpadFullScale);
        float t = Mathf.Clamp01((mag - dpadDeadZone) / (denom - dpadDeadZone));
        Vector2 outVec = raw.normalized * t;

        if (useFourWay)
        {
            if (Mathf.Abs(outVec.x) >= Mathf.Abs(outVec.y))
                outVec = new Vector2(Mathf.Sign(outVec.x) * Mathf.Abs(outVec.x), 0f);
            else
                outVec = new Vector2(0f, Mathf.Sign(outVec.y) * Mathf.Abs(outVec.y));
        }

        Dpad = outVec;
    }
}
