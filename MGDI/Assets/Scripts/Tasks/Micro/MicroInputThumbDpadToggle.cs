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
    [SerializeField] private float dpadDeadZone = 0.22f;
    [SerializeField] private float dpadFullScale = 0.85f;
    [SerializeField] private bool useFourWay = false;
    [SerializeField] private bool requireEngagedForDpad = true;

    [Header("D-pad calibration")]
    [SerializeField] private bool calibrateCenterOnEngage = true;
    [SerializeField] private float dpadOutputLerp = 12f;

    [Header("Axis options")]
    [Tooltip("Flip X if the direction feels reversed.")]
    [SerializeField] private bool invertX = false;

    [Tooltip("Flip Y if the direction feels reversed.")]
    [SerializeField] private bool invertY = false;

    // Outputs
    public bool IsEngaged { get; private set; } = false;
    public bool ZMode { get; private set; } = false;
    public Vector2 Dpad { get; private set; } = Vector2.zero;

    public bool EngageToggledThisFrame { get; private set; } = false;
    public bool ZModeToggledThisFrame { get; private set; } = false;

    // pinch state machine
    private bool _pinching = false;
    private PinchKind _winner = PinchKind.None;
    private float _cooldownUntil = 0f;

    // D-pad center calibration + smoothing
    private Vector2 _uvCenter = Vector2.zero;
    private bool _hasCenter = false;
    private Vector2 _dpadSm = Vector2.zero;
    private bool _prevEngaged = false;

    void ResetFrameOutputs()
    {
        EngageToggledThisFrame = false;
        ZModeToggledThisFrame = false;
    }

    void Update()
    {
        ResetFrameOutputs();

        // allow D-pad during cooldown (if engaged), but block new toggles
        bool canToggle = Time.time >= _cooldownUntil;

        if (remoteHand == null || remoteHand.remoteByIndex == null || remoteHand.remoteByIndex.Length < 18)
        {
            HandleCenterCalibrationEdge();
            Dpad = Vector2.zero;
            return;
        }

        Transform thumbTip = remoteHand.remoteByIndex[4];
        Transform indexTip = remoteHand.remoteByIndex[8];
        Transform middleTip = remoteHand.remoteByIndex[12];

        if (thumbTip != null && indexTip != null && middleTip != null && canToggle)
        {
            float dTI = Vector3.Distance(thumbTip.position, indexTip.position);
            float dTM = Vector3.Distance(thumbTip.position, middleTip.position);

            // 1) Pinch toggle (winner/lockout)
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
        }

        HandleCenterCalibrationEdge();
        UpdateDpad();
    }

    void HandleCenterCalibrationEdge()
    {
        if (!calibrateCenterOnEngage)
        {
            _prevEngaged = IsEngaged;
            return;
        }

        if (IsEngaged && !_prevEngaged)
        {
            _hasCenter = false; // capture center on next UpdateDpad()
        }
        else if (!IsEngaged)
        {
            _hasCenter = false;
            _dpadSm = Vector2.zero;
        }

        _prevEngaged = IsEngaged;
    }

    void UpdateDpad()
    {
        if (requireEngagedForDpad && !IsEngaged)
        {
            Dpad = Vector2.zero;
            _dpadSm = Vector2.zero;
            return;
        }

        Transform thumbTip = remoteHand.remoteByIndex[4];
        Transform wrist = remoteHand.remoteByIndex[0];

        Transform indexMcp = remoteHand.remoteByIndex[5];
        Transform middleMcp = remoteHand.remoteByIndex[9];
        Transform ringMcp = remoteHand.remoteByIndex[13];
        Transform pinkyMcp = remoteHand.remoteByIndex[17];

        if (thumbTip == null || wrist == null || indexMcp == null || middleMcp == null || ringMcp == null || pinkyMcp == null)
            return;

        // Palm center near finger-palm junction
        Vector3 centerW = (indexMcp.position + middleMcp.position + ringMcp.position + pinkyMcp.position) * 0.25f;

        // u axis: across palm
        Vector3 uAxis = pinkyMcp.position - indexMcp.position;
        if (uAxis.sqrMagnitude < 1e-8f) return;
        uAxis.Normalize();

        // v axis: from wrist to palm center (more stable than wrist->middleMcp)
        Vector3 vAxis = centerW - wrist.position;
        vAxis = vAxis - Vector3.Dot(vAxis, uAxis) * uAxis;
        if (vAxis.sqrMagnitude < 1e-8f) return;
        vAxis.Normalize();

        Vector3 rel = thumbTip.position - centerW;

        float palmWidth = Mathf.Max(0.03f, Vector3.Distance(indexMcp.position, pinkyMcp.position));
        float palmHeight = Mathf.Max(0.04f, Vector3.Distance(wrist.position, centerW));

        float u = Vector3.Dot(rel, uAxis) / palmWidth;
        float v = Vector3.Dot(rel, vAxis) / palmHeight;

        if (invertX) u = -u;
        if (invertY) v = -v;

        Vector2 uv = new Vector2(u, v);

        if (calibrateCenterOnEngage && !_hasCenter)
        {
            _uvCenter = uv;
            _hasCenter = true;
        }

        Vector2 raw = calibrateCenterOnEngage ? (uv - _uvCenter) : uv;

        Vector2 outVec = Vector2.zero;
        float mag = raw.magnitude;

        if (mag >= dpadDeadZone)
        {
            float denom = Mathf.Max(dpadDeadZone + 1e-6f, dpadFullScale);
            float t = Mathf.Clamp01((mag - dpadDeadZone) / (denom - dpadDeadZone));
            outVec = raw.normalized * t;

            if (useFourWay)
            {
                if (Mathf.Abs(outVec.x) >= Mathf.Abs(outVec.y))
                    outVec = new Vector2(Mathf.Sign(outVec.x) * Mathf.Abs(outVec.x), 0f);
                else
                    outVec = new Vector2(0f, Mathf.Sign(outVec.y) * Mathf.Abs(outVec.y));
            }
        }

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float k = 1f - Mathf.Exp(-Mathf.Max(0.01f, dpadOutputLerp) * dt);
        _dpadSm = Vector2.Lerp(_dpadSm, outVec, k);

        Dpad = _dpadSm;
    }
}
