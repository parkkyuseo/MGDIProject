using UnityEngine;

public class MicroThumbIndexSliderInput : MonoBehaviour
{
    public enum AxisMode { X = 0, Y = 1, Z = 2 }

    [Header("References")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("Index axis (projection)")]
    [Tooltip("If true, thumb position is projected to index MCP->TIP axis (recommended).")]
    [SerializeField] private bool useProjection = true; // (현재 구현은 projection 전제로 동작)

    [Header("Neutral (t0) handling")]
    [Tooltip("If true, neutral t0 is fixed at 0.5 (recommended for stability).")]
    [SerializeField] private bool useFixedNeutralMidpoint = true;
    [SerializeField] private float fixedNeutralT = 0.50f;

    [Header("Slider shaping")]
    [SerializeField] private float deadZoneT = 0.12f;
    [SerializeField] private float fullScaleT = 0.40f;
    [SerializeField] private float gamma = 1.6f;
    [SerializeField] private float outputLerp = 14f;
    [SerializeField] private float maxOutputRatePerSec = 6f;
    [SerializeField] private bool invertAxis = false;

    [Header("Tap detection (Thumb-Middle touch) - normalized")]
    [Tooltip("Touch DOWN if (thumb-middle distance / middle length) <= this.")]
    [SerializeField] private float touchOnRatio = 0.55f;
    [Tooltip("Touch UP if (thumb-middle distance / middle length) >= this. Must be > touchOnRatio.")]
    [SerializeField] private float touchOffRatio = 0.85f;
    [SerializeField] private float minFingerLenMeters = 0.03f;

    [Tooltip("Debounce time for DOWN/UP (seconds).")]
    [SerializeField] private float debounceSeconds = 0.06f;

    [Tooltip("Max time between two taps to count as double tap (seconds).")]
    [SerializeField] private float doubleTapWindowSec = 0.30f;

    [Tooltip("Cooldown after accepting a tap sequence (seconds).")]
    [SerializeField] private float tapCooldownSec = 0.18f;

    [Header("Tap gating (legacy)")]
    [Tooltip("If true, taps are only recognized near the center zone.")]
    [SerializeField] private bool gateTapToCenterZone = true;
    [SerializeField] private float tapCenterBand = 0.10f;

    [Tooltip("Extra minimum interval between accepted taps (seconds). NOTE: patched to not block 2nd tap in a double-tap.")]
    [SerializeField] private float tapMinIntervalSec = 0.08f;

    [Header("Slider <-> Tap separation (recommended)")]
    [Tooltip("Block tap detection while thumb is close to the index axis (treat as 'sliding').")]
    [SerializeField] private bool blockTapsWhileThumbOnIndex = true;

    [Tooltip("Thumb is considered 'on index' if distance to index axis <= this (meters).")]
    [SerializeField] private float thumbOnIndexDist = 0.018f;

    [Tooltip("Once 'on index', it stays on until distance >= this (meters). Must be > thumbOnIndexDist.")]
    [SerializeField] private float thumbOffIndexDist = 0.028f;

    [Header("Tap signal stabilization")]
    [Tooltip("Use bone-sum length (MCP->PIP->DIP->TIP) for middle length (less pose-dependent).")]
    [SerializeField] private bool useBoneSumForMiddleLength = true;

    [Tooltip("Low-pass filter for dNorm (higher = snappier). 0 = no filtering.")]
    [SerializeField] private float touchDistLerp = 25f;

    [Header("Mode switching rules")]
    [SerializeField] private bool singleTapTogglesXY = true;
    [SerializeField] private bool doubleTapTogglesZ = true;

    [Header("Debug (read-only)")]
    [SerializeField] private bool debug = false;
    [SerializeField] private float debug_t = -1f;
    [SerializeField] private float debug_touchDist = -1f;
    [SerializeField] private float debug_thumbIndexDist = -1f;
    [SerializeField] private bool debug_thumbOnIndex = false;
    [SerializeField] private string debug_state = "";

    // Outputs
    public AxisMode Mode { get; private set; } = AxisMode.X;
    public float AxisValue { get; private set; } = 0f;
    public bool SingleTapThisFrame { get; private set; } = false;
    public bool DoubleTapThisFrame { get; private set; } = false;

    // Debug getters for HUD
    public float Debug_t => debug_t;
    public float Debug_touchDist => debug_touchDist;
    public string Debug_state => debug_state;

    // Internal: neutral point in t
    float _tNeutral = 0.5f;
    bool _neutralCaptured = false;

    // Output smoothing state
    float _axisSm = 0f;

    // Tap FSM
    enum TouchFSM { Idle, DownLatched }
    TouchFSM _touchFsm = TouchFSM.Idle;
    float _downHeld = 0f;
    float _upHeld = 0f;

    int _tapCount = 0;
    float _tapWindowUntil = 0f;
    float _tapCooldownUntil = 0f;
    float _lastTapAcceptedTime = -999f;

    // Touch distance filter
    float _dNormSm = -1f;

    // Thumb-on-index latch (hysteresis)
    bool _thumbOnIndex = false;

    void ResetFramePulses()
    {
        SingleTapThisFrame = false;
        DoubleTapThisFrame = false;
    }

    void Update()
    {
        ResetFramePulses();
        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (!HasRequiredJoints())
        {
            AxisValue = 0f;
            _axisSm = 0f;

            // IMPORTANT: still finalize pending single tap even if tracking momentarily drops
            FinalizeSingleTapIfDue();

            if (debug) debug_state = "Missing joints";
            return;
        }

        // 1) Compute t + thumb->index axis distance
        float t, thumbIndexDist;
        ComputeProjectedTAndThumbIndexDist(out t, out thumbIndexDist);
        debug_t = t;
        debug_thumbIndexDist = thumbIndexDist;

        // 2) Update thumb-on-index latch (hysteresis)
        if (_thumbOnIndex)
        {
            if (thumbIndexDist >= Mathf.Max(thumbOffIndexDist, thumbOnIndexDist + 0.001f))
                _thumbOnIndex = false;
        }
        else
        {
            if (thumbIndexDist <= Mathf.Max(0f, thumbOnIndexDist))
                _thumbOnIndex = true;
        }
        debug_thumbOnIndex = _thumbOnIndex;

        // 3) Neutral handling
        if (useFixedNeutralMidpoint)
        {
            _tNeutral = fixedNeutralT;
            _neutralCaptured = true;
        }
        else
        {
            if (!_neutralCaptured)
            {
                _tNeutral = t;
                _neutralCaptured = true;
            }
        }

        // 4) Convert t -> signed axis value in [-1..1]
        float rawAxis = TToAxisValue(t, _tNeutral);
        if (invertAxis) rawAxis = -rawAxis;

        // (추천) thumb가 index에서 떨어져 탭을 하는 동안에는 슬라이더 출력이 튀지 않게 0으로 끌어주기
        if (blockTapsWhileThumbOnIndex && !_thumbOnIndex)
        {
            rawAxis = 0f;
        }

        // 5) Smooth + rate limit
        float k = 1f - Mathf.Exp(-Mathf.Max(0.01f, outputLerp) * dt);
        float target = Mathf.Lerp(_axisSm, rawAxis, k);

        if (maxOutputRatePerSec > 0f)
        {
            float maxStep = maxOutputRatePerSec * dt;
            float d = target - _axisSm;
            d = Mathf.Clamp(d, -maxStep, maxStep);
            target = _axisSm + d;
        }

        _axisSm = target;
        AxisValue = _axisSm;

        // 6) Tap detection + mode switching (thumb-middle)
        HandleTapFSM(dt, t);

        if (SingleTapThisFrame && singleTapTogglesXY)
            ToggleXY();

        if (DoubleTapThisFrame && doubleTapTogglesZ)
            ToggleZ();
    }

    bool HasRequiredJoints()
    {
        if (remoteHand == null) return false;
        if (remoteHand.remoteByIndex == null) return false;
        if (remoteHand.remoteByIndex.Length < 13) return false;

        // slider: 4,5,8
        if (remoteHand.remoteByIndex[4] == null) return false;  // thumb tip
        if (remoteHand.remoteByIndex[5] == null) return false;  // index mcp
        if (remoteHand.remoteByIndex[8] == null) return false;  // index tip

        // tap: 12,9
        if (remoteHand.remoteByIndex[12] == null) return false; // middle tip
        if (remoteHand.remoteByIndex[9] == null) return false;  // middle mcp

        return true;
    }

    void ComputeProjectedTAndThumbIndexDist(out float t, out float thumbIndexDist)
    {
        // Use the same thumb tip source as tap if fast tips are available
        Vector3 thumb = (remoteHand != null && remoteHand.fastTipsReady)
            ? remoteHand.thumbTipFast
            : remoteHand.remoteByIndex[4].position;

        Vector3 mcp = remoteHand.remoteByIndex[5].position;
        Vector3 tip = remoteHand.remoteByIndex[8].position;

        Vector3 axis = tip - mcp;
        float denom = axis.sqrMagnitude;
        if (denom < 1e-8f)
        {
            t = 0.5f;
            thumbIndexDist = float.PositiveInfinity;
            return;
        }

        t = Vector3.Dot(thumb - mcp, axis) / denom;
        t = Mathf.Clamp01(t);

        Vector3 closest = mcp + t * axis;
        thumbIndexDist = Vector3.Distance(thumb, closest);
    }

    float TToAxisValue(float t, float t0)
    {
        float d = t - t0;
        float ad = Mathf.Abs(d);

        float dz = Mathf.Max(0f, deadZoneT);
        float fs = Mathf.Max(dz + 1e-4f, fullScaleT);

        if (ad < dz) return 0f;

        float x = Mathf.Clamp01((ad - dz) / (fs - dz));

        if (gamma > 0.01f && Mathf.Abs(gamma - 1f) > 1e-3f)
            x = Mathf.Pow(x, gamma);

        return Mathf.Sign(d) * x;
    }

    void HandleTapFSM(float dt, float t)
    {
        // Always finalize pending single tap even if we are currently blocking new taps
        // (prevents "stuck waiting" when user starts sliding right after a tap)
        // We'll call again at the end too; harmless.
        FinalizeSingleTapIfDue();

        // Cooldown blocks starting/accepting taps
        bool inCooldown = Time.time < _tapCooldownUntil;

        // IMPORTANT PATCH:
        // tapMinIntervalSec should NOT block the 2nd tap in a double-tap.
        bool minIntervalOk = (_tapCount == 1) || (Time.time - _lastTapAcceptedTime >= tapMinIntervalSec);

        // Center gate: use neutral as center (fixes inconsistency when useFixedNeutralMidpoint=false)
        float center = _tNeutral;

        bool centerOk = true;
        if (gateTapToCenterZone)
        {
            // If thumb is off index (we are in tap posture), t is less meaningful -> don't block taps by center band.
            if (_thumbOnIndex)
                centerOk = Mathf.Abs(t - center) <= tapCenterBand;
        }

        // Strong separation: while thumb is on index (sliding posture), block starting taps
        bool contextOk = true;
        if (blockTapsWhileThumbOnIndex && _thumbOnIndex)
            contextOk = false;

        bool gateForStart = !inCooldown && minIntervalOk && centerOk && contextOk;

        // Compute normalized touch distance (thumb-middle)
        float dNorm = ComputeThumbMiddleNormalizedDistance(dt);
        debug_touchDist = dNorm;

        bool downCond = dNorm <= touchOnRatio;
        bool upCond = dNorm >= touchOffRatio;

        // If we are blocking taps due to context, cancel any in-progress latch to prevent "delayed tap"
        if (!contextOk && _touchFsm != TouchFSM.Idle)
        {
            ResetTouchFSM();
            if (debug) debug_state = "Tap: canceled (thumb on index)";
        }

        if (_touchFsm == TouchFSM.Idle)
        {
            _upHeld = 0f;

            if (!gateForStart)
            {
                _downHeld = 0f;
                if (debug)
                {
                    if (inCooldown) debug_state = "Tap: cooldown";
                    else if (!minIntervalOk) debug_state = "Tap: blocked (min interval)";
                    else if (!centerOk) debug_state = "Tap: blocked (not in center band)";
                    else if (!contextOk) debug_state = "Tap: blocked (thumb on index)";
                }
            }
            else
            {
                if (downCond)
                {
                    _downHeld += dt;
                    if (_downHeld >= debounceSeconds)
                    {
                        _touchFsm = TouchFSM.DownLatched;
                        _downHeld = 0f;
                        if (debug) debug_state = "Tap: DOWN latched";
                    }
                }
                else
                {
                    _downHeld = 0f;
                    if (debug) debug_state = "Tap: ready";
                }
            }
        }
        else // DownLatched
        {
            _downHeld = 0f;

            if (upCond)
            {
                _upHeld += dt;
                if (_upHeld >= debounceSeconds)
                {
                    _upHeld = 0f;
                    _touchFsm = TouchFSM.Idle;

                    // Accept this tap only if not in cooldown
                    if (!inCooldown)
                        RegisterTap();
                    else if (debug)
                        debug_state = "Tap: ignored (cooldown)";
                }
            }
            else
            {
                _upHeld = 0f;
            }
        }

        // finalize single tap if window expires
        FinalizeSingleTapIfDue();
    }

    float ComputeThumbMiddleNormalizedDistance(float dt)
    {
        Vector3 thumbPos, middleTipPos;

        if (remoteHand != null && remoteHand.fastTipsReady)
        {
            thumbPos = remoteHand.thumbTipFast;
            middleTipPos = remoteHand.middleTipFast;
        }
        else
        {
            thumbPos = remoteHand.remoteByIndex[4].position;
            middleTipPos = remoteHand.remoteByIndex[12].position;
        }

        Vector3 middleMcpPos = remoteHand.remoteByIndex[9].position;

        float dTM = Vector3.Distance(thumbPos, middleTipPos);
        float middleLen;

        if (useBoneSumForMiddleLength
            && remoteHand.remoteByIndex.Length >= 13
            && remoteHand.remoteByIndex[10] != null
            && remoteHand.remoteByIndex[11] != null)
        {
            Vector3 pip = remoteHand.remoteByIndex[10].position;
            Vector3 dip = remoteHand.remoteByIndex[11].position;

            // Bone-sum is much less sensitive to finger flex than MCP<->TIP chord
            middleLen =
                Vector3.Distance(middleMcpPos, pip) +
                Vector3.Distance(pip, dip) +
                Vector3.Distance(dip, middleTipPos);
        }
        else
        {
            middleLen = Vector3.Distance(middleMcpPos, middleTipPos);
        }

        middleLen = Mathf.Max(minFingerLenMeters, middleLen);
        float dNorm = dTM / middleLen;

        // Optional low-pass on dNorm
        if (touchDistLerp > 0.01f)
        {
            float k = 1f - Mathf.Exp(-touchDistLerp * dt);
            if (_dNormSm < 0f) _dNormSm = dNorm;
            else _dNormSm = Mathf.Lerp(_dNormSm, dNorm, k);
            return _dNormSm;
        }

        _dNormSm = dNorm;
        return dNorm;
    }

    void ResetTouchFSM()
    {
        _touchFsm = TouchFSM.Idle;
        _downHeld = 0f;
        _upHeld = 0f;
    }

    void FinalizeSingleTapIfDue()
    {
        if (_tapCount == 1 && Time.time >= _tapWindowUntil)
        {
            _tapCount = 0;
            _tapWindowUntil = 0f;
            SingleTapThisFrame = true;

            _tapCooldownUntil = Time.time + Mathf.Max(0f, tapCooldownSec);
            _lastTapAcceptedTime = Time.time;

            if (debug) debug_state = "Tap: SINGLE";
        }
    }

    void RegisterTap()
    {
        _lastTapAcceptedTime = Time.time;

        if (_tapCount == 0)
        {
            _tapCount = 1;
            _tapWindowUntil = Time.time + Mathf.Max(0.05f, doubleTapWindowSec);
            if (debug) debug_state = "Tap: first";
            return;
        }

        if (_tapCount == 1 && Time.time <= _tapWindowUntil)
        {
            _tapCount = 0;
            _tapWindowUntil = 0f;
            DoubleTapThisFrame = true;

            _tapCooldownUntil = Time.time + Mathf.Max(0f, tapCooldownSec);

            if (debug) debug_state = "Tap: DOUBLE";
            return;
        }

        _tapCount = 1;
        _tapWindowUntil = Time.time + Mathf.Max(0.05f, doubleTapWindowSec);
        if (debug) debug_state = "Tap: first (late reset)";
    }

    void ToggleXY()
    {
        if (Mode == AxisMode.Z)
            return;

        Mode = (Mode == AxisMode.X) ? AxisMode.Y : AxisMode.X;

        _axisSm = 0f;
        AxisValue = 0f;

        if (!useFixedNeutralMidpoint)
            _neutralCaptured = false;
    }

    void ToggleZ()
    {
        Mode = (Mode == AxisMode.Z) ? AxisMode.X : AxisMode.Z;

        _axisSm = 0f;
        AxisValue = 0f;

        if (!useFixedNeutralMidpoint)
            _neutralCaptured = false;
    }
}
