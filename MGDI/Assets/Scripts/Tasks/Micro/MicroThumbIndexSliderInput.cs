using UnityEngine;

public class MicroThumbIndexSliderInput : MonoBehaviour
{
    public enum AxisMode { X = 0, Y = 1, Z = 2 }

    public enum GestureUpdateSource
    {
        EveryUnityFrame = 0,
        RenderFrameId = 1,  // recommended when RemoteHandRuntime uses interpolation buffer
        SampleId = 2        // useful in immediate mode (no interpolation buffer)
    }

    public enum DistancePlane
    {
        None3D = 0,
        Camera = 1,
        Palm = 2
    }

    [Header("References")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("Index axis (projection)")]
    [Tooltip("If true, thumb position is projected to index MCP->TIP axis (recommended).")]
    [SerializeField] private bool useProjection = true;

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

    [Header("Gesture update timing")]
    [Tooltip("Which frame-id to use when updating the touch signal (dNorm). RenderFrameId is recommended with interpolation buffer.")]
    [SerializeField] private GestureUpdateSource gestureUpdateSource = GestureUpdateSource.RenderFrameId;

    [Header("Depth-robust distances")]
    [Tooltip("Plane used for thumb-middle and thumb-on-index distance calculations.")]
    [SerializeField] private DistancePlane distancePlane = DistancePlane.Palm;

    [Tooltip("If true, compute thumb-on-index distance in the selected plane (reduces depth jitter).")]
    [SerializeField] private bool usePlaneForThumbIndexDistance = true;

    [Tooltip("If true, use point-to-segment distance (thumb -> middle finger segment) instead of tip-to-tip distance.")]
    [SerializeField] private bool useMiddleSegmentDistance = true;

    [Header("Tap detection (Thumb-Middle touch) - normalized")]
    [Tooltip("Touch DOWN if (distance / middle length) <= this.")]
    [SerializeField] private float touchOnRatio = 0.55f;

    [Tooltip("Touch UP if (distance / middle length) >= this. Must be > touchOnRatio.")]
    [SerializeField] private float touchOffRatio = 0.85f;

    [Tooltip("Minimum middle length to avoid divide-by-zero (meters).")]
    [SerializeField] private float minFingerLenMeters = 0.03f;

    [Tooltip("Debounce time for DOWN/UP (seconds).")]
    [SerializeField] private float debounceSeconds = 0.06f;

    [Tooltip("Max time between two taps to count as double tap (seconds).")]
    [SerializeField] private float doubleTapWindowSec = 0.30f;

    [Tooltip("Cooldown after accepting a tap sequence (seconds).")]
    [SerializeField] private float tapCooldownSec = 0.18f;

    [Header("Touch signal stabilization")]
    [Tooltip("Low-pass filter for dNorm (higher = snappier). 0 = no filtering.")]
    [SerializeField] private float touchDistLerp = 25f;

    [Header("Touch velocity gate (optional)")]
    [Tooltip("If true, require the right direction of change near thresholds (reduces jitter-based toggles).")]
    [SerializeField] private bool useTouchVelocityGate = true;

    [Tooltip("Min approach speed (ratio/sec) required near threshold to accept DOWN.")]
    [SerializeField] private float minApproachSpeedRatioPerSec = 0.25f;

    [Tooltip("Min separation speed (ratio/sec) required near threshold to accept UP.")]
    [SerializeField] private float minSeparationSpeedRatioPerSec = 0.25f;

    [Tooltip("Velocity gate applies only within this band around thresholds (ratio units).")]
    [SerializeField] private float velocityGateBand = 0.10f;

    [Header("Tap gating (center zone)")]
    [Tooltip("If true, taps are only recognized near the center zone (only applied while thumb is on index).")]
    [SerializeField] private bool gateTapToCenterZone = true;

    [Tooltip("Require |t - t0| <= this to accept a tap (while thumb is on index).")]
    [SerializeField] private float tapCenterBand = 0.10f;

    [Tooltip("Extra minimum interval between accepted taps (seconds). NOTE: does not block 2nd tap while waiting for double tap).")]
    [SerializeField] private float tapMinIntervalSec = 0.08f;

    [Header("Slider <-> Tap separation (recommended)")]
    [Tooltip("Block tap detection while thumb is close to the index axis (treat as 'sliding').")]
    [SerializeField] private bool blockTapsWhileThumbOnIndex = true;

    [Tooltip("Thumb is considered 'on index' if distance to index axis <= this (meters).")]
    [SerializeField] private float thumbOnIndexDist = 0.018f;

    [Tooltip("Once 'on index', it stays on until distance >= this (meters). Must be > thumbOnIndexDist.")]
    [SerializeField] private float thumbOffIndexDist = 0.028f;

    [Header("Long press (recommended for webcam)")]
    [Tooltip("If true, long press toggles Z mode (recommended).")]
    [SerializeField] private bool longPressTogglesZ = true;

    [Tooltip("Hold time to trigger long press (seconds).")]
    [SerializeField] private float longPressSec = 0.30f;

    [Header("Mode switching rules")]
    [Tooltip("Single tap toggles between X and Y.")]
    [SerializeField] private bool singleTapTogglesXY = true;

    [Tooltip("Double tap toggles Z mode (enter/exit). Exiting returns to X.")]
    [SerializeField] private bool doubleTapTogglesZ = false; // webcam에서는 보통 OFF 추천

    [Header("Debug (read-only)")]
    [SerializeField] private bool debug = false;
    [SerializeField] private float debug_t = -1f;
    [SerializeField] private float debug_touchDist = -1f; // ratio
    [SerializeField] private float debug_touchVel = 0f;   // ratio/sec
    [SerializeField] private float debug_thumbIndexDist = -1f;
    [SerializeField] private bool debug_thumbOnIndex = false;
    [SerializeField] private string debug_state = "";

    // Outputs
    public AxisMode Mode { get; private set; } = AxisMode.X;

    // Signed output in [-1..1]. Positive means toward TIP direction, negative toward MCP direction.
    public float AxisValue { get; private set; } = 0f;

    public bool SingleTapThisFrame { get; private set; } = false;
    public bool DoubleTapThisFrame { get; private set; } = false;
    public bool LongPressThisFrame { get; private set; } = false;

    // Debug getters for HUD / logs
    public float Debug_t => debug_t;
    public float Debug_touchDist => debug_touchDist;
    public string Debug_state => debug_state;

    // Backward-compatible debug getters (safe even if HUD uses them)
    public float Debug_tNeutral => _tNeutral;
    public int Debug_tapCount => _tapCount;
    public float Debug_tapWindowRemaining => Mathf.Max(0f, _tapWindowUntil - Time.time);
    public bool Debug_touchDownLatched => (_touchFsm == TouchFSM.DownLatched);
    public bool Debug_inTapCooldown => (Time.time < _tapCooldownUntil);
    public float Debug_lastTapAcceptedAge => Mathf.Max(0f, Time.time - _lastTapAcceptedTime);

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

    // Touch signal state
    float _dNormSm = -1f;
    float _dNormPrev = -1f;
    float _dNormVel = 0f;
    float _lastTouchSignalTime = -999f;
    int _lastGestureId = -1;

    // Long press
    float _pressHeld = 0f;
    bool _longPressFired = false;

    // Thumb-on-index latch (hysteresis)
    bool _thumbOnIndex = false;

    void ResetFramePulses()
    {
        SingleTapThisFrame = false;
        DoubleTapThisFrame = false;
        LongPressThisFrame = false;
    }

    void Update()
    {
        ResetFramePulses();

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (!HasRequiredJoints())
        {
            AxisValue = 0f;
            _axisSm = 0f;

            // still finalize pending single tap if due
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

        // When thumb is NOT on index (tap posture), freeze axis output to 0 to avoid surprise drift
        if (blockTapsWhileThumbOnIndex && !_thumbOnIndex)
            rawAxis = 0f;

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

        // 6) Tap detection + mode switching
        HandleTapFSM(dt, t);

        if (SingleTapThisFrame && singleTapTogglesXY)
            ToggleXY();

        // Long press gets priority over double tap
        if (LongPressThisFrame && longPressTogglesZ)
        {
            ToggleZ();
        }
        else if (DoubleTapThisFrame && doubleTapTogglesZ)
        {
            ToggleZ();
        }
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
        Vector3 thumb = remoteHand.remoteByIndex[4].position;
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

        // slider t in 3D (more faithful)
        t = Vector3.Dot(thumb - mcp, axis) / denom;
        t = Mathf.Clamp01(t);

        // thumb-on-index distance (optionally plane-projected)
        if (usePlaneForThumbIndexDistance && distancePlane != DistancePlane.None3D && TryGetDistancePlane(out Vector3 origin, out Vector3 normal))
        {
            Vector3 thumbP = ProjectToPlane(thumb, origin, normal);
            Vector3 mcpP = ProjectToPlane(mcp, origin, normal);
            Vector3 tipP = ProjectToPlane(tip, origin, normal);

            Vector3 axisP = tipP - mcpP;
            float denomP = axisP.sqrMagnitude;

            if (denomP < 1e-8f)
            {
                thumbIndexDist = Vector3.Distance(thumbP, mcpP);
            }
            else
            {
                float tp = Vector3.Dot(thumbP - mcpP, axisP) / denomP;
                tp = Mathf.Clamp01(tp);
                Vector3 closestP = mcpP + tp * axisP;
                thumbIndexDist = Vector3.Distance(thumbP, closestP);
            }
        }
        else
        {
            Vector3 closest = mcp + t * axis;
            thumbIndexDist = Vector3.Distance(thumb, closest);
        }
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
        // Always finalize pending single tap even if we block starting taps this frame
        FinalizeSingleTapIfDue();

        bool inCooldown = Time.time < _tapCooldownUntil;

        // tapMinIntervalSec should NOT block the 2nd tap in a double-tap window
        bool minIntervalOk = (_tapCount == 1) || (Time.time - _lastTapAcceptedTime >= tapMinIntervalSec);

        // Center gate uses neutral as center (t0). Only apply while thumb is ON index.
        bool centerOk = true;
        if (gateTapToCenterZone && _thumbOnIndex)
        {
            centerOk = Mathf.Abs(t - _tNeutral) <= tapCenterBand;
        }

        // Strong separation: block starting taps while thumb is on index (sliding posture)
        bool contextOk = true;
        if (blockTapsWhileThumbOnIndex && _thumbOnIndex)
            contextOk = false;

        bool gateForStart = !inCooldown && minIntervalOk && centerOk && contextOk;

        // Update touch signal (dNorm) with gating by frame id
        float dNorm = UpdateTouchSignal(dt);
        debug_touchDist = dNorm;
        debug_touchVel = _dNormVel;

        bool downCond = dNorm <= touchOnRatio;
        bool upCond = dNorm >= touchOffRatio;

        // Optional velocity gate near thresholds (reduces jitter-triggered transitions)
        if (useTouchVelocityGate)
        {
            if (downCond && (touchOnRatio - dNorm) < velocityGateBand)
            {
                if (_dNormVel > -Mathf.Max(0f, minApproachSpeedRatioPerSec))
                    downCond = false;
            }

            if (upCond && (dNorm - touchOffRatio) < velocityGateBand)
            {
                if (_dNormVel < Mathf.Max(0f, minSeparationSpeedRatioPerSec))
                    upCond = false;
            }
        }

        // If we are blocking taps due to context, cancel any in-progress latch to prevent "delayed tap"
        if (!contextOk && _touchFsm != TouchFSM.Idle)
        {
            ResetTouchFSM();
            if (debug) debug_state = "Tap: canceled (thumb on index)";
        }

        if (_touchFsm == TouchFSM.Idle)
        {
            _upHeld = 0f;

            // reset long press tracking in idle
            _pressHeld = 0f;
            _longPressFired = false;

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
                        _pressHeld = 0f;
                        _longPressFired = false;

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

            // long press while held (only if not already fired)
            if (longPressTogglesZ && !_longPressFired && !inCooldown && contextOk)
            {
                _pressHeld += dt;
                if (_pressHeld >= Mathf.Max(0.05f, longPressSec))
                {
                    _longPressFired = true;
                    LongPressThisFrame = true;

                    // cancel any pending tap sequence
                    _tapCount = 0;
                    _tapWindowUntil = 0f;

                    // enter cooldown
                    _tapCooldownUntil = Time.time + Mathf.Max(0f, tapCooldownSec);
                    _lastTapAcceptedTime = Time.time;

                    // prevent repeated firing while holding
                    // keep FSM latched, but release won't register tap
                    if (debug) debug_state = "Tap: LONG";
                }
            }

            if (upCond)
            {
                _upHeld += dt;
                if (_upHeld >= debounceSeconds)
                {
                    _upHeld = 0f;
                    _touchFsm = TouchFSM.Idle;

                    // If long press fired, do not register tap on release
                    if (!_longPressFired)
                    {
                        if (!inCooldown)
                            RegisterTap();
                        else if (debug)
                            debug_state = "Tap: ignored (cooldown)";
                    }
                    else
                    {
                        // after long press, ignore release for tap
                        if (debug) debug_state = "Tap: release (after long)";
                    }

                    // reset long press tracking
                    _pressHeld = 0f;
                    _longPressFired = false;
                }
            }
            else
            {
                _upHeld = 0f;
            }
        }

        FinalizeSingleTapIfDue();
    }

    float UpdateTouchSignal(float dt)
    {
        // Decide whether we should recompute measurement this frame
        int idNow = 0;
        bool shouldUpdate = true;

        if (remoteHand != null)
        {
            if (gestureUpdateSource == GestureUpdateSource.RenderFrameId)
            {
                idNow = remoteHand.RenderFrameId;
                shouldUpdate = (idNow != _lastGestureId);
            }
            else if (gestureUpdateSource == GestureUpdateSource.SampleId)
            {
                idNow = remoteHand.SampleId;
                shouldUpdate = (idNow != _lastGestureId);
            }
            else
            {
                shouldUpdate = true;
            }
        }

        if (!shouldUpdate && _dNormSm >= 0f)
        {
            // no new measurement => treat velocity as 0
            _dNormVel = 0f;
            return _dNormSm;
        }

        _lastGestureId = idNow;

        // --- Fetch joints (filtered pose) ---
        Vector3 thumb = remoteHand.remoteByIndex[4].position;
        Vector3 midTip = remoteHand.remoteByIndex[12].position;
        Vector3 midMcp = remoteHand.remoteByIndex[9].position;

        Vector3 midPip = (remoteHand.remoteByIndex.Length > 10 && remoteHand.remoteByIndex[10] != null)
            ? remoteHand.remoteByIndex[10].position
            : midMcp;

        Vector3 midDip = (remoteHand.remoteByIndex.Length > 11 && remoteHand.remoteByIndex[11] != null)
            ? remoteHand.remoteByIndex[11].position
            : midTip;

        // --- Middle length (bone-sum in 3D; less pose-dependent than MCP<->TIP chord) ---
        float middleLen =
            Vector3.Distance(midMcp, midPip) +
            Vector3.Distance(midPip, midDip) +
            Vector3.Distance(midDip, midTip);

        middleLen = Mathf.Max(minFingerLenMeters, middleLen);

        // --- Distance plane projection (optional) for NUMERATOR only ---
        Vector3 thumbD = thumb;
        Vector3 midTipD = midTip;
        Vector3 midPipD = midPip;
        Vector3 midDipD = midDip;

        if (distancePlane != DistancePlane.None3D && TryGetDistancePlane(out Vector3 origin, out Vector3 normal))
        {
            thumbD = ProjectToPlane(thumbD, origin, normal);
            midTipD = ProjectToPlane(midTipD, origin, normal);
            midPipD = ProjectToPlane(midPipD, origin, normal);
            midDipD = ProjectToPlane(midDipD, origin, normal);
        }

        float dTM;

        if (useMiddleSegmentDistance)
        {
            // Use min distance to segments (DIP->TIP and PIP->TIP) to be more stable than tip-to-tip
            float d1 = DistancePointToSegment(thumbD, midDipD, midTipD);
            float d2 = DistancePointToSegment(thumbD, midPipD, midTipD);
            dTM = Mathf.Min(d1, d2);
        }
        else
        {
            dTM = Vector3.Distance(thumbD, midTipD);
        }

        float dNormRaw = dTM / middleLen;

        // Low-pass filter for dNorm
        float dNormOut = dNormRaw;
        if (touchDistLerp > 0.01f)
        {
            float k = 1f - Mathf.Exp(-touchDistLerp * dt);
            if (_dNormSm < 0f) _dNormSm = dNormRaw;
            else _dNormSm = Mathf.Lerp(_dNormSm, dNormRaw, k);

            dNormOut = _dNormSm;
        }
        else
        {
            _dNormSm = dNormRaw;
        }

        // Velocity (ratio/sec) computed from filtered signal
        float now = Time.time;
        float dts = ( _lastTouchSignalTime < 0f ) ? dt : Mathf.Max(1e-4f, now - _lastTouchSignalTime);

        if (_dNormPrev < 0f)
        {
            _dNormVel = 0f;
        }
        else
        {
            _dNormVel = (dNormOut - _dNormPrev) / dts;
        }

        _dNormPrev = dNormOut;
        _lastTouchSignalTime = now;

        return dNormOut;
    }

    static float DistancePointToSegment(Vector3 p, Vector3 a, Vector3 b)
    {
        Vector3 ab = b - a;
        float denom = ab.sqrMagnitude;
        if (denom < 1e-10f) return Vector3.Distance(p, a);

        float t = Vector3.Dot(p - a, ab) / denom;
        t = Mathf.Clamp01(t);
        Vector3 c = a + t * ab;
        return Vector3.Distance(p, c);
    }

    bool TryGetDistancePlane(out Vector3 origin, out Vector3 normal)
    {
        origin = Vector3.zero;
        normal = Vector3.forward;

        if (remoteHand == null || remoteHand.remoteByIndex == null || remoteHand.remoteByIndex.Length < 1)
            return false;

        // origin: wrist is a good stable anchor
        Transform wrist = remoteHand.remoteByIndex[0];
        if (wrist != null) origin = wrist.position;

        if (distancePlane == DistancePlane.Palm)
        {
            // Use palm plane: cross( (indexMCP-wrist), (pinkyMCP-wrist) )
            Transform idx = (remoteHand.remoteByIndex.Length > 5) ? remoteHand.remoteByIndex[5] : null;
            Transform pinky = (remoteHand.remoteByIndex.Length > 17) ? remoteHand.remoteByIndex[17] : null;

            if (wrist != null && idx != null && pinky != null)
            {
                Vector3 a = idx.position - wrist.position;
                Vector3 b = pinky.position - wrist.position;
                Vector3 n = Vector3.Cross(a, b);
                if (n.sqrMagnitude > 1e-8f)
                {
                    normal = n.normalized;
                    origin = wrist.position;
                    return true;
                }
            }

            // fallback to camera plane if palm plane cannot be built
        }

        // Camera plane fallback / explicit camera plane
        Camera cam = Camera.main;
        if (cam != null)
        {
            Vector3 f = cam.transform.forward;
            if (f.sqrMagnitude < 1e-8f) f = Vector3.forward;
            normal = f.normalized;
            // origin can still be wrist (projection difference cancels for distances)
            return true;
        }

        return false;
    }

    static Vector3 ProjectToPlane(Vector3 p, Vector3 planeOrigin, Vector3 planeNormal)
    {
        Vector3 n = planeNormal;
        if (n.sqrMagnitude < 1e-8f) return p;
        n.Normalize();
        float d = Vector3.Dot(p - planeOrigin, n);
        return p - d * n;
    }

    void ResetTouchFSM()
    {
        _touchFsm = TouchFSM.Idle;
        _downHeld = 0f;
        _upHeld = 0f;

        _pressHeld = 0f;
        _longPressFired = false;
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
