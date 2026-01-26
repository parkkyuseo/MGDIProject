using UnityEngine;

public class MicroThumbIndexSliderInput : MonoBehaviour
{
    public enum AxisMode
    {
        XY = 0,   // Slide -> X, Twist -> Y
        Z = 1     // Slide -> Z, Twist -> Y
    }

    [Header("References")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("Slider (thumb slide) shaping")]
    [SerializeField] private float deadZoneT = 0.12f;
    [SerializeField] private float fullScaleT = 0.40f;
    [SerializeField] private float gamma = 1.6f;
    [SerializeField] private float outputLerp = 14f;
    [SerializeField] private float maxOutputRatePerSec = 6f;

    [Header("Neutral (t0) handling")]
    [SerializeField] private bool useFixedNeutralMidpoint = true;
    [SerializeField] private float fixedNeutralT = 0.50f;
    [SerializeField] private bool invertSlide = false;

    [Header("Twist (wrist) -> Y")]
    [SerializeField] private float twistMaxAbsDegForNormalize = 45f;
    [SerializeField] private float twistDeadZoneNorm = 0.06f;
    [SerializeField] private float twistOutputLerp = 16f;
    [SerializeField] private bool invertTwist = false;

    [Header("ThumbOnIndex detection (robust)")]
    [SerializeField] private float thumbOnIndexDist = 0.018f;
    [SerializeField] private float thumbOffIndexDist = 0.028f;
    [SerializeField] private float offConfirmSec = 0.10f;
    [SerializeField] private float onConfirmSec = 0.06f;
    [SerializeField] private float thumbIndexDistLerp = 18f;

    [Header("Twist cross-talk suppression")]
    [Tooltip("If |twist velocity| is above this, treat as 'active twisting' (deg/sec).")]
    [SerializeField] private float twistActiveVelDegPerSec = 120f;

    [Tooltip("During active twisting, ignore small t changes (prevents slide drift while twisting).")]
    [SerializeField] private float tJitterHoldDelta = 0.010f;

    [Tooltip("During active twisting, require larger distance to declare OFF (meters).")]
    [SerializeField] private float offExtraMarginWhileTwisting = 0.010f;

    [Header("Mode toggle (single toggle)")]
    [Tooltip("Hold thumb OFF index for this long, then re-attach to toggle XY<->Z.")]
    [SerializeField] private float zToggleHoldSec = 0.60f;

    [Tooltip("Cooldown after toggling mode (sec).")]
    [SerializeField] private float zToggleCooldownSec = 0.30f;

    [Tooltip("Require slide near neutral when arming mode toggle (prevents accidental toggles while sliding).")]
    [SerializeField] private bool requireNeutralForZToggle = true;

    [SerializeField] private float zToggleNeutralGate = 0.20f;

    [Header("Debug (read-only)")]
    [SerializeField] private bool debug = false;
    [SerializeField] private float debug_tRaw = -1f;
    [SerializeField] private float debug_tUsed = -1f;
    [SerializeField] private float debug_thumbIndexDistRaw = -1f;
    [SerializeField] private float debug_thumbIndexDistSm = -1f;
    [SerializeField] private bool debug_thumbOnIndex = false;
    [SerializeField] private float debug_twistDeg = 0f;
    [SerializeField] private float debug_twistVelDegPerSec = 0f;
    [SerializeField] private float debug_offHeldSec = 0f;
    [SerializeField] private bool debug_zArmed = false;
    [SerializeField] private string debug_state = "";

    // Outputs
    public AxisMode Mode { get; private set; } = AxisMode.XY;

    // Backward compatibility: slide value (always)
    public float AxisValue { get; private set; } = 0f;

    // 3-axis outputs
    public float AxisX { get; private set; } = 0f; // slide when Mode==XY
    public float AxisY { get; private set; } = 0f; // twist always
    public float AxisZ { get; private set; } = 0f; // slide when Mode==Z

    // HUD/debug access
    public float Debug_t => debug_tUsed;
    public float Debug_tRaw => debug_tRaw;
    public float Debug_thumbIndexDist => debug_thumbIndexDistSm;
    public bool Debug_thumbOnIndex => debug_thumbOnIndex;
    public float Debug_offHeldSec => debug_offHeldSec;
    public bool Debug_zArmed => debug_zArmed;
    public string Debug_state => debug_state;

    // Internal
    float _tNeutral = 0.5f;
    bool _neutralCaptured = false;

    float _slideSm = 0f;
    float _twistSm = 0f;

    bool _thumbOnIndexStable = true;
    float _offHeld = 0f;
    float _onHeld = 0f;

    float _thumbIndexDistSm = -1f;
    float _tPrevUsed = 0.5f;

    float _twistPrevDeg = 0f;
    float _twistPrevTime = -999f;

    bool _inOffSegment = false;
    float _offStartTime = -1f;
    bool _modeToggleArmed = false;
    float _modeToggleCooldownUntil = -999f;

    int _lastRenderFrameId = -1;

    // Reusable buffer for index polyline points: MCP -> (PIP) -> (DIP) -> TIP
    readonly Vector3[] _indexPolylinePts = new Vector3[4];

    void Update()
    {
        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (!HasRequiredJoints())
        {
            AxisX = AxisY = AxisZ = AxisValue = 0f;
            _slideSm = _twistSm = 0f;
            if (debug) debug_state = "Missing joints";
            return;
        }

        // Gate by RenderFrameId to avoid treating repeated Unity frames as new hand poses
        if (remoteHand != null)
        {
            int rid = remoteHand.RenderFrameId;
            if (rid == _lastRenderFrameId)
            {
                UpdateOutputs(dt, poseIsFresh: false);
                return;
            }
            _lastRenderFrameId = rid;
        }

        UpdateOutputs(dt, poseIsFresh: true);
    }

    void UpdateOutputs(float dt, bool poseIsFresh)
    {
        if (debug) debug_state = "";

        // Joints
        Vector3 thumb = remoteHand.remoteByIndex[4].position;

        // Index joints (5,6,7,8 in MediaPipe-style; 6/7 may be null depending on your runtime)
        Vector3 idxMcp = remoteHand.remoteByIndex[5].position;
        Vector3 idxTip = remoteHand.remoteByIndex[8].position;

        bool hasPip = (remoteHand.remoteByIndex.Length > 6 && remoteHand.remoteByIndex[6] != null);
        bool hasDip = (remoteHand.remoteByIndex.Length > 7 && remoteHand.remoteByIndex[7] != null);

        int nPts = 0;
        _indexPolylinePts[nPts++] = idxMcp;
        if (hasPip) _indexPolylinePts[nPts++] = remoteHand.remoteByIndex[6].position;
        if (hasDip) _indexPolylinePts[nPts++] = remoteHand.remoteByIndex[7].position;
        _indexPolylinePts[nPts++] = idxTip;

        // t + thumb-index distance (polyline-based; falls back to MCP->TIP if PIP/DIP unavailable)
        float distRaw;
        float tRaw = ProjectToPolylineT(thumb, _indexPolylinePts, nPts, out distRaw);
        debug_tRaw = tRaw;
        debug_thumbIndexDistRaw = distRaw;

        // smooth distance
        float distSm = distRaw;
        if (thumbIndexDistLerp > 0.01f)
        {
            float k = 1f - Mathf.Exp(-thumbIndexDistLerp * dt);
            if (_thumbIndexDistSm < 0f) _thumbIndexDistSm = distSm;
            else _thumbIndexDistSm = Mathf.Lerp(_thumbIndexDistSm, distSm, k);
            distSm = _thumbIndexDistSm;
        }
        else
        {
            _thumbIndexDistSm = distSm;
        }
        debug_thumbIndexDistSm = distSm;

        // twist + velocity (poseIsFresh-aware to avoid velocity spikes when pose is unchanged)
        float twistDeg = (remoteHand != null) ? remoteHand.TwistDegrees : 0f;
        debug_twistDeg = twistDeg;

        float now = Time.time;
        float twistVel = 0f;

        if (poseIsFresh)
        {
            float dtTw = (_twistPrevTime < 0f) ? dt : Mathf.Max(1e-4f, now - _twistPrevTime);
            twistVel = (twistDeg - _twistPrevDeg) / dtTw;

            _twistPrevDeg = twistDeg;
            _twistPrevTime = now;
        }
        else
        {
            // Do not update prev time/deg here; let dtTw accumulate until next fresh pose arrives.
            twistVel = 0f;
        }

        debug_twistVelDegPerSec = twistVel;

        bool twistActive = Mathf.Abs(twistVel) >= Mathf.Max(0f, twistActiveVelDegPerSec);

        // update thumb-on-index stable (suppresses false OFF during twist)
        bool prevThumbOn = _thumbOnIndexStable;
        UpdateThumbOnIndexStable(dt, distSm, twistActive);
        debug_thumbOnIndex = _thumbOnIndexStable;

        // OFF segment edges for mode toggle
        if (prevThumbOn && !_thumbOnIndexStable)
        {
            _inOffSegment = true;
            _offStartTime = Time.time;
            _modeToggleArmed = false;
            if (debug) debug_state = "Edge: ON->OFF";
        }
        else if (!prevThumbOn && _thumbOnIndexStable)
        {
            if (_inOffSegment)
            {
                _inOffSegment = false;

                bool inCooldown = Time.time < _modeToggleCooldownUntil;
                if (!inCooldown && _modeToggleArmed)
                {
                    ToggleModeXY_Z();
                    _modeToggleCooldownUntil = Time.time + Mathf.Max(0f, zToggleCooldownSec);
                    if (debug) debug_state = "Mode toggled (XY<->Z)";
                }
                else
                {
                    if (debug) debug_state = inCooldown ? "Reattach ignored (cooldown)" : "Reattach (no arm)";
                }
            }
        }

        // Arm mode toggle while stable OFF
        debug_offHeldSec = 0f;
        debug_zArmed = _modeToggleArmed;

        if (_inOffSegment && !_thumbOnIndexStable)
        {
            float offHeld = Mathf.Max(0f, Time.time - _offStartTime);
            debug_offHeldSec = offHeld;

            bool inCooldown = Time.time < _modeToggleCooldownUntil;
            if (!inCooldown && !_modeToggleArmed)
            {
                bool neutralOk = true;
                if (requireNeutralForZToggle)
                    neutralOk = Mathf.Abs(_slideSm) <= Mathf.Max(0f, zToggleNeutralGate);

                if (offHeld >= Mathf.Max(0.05f, zToggleHoldSec) && neutralOk)
                {
                    _modeToggleArmed = true;
                    debug_zArmed = true;
                    if (debug) debug_state = "Mode toggle armed (hold off)";
                }
            }
        }

        // Stabilize t against twist cross-talk
        float tUsed = tRaw;
        if (twistActive)
        {
            if (Mathf.Abs(tRaw - _tPrevUsed) <= Mathf.Max(0f, tJitterHoldDelta))
                tUsed = _tPrevUsed;
        }
        _tPrevUsed = tUsed;
        debug_tUsed = tUsed;

        // Slide target from tUsed (only when thumb ON index; otherwise hold)
        float slideTarget = _slideSm;

        if (_thumbOnIndexStable)
        {
            if (useFixedNeutralMidpoint)
            {
                _tNeutral = fixedNeutralT;
                _neutralCaptured = true;
            }
            else
            {
                if (!_neutralCaptured)
                {
                    _tNeutral = tUsed;
                    _neutralCaptured = true;
                }
            }

            float rawSlide = TToAxisValue(tUsed, _tNeutral);
            if (invertSlide) rawSlide = -rawSlide;
            slideTarget = rawSlide;
        }
        else
        {
            slideTarget = _slideSm; // hold
        }

        // Smooth slide + rate limit
        {
            float k = 1f - Mathf.Exp(-Mathf.Max(0.01f, outputLerp) * dt);
            float candidate = Mathf.Lerp(_slideSm, slideTarget, k);

            if (maxOutputRatePerSec > 0f)
            {
                float maxStep = maxOutputRatePerSec * dt;
                float d = candidate - _slideSm;
                d = Mathf.Clamp(d, -maxStep, maxStep);
                candidate = _slideSm + d;
            }
            _slideSm = candidate;
        }

        // Twist normalize (always maps to Y)
        float twistTarget = 0f;
        {
            float denom = Mathf.Max(1e-3f, Mathf.Abs(twistMaxAbsDegForNormalize));
            float v = Mathf.Clamp(twistDeg / denom, -1f, 1f);

            float dz = Mathf.Max(0f, twistDeadZoneNorm);
            if (Mathf.Abs(v) < dz) v = 0f;
            else
            {
                float s = (Mathf.Abs(v) - dz) / Mathf.Max(1e-4f, (1f - dz));
                v = Mathf.Sign(v) * Mathf.Clamp01(s);
            }

            if (invertTwist) v = -v;
            twistTarget = v;
        }

        // Smooth twist output
        if (twistOutputLerp > 0.01f)
        {
            float k = 1f - Mathf.Exp(-twistOutputLerp * dt);
            _twistSm = Mathf.Lerp(_twistSm, twistTarget, k);
        }
        else
        {
            _twistSm = twistTarget;
        }

        // Publish outputs
        AxisValue = _slideSm;  // always slide value
        AxisY = _twistSm;      // twist always Y

        if (Mode == AxisMode.XY)
        {
            AxisX = _slideSm;  // slide -> X
            AxisZ = 0f;
        }
        else // Mode == AxisMode.Z
        {
            AxisX = 0f;
            AxisZ = _slideSm;  // slide -> Z
        }

        if (debug && string.IsNullOrEmpty(debug_state))
            debug_state = (Mode == AxisMode.XY) ? "Mode=XY (Slide->X, Twist->Y)" : "Mode=Z (Slide->Z, Twist->Y)";
    }

    void ToggleModeXY_Z()
    {
        Mode = (Mode == AxisMode.XY) ? AxisMode.Z : AxisMode.XY;

        // Optional: clear slide to avoid surprise if used by different axis after toggle
        _slideSm = 0f;
        AxisX = 0f;
        AxisZ = 0f;
        AxisValue = 0f;
    }

    void UpdateThumbOnIndexStable(float dt, float distSm, bool twistActive)
    {
        bool onCandidate = distSm <= Mathf.Max(0f, thumbOnIndexDist);

        float offThresh = Mathf.Max(thumbOffIndexDist, thumbOnIndexDist + 0.001f);
        if (twistActive)
            offThresh += Mathf.Max(0f, offExtraMarginWhileTwisting);

        bool offCandidate = distSm >= offThresh;

        if (_thumbOnIndexStable)
        {
            if (offCandidate)
            {
                _offHeld += dt;
                _onHeld = 0f;
                if (_offHeld >= Mathf.Max(0.01f, offConfirmSec))
                {
                    _thumbOnIndexStable = false;
                    _offHeld = 0f;
                }
            }
            else
            {
                _offHeld = 0f;
                _onHeld = 0f;
            }
        }
        else
        {
            if (onCandidate)
            {
                _onHeld += dt;
                _offHeld = 0f;
                if (_onHeld >= Mathf.Max(0.01f, onConfirmSec))
                {
                    _thumbOnIndexStable = true;
                    _onHeld = 0f;
                }
            }
            else
            {
                _onHeld = 0f;
                _offHeld = 0f;
            }
        }
    }

    // Polyline-based projection:
    // Returns t in [0,1] along the polyline length, and minDist as the closest distance to any segment.
    // This makes both t and distance more stable when the index finger is bent.
    static float ProjectToPolylineT(Vector3 p, Vector3[] pts, int count, out float minDist)
    {
        minDist = float.PositiveInfinity;

        if (pts == null || count < 2)
        {
            minDist = 0f;
            return 0.5f;
        }

        float totalLen = 0f;
        for (int i = 0; i < count - 1; i++)
            totalLen += Vector3.Distance(pts[i], pts[i + 1]);

        if (totalLen < 1e-6f)
        {
            minDist = Vector3.Distance(p, pts[0]);
            return 0.5f;
        }

        float bestS = 0f;
        float sAccum = 0f;

        for (int i = 0; i < count - 1; i++)
        {
            Vector3 a = pts[i];
            Vector3 b = pts[i + 1];
            Vector3 ab = b - a;

            float denom = ab.sqrMagnitude;
            float u = (denom < 1e-10f) ? 0f : Vector3.Dot(p - a, ab) / denom;
            u = Mathf.Clamp01(u);

            Vector3 c = a + u * ab;
            float d = Vector3.Distance(p, c);

            float segLen = (denom < 1e-10f) ? 0f : Mathf.Sqrt(denom);

            if (d < minDist)
            {
                minDist = d;
                bestS = sAccum + u * segLen;
            }

            sAccum += segLen;
        }

        return Mathf.Clamp01(bestS / totalLen);
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

    bool HasRequiredJoints()
    {
        if (remoteHand == null) return false;
        if (remoteHand.remoteByIndex == null) return false;
        if (remoteHand.remoteByIndex.Length < 9) return false;

        if (remoteHand.remoteByIndex[4] == null) return false; // thumb tip
        if (remoteHand.remoteByIndex[5] == null) return false; // index mcp
        if (remoteHand.remoteByIndex[8] == null) return false; // index tip

        // index PIP (6) / DIP (7) are optional; script will fall back to MCP->TIP segment if missing
        return true;
    }
}
