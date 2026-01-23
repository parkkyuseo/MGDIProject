using UnityEngine;

public class MicroThumbIndexSliderInput : MonoBehaviour
{
    public enum AxisMode { X = 0, Y = 1, Z = 2 }

    public enum DistancePlane
    {
        None3D = 0,
        Camera = 1,
        Palm = 2
    }

    [Header("References")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("Index axis (projection)")]
    [Tooltip("Thumb position projected to index MCP->TIP axis (0..1).")]
    [SerializeField] private bool useProjection = true;

    [Header("Neutral (t0) handling")]
    [Tooltip("If true, neutral t0 is fixed (recommended).")]
    [SerializeField] private bool useFixedNeutralMidpoint = true;
    [SerializeField] private float fixedNeutralT = 0.50f;

    [Header("Slider shaping")]
    [SerializeField] private float deadZoneT = 0.12f;
    [SerializeField] private float fullScaleT = 0.40f;
    [SerializeField] private float gamma = 1.6f;
    [SerializeField] private float outputLerp = 14f;
    [SerializeField] private float maxOutputRatePerSec = 6f;
    [SerializeField] private bool invertAxis = false;

    [Header("ThumbOnIndex detection (distance to index axis)")]
    [Tooltip("Plane used for thumb-index distance (depth-robust).")]
    [SerializeField] private DistancePlane distancePlane = DistancePlane.Palm;

    [Tooltip("Compute thumb->index-axis distance in the selected plane when possible.")]
    [SerializeField] private bool usePlaneForThumbIndexDistance = true;

    [Tooltip("Consider ON (thumb on index) if dist <= this (meters).")]
    [SerializeField] private float thumbOnIndexDist = 0.018f;

    [Tooltip("Consider OFF (thumb off index) if dist >= this (meters). Must be > thumbOnIndexDist.")]
    [SerializeField] private float thumbOffIndexDist = 0.028f;

    [Header("ThumbOnIndex stabilization (hysteresis + hold confirm)")]
    [Tooltip("Time to confirm ON candidate (seconds).")]
    [SerializeField] private float onConfirmSec = 0.05f;

    [Tooltip("Time to confirm OFF candidate (seconds).")]
    [SerializeField] private float offConfirmSec = 0.07f;

    [Header("Mode selection using OFF pulses + long OFF hold")]
    [Tooltip("Valid short OFF duration min (seconds).")]
    [SerializeField] private float pulseMinOffSec = 0.06f;

    [Tooltip("Valid short OFF duration max (seconds).")]
    [SerializeField] private float pulseMaxOffSec = 0.22f;

    [Tooltip("Second OFF pulse must happen within this window after first pulse (seconds).")]
    [SerializeField] private float doublePulseWindowSec = 0.35f;

    [Tooltip("OFF hold duration to arm Z toggle (seconds). Toggle happens on re-attach (OFF->ON).")]
    [SerializeField] private float zHoldSec = 0.35f;

    [Tooltip("Cooldown after a mode action to avoid rapid retriggers (seconds).")]
    [SerializeField] private float modeCooldownSec = 0.18f;

    [Header("Mode mapping")]
    [Tooltip("1 short OFF pulse selects X (immediately on re-attach).")]
    [SerializeField] private bool onePulseSelectsX = true;

    [Tooltip("2 short OFF pulses selects Y (on second re-attach).")]
    [SerializeField] private bool twoPulsesSelectY = true;

    [Tooltip("Long OFF hold selects/toggles Z (on re-attach).")]
    [SerializeField] private bool longHoldTogglesZ = true;

    [Tooltip("If true, short pulses can change mode even while in Z. If false, pulses are ignored in Z.")]
    [SerializeField] private bool pulsesAffectZ = false;

    [Header("Debug (read-only)")]
    [SerializeField] private bool debug = false;
    [SerializeField] private float debug_t = -1f;
    [SerializeField] private float debug_thumbIndexDist = -1f;
    [SerializeField] private bool debug_thumbOnIndex = false;
    [SerializeField] private int debug_pulseCount = 0;
    [SerializeField] private float debug_offHeldSec = 0f;
    [SerializeField] private float debug_doubleWindowRem = 0f;
    [SerializeField] private string debug_state = "";

    // Outputs
    public AxisMode Mode { get; private set; } = AxisMode.X;
    public float AxisValue { get; private set; } = 0f;

    // Optional: HUD/debug getters
    public float Debug_t => debug_t;
    public float Debug_thumbIndexDist => debug_thumbIndexDist;
    public bool Debug_thumbOnIndex => debug_thumbOnIndex;
    public int Debug_pulseCount => debug_pulseCount;
    public float Debug_offHeldSec => debug_offHeldSec;
    public float Debug_doubleWindowRemaining => debug_doubleWindowRem;
    public string Debug_state => debug_state;

    // Internal: neutral point
    float _tNeutral = 0.5f;
    bool _neutralCaptured = false;

    // Slider smoothing
    float _axisSm = 0f;

    // Stable thumb-on-index state (after confirm holds)
    bool _thumbOnIndexStable = false;
    float _onHeld = 0f;
    float _offHeld = 0f;

    // OFF segment tracking
    bool _inOffSegment = false;
    float _offStartTime = -1f;
    bool _zArmed = false;

    // Pulse selection
    int _pulseCount = 0;
    float _doubleWindowUntil = 0f;

    // Cooldown
    float _modeCooldownUntil = 0f;

    // Frame gating (update thumb state only on new rendered pose)
    int _lastRenderFrameId = -1;

    void Update()
    {
        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (!HasRequiredJoints())
        {
            AxisValue = 0f;
            _axisSm = 0f;
            if (debug) debug_state = "Missing joints";
            return;
        }

        // (Recommended) Gate state updates by RenderFrameId to avoid reprocessing same pose many times
        if (remoteHand != null)
        {
            int rid = remoteHand.RenderFrameId;
            if (rid == _lastRenderFrameId)
            {
                // Still run slider smoothing each Unity frame (looks nicer),
                // but do not advance thumb-on-index FSM using duplicated data.
                UpdateSliderOnly(dt);
                UpdateDebugWindowTimers();
                return;
            }
            _lastRenderFrameId = rid;
        }

        // Full update (new pose)
        float t, thumbIndexDist;
        ComputeProjectedTAndThumbIndexDist(out t, out thumbIndexDist);
        debug_t = t;
        debug_thumbIndexDist = thumbIndexDist;

        // Update stable thumb-on-index state (hysteresis + confirm holds)
        UpdateThumbOnIndexStable(dt, thumbIndexDist);

        // Update slider output (freeze to 0 while OFF, so no accidental movement during mode select)
        UpdateSliderWithKnownT(dt, t);

        // Process mode selection on stable OFF/ON transitions
        UpdateModeSelectionFSM();

        UpdateDebugWindowTimers();
    }

    void UpdateSliderOnly(float dt)
    {
        // Use last computed t if possible; if not, recompute quickly
        float t = debug_t;
        if (t < 0f || t > 1f)
        {
            float dummy;
            ComputeProjectedTAndThumbIndexDist(out t, out dummy);
            debug_t = t;
        }

        UpdateSliderWithKnownT(dt, t);
    }

    void UpdateSliderWithKnownT(float dt, float t)
    {
        // Neutral handling
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

        float rawAxis = TToAxisValue(t, _tNeutral);
        if (invertAxis) rawAxis = -rawAxis;

        // Freeze axis while OFF (mode selection posture)
        if (!_thumbOnIndexStable)
            rawAxis = 0f;

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
    }

    void UpdateThumbOnIndexStable(float dt, float thumbIndexDist)
    {
        bool onCandidate = thumbIndexDist <= Mathf.Max(0f, thumbOnIndexDist);
        bool offCandidate = thumbIndexDist >= Mathf.Max(thumbOffIndexDist, thumbOnIndexDist + 0.001f);

        // If neither candidate is strongly true, do not accumulate holds (prevents jitter in the mid band)
        if (!onCandidate && !offCandidate)
        {
            _onHeld = 0f;
            _offHeld = 0f;
            if (debug) debug_state = "ThumbState: in hysteresis band";
            debug_thumbOnIndex = _thumbOnIndexStable;
            return;
        }

        bool prevStable = _thumbOnIndexStable;

        if (_thumbOnIndexStable)
        {
            // Currently ON: look for confirmed OFF
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
            // Currently OFF: look for confirmed ON
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

        debug_thumbOnIndex = _thumbOnIndexStable;

        // Transition bookkeeping (stable edges)
        if (prevStable != _thumbOnIndexStable)
        {
            if (!_thumbOnIndexStable)
            {
                // ON -> OFF (stable): start OFF segment
                _inOffSegment = true;
                _offStartTime = Time.time;
                _zArmed = false;
                if (debug) debug_state = "Edge: ON->OFF (start segment)";
            }
            else
            {
                // OFF -> ON (stable): end OFF segment (mode action happens later)
                if (debug) debug_state = "Edge: OFF->ON (end segment)";
            }
        }

        // While stable OFF, arm long-hold Z
        if (_inOffSegment && !_thumbOnIndexStable)
        {
            float offHeld = Mathf.Max(0f, Time.time - _offStartTime);
            debug_offHeldSec = offHeld;

            if (longHoldTogglesZ && !_zArmed && offHeld >= Mathf.Max(0.05f, zHoldSec))
            {
                _zArmed = true;
                if (debug) debug_state = "Z armed (long OFF hold)";
            }
        }
        else
        {
            debug_offHeldSec = 0f;
        }
    }

    void UpdateModeSelectionFSM()
    {
        // Cooldown blocks actions, but still allow state tracking
        bool inCooldown = Time.time < _modeCooldownUntil;

        // If we are in an OFF segment and come back ON, decide what that OFF meant
        if (_inOffSegment && _thumbOnIndexStable)
        {
            _inOffSegment = false;

            float offDur = Mathf.Max(0f, Time.time - _offStartTime);
            _offStartTime = -1f;

            // If we're in cooldown, ignore the segment end (prevents chatter)
            if (inCooldown)
            {
                if (debug) debug_state = "Mode: ignored (cooldown)";
                return;
            }

            // Long hold -> Toggle Z on re-attach
            if (_zArmed && longHoldTogglesZ)
            {
                _zArmed = false;

                ToggleZ();
                _modeCooldownUntil = Time.time + Mathf.Max(0f, modeCooldownSec);

                // Clear pulse sequence
                _pulseCount = 0;
                _doubleWindowUntil = 0f;

                if (debug) debug_state = "Mode: Z (long hold)";
                return;
            }

            _zArmed = false;

            // Short OFF pulse?
            if (offDur >= pulseMinOffSec && offDur <= pulseMaxOffSec)
            {
                // If currently in Z and pulses should not affect it, ignore
                if (Mode == AxisMode.Z && !pulsesAffectZ)
                {
                    if (debug) debug_state = "Pulse ignored (in Z)";
                    return;
                }

                RegisterPulse();
                return;
            }

            // Otherwise: not a valid pulse, do nothing
            if (debug) debug_state = $"Segment ignored (offDur={offDur:F2}s)";
        }

        // If window expired with exactly 1 pulse, keep X selection already applied (we apply immediately),
        // just clear the pulse state to avoid lingering.
        if (_pulseCount == 1 && Time.time >= _doubleWindowUntil)
        {
            _pulseCount = 0;
            _doubleWindowUntil = 0f;

            if (debug) debug_state = "Pulse window expired (cleared)";
        }
    }

    void RegisterPulse()
    {
        // First pulse: select X immediately, and open double window
        if (_pulseCount == 0)
        {
            _pulseCount = 1;
            _doubleWindowUntil = Time.time + Mathf.Max(0.10f, doublePulseWindowSec);

            if (onePulseSelectsX)
            {
                SetModeX();
                _modeCooldownUntil = Time.time + Mathf.Max(0f, modeCooldownSec);
            }

            if (debug) debug_state = "Pulse #1 -> X selected (window open)";
            return;
        }

        // Second pulse within window: select Y
        if (_pulseCount == 1 && Time.time <= _doubleWindowUntil)
        {
            _pulseCount = 0;
            _doubleWindowUntil = 0f;

            if (twoPulsesSelectY)
            {
                SetModeY();
                _modeCooldownUntil = Time.time + Mathf.Max(0f, modeCooldownSec);
            }

            if (debug) debug_state = "Pulse #2 -> Y selected";
            return;
        }

        // Late pulse: treat as new first pulse
        _pulseCount = 1;
        _doubleWindowUntil = Time.time + Mathf.Max(0.10f, doublePulseWindowSec);

        if (onePulseSelectsX)
        {
            SetModeX();
            _modeCooldownUntil = Time.time + Mathf.Max(0f, modeCooldownSec);
        }

        if (debug) debug_state = "Pulse late -> X selected (window reset)";
    }

    void UpdateDebugWindowTimers()
    {
        debug_pulseCount = _pulseCount;
        debug_doubleWindowRem = Mathf.Max(0f, _doubleWindowUntil - Time.time);
    }

    // --------------------------
    // Mode actions
    // --------------------------
    void SetModeX()
    {
        if (Mode == AxisMode.Z && !pulsesAffectZ) return;

        Mode = AxisMode.X;
        RecenterOutputAfterModeChange();
    }

    void SetModeY()
    {
        if (Mode == AxisMode.Z && !pulsesAffectZ) return;

        Mode = AxisMode.Y;
        RecenterOutputAfterModeChange();
    }

    void ToggleZ()
    {
        Mode = (Mode == AxisMode.Z) ? AxisMode.X : AxisMode.Z;
        RecenterOutputAfterModeChange();
    }

    void RecenterOutputAfterModeChange()
    {
        _axisSm = 0f;
        AxisValue = 0f;

        if (!useFixedNeutralMidpoint)
            _neutralCaptured = false;
    }

    // --------------------------
    // Geometry
    // --------------------------
    bool HasRequiredJoints()
    {
        if (remoteHand == null) return false;
        if (remoteHand.remoteByIndex == null) return false;
        if (remoteHand.remoteByIndex.Length < 9) return false;

        // thumb tip (4), index mcp (5), index tip (8)
        if (remoteHand.remoteByIndex[4] == null) return false;
        if (remoteHand.remoteByIndex[5] == null) return false;
        if (remoteHand.remoteByIndex[8] == null) return false;

        // wrist (0) required for palm plane origin
        if (remoteHand.remoteByIndex[0] == null) return false;

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

        // 1) slider t in 3D (stable enough; keeps control direction consistent)
        t = Vector3.Dot(thumb - mcp, axis) / denom;
        t = Mathf.Clamp01(t);

        // 2) thumb->axis distance (optionally plane-projected)
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

    bool TryGetDistancePlane(out Vector3 origin, out Vector3 normal)
    {
        origin = Vector3.zero;
        normal = Vector3.forward;

        if (remoteHand == null || remoteHand.remoteByIndex == null || remoteHand.remoteByIndex.Length < 1)
            return false;

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
            // fallback to camera plane below
        }

        Camera cam = Camera.main;
        if (cam != null)
        {
            Vector3 f = cam.transform.forward;
            if (f.sqrMagnitude < 1e-8f) f = Vector3.forward;
            normal = f.normalized;
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
}
