using UnityEngine;



public class RemoteHandRuntime : MonoBehaviour

{

    [Header("Drivers (21 joints)")]

    public Transform[] remoteByIndex = new Transform[21]; // 0..20 (WRIST..PINKY_TIP)



    [Header("Palm frame drivers (for Wrist_Aim target)")]

    public Transform palmFwd; // Remote_PALM_FWD

    public Transform palmUp;  // Remote_PALM_UP



    [Header("Wrist Aim helper")]

    [Tooltip("Distance from wrist to aim target (m)")]

    public float palmAimDistance = 0.08f; // ~5-10cm recommended



    [Header("Options")]

    public bool isLeft = false;          // left/right hand flag (from sender)

    public bool manualTestMode = false;  // ignore incoming data when true



    [Header("Smoothing (positions)")]

    public float cutoffHz = 10f;         // LPF cutoff (8~12 recommended)

    public float maxStepMeters = 0.08f;  // per-frame position clamp



    [Header("Rig arming (0..1) — set autoArm=false to arm manually")]

    public HandRigArmer armer;

    public float firstValidDistance = 0.02f;

    public bool autoArm = false;         // false means you will call ArmNow() manually



    // ====== Preprocess / alignment / gating ======

    [Header("Preprocess (toggle as needed)")]

    public bool mmToMeters = false;      // true if incoming data is in millimeters

    public bool flipZ = false;           // swap RH/LH by flipping Z

    public bool flipY = false;           // flip Y axis

    public Transform hToUnity = null;    // optional H-world to Unity transform

    public bool yaw180 = false;          // rotate 180 degrees around Y (flip X/Z)

    public bool flipX = false;           // flip X axis only



    [Header("Validation / Gating")]

    public bool gateUntilSane = true;    // block until first sane wrist arrives

    public float saneMinRadius = 0.05f;

    public float saneMaxRadius = 10.0f;

    public float saneMinY = -1.0f;

    public float saneMaxY = 3.0f;



    [Header("Initial Offset (Option B) — keep proxy fixed")]

    public bool addInitialOffset = true; // capture initial offset on first valid frame

    public Transform rWrist;             // ProxyHandR/R_Wrist anchor (required)

    bool offsetCaptured = false;

    Vector3 initialOffset = Vector3.zero;

    Vector3 lastPreOffsetWrist = Vector3.zero; // wrist after preprocess, before offset



    [Header("Hand Scale (radial, from WRIST)")]

    public bool applyHandScale = false;  // enable hand scaling (default Off)

    public bool autoScaleOnce = false;   // auto capture on first valid frame

    public Transform[] skeletonByIndex = null; // 0..20 skeleton joints for auto scale

    public float manualHandScale = 1.0f; // manual scale if auto is off

    public float minHandScale = 0.5f, maxHandScale = 4.0f;

    bool scaleCaptured = false;

    float handScale = 1.0f;



    [Header("Extra offset from camera (for starting distance)")]

    [Tooltip("Optional forward/up offset from camera before capturing initial offset.")]

    public bool useExtraCameraOffset = true;

    public float extraForwardMeters = 0.25f;   // ~25cm forward

    public float extraUpMeters = 0.00f;        // vertical tweak





    // ====== Tip auto look / splay (default Off; enable if needed) ======

    [Header("Tip Auto Look & Roll (optional)")]

    public bool tipAutoLook = false;     // Off by default; rarely needed

    [Range(-60, 60)] public float rollThumb = -20f;

    [Range(-60, 60)] public float rollIndex = -10f;

    [Range(-60, 60)] public float rollMiddle = 0f;

    [Range(-60, 60)] public float rollRing = 8f;

    [Range(-60, 60)] public float rollPinky = 15f;



    [Header("Splay (optional)")]

    public bool applySplay = false;

    public float splayThumb = 0.010f;  // 10mm

    public float splayIndex = 0.005f;  // 5mm

    public float splayRing = 0.005f;   // 5mm

    public float splayPinky = 0.010f;  // 10mm



    [Header("Jitter control (per-finger pos)")]

    public float cutoffHzTips = 6f;

    public float maxStepTips = 0.015f;

    [Range(0f, 1f)] public float tipRotLerp = 0.18f;



    [Header("Palm/Wrist stabilization (rotation)")]

    public bool computePalmFrame = true; // compute palmFwd/Up transforms

    public bool stabilizeWrist = true;   // apply angular clamps/smoothing

    [Range(0f, 1f)] public float palmSlerp = 0.18f;

    public float maxPalmDegPerSec = 540f;

    public float maxRollDegPerSec = 240f;

    [Range(0.0f, 1.0f)] public float nearParallelDot = 0.92f;



    [Header("Debug logging")]

    public bool logFirstRaw = true;

    public bool logFirstAfter = true;

    public int debugEveryNFrames = 0;



    // === Aim (position-only) ===

    [Header("Aim (position-only; no roll/up)")]

    [Tooltip("If true, palmFwd moves position only (UpType=None on Wrist_Aim).")]

    public bool aimPositionOnly = true;      // default ON

    [Tooltip("If true, aim from WRIST->MIDDLE_MCP vector.")]

    public bool aimFromVelocity = true;      // velocity-based aim

    public float aimVelMinSpeed = 0.05f;     // m/s threshold for velocity aim

    [Range(0f, 1f)] public float aimDirLerp = 0.20f; // aim direction smoothing

    public float aimGlitchThreshDeg = 75f;   // discard sudden aim flips above this

    public float aimMaxDegPerSec = 225f;     // clamp aim turn rate per second



    Vector3 _wristPrev; bool _haveWristPrev = false;

    Vector3 _aimDirSm = Vector3.zero;



    // ===== Internal state =====

    Quaternion[] _prevTipRot = new Quaternion[21];

    bool _tipRotInit = false;

    bool IsTip(int i) => (i == 4 || i == 8 || i == 12 || i == 16 || i == 20);



    [Header("TEST (read-only helpers)")]

    public Transform remoteWrist;   // RemoteHand/Drivers/Remote_WRIST

    public Transform proxyRoot;     // ProxyHandR

    public Transform rightHand;     // ProxyHandR/RightHand



    Vector3 _pDrv, _pBone, _pProxy, _pMesh;



    Vector3[] prev = new Vector3[21];

    bool init = false;



    Quaternion prevPalmRot;

    bool palmInit = false;



    bool firstArmed = false;

    bool firstValid = false;

    bool firstRawLogged = false;

    bool firstAfterLogged = false;

    int dbgFrame = 0;



    // ===== Main entry: apply streamed joints =====

    public void ApplyWorldPositions(Vector3[] worldPos)

    {

        if (manualTestMode || worldPos == null || worldPos.Length < 21) return;



        // (A) RAW first-frame logging

        if (logFirstRaw && !firstRawLogged)

        {

            firstRawLogged = true;

            var w0 = worldPos[0];

            var w8 = worldPos[8];

            float span = Vector3.Distance(w0, w8);

            var cam = Camera.main ? Camera.main.transform : null;

            Vector3 camLocal = cam ? cam.InverseTransformPoint(w0) : w0;

            //DebugHUD_LogSafe($"[DATA-RAW] WRIST:{w0}  span:{span:F3}m  camLocal:{camLocal}");

        }



        // (B) Preprocess (units/axes/session)

        PreprocessInPlace(worldPos);



        // (B-2) Hand scale capture (once)

        if (!scaleCaptured)

        {

            if (autoScaleOnce)

            {

                float r = RemoteAvgSpan(worldPos);

                float k = SkelAvgSpan();

                if (r > 1e-4f && k > 1e-4f)

                {

                    handScale = Mathf.Clamp(k / r, minHandScale, maxHandScale);

                    scaleCaptured = true;

                    //DebugHUD_LogSafe($"[SCALE] auto handScale={handScale:F3} (remote={r:F3}, skel={k:F3})");

                }

            }

            else

            {

                handScale = Mathf.Clamp(manualHandScale, minHandScale, maxHandScale);

                scaleCaptured = true;

                //DebugHUD_LogSafe($"[SCALE] manual handScale={handScale:F3}");

            }

        }



        // (B-3) Apply scale (optional)

        if (applyHandScale && scaleCaptured) ApplyHandScaleInPlace(worldPos, handScale);



        // (C) Save wrist before offset

        lastPreOffsetWrist = worldPos[0];



        // (D) Gate on sanity

        if (gateUntilSane && !firstValid)

        {

            if (!IsSane(worldPos[0])) return;

            firstValid = true;

            init = false; // reset smoothing

        }



        // (E) Capture initial offset

        if (addInitialOffset && !offsetCaptured)

        {

            if (!gateUntilSane || firstValid)

            {

                if (rWrist != null)

                {

                    // anchor = rWrist + optional camera forward/up offset

                    Vector3 anchorPos = rWrist.position;



                    if (useExtraCameraOffset && Camera.main != null)

                    {

                        var cam = Camera.main.transform;

                        anchorPos += cam.forward * extraForwardMeters

                                     + cam.up * extraUpMeters;

                    }



                    initialOffset = anchorPos - lastPreOffsetWrist;



                    offsetCaptured = true;

                    init = false; // reset smoothing after offset capture

                    //DebugHUD_LogSafe($"[OFFSET] captured {initialOffset}  anchorPos={anchorPos}");



                }

                else

                {

                    //DebugHUD_LogSafe("[OFFSET] rWrist is null - cannot capture initial offset");

                }

            }

        }



        // (F) Apply offset

        if (addInitialOffset && offsetCaptured)

            for (int i = 0; i < 21; i++) worldPos[i] += initialOffset;



        // (G) AFTER first-frame logging

        if (logFirstAfter && !firstAfterLogged)

        {

            firstAfterLogged = true;

            var w0 = worldPos[0];

            var w8 = worldPos[8];

            float span = Vector3.Distance(w0, w8);

            var cam = Camera.main ? Camera.main.transform : null;

            Vector3 camLocal = cam ? Camera.main.transform.InverseTransformPoint(w0) : w0;

            //DebugHUD_LogSafe($"[DATA-AFTER] WRIST:{w0}  span:{span:F3}m  camLocal:{camLocal}");

        }



        // (H) Periodic debug (optional)

        if (debugEveryNFrames > 0 && ((dbgFrame++ % debugEveryNFrames) == 0))

        {

            var w0 = worldPos[0];

            var cam = Camera.main ? Camera.main.transform : null;

            Vector3 camLocal = cam ? cam.InverseTransformPoint(w0) : w0;

            // DebugHUD_LogSafe($"[DATA-STREAM] worldY:{w0.y:F3} camLocal:{camLocal}");

        }



        // (I) Optional splay

        ApplySplayInPlace(worldPos);



        // (J) Smooth + step clamp + apply to Remote_* (positions only)

        SmoothAndApply(worldPos);



        // (K) Aim targets

        if (computePalmFrame)

        {

            if (aimPositionOnly) UpdateAimPositionOnly();  // position-only mode

            else UpdatePalmFrame_Full();                   // full rotation mode

        }



        // (L) Optional tip rotations

        if (tipAutoLook) UpdateFingerTipFrames();



        // (M) Auto-arm once valid

        if (autoArm && !firstArmed && FrameLooksValid(worldPos) && (!gateUntilSane || firstValid))

        {

            if (armer) armer.ArmNow();

            firstArmed = true;

        }



        // ===== TEST logging =====

        if (!remoteWrist || !rWrist || !proxyRoot || !rightHand) return;



        var d = remoteWrist.position;

        var b = rWrist.position;

        var p = proxyRoot.position;

        var m = rightHand.position;



        if ((d - _pDrv).sqrMagnitude > 1e-6f ||

            (b - _pBone).sqrMagnitude > 1e-6f ||

            (p - _pProxy).sqrMagnitude > 1e-6f ||

            (m - _pMesh).sqrMagnitude > 1e-6f)

        {

            // DebugHUD_LogSafe($"[WHO-MOVED] DRV:{d}  BONE:{b}  PROXY:{p}  MESH:{m}");

            _pDrv = d; _pBone = b; _pProxy = p; _pMesh = m;

        }

    }



    // --- Smoothing + step clamp ---

    void SmoothAndApply(Vector3[] inPos)

    {

        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);



        for (int i = 0; i < 21; i++)

        {

            // tips use lower cutoff and tighter step clamp

            float cutoff = IsTip(i) ? cutoffHzTips : cutoffHz;

            float omega = 2f * Mathf.PI * cutoff;

            float alpha = omega * dt / (1f + omega * dt);

            alpha = IsTip(i) ? Mathf.Clamp(alpha, 0.02f, 0.30f) : Mathf.Clamp01(alpha);



            Vector3 v = inPos[i];

            if (!init) prev[i] = v;



            Vector3 raw = Vector3.Lerp(prev[i], v, alpha);



            float stepCap = IsTip(i) ? maxStepTips : maxStepMeters;

            Vector3 d = raw - prev[i];

            float m = d.magnitude;

            if (m > stepCap) raw = prev[i] + d.normalized * stepCap;



            // apply only to Remote_* driver transforms (positions only)

            remoteByIndex[i].position = raw;

            prev[i] = raw;

        }

        init = true;

    }



    // === Position-only aim: place forward target (UpType=None) ===

    void UpdateAimPositionOnly()

    {

        Vector3 w = remoteByIndex[0].position; // WRIST (smoothed)

        Vector3 dir;



        // 1) derive direction

        if (aimFromVelocity)

        {

            if (!_haveWristPrev) { _wristPrev = w; _haveWristPrev = true; }

            float dt = Mathf.Max(Time.deltaTime, 1e-4f);

            Vector3 v = (w - _wristPrev) / dt; // m/s

            _wristPrev = w;



            if (v.magnitude >= Mathf.Max(1e-4f, aimVelMinSpeed)) dir = v;

            else dir = (remoteByIndex[9].position - w); // fallback WRIST->MIDDLE_MCP

        }

        else

        {

            dir = (remoteByIndex[9].position - w); // pose-based

        }



        if (dir.sqrMagnitude < 1e-8f) return;



        // 2) normalize and keep aim in camera front hemisphere

        Vector3 dN = dir.normalized;

        var cam = Camera.main;

        if (cam != null)

        {

            Vector3 camFwd = cam.transform.forward;

            float dot = Vector3.Dot(dN, camFwd);

            if (dot < 0f)

            {

                if (_aimDirSm.sqrMagnitude > 1e-6f)

                    dN = _aimDirSm.normalized; // keep previous aim to avoid flipping behind camera

                else

                    dN = camFwd;               // fallback to camera forward if no history

            }

        }



        // 2b) glitch rejection and angular clamp vs previous aim

        if (_aimDirSm.sqrMagnitude > 1e-6f)

        {

            Vector3 prevDir = _aimDirSm.normalized;

            float angleDeg = Vector3.Angle(prevDir, dN);



            if (angleDeg > aimGlitchThreshDeg)

            {

                dN = prevDir; // drop sudden flip

            }

            else

            {

                float dt = Mathf.Max(Time.deltaTime, 1f / 120f);

                float maxStepDeg = aimMaxDegPerSec * dt;

                if (angleDeg > maxStepDeg && angleDeg > 1e-3f)

                {

                    float t = maxStepDeg / angleDeg;

                    dN = Vector3.Slerp(prevDir, dN, t);

                }

            }

        }



        if (_aimDirSm == Vector3.zero) _aimDirSm = dN;

        else _aimDirSm = Vector3.Slerp(_aimDirSm, dN, Mathf.Clamp01(aimDirLerp));



        // 3) place forward target (no roll/up)

        Vector3 aimPos = w + _aimDirSm * Mathf.Max(0.01f, palmAimDistance);

        palmFwd.position = aimPos;



        // UpType=None: set palmUp to vertical for debug only

        if (palmUp) palmUp.position = w + Vector3.up * Mathf.Max(0.01f, palmAimDistance);

    }



    // --- Full palm frame with rotation stabilization ---

    void UpdatePalmFrame_Full()

    {

        // joints

        Vector3 w = remoteByIndex[0].position;   // WRIST

        Vector3 i = remoteByIndex[5].position;   // INDEX_MCP

        Vector3 m = remoteByIndex[9].position;   // MIDDLE_MCP

        Vector3 p = remoteByIndex[17].position;  // PINKY_MCP



        // forward/up candidates

        Vector3 fwd = (m - w);

        if (fwd.sqrMagnitude < 1e-10f) return;

        fwd.Normalize();



        Vector3 upC = isLeft ? Vector3.Cross(i - w, p - w) : Vector3.Cross(p - w, i - w);

        if (upC.sqrMagnitude < 1e-10f) upC = prevPalmRot * Vector3.up; // fallback if degenerate

        upC.Normalize();



        // orthonormal basis

        Vector3 side = Vector3.Cross(upC, fwd);

        if (side.sqrMagnitude < 1e-10f && palmInit)

            side = Vector3.Cross(prevPalmRot * Vector3.up, fwd);

        side.Normalize();

        Vector3 up = Vector3.Cross(fwd, side).normalized;



        // target rotation

        Quaternion target = Quaternion.LookRotation(fwd, up);



        if (stabilizeWrist && palmInit)

        {

            // avoid 180 flips

            if (Quaternion.Dot(prevPalmRot, target) < 0f)

                target = new Quaternion(-target.x, -target.y, -target.z, -target.w);



            // clamp angular speed

            float ang = Quaternion.Angle(prevPalmRot, target);

            float maxStep = maxPalmDegPerSec * Mathf.Max(Time.deltaTime, 1f / 120f);

            float tStep = (ang > 1e-5f) ? Mathf.Clamp01(maxStep / ang) : 1f;

            Quaternion clamped = Quaternion.Slerp(prevPalmRot, target, tStep);



            // clamp roll drift

            if (maxRollDegPerSec > 0f)

            {

                Vector3 prevUpProj = Vector3.ProjectOnPlane(prevPalmRot * Vector3.up, fwd).normalized;

                Vector3 newUpProj = Vector3.ProjectOnPlane(clamped * Vector3.up, fwd).normalized;

                if (prevUpProj.sqrMagnitude > 1e-10f && newUpProj.sqrMagnitude > 1e-10f)

                {

                    float rollDelta = Vector3.SignedAngle(prevUpProj, newUpProj, fwd);

                    float maxRoll = maxRollDegPerSec * Mathf.Max(Time.deltaTime, 1f / 120f);

                    float rollClamped = Mathf.Clamp(rollDelta, -maxRoll, +maxRoll);

                    float correction = rollClamped - rollDelta;

                    clamped = Quaternion.AngleAxis(correction, fwd) * clamped;

                }

            }



            // final smoothing

            target = Quaternion.Slerp(prevPalmRot, clamped, Mathf.Clamp01(palmSlerp));

        }



        prevPalmRot = target;

        palmInit = true;



        // place palmFwd/palmUp relative to wrist

        palmFwd.SetPositionAndRotation(w + fwd * palmAimDistance, target);

        palmUp.SetPositionAndRotation(w + (prevPalmRot * Vector3.up) * palmAimDistance, target);

    }



    // --- Gating helper ---

    bool FrameLooksValid(Vector3[] pos)

    {

        float d1 = Vector3.Distance(pos[0], pos[5]);  // WRIST->INDEX_MCP

        float d2 = Vector3.Distance(pos[0], pos[9]);  // WRIST->MIDDLE_MCP

        return (d1 > firstValidDistance && d2 > firstValidDistance);

    }



    // === Preprocess (mm->m, flips, transform) ===

    void PreprocessInPlace(Vector3[] a)

    {

        // 1) units

        if (mmToMeters) for (int i = 0; i < 21; i++) a[i] *= 0.001f;



        // 2) flips

        if (yaw180 || flipX || flipY || flipZ)

        {

            for (int i = 0; i < 21; i++)

            {

                var p = a[i];

                if (yaw180) { p.x = -p.x; p.z = -p.z; }  // Yaw 180 degrees
                else

                {

                    if (flipX) p.x = -p.x;              // flip X

                    if (flipZ) p.z = -p.z;              // flip Z (swap handedness)

                }

                if (flipY) p.y = -p.y;                  // flip Y

                a[i] = p;

            }

        }



        // 3) transform H->Unity

        if (hToUnity)

            for (int i = 0; i < 21; i++)

                a[i] = hToUnity.TransformPoint(a[i]);

    }



    // === Basic sanity check ===

    bool IsSane(Vector3 p)

    {

        if (!IsFinite(p)) return false;

        float d = p.magnitude;

        if (d < saneMinRadius || d > saneMaxRadius) return false;

        if (p.y < saneMinY || p.y > saneMaxY) return false;

        return true;

    }

    static bool IsFinite(Vector3 v) => IsFinite(v.x) && IsFinite(v.y) && IsFinite(v.z);

    static bool IsFinite(float f) => !(float.IsNaN(f) || float.IsInfinity(f));



    // === Context menu helpers: offsets ===

    [ContextMenu("Offset/Clear & Re-arm")]

    public void ContextClearAndRearm()

    {

        offsetCaptured = false;

        initialOffset = Vector3.zero;

        firstValid = false;

        firstArmed = false;

        init = false;

        //DebugHUD_LogSafe("[OFFSET] cleared; will capture again on next valid frame.");

    }



    [ContextMenu("Offset/Recapture Now (use last pre-offset wrist)")]

    public void ContextRecaptureNow()

    {

        if (rWrist == null)

        {

            DebugHUD_LogSafe("[OFFSET] rWrist null - cannot recapture.");

            return;

        }

        initialOffset = rWrist.position - lastPreOffsetWrist;

        offsetCaptured = true;

        init = false;

        //DebugHUD_LogSafe($"[OFFSET] recaptured {initialOffset}");

    }



    // Average remote span (wrist -> tips)

    float RemoteAvgSpan(Vector3[] a)

    {

        int[] tips = { 4, 8, 12, 16, 20 }; // THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP

        float s = 0f; int n = 0;

        foreach (int idx in tips) { s += Vector3.Distance(a[0], a[idx]); n++; }

        return n > 0 ? s / n : 0f;

    }



    // Average skeleton span (optional)

    float SkelAvgSpan()

    {

        if (skeletonByIndex == null || skeletonByIndex.Length < 21) return 0f;

        if (!skeletonByIndex[0]) return 0f;

        int[] tips = { 4, 8, 12, 16, 20 };

        float s = 0f; int n = 0;

        foreach (int idx in tips)

        {

            if (skeletonByIndex[idx])

            {

                s += Vector3.Distance(skeletonByIndex[0].position, skeletonByIndex[idx].position);

                n++;

            }

        }

        return n > 0 ? s / n : 0f;

    }



    // Apply radial hand scale around wrist

    void ApplyHandScaleInPlace(Vector3[] a, float s)

    {

        if (Mathf.Abs(s - 1f) < 1e-4f) return;

        Vector3 w = a[0];

        for (int i = 1; i < 21; i++) a[i] = w + (a[i] - w) * s;

    }



    void UpdateFingerTipFrames()

    {

        if (!tipAutoLook || palmUp == null) return;

        Vector3 up = palmUp.up;



        int[,] pairs = new int[,] { { 1, 4 }, { 5, 8 }, { 9, 12 }, { 13, 16 }, { 17, 20 } }; // [MCP, TIP]

        float[] rolls = new float[] { rollThumb, rollIndex, rollMiddle, rollRing, rollPinky };



        for (int k = 0; k < 5; k++)

        {

            int mcp = pairs[k, 0], tip = pairs[k, 1];

            var mcpT = remoteByIndex[mcp];

            var tipT = remoteByIndex[tip];

            if (!mcpT || !tipT) continue;



            Vector3 dir = tipT.position - mcpT.position;

            if (dir.sqrMagnitude < 1e-8f) continue;



            Quaternion target = Quaternion.LookRotation(dir.normalized, up)

                                * Quaternion.AngleAxis(rolls[k], Vector3.forward);



            if (!_tipRotInit) { _prevTipRot[tip] = target; }



            Quaternion smoothed = Quaternion.Slerp(_prevTipRot[tip], target, Mathf.Clamp01(tipRotLerp));

            _prevTipRot[tip] = smoothed;



            // apply to TIP driver only (Rig_Fingers handles bones)

            tipT.rotation = smoothed;

        }

        _tipRotInit = true;

    }



    void ApplySplayInPlace(Vector3[] a)

    {

        if (!applySplay || palmUp == null) return;



        Vector3 w = a[0];                   // WRIST

        Vector3 fwd = (a[9] - w);           // WRIST->MIDDLE_MCP direction

        if (fwd.sqrMagnitude < 1e-6f) return;



        Vector3 side = Vector3.Cross(palmUp.up, fwd).normalized; // lateral direction

        int TH_CMC = 1, TH_TIP = 4, IX_MCP = 5, IX_TIP = 8, RG_MCP = 13, RG_TIP = 16, PK_MCP = 17, PK_TIP = 20;

        float s = isLeft ? -1f : 1f;



        a[TH_CMC] += -s * side * splayThumb; a[TH_TIP] += -s * side * splayThumb;

        a[IX_MCP] += -s * side * splayIndex; a[IX_TIP] += -s * side * splayIndex;

        a[RG_MCP] += s * side * splayRing; a[RG_TIP] += s * side * splayRing;

        a[PK_MCP] += s * side * splayPinky; a[PK_TIP] += s * side * splayPinky;

    }



    // Safe logging to DebugHUD (if present) else Unity log

    void DebugHUD_LogSafe(string s)

    {

        try { DebugHUD.Log(s); } catch { Debug.Log(s); }

    }

}





