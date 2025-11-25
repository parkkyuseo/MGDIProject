using UnityEngine;

public class RemoteHandRuntime : MonoBehaviour
{
    [Header("Drivers (21 joints)")]
    public Transform[] remoteByIndex = new Transform[21]; // 0..20 (WRIST..PINKY_TIP)

    [Header("Palm frame drivers (for Wrist_Aim target)")]
    public Transform palmFwd; // Remote_PALM_FWD
    public Transform palmUp;  // Remote_PALM_UP

    [Header("Wrist Aim helper")]
    [Tooltip("Wrist에서 손끝 방향으로 띄울 거리(m)")]
    public float palmAimDistance = 0.08f; // 5~10cm 권장

    [Header("Options")]
    public bool isLeft = false;          // 송신부에서 true/false 세팅 유지
    public bool manualTestMode = false;  // 테스트 모드(외부 입력 무시)

    [Header("Smoothing (positions)")]
    public float cutoffHz = 10f;         // 8~12 권장
    public float maxStepMeters = 0.08f;  // 프레임당 최대 이동 제한(스파이크 컷)

    [Header("Rig arming (0→1) — 필요 없으면 autoArm=false")]
    public HandRigArmer armer;
    public float firstValidDistance = 0.02f;
    public bool autoArm = false;         // false면 절대 ArmNow 호출 안 함(디버그용)

    // ====== 전처리/정렬/게이트 ======
    [Header("Preprocess (toggle as needed)")]
    public bool mmToMeters = false;      // 수신 값이 mm라면 true
    public bool flipZ = false;           // RH↔LH 교정(자주 사용)
    public bool flipY = false;           // 드물게 Y 반전 필요시
    public Transform hToUnity = null;    // (선택) H-world→Unity-world 정렬 트랜스폼
    public bool yaw180 = false;          // Y축 180° 회전(= X,Z 반전)
    public bool flipX = false;           // X만 단독 반전

    [Header("Validation / Gating")]
    public bool gateUntilSane = true;    // 첫 유효 좌표 전에는 적용 안 함
    public float saneMinRadius = 0.05f;
    public float saneMaxRadius = 10.0f;
    public float saneMinY = -1.0f;
    public float saneMaxY = 3.0f;

    [Header("Initial Offset (Option B) — keep proxy fixed")]
    public bool addInitialOffset = true; // 첫 유효 프레임에 오프셋 캡처하여 이후 모든 점에 더함
    public Transform rWrist;             // ProxyHandR/R_Wrist (본 루트) — 반드시 할당
    bool offsetCaptured = false;
    Vector3 initialOffset = Vector3.zero;
    Vector3 lastPreOffsetWrist = Vector3.zero; // 전처리 후, 오프셋 적용 전의 WRIST(참조용)

    [Header("Hand Scale (radial, from WRIST)")]
    public bool applyHandScale = false;  // 실제 적용할지 여부(기본 Off)
    public bool autoScaleOnce = false;   // true면 첫 유효 프레임에 자동 캡처
    public Transform[] skeletonByIndex = null; // 0..20 → 실제 본 매핑(있으면 자동 스케일에 사용)
    public float manualHandScale = 1.0f; // 수동 스케일(자동 미사용 시)
    public float minHandScale = 0.5f, maxHandScale = 4.0f;
    bool scaleCaptured = false;
    float handScale = 1.0f;

    [Header("Extra offset from camera (for starting distance)")]
    [Tooltip("홀로렌즈 머리 기준 앞으로 얼마나 더 당길지 (미터). 0이면 현재 rWrist 위치를 그대로 기준점으로 사용.")]
    public bool useExtraCameraOffset = true;
    public float extraForwardMeters = 0.25f;   // 25cm 정도 앞으로
    public float extraUpMeters = 0.00f;   // 필요하면 위/아래로 조절


    // ====== 팁 자동 회전/스플레이(기본 Off; 필요시만 On) ======
    [Header("Tip Auto Look & Roll (optional)")]
    public bool tipAutoLook = false;     // 기본 Off (리그 손가락 안 쓰면 보통 불필요)
    [Range(-60, 60)] public float rollThumb = -20f;
    [Range(-60, 60)] public float rollIndex = -10f;
    [Range(-60, 60)] public float rollMiddle = 0f;
    [Range(-60, 60)] public float rollRing = 8f;
    [Range(-60, 60)] public float rollPinky = 15f;

    [Header("Splay (optional)")]
    public bool applySplay = false;
    public float splayThumb = 0.010f;  // 10mm
    public float splayIndex = 0.005f;  // 5mm
    public float splayRing = 0.005f;  // 5mm
    public float splayPinky = 0.010f;  // 10mm

    [Header("Jitter control (per-finger pos)")]
    public float cutoffHzTips = 6f;
    public float maxStepTips = 0.015f;
    [Range(0f, 1f)] public float tipRotLerp = 0.18f;

    [Header("Palm/Wrist stabilization (회전 안정화)")]
    public bool computePalmFrame = true; // palmFwd/Up 계산 자체를 켜/끔
    public bool stabilizeWrist = true;   // 아래 각속도/롤 캡/스무딩 적용 여부
    [Range(0f, 1f)] public float palmSlerp = 0.18f;
    public float maxPalmDegPerSec = 540f;
    public float maxRollDegPerSec = 240f;
    [Range(0.0f, 1.0f)] public float nearParallelDot = 0.92f;

    [Header("Debug logging")]
    public bool logFirstRaw = true;
    public bool logFirstAfter = true;
    public int debugEveryNFrames = 0;

    // === NEW: 회전 제거 모드(조준점만) ===
    [Header("Aim (position-only; no roll/up)")]
    [Tooltip("True면 palmFwd는 '위치'만 갱신. Wrist_Aim은 UpType=None으로 타깃 위치만 조준하게 함.")]
    public bool aimPositionOnly = true;      // <- 기본 ON (요청 모드)
    [Tooltip("True면 손목의 이동 방향(속도)으로 조준. False면 손 모양(WRIST→MIDDLE_MCP)로 조준.")]
    public bool aimFromVelocity = true;      // 이동 방향 조준
    public float aimVelMinSpeed = 0.05f;     // m/s 이상일 때만 속도 사용
    [Range(0f, 1f)] public float aimDirLerp = 0.20f; // 조준 방향 저역 필터

    Vector3 _wristPrev; bool _haveWristPrev = false;
    Vector3 _aimDirSm = Vector3.zero;

    // ===== 내부 상태 =====
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

    // ===== UDP 호출부 (그대로 사용) =====
    public void ApplyWorldPositions(Vector3[] worldPos)
    {
        if (manualTestMode || worldPos == null || worldPos.Length < 21) return;

        // (A) RAW 1회 로그
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

        // (B) 전처리(단위/축/세션 정렬)
        PreprocessInPlace(worldPos);

        // (B-2) 손 스케일 결정(한 번만)
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

        // (B-3) 스케일 적용(옵션)
        if (applyHandScale && scaleCaptured) ApplyHandScaleInPlace(worldPos, handScale);

        // (C) 전처리 후 기준 WRIST 저장(오프셋 계산용)
        lastPreOffsetWrist = worldPos[0];

        // (D) 첫 유효 프레임 게이트
        if (gateUntilSane && !firstValid)
        {
            if (!IsSane(worldPos[0])) return;
            firstValid = true;
            init = false; // 스무딩 기준 리셋
        }

        // (E) 초기 오프셋 캡처
        if (addInitialOffset && !offsetCaptured)
        {
            if (!gateUntilSane || firstValid)
            {
                if (rWrist != null)
                {
                    // ★ 기준점: rWrist.position에서 카메라 앞쪽으로 extraForwardMeters만큼 더 나간 위치
                    Vector3 anchorPos = rWrist.position;

                    if (useExtraCameraOffset && Camera.main != null)
                    {
                        var cam = Camera.main.transform;
                        anchorPos += cam.forward * extraForwardMeters
                                     + cam.up * extraUpMeters;
                    }

                    initialOffset = anchorPos - lastPreOffsetWrist;

                    offsetCaptured = true;
                    init = false; // 다음 프레임 스무딩 기준 재설정
                    //DebugHUD_LogSafe($"[OFFSET] captured {initialOffset}  anchorPos={anchorPos}");

                }
                else
                {
                    //DebugHUD_LogSafe("[OFFSET] rWrist is null — cannot capture initial offset");
                }
            }
        }

        // (F) 오프셋 적용
        if (addInitialOffset && offsetCaptured)
            for (int i = 0; i < 21; i++) worldPos[i] += initialOffset;

        // (G) AFTER 1회 로그
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

        // (H) 연속 스트리밍 로그(선택)
        if (debugEveryNFrames > 0 && ((dbgFrame++ % debugEveryNFrames) == 0))
        {
            var w0 = worldPos[0];
            var cam = Camera.main ? Camera.main.transform : null;
            Vector3 camLocal = cam ? cam.InverseTransformPoint(w0) : w0;
            // DebugHUD_LogSafe($"[DATA-STREAM] worldY:{w0.y:F3} camLocal:{camLocal}");
        }

        // (I) TIP 벌림/정렬 등 (옵션)
        ApplySplayInPlace(worldPos);

        // (J) 좌표 스무딩 + 스텝 캡 + 드라이버에 대입(= 오직 Remote_*에만!)
        SmoothAndApply(worldPos);

        // (K) 손바닥 프레임(= Wrist_Aim 타깃)
        if (computePalmFrame)
        {
            if (aimPositionOnly) UpdateAimPositionOnly();  // 회전 제거 모드
            else UpdatePalmFrame_Full();   // 기존(회전 포함) 모드
        }

        // (L) 손가락 TIP 방향만(드라이버 회전) — 기본 Off
        if (tipAutoLook) UpdateFingerTipFrames();

        // (M) 첫 유효 프레임에 리그 켜기(원하면 사용)
        if (autoArm && !firstArmed && FrameLooksValid(worldPos) && (!gateUntilSane || firstValid))
        {
            if (armer) armer.ArmNow();
            firstArmed = true;
        }

        // ===== TEST 로그 =====
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

    // --- 내부: 스무딩 + 스텝 클램프 ---
    void SmoothAndApply(Vector3[] inPos)
    {
        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);

        for (int i = 0; i < 21; i++)
        {
            // TIP엔 더 낮은 컷오프/작은 스텝캡
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

            // ★ 오직 Remote_* 드라이버에만 position을 씀 (본/메시는 절대 X)
            remoteByIndex[i].position = raw;
            prev[i] = raw;
        }
        init = true;
    }

    // === NEW: 회전 제거 모드 — 타깃 '위치'만 갱신 (UpType=None 가정) ===
    void UpdateAimPositionOnly()
    {
        Vector3 w = remoteByIndex[0].position; // WRIST (스무딩 후)
        Vector3 dir;

        // 1) 조준 방향 계산
        if (aimFromVelocity)
        {
            if (!_haveWristPrev) { _wristPrev = w; _haveWristPrev = true; }
            float dt = Mathf.Max(Time.deltaTime, 1e-4f);
            Vector3 v = (w - _wristPrev) / dt; // m/s
            _wristPrev = w;

            if (v.magnitude >= Mathf.Max(1e-4f, aimVelMinSpeed)) dir = v;
            else dir = (remoteByIndex[9].position - w); // WRIST→MIDDLE_MCP 폴백
        }
        else
        {
            dir = (remoteByIndex[9].position - w); // 손 모양 기반
        }

        if (dir.sqrMagnitude < 1e-8f) return;

        // 2) 방향 스무딩(선택)
        Vector3 dN = dir.normalized;
        if (_aimDirSm == Vector3.zero) _aimDirSm = dN;
        else _aimDirSm = Vector3.Slerp(_aimDirSm, dN, Mathf.Clamp01(aimDirLerp));

        // 3) 타깃 '위치'만 갱신 (회전은 건드리지 않음)
        Vector3 aimPos = w + _aimDirSm * Mathf.Max(0.01f, palmAimDistance);
        palmFwd.position = aimPos;

        // UpType=None이면 palmUp은 쓰이지 않지만, 디버깅용으로 올려 둠
        if (palmUp) palmUp.position = w + Vector3.up * Mathf.Max(0.01f, palmAimDistance);
        // 회전은 의도적으로 설정하지 않음(UpType=None에서 무시됨)
    }

    // --- 기존: 손바닥 프레임 업데이트(회전 포함) ---
    void UpdatePalmFrame_Full()
    {
        // 데이터
        Vector3 w = remoteByIndex[0].position;   // WRIST
        Vector3 i = remoteByIndex[5].position;   // INDEX_MCP
        Vector3 m = remoteByIndex[9].position;   // MIDDLE_MCP
        Vector3 p = remoteByIndex[17].position;  // PINKY_MCP

        // 전방/업 후보
        Vector3 fwd = (m - w);
        if (fwd.sqrMagnitude < 1e-10f) return;
        fwd.Normalize();

        Vector3 upC = isLeft ? Vector3.Cross(i - w, p - w) : Vector3.Cross(p - w, i - w);
        if (upC.sqrMagnitude < 1e-10f) upC = prevPalmRot * Vector3.up; // 완전 휑하면 이전 업 사용
        upC.Normalize();

        // 직교기저 재구성
        Vector3 side = Vector3.Cross(upC, fwd);
        if (side.sqrMagnitude < 1e-10f && palmInit)
            side = Vector3.Cross(prevPalmRot * Vector3.up, fwd);
        side.Normalize();
        Vector3 up = Vector3.Cross(fwd, side).normalized;

        // 목표 회전
        Quaternion target = Quaternion.LookRotation(fwd, up);

        if (stabilizeWrist && palmInit)
        {
            // 쿼터니언 부호 정리
            if (Quaternion.Dot(prevPalmRot, target) < 0f)
                target = new Quaternion(-target.x, -target.y, -target.z, -target.w);

            // 전체 각속도 제한
            float ang = Quaternion.Angle(prevPalmRot, target);
            float maxStep = maxPalmDegPerSec * Mathf.Max(Time.deltaTime, 1f / 120f);
            float tStep = (ang > 1e-5f) ? Mathf.Clamp01(maxStep / ang) : 1f;
            Quaternion clamped = Quaternion.Slerp(prevPalmRot, target, tStep);

            // 롤 제한
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

            // 저주파 스무딩
            target = Quaternion.Slerp(prevPalmRot, clamped, Mathf.Clamp01(palmSlerp));
        }

        prevPalmRot = target;
        palmInit = true;

        // Wrist_Aim이 "위치"를 조준할 수 있게, 손목 기준으로 살짝 띄워 배치
        palmFwd.SetPositionAndRotation(w + fwd * palmAimDistance, target);
        palmUp.SetPositionAndRotation(w + (prevPalmRot * Vector3.up) * palmAimDistance, target);
    }

    // --- 내부: 첫 프레임 유효성 체크(간단) ---
    bool FrameLooksValid(Vector3[] pos)
    {
        float d1 = Vector3.Distance(pos[0], pos[5]);  // WRIST↔INDEX_MCP
        float d2 = Vector3.Distance(pos[0], pos[9]);  // WRIST↔MIDDLE_MCP
        return (d1 > firstValidDistance && d2 > firstValidDistance);
    }

    // === 전처리 (mm→m, 축 반전, 세션 정렬) ===
    void PreprocessInPlace(Vector3[] a)
    {
        // 1) 단위
        if (mmToMeters) for (int i = 0; i < 21; i++) a[i] *= 0.001f;

        // 2) 축 반전/회전
        if (yaw180 || flipX || flipY || flipZ)
        {
            for (int i = 0; i < 21; i++)
            {
                var p = a[i];
                if (yaw180) { p.x = -p.x; p.z = -p.z; }  // Yaw 180°(세계 Y 축 기준)
                else
                {
                    if (flipX) p.x = -p.x;              // 좌↔우
                    if (flipZ) p.z = -p.z;              // 앞↔뒤 (RH↔LH 전환 대표)
                }
                if (flipY) p.y = -p.y;                  // 위↔아래
                a[i] = p;
            }
        }

        // 3) 세션 정렬(H→Unity)
        if (hToUnity)
            for (int i = 0; i < 21; i++)
                a[i] = hToUnity.TransformPoint(a[i]);
    }

    // === 좌표 sanity 검사 ===
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

    // === 편의 기능: 오프셋 재설정 ===
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
            DebugHUD_LogSafe("[OFFSET] rWrist null — cannot recapture.");
            return;
        }
        initialOffset = rWrist.position - lastPreOffsetWrist;
        offsetCaptured = true;
        init = false;
        //DebugHUD_LogSafe($"[OFFSET] recaptured {initialOffset}");
    }

    // 리모트(수신) 손목→팁 평균 거리
    float RemoteAvgSpan(Vector3[] a)
    {
        int[] tips = { 4, 8, 12, 16, 20 }; // THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP
        float s = 0f; int n = 0;
        foreach (int idx in tips) { s += Vector3.Distance(a[0], a[idx]); n++; }
        return n > 0 ? s / n : 0f;
    }

    // 스켈레톤 손목→팁 평균 거리(옵션)
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

    // 손 스케일 적용(손목을 기준으로 방사형 스케일)
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

            // ★ TIP '드라이버' 회전만 건드림(본 회전 X) — Rig_Fingers가 꺼져있다면 메시엔 영향 없음
            tipT.rotation = smoothed;
        }
        _tipRotInit = true;
    }

    void ApplySplayInPlace(Vector3[] a)
    {
        if (!applySplay || palmUp == null) return;

        Vector3 w = a[0];                   // WRIST
        Vector3 fwd = (a[9] - w);           // WRIST→MIDDLE_MCP 대략 전방
        if (fwd.sqrMagnitude < 1e-6f) return;

        Vector3 side = Vector3.Cross(palmUp.up, fwd).normalized; // 손바닥 좌우
        int TH_CMC = 1, TH_TIP = 4, IX_MCP = 5, IX_TIP = 8, RG_MCP = 13, RG_TIP = 16, PK_MCP = 17, PK_TIP = 20;
        float s = isLeft ? -1f : 1f;

        a[TH_CMC] += -s * side * splayThumb; a[TH_TIP] += -s * side * splayThumb;
        a[IX_MCP] += -s * side * splayIndex; a[IX_TIP] += -s * side * splayIndex;
        a[RG_MCP] += s * side * splayRing; a[RG_TIP] += s * side * splayRing;
        a[PK_MCP] += s * side * splayPinky; a[PK_TIP] += s * side * splayPinky;
    }

    // DebugHUD가 OnGUI 레이아웃 에러를 낼 수 있어 try/catch 래핑
    void DebugHUD_LogSafe(string s)
    {
        try { DebugHUD.Log(s); } catch { Debug.Log(s); }
    }
}
