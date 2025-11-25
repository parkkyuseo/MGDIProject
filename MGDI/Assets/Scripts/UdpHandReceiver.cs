using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Globalization;
using System.Text.RegularExpressions;
using UnityEngine;

#region DTOs (필요 최소)
[Serializable] public class GestureDto { public string name; public float score; }
[Serializable] public class PointDto { public string name; public float x, y, z; }
[Serializable]
public class HandPointsDto
{
    public string type;      // "hand_points"
    public string hand;      // "right"/"left"/...
    public long ts;
    public int frame_id;
    public PointDto[] points;
    public bool valid = true;

    public long ts_src_ms;
    public bool detected_now;
    public bool interpolated;
    public float palm_cz;
    public GestureDto gesture;
}
[Serializable] class TypeOnly { public string type; }
#endregion

public class UdpHandReceiver : MonoBehaviour
{
    // ========= 1) Remote 21 joints → RemoteHandRuntime =========
    [Header("Remote joints (21) → RemoteHandRuntime")]
    public bool forwardRemoteJoints = true;
    public RemoteHandRuntime remoteLeft;
    public RemoteHandRuntime remoteRight;
    public bool remoteHoldLastOnMissing = true;

    readonly object _remoteLock = new object();
    Vector3[] _remoteLastLeft = new Vector3[21];
    Vector3[] _remoteLastRight = new Vector3[21];
    bool _remoteHasLastLeft = false, _remoteHasLastRight = false;
    Vector3[] _remotePendingLeft = null;
    Vector3[] _remotePendingRight = null;
    bool _remotePendingFlagLeft = false, _remotePendingFlagRight = false;

    static readonly Regex s_joint4Regex = new Regex(
        @"\[\s*(?<x>null|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*" +
        @"(?<y>null|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*" +
        @"(?<z>null|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*" +
        @"(?<c>null|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]",
        RegexOptions.Singleline | RegexOptions.CultureInvariant
    );

    [Header("Incoming coords")]
    public bool invertIncomingY = false;

    [Header("Input rotation offset (sensor → Unity)")]
    [Tooltip("웹캠/Holo 좌표계가 Unity 월드와 축이 다를 때, 전체 joints에 적용할 고정 회전 오프셋입니다.")]
    public bool applyInputRotation = false;

    [Tooltip("센서 좌표계를 Unity 기준에 맞추기 위한 Euler 오프셋(도 단위, XYZ 축 순서)")]
    public Vector3 inputEulerOffset = Vector3.zero;

    // ========= 2) Grip → Animator(BlendTree: Grip01) =========
    [Header("Grip → Animator(BlendTree: Grip01)")]
    public bool forwardGripToAnimator = true;
    [SerializeField] GripParamDriver gripDriver;

    public bool gripUseModelGesture = true;
    public bool gripUseGeometry = true;
    [Range(0f, 1f)] public float gestureScoreTau = 0.65f;

    [Header("Grip thresholds (geometry)")]
    public float grip_TH_OPEN = 1.10f;
    public float grip_TH_CLOSE = 0.70f;

    [Header("Grip smoothing")]
    public float gripThUp = 0.65f;
    public float gripThDn = 0.35f;
    public float gripTau = 0.10f;
    public float gripDropoutSec = 0.30f;

    // 내부 그립 상태 (Animator용 + 이벤트 전파)
    float _gripSmoothed = 0f;    // 0..1 (LPF 결과)
    bool _gripLatched = false; // 히스테리시스 결과

    // ---------- [GRIP-STATE] 외부에 노출되는 그립 상태 ----------
    public enum GripState { Unknown, Open, Closed }
    GripState _gripState = GripState.Unknown;
    public GripState CurrentGripState => _gripState;

    public static event Action<GripState> OnGripStateChanged; // ProxyHandGrabber 호환
    [SerializeField] bool logGripChanges = false;
    float _gripLastRxTime = -1f;

    // ========= 3) Network =========
    [Header("Network")]
    public int listenPort = 33333;

    UdpClient _udp;
    IPEndPoint _any;
    Thread _recvThread;
    volatile bool _running;

    readonly object _pktLock = new object();
    HandPointsDto _latest;
    int _latestSeq = 0, _appliedSeq = -1;

    static readonly System.Diagnostics.Stopwatch s_sw = System.Diagnostics.Stopwatch.StartNew();
    static long NowMs() => s_sw.ElapsedMilliseconds;
    long _lastRecvMs = 0;

    void Start()
    {
        try
        {
            _udp = new UdpClient(listenPort);
            _any = new IPEndPoint(IPAddress.Any, 0);
            _running = true;
            _recvThread = new Thread(RecvLoop) { IsBackground = true };
            _recvThread.Start();
            Debug.Log($"[UdpHandReceiver] Listening UDP :{listenPort}");
        }
        catch (Exception e)
        {
            Debug.LogError("[UdpHandReceiver] UDP init failed: " + e.Message);
            enabled = false;
        }

        _gripLastRxTime = Time.unscaledTime;
    }

    void OnApplicationQuit()
    {
        _running = false;
        try { _udp?.Close(); } catch { }
        try { _recvThread?.Join(100); } catch { }
    }

    // ================= Recv (worker) =================
    void RecvLoop()
    {
        while (_running)
        {
            try
            {
                var data = _udp.Receive(ref _any);
                var json = Encoding.UTF8.GetString(data);

                TypeOnly tOnly = null;
                try { tOnly = JsonUtility.FromJson<TypeOnly>(json); } catch { }
                string ty = (tOnly != null && !string.IsNullOrEmpty(tOnly.type)) ? tOnly.type : "";

                if (!string.Equals(ty, "hand_points", StringComparison.OrdinalIgnoreCase))
                {
                    System.Threading.Interlocked.Exchange(ref _lastRecvMs, NowMs());
                    continue;
                }

                var pkt = JsonUtility.FromJson<HandPointsDto>(json);
                if (pkt == null) { System.Threading.Interlocked.Exchange(ref _lastRecvMs, NowMs()); continue; }

                // 21 joints 파싱 → 보류 버퍼
                if (forwardRemoteJoints && TryParseJointsFromJson(json, out var joints, out var confs))
                {
                    bool isLeft = false;
                    if (!string.IsNullOrEmpty(pkt.hand))
                        isLeft = pkt.hand.Trim().ToLowerInvariant().StartsWith("l");
                    CommitRemotePending(joints, confs, isLeft);
                }

                lock (_pktLock) { _latest = pkt; _latestSeq++; }
                System.Threading.Interlocked.Exchange(ref _lastRecvMs, NowMs());
            }
            catch (SocketException) { /* ignore */ }
            catch (Exception ex) { Debug.LogWarning("[UdpHandReceiver] recv err: " + ex.Message); }
        }
    }

    // ================= Update (main) =================
    void Update()
    {
        // 최신 hand_points
        HandPointsDto pkt = null;
        int seq = _latestSeq;
        lock (_pktLock)
        {
            if (seq != _appliedSeq && _latest != null)
            {
                pkt = _latest;
                _appliedSeq = seq;
            }
        }

        // 1) RemoteHandRuntime로 21점 적용
        FlushRemoteHandsIfPending();

        // 2) Grip → Animator + [GRIP-STATE] 이벤트/프로퍼티 업데이트
        if (pkt != null)
        {
            float? conf = ComputeGripConfidence01(pkt); // 0..1 (닫힘 확률)

            if (conf.HasValue)
            {
                _gripLastRxTime = Time.unscaledTime;

                // 이전 래치 값 보관
                bool latchedBefore = _gripLatched;

                // 히스테리시스 래치
                if (!_gripLatched && conf.Value >= gripThUp) _gripLatched = true;
                else if (_gripLatched && conf.Value <= gripThDn) _gripLatched = false;

                // [GRIP-STATE] 상태 전이 감지 → 이벤트
                if (_gripLatched != latchedBefore || _gripState == GripState.Unknown)
                {
                    SetGripState(_gripLatched ? GripState.Closed : GripState.Open);
                }

                float target01 = _gripLatched ? 1f : 0f;

                // 저역필터(LPF) – 프레임률 무관
                float a = 1f - Mathf.Exp(-Time.unscaledDeltaTime / Mathf.Max(1e-4f, gripTau));
                _gripSmoothed = Mathf.Lerp(_gripSmoothed, target01, a);

                if (forwardGripToAnimator && gripDriver != null)
                    gripDriver.SetGrip01(_gripSmoothed);
            }
        }
    }

    void LateUpdate()
    {
        // 드롭아웃 → 자동 OPEN
        if (Time.unscaledTime - _gripLastRxTime > gripDropoutSec)
        {
            float a = 1f - Mathf.Exp(-Time.unscaledDeltaTime / Mathf.Max(1e-4f, gripTau));
            _gripSmoothed = Mathf.Lerp(_gripSmoothed, 0f, a);
            if (forwardGripToAnimator && gripDriver != null)
                gripDriver.SetGrip01(_gripSmoothed);

            // [GRIP-STATE] 상태도 Open으로 복귀 (한 번만)
            if (_gripState != GripState.Open)
                SetGripState(GripState.Open);

            _gripLatched = false;
        }
    }

    // ---------- [GRIP-STATE] 상태 설정 + 이벤트 ----------
    void SetGripState(GripState s)
    {
        if (_gripState == s) return;
        _gripState = s;
        if (logGripChanges) Debug.Log($"[Grip] {_gripState}");
        try { OnGripStateChanged?.Invoke(_gripState); } catch { /* ignore */ }
    }

    // ================= Grip helpers =================
    // MediaPipe 인덱스: 0 WRIST, 5 INDEX_MCP, 8 INDEX_TIP, 9 MIDDLE_MCP, 17 PINKY_MCP
    const int IDX_WRIST = 0, IDX_INDEX_MCP = 5, IDX_INDEX_TIP = 8, IDX_MIDDLE_MCP = 9, IDX_PINKY_MCP = 17;

    float? ComputeGripConfidence01(HandPointsDto pkt)
    {
        // 1) 모델 제스처 우선
        if (gripUseModelGesture && pkt.gesture != null)
        {
            var name = (pkt.gesture.name ?? "").ToLowerInvariant();
            float s = Mathf.Clamp01(pkt.gesture.score);

            bool isClosed = name.Contains("fist") || name.Contains("closed") || name.Contains("close")
                         || name.Contains("clench") || name.Contains("grip");
            bool isOpen = name.Contains("open") || name.Contains("palm");

            if (pkt.gesture.score >= gestureScoreTau)
            {
                if (isClosed) return s;      // 닫힘 확신 ↑
                if (isOpen) return 1f - s; // 열림이면 닫힘확률 ↓
            }
        }

        if (!gripUseGeometry) return null;

        // 2) 기하 폴백: joints hold‑last에서 rTip 산출
        bool isLeft = !string.IsNullOrEmpty(pkt.hand) && pkt.hand.Trim().ToLowerInvariant().StartsWith("l");
        Vector3[] last = null; bool has = false;
        lock (_remoteLock)
        {
            if (isLeft && _remoteHasLastLeft) { last = _remoteLastLeft; has = true; }
            else if (!isLeft && _remoteHasLastRight) { last = _remoteLastRight; has = true; }
        }
        if (!has || last == null || last.Length < 21) return null;

        Vector3 wrist = last[IDX_WRIST];
        Vector3 mcpI = last[IDX_INDEX_MCP];
        Vector3 mcpM = last[IDX_MIDDLE_MCP];
        Vector3 mcpP = last[IDX_PINKY_MCP];
        Vector3 tipI = last[IDX_INDEX_TIP];
        Vector3 palm = (mcpI + mcpM + mcpP) / 3f; // palm approx

        float handSize = (palm - wrist).magnitude;
        if (handSize < 1e-4f) return null;

        float rTip = (tipI - palm).magnitude / handSize;

        float denom = Mathf.Max(1e-4f, grip_TH_OPEN - grip_TH_CLOSE);
        float confClosed = Mathf.Clamp01((grip_TH_OPEN - rTip) / denom); // rTip 작을수록(=쥠) 1

        return confClosed;
    }

    // ================= joints parsing & forwarding =================
    bool TryParseJointsFromJson(string json, out Vector3?[] joints, out float[] confs)
    {
        joints = null; confs = null;

        int keyIdx = json.IndexOf("\"joints\"", StringComparison.Ordinal);
        if (keyIdx < 0) return false;

        int start = json.IndexOf('[', keyIdx);
        if (start < 0) return false;

        int depth = 0, end = -1;
        for (int i = start; i < json.Length; i++)
        {
            char c = json[i];
            if (c == '[') depth++;
            else if (c == ']')
            {
                depth--;
                if (depth == 0) { end = i; break; }
            }
        }
        if (end < 0) return false;

        string arr = json.Substring(start, end - start + 1);
        var mc = s_joint4Regex.Matches(arr);
        if (mc.Count == 0) return false;

        joints = new Vector3?[21];
        confs = new float[21];

        int count = Mathf.Min(21, mc.Count);
        for (int i = 0; i < count; i++)
        {
            var m = mc[i];
            string sx = m.Groups["x"].Value;
            string sy = m.Groups["y"].Value;
            string sz = m.Groups["z"].Value;
            string sc = m.Groups["c"].Value;

            bool xNull = string.Equals(sx, "null", StringComparison.OrdinalIgnoreCase);
            bool yNull = string.Equals(sy, "null", StringComparison.OrdinalIgnoreCase);
            bool zNull = string.Equals(sz, "null", StringComparison.OrdinalIgnoreCase);

            float cx = 0f, cy = 0f, cz = 0f, cc = 0f;
            if (!xNull) float.TryParse(sx, NumberStyles.Float, CultureInfo.InvariantCulture, out cx);
            if (!yNull) float.TryParse(sy, NumberStyles.Float, CultureInfo.InvariantCulture, out cy);
            if (!zNull) float.TryParse(sz, NumberStyles.Float, CultureInfo.InvariantCulture, out cz);
            float.TryParse(sc, NumberStyles.Float, CultureInfo.InvariantCulture, out cc);

            if (!xNull && !yNull && !zNull && cc > 0f)
            {
                var v = new Vector3(cx, cy, cz);
                if (invertIncomingY) v.y = -v.y;

                // 추가: 고정 회전 오프셋
                if (applyInputRotation)
                {
                    var qOff = Quaternion.Euler(inputEulerOffset);
                    v = qOff * v;
                }

                joints[i] = v;
            }
            else joints[i] = null;

            confs[i] = cc;
        }

        for (int i = count; i < 21; i++) { joints[i] = null; confs[i] = 0f; }
        return true;
    }

    void CommitRemotePending(Vector3?[] parsed, float[] confs, bool isLeft)
    {
        if (parsed == null || confs == null || parsed.Length < 21 || confs.Length < 21) return;

        lock (_remoteLock)
        {
            if (isLeft)
            {
                if (_remotePendingLeft == null) _remotePendingLeft = new Vector3[21];

                bool anyValid = false;
                for (int i = 0; i < 21; i++)
                {
                    if (parsed[i].HasValue) { _remoteLastLeft[i] = parsed[i].Value; anyValid = true; }
                    _remotePendingLeft[i] = (remoteHoldLastOnMissing && (_remoteHasLastLeft || parsed[i].HasValue))
                        ? _remoteLastLeft[i]
                        : _remotePendingLeft[i];
                }
                if (anyValid || _remoteHasLastLeft) { _remoteHasLastLeft = true; _remotePendingFlagLeft = true; }
            }
            else
            {
                if (_remotePendingRight == null) _remotePendingRight = new Vector3[21];

                bool anyValid = false;
                for (int i = 0; i < 21; i++)
                {
                    if (parsed[i].HasValue) { _remoteLastRight[i] = parsed[i].Value; anyValid = true; }
                    _remotePendingRight[i] = (remoteHoldLastOnMissing && (_remoteHasLastRight || parsed[i].HasValue))
                        ? _remoteLastRight[i]
                        : _remotePendingRight[i];
                }
                if (anyValid || _remoteHasLastRight) { _remoteHasLastRight = true; _remotePendingFlagRight = true; }
            }
        }
    }

    void FlushRemoteHandsIfPending()
    {
        if (!forwardRemoteJoints) return;

        lock (_remoteLock)
        {
            if (_remotePendingFlagLeft && remoteLeft != null && _remotePendingLeft != null)
            {
                remoteLeft.isLeft = true;
                remoteLeft.ApplyWorldPositions(_remotePendingLeft);
                _remotePendingFlagLeft = false;
            }
            if (_remotePendingFlagRight && remoteRight != null && _remotePendingRight != null)
            {
                remoteRight.isLeft = false;
                remoteRight.ApplyWorldPositions(_remotePendingRight);
                _remotePendingFlagRight = false;
            }
        }
    }
}
