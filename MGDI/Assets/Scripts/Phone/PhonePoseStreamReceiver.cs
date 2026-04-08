using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Diagnostics;          // Stopwatch
using UnityEngine;

using UDebug = UnityEngine.Debug;  // System.Diagnostics.Debug와 이름 충돌 방지

public class PhonePoseStreamReceiver : MonoBehaviour
{
    [Header("UDP Listen")]
    [SerializeField] private int listenPort = 5555;

    [Header("Debug")]
    [SerializeField] private bool logPacketsPerSec = true;

    [Serializable]
    private struct PosePacket
    {
        public double t;
        public float px, py, pz;
        public float qx, qy, qz, qw;

        public bool hold;        // macro
        public bool toggle;      // micro grab toggle state
        public int modeToggleId; // micro placement plane toggle (event id)
        public int tripleTapId;  // micro triple-tap event id

        public float ax;
        public float ay;
        public bool drag;
    }

    private readonly object _lock = new object();

    private bool _hasPhonePose;
    private Pose _phonePose;
    private bool _hasPrevPhonePose;
    private Pose _prevPhonePose;
    private double _cumulativePathLengthMeters;

    private bool _hold;
    private bool _toggle;
    private int _modeToggleId;
    private int _tripleTapId;
    private float _ax;
    private float _ay;
    private bool _drag;

    // ---- thread-safe monotonic timestamp ----
    private long _lastRxStamp; // Stopwatch ticks
    private long _prevRxStamp; // Stopwatch ticks
    private static readonly double _invStopwatchFreq = 1.0 / Stopwatch.Frequency;

    private UdpClient _udp;
    private Thread _rxThread;
    private volatile bool _running;

    private int _pktCount;
    private float _nextPktsLogTime;

    public bool HasPhonePose { get { lock (_lock) return _hasPhonePose; } }
    public Pose LatestPhonePose { get { lock (_lock) return _phonePose; } }
    public double CumulativePathLengthMeters { get { lock (_lock) return _cumulativePathLengthMeters; } }

    public bool LatestHold { get { lock (_lock) return _hold; } }
    public bool LatestToggle { get { lock (_lock) return _toggle; } }

    public int LatestModeToggleId { get { lock (_lock) return _modeToggleId; } }
    public int LatestTripleTapId { get { lock (_lock) return _tripleTapId; } }

    public float LatestAx { get { lock (_lock) return _ax; } }
    public float LatestAy { get { lock (_lock) return _ay; } }
    public bool LatestDrag { get { lock (_lock) return _drag; } }

    public float SecondsSinceLastRx
    {
        get
        {
            lock (_lock)
            {
                if (_lastRxStamp == 0) return float.PositiveInfinity;
                long dtTicks = Stopwatch.GetTimestamp() - _lastRxStamp;
                double dtSec = dtTicks * _invStopwatchFreq;
                return (float)Math.Max(0.0, dtSec);
            }
        }
    }

    public bool TryGetPhoneMotionEstimate(
        out Pose latestPose,
        out Vector3 linearVelocityMetersPerSec,
        out Vector3 angularVelocityDegPerSec,
        out float ageSec)
    {
        lock (_lock)
        {
            latestPose = _phonePose;
            linearVelocityMetersPerSec = Vector3.zero;
            angularVelocityDegPerSec = Vector3.zero;

            if (_lastRxStamp == 0)
            {
                ageSec = float.PositiveInfinity;
                return false;
            }

            long dtTicks = Stopwatch.GetTimestamp() - _lastRxStamp;
            ageSec = (float)Math.Max(0.0, dtTicks * _invStopwatchFreq);

            if (!_hasPhonePose || !_hasPrevPhonePose || _prevRxStamp == 0)
                return _hasPhonePose;

            double sampleDtSec = (_lastRxStamp - _prevRxStamp) * _invStopwatchFreq;
            if (sampleDtSec <= 1e-4)
                return true;

            float dt = (float)sampleDtSec;
            linearVelocityMetersPerSec = (_phonePose.position - _prevPhonePose.position) / dt;

            Quaternion dq = _phonePose.rotation * Quaternion.Inverse(_prevPhonePose.rotation);
            dq.ToAngleAxis(out float angDeg, out Vector3 axis);
            if (float.IsNaN(axis.x) || float.IsNaN(axis.y) || float.IsNaN(axis.z))
                return true;

            if (angDeg > 180f)
                angDeg -= 360f;

            if (Mathf.Abs(angDeg) > 1e-4f && axis.sqrMagnitude > 1e-6f)
                angularVelocityDegPerSec = axis.normalized * (angDeg / dt);

            return true;
        }
    }

    void Start()
    {
        _udp = new UdpClient(listenPort);
        _udp.Client.ReceiveTimeout = 1000;

        _running = true;
        _rxThread = new Thread(ReceiveLoop) { IsBackground = true };
        _rxThread.Start();

        UDebug.Log($"[PhonePoseStreamReceiver] Listening UDP :{listenPort}");
    }

    void OnDestroy()
    {
        _running = false;

        try { _udp?.Close(); } catch { }
        _udp = null;

        try { _rxThread?.Join(200); } catch { }
        _rxThread = null;
    }

    void Update()
    {
        if (logPacketsPerSec && Time.unscaledTime >= _nextPktsLogTime)
        {
            _nextPktsLogTime = Time.unscaledTime + 1f;
            int c = _pktCount;
            _pktCount = 0;
            UDebug.Log($"[PhonePoseStreamReceiver] pkts/sec ~ {c}");
        }
    }

    private void ReceiveLoop()
    {
        var any = new IPEndPoint(IPAddress.Any, 0);

        while (_running)
        {
            try
            {
                byte[] data = _udp.Receive(ref any);
                if (data == null || data.Length == 0) continue;

                string json = Encoding.UTF8.GetString(data);
                PosePacket pkt = JsonUtility.FromJson<PosePacket>(json);

                Quaternion q = new Quaternion(pkt.qx, pkt.qy, pkt.qz, pkt.qw);
                float qMag = Mathf.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
                if (qMag > 1e-6f)
                {
                    float inv = 1f / qMag;
                    q = new Quaternion(q.x * inv, q.y * inv, q.z * inv, q.w * inv);
                }
                else
                {
                    q = Quaternion.identity;
                }

                if (float.IsNaN(q.x) || float.IsNaN(q.y) || float.IsNaN(q.z) || float.IsNaN(q.w))
                    continue;

                Pose phonePose = new Pose(
                    new Vector3(pkt.px, pkt.py, pkt.pz),
                    q
                );

                long nowStamp = Stopwatch.GetTimestamp();

                lock (_lock)
                {
                    if (_hasPhonePose)
                    {
                        _cumulativePathLengthMeters += Vector3.Distance(_phonePose.position, phonePose.position);
                        _prevPhonePose = _phonePose;
                        _prevRxStamp = _lastRxStamp;
                        _hasPrevPhonePose = true;
                    }

                    _phonePose = phonePose;
                    _hasPhonePose = true;

                    _hold = pkt.hold;
                    _toggle = pkt.toggle;
                    _modeToggleId = pkt.modeToggleId;
                    _tripleTapId = pkt.tripleTapId;

                    _ax = pkt.ax;
                    _ay = pkt.ay;
                    _drag = pkt.drag;

                    _lastRxStamp = nowStamp;
                }

                _pktCount++;
            }
            catch (SocketException) { }
            catch (Exception e)
            {
                UDebug.LogWarning($"[PhonePoseStreamReceiver] RX error: {e.Message}");
            }
        }
    }
}
