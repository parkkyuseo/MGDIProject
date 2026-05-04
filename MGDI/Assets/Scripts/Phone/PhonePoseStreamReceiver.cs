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

        // Optional phone-side QR/marker pose in the same AR session frame as px/py/pz.
        // If present once, it is retained so the phone and HoloLens do not need to see
        // the QR marker at the same time.
        public bool mvis;
        public string mname;
        public float mx, my, mz;
        public float mqx, mqy, mqz, mqw;

        public bool qrCalibrated;
        public float dx_qr, dy_qr, dz_qr;
        public float dqx_qr, dqy_qr, dqz_qr, dqw_qr;

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
    private bool _hasPhoneMarkerPose;
    private bool _latestPhoneMarkerVisible;
    private Pose _phoneMarkerPose;
    private string _phoneMarkerName;
    private bool _hasQrRelativePhonePose;
    private Pose _qrRelativePhonePose;
    private bool _hasQrDeltaPose;
    private Pose _qrDeltaPose;
    private long _lastMarkerRxStamp;

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
    public bool HasPhoneMarker { get { lock (_lock) return _hasPhoneMarkerPose; } }
    public bool LatestMarkerVisible { get { lock (_lock) return _latestPhoneMarkerVisible; } }
    public Pose LatestPhoneMarkerPose { get { lock (_lock) return _phoneMarkerPose; } }
    public string LatestPhoneMarkerName { get { lock (_lock) return _phoneMarkerName; } }
    public bool HasQrRelativePhonePose { get { lock (_lock) return _hasQrRelativePhonePose; } }
    public Pose LatestQrRelativePhonePose { get { lock (_lock) return _qrRelativePhonePose; } }
    public bool HasQrDeltaPose { get { lock (_lock) return _hasQrDeltaPose; } }
    public Pose LatestQrDeltaPose { get { lock (_lock) return _qrDeltaPose; } }

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

    public float SecondsSinceLastPhoneMarker
    {
        get
        {
            lock (_lock)
            {
                if (_lastMarkerRxStamp == 0) return float.PositiveInfinity;
                long dtTicks = Stopwatch.GetTimestamp() - _lastMarkerRxStamp;
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

                Quaternion q = NormalizePacketQuaternion(new Quaternion(pkt.qx, pkt.qy, pkt.qz, pkt.qw));
                Vector3 phonePosition = new Vector3(pkt.px, pkt.py, pkt.pz);
                if (!IsFinite(q) || !IsFinite(phonePosition))
                    continue;

                Pose phonePose = new Pose(
                    phonePosition,
                    q
                );

                long nowStamp = Stopwatch.GetTimestamp();
                bool hasMarkerPacket = false;
                Pose markerPose = new Pose(Vector3.zero, Quaternion.identity);
                bool hasQrDeltaPacket = false;
                Pose qrDeltaPose = new Pose(Vector3.zero, Quaternion.identity);

                if (pkt.mvis)
                {
                    bool hasMarkerRotation = TryNormalizePacketQuaternion(
                        new Quaternion(pkt.mqx, pkt.mqy, pkt.mqz, pkt.mqw),
                        out Quaternion mq);
                    Vector3 markerPosition = new Vector3(pkt.mx, pkt.my, pkt.mz);
                    if (hasMarkerRotation && IsFinite(markerPosition))
                    {
                        markerPose = new Pose(markerPosition, mq);
                        hasMarkerPacket = true;
                    }
                }

                if (pkt.qrCalibrated)
                {
                    bool hasDeltaRotation = TryNormalizePacketQuaternion(
                        new Quaternion(pkt.dqx_qr, pkt.dqy_qr, pkt.dqz_qr, pkt.dqw_qr),
                        out Quaternion qrDeltaRotation);
                    Vector3 qrDeltaPosition = new Vector3(pkt.dx_qr, pkt.dy_qr, pkt.dz_qr);
                    if (hasDeltaRotation && IsFinite(qrDeltaPosition))
                    {
                        qrDeltaPose = new Pose(qrDeltaPosition, qrDeltaRotation);
                        hasQrDeltaPacket = true;
                    }
                }

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

                    _latestPhoneMarkerVisible = hasMarkerPacket;
                    if (hasMarkerPacket)
                    {
                        _phoneMarkerPose = markerPose;
                        _phoneMarkerName = pkt.mname;
                        _hasPhoneMarkerPose = true;
                        _lastMarkerRxStamp = nowStamp;
                    }

                    if (_hasPhoneMarkerPose)
                    {
                        Pose qrRelativePose = MakeRelativePose(_phoneMarkerPose, phonePose);
                        if (IsFinite(qrRelativePose.position) && IsFinite(qrRelativePose.rotation))
                        {
                            _qrRelativePhonePose = qrRelativePose;
                            _hasQrRelativePhonePose = true;
                        }
                    }

                    if (hasQrDeltaPacket)
                    {
                        _qrDeltaPose = qrDeltaPose;
                        _hasQrDeltaPose = true;
                    }

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

    private static Quaternion NormalizePacketQuaternion(Quaternion q)
    {
        if (TryNormalizePacketQuaternion(q, out Quaternion normalized))
            return normalized;

        return Quaternion.identity;
    }

    private static bool TryNormalizePacketQuaternion(Quaternion q, out Quaternion normalized)
    {
        normalized = Quaternion.identity;
        if (!IsFinite(q))
            return false;

        float qMag = Mathf.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        if (qMag > 1e-6f)
        {
            float inv = 1f / qMag;
            normalized = new Quaternion(q.x * inv, q.y * inv, q.z * inv, q.w * inv);
            return IsFinite(normalized);
        }

        return false;
    }

    private static Pose MakeRelativePose(Pose parent, Pose child)
    {
        Quaternion invParentRot = Quaternion.Inverse(parent.rotation);
        return new Pose(
            invParentRot * (child.position - parent.position),
            invParentRot * child.rotation
        );
    }

    private static bool IsFinite(Vector3 v)
    {
        return IsFinite(v.x) && IsFinite(v.y) && IsFinite(v.z);
    }

    private static bool IsFinite(Quaternion q)
    {
        return IsFinite(q.x) && IsFinite(q.y) && IsFinite(q.z) && IsFinite(q.w);
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
