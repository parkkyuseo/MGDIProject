using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

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

        public bool hold;       // macro
        public bool toggle;     // micro grab toggle
        public bool modeToggle; // micro placement plane toggle (one-shot)
        public float ax;        // micro analog axis x (-1..1)
        public float ay;        // micro analog axis y (-1..1)
        public bool drag;       // micro analog active
    }

    private readonly object _lock = new object();

    private bool _hasPhonePose;
    private Pose _phonePose;

    private bool _hold;
    private bool _toggle;
    private bool _modeToggle;
    private float _ax;
    private float _ay;
    private bool _drag;

    private float _lastRxRealtime;

    private UdpClient _udp;
    private Thread _rxThread;
    private volatile bool _running;

    private int _pktCount;
    private float _nextPktsLogTime;

    public bool HasPhonePose { get { lock (_lock) return _hasPhonePose; } }
    public Pose LatestPhonePose { get { lock (_lock) return _phonePose; } }

    public bool LatestHold { get { lock (_lock) return _hold; } }
    public bool LatestToggle { get { lock (_lock) return _toggle; } }
    public bool LatestModeToggle { get { lock (_lock) return _modeToggle; } }

    public float LatestAx { get { lock (_lock) return _ax; } }
    public float LatestAy { get { lock (_lock) return _ay; } }
    public bool LatestDrag { get { lock (_lock) return _drag; } }

    public float SecondsSinceLastRx { get { lock (_lock) return Time.unscaledTime - _lastRxRealtime; } }

    void Start()
    {
        _udp = new UdpClient(listenPort);
        _udp.Client.ReceiveTimeout = 1000;

        _running = true;
        _rxThread = new Thread(ReceiveLoop) { IsBackground = true };
        _rxThread.Start();

        Debug.Log($"[PhonePoseStreamReceiver] Listening UDP :{listenPort}");
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
            Debug.Log($"[PhonePoseStreamReceiver] pkts/sec ~ {c}");
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

                Pose phonePose = new Pose(
                    new Vector3(pkt.px, pkt.py, pkt.pz),
                    new Quaternion(pkt.qx, pkt.qy, pkt.qz, pkt.qw)
                );

                lock (_lock)
                {
                    _phonePose = phonePose;
                    _hasPhonePose = true;

                    _hold = pkt.hold;
                    _toggle = pkt.toggle;
                    _modeToggle = pkt.modeToggle;

                    _ax = pkt.ax;
                    _ay = pkt.ay;
                    _drag = pkt.drag;

                    _lastRxRealtime = Time.unscaledTime;
                }

                _pktCount++;
            }
            catch (SocketException) { }
            catch (Exception e)
            {
                Debug.LogWarning($"[PhonePoseStreamReceiver] RX error: {e.Message}");
            }
        }
    }
}
