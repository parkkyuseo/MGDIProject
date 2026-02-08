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
    [SerializeField] private bool logInputChanges = false;   // hold/toggle/swipe 변화 로그
    [SerializeField] private bool logPoseOccasionally = false; // 위치/회전 가끔 로그

    [Serializable]
    private struct PosePacket
    {
        public double t;
        public float px, py, pz;
        public float qx, qy, qz, qw;

        public bool hold;    // macro
        public bool toggle;  // micro
        public int swipe;    // 0 none, 1 up, 2 down, 3 left, 4 right
    }

    private readonly object _lock = new object();

    private bool _hasPhonePose;
    private Pose _phonePose;

    private bool _hold;
    private bool _toggle;
    private int _swipe;

    private double _lastPacketTime; // sender timestamp (pkt.t) if needed
    private float _lastRxRealtime;  // local realtime when received

    private UdpClient _udp;
    private Thread _rxThread;
    private volatile bool _running;

    private int _pktCount;
    private float _nextPktsLogTime;

    private bool _dbgHold;
    private bool _dbgToggle;
    private int _dbgSwipe;
    private float _nextPoseLogTime;

    // -----------------------------
    // Public getters
    // -----------------------------
    public bool HasPhonePose
    {
        get { lock (_lock) return _hasPhonePose; }
    }

    public Pose LatestPhonePose
    {
        get { lock (_lock) return _phonePose; }
    }

    public bool LatestHold
    {
        get { lock (_lock) return _hold; }
    }

    public bool LatestToggle
    {
        get { lock (_lock) return _toggle; }
    }

    public int LatestSwipe
    {
        get { lock (_lock) return _swipe; }
    }

    public float SecondsSinceLastRx
    {
        get { lock (_lock) return Time.unscaledTime - _lastRxRealtime; }
    }

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
        // packets/sec log
        if (logPacketsPerSec && Time.unscaledTime >= _nextPktsLogTime)
        {
            _nextPktsLogTime = Time.unscaledTime + 1f;
            int c = _pktCount;
            _pktCount = 0;
            Debug.Log($"[PhonePoseStreamReceiver] pkts/sec ~ {c}");
        }

        // input changes log (main thread, safe)
        if (logInputChanges)
        {
            bool h, t;
            int s;
            lock (_lock)
            {
                h = _hold;
                t = _toggle;
                s = _swipe;
            }

            if (h != _dbgHold || t != _dbgToggle || s != _dbgSwipe)
            {
                _dbgHold = h;
                _dbgToggle = t;
                _dbgSwipe = s;
                DebugHUD.Log($"[PhoneRX] hold={h} toggle={t} swipe={s}");
            }
        }

        // occasional pose log
        if (logPoseOccasionally && Time.unscaledTime >= _nextPoseLogTime)
        {
            _nextPoseLogTime = Time.unscaledTime + 0.5f;

            Pose p;
            lock (_lock) p = _phonePose;

            Vector3 pos = p.position;
            Vector3 eul = p.rotation.eulerAngles;
            DebugHUD.Log($"[PhoneRX] pos=({pos.x:F3},{pos.y:F3},{pos.z:F3}) rot=({eul.x:F1},{eul.y:F1},{eul.z:F1})");
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
                    _swipe = pkt.swipe;

                    _lastPacketTime = pkt.t;
                    _lastRxRealtime = Time.unscaledTime;
                }

                _pktCount++;
            }
            catch (SocketException)
            {
                // timeout, continue
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[PhonePoseStreamReceiver] RX error: {e.Message}");
            }
        }
    }
}
