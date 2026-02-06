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

        // Phase 1: marker
        public bool mvis;
        public string mname;
        public float mx, my, mz;
        public float mqx, mqy, mqz, mqw;
    }

    private readonly object _lock = new object();

    private bool _hasPhonePose;
    private Pose _phonePose;

    private bool _hasPhoneMarker;
    private Pose _phoneMarkerPose;
    private string _markerName = "";

    private UdpClient _udp;
    private Thread _rxThread;
    private volatile bool _running;

    private int _pktCount;
    private float _nextLogTime;

    public bool HasPhonePose
    {
        get { lock (_lock) return _hasPhonePose; }
    }

    public Pose LatestPhonePose
    {
        get { lock (_lock) return _phonePose; }
    }

    public bool HasPhoneMarker
    {
        get { lock (_lock) return _hasPhoneMarker; }
    }

    public Pose LatestPhoneMarkerPose
    {
        get { lock (_lock) return _phoneMarkerPose; }
    }

    public string LatestMarkerName
    {
        get { lock (_lock) return _markerName; }
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
        if (!logPacketsPerSec) return;

        if (Time.unscaledTime >= _nextLogTime)
        {
            _nextLogTime = Time.unscaledTime + 1f;
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

                bool mvis = pkt.mvis;
                Pose markerPose = new Pose(
                    new Vector3(pkt.mx, pkt.my, pkt.mz),
                    new Quaternion(pkt.mqx, pkt.mqy, pkt.mqz, pkt.mqw)
                );

                lock (_lock)
                {
                    _phonePose = phonePose;
                    _hasPhonePose = true;

                    _hasPhoneMarker = mvis;
                    if (mvis)
                    {
                        _phoneMarkerPose = markerPose;
                        _markerName = pkt.mname ?? "";
                    }
                }

                _pktCount++;
            }
            catch (SocketException)
            {
                // timeout, continue
            }
            catch (Exception e)
            {
                // keep receiver alive even if parsing fails intermittently
                // message is short to avoid log spam
                Debug.LogWarning($"[PhonePoseStreamReceiver] RX error: {e.Message}");
            }
        }
    }
}
