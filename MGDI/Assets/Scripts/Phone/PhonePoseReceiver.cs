using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class PhonePoseReceiver : MonoBehaviour
{
    [Header("UDP Listen")]
    [SerializeField] private int listenPort = 5555;

    [Header("Apply To")]
    [SerializeField] private Transform target; // PhoneProxyCube

    [Header("Mapping (Phase 0)")]
    [SerializeField] private float positionGain = 1.0f;     // meters -> meters
    [SerializeField] private bool applyRelativePose = true; // use first packet as origin

    [Header("Smoothing")]
    [SerializeField] private float posLerp = 18f;
    [SerializeField] private float rotLerp = 18f;

    [Serializable]
    private struct PosePacket
    {
        public double t;
        public float px, py, pz;
        public float qx, qy, qz, qw;

        // Phase 1 marker
        public bool mvis;
        public string mname;
        public float mx, my, mz;
        public float mqx, mqy, mqz, mqw;
    }

    private UdpClient _udp;
    private Thread _rxThread;
    private volatile bool _running;

    private readonly object _lock = new object();
    private bool _hasPose;
    private Vector3 _pLatest;
    private Quaternion _qLatest;

    private bool _hasOrigin;
    private Vector3 _p0;
    private Quaternion _q0;
    private Vector3 _targetStartPos;
    private Quaternion _targetStartRot;

    private int _pktCount;
    private float _nextLogTime;

    private bool _hasMarker;
    private Vector3 _mLatest;
    private Quaternion _mqLatest;

    public bool HasPhonePose => _hasPose;
    public Pose LatestPhonePose
    {
        get { lock (_lock) return new Pose(_pLatest, _qLatest); }
    }

    public bool HasPhoneMarker => _hasMarker;
    public Pose LatestPhoneMarkerPose
    {
        get { lock (_lock) return new Pose(_mLatest, _mqLatest); }
    }
    public bool LatestMarkerVisible
    {
        get { lock (_lock) return _hasMarker; }
    }

    void Start()
    {
        if (target == null)
        {
            DebugHUD.Log("[PhonePoseReceiver] Target not assigned.");
            enabled = false;
            return;
        }

        _targetStartPos = target.position;
        _targetStartRot = target.rotation;

        _udp = new UdpClient(listenPort);
        _udp.Client.ReceiveTimeout = 1000;

        _running = true;
        _rxThread = new Thread(ReceiveLoop) { IsBackground = true };
        _rxThread.Start();

        DebugHUD.Log($"[PhonePoseReceiver] Listening UDP :{listenPort}");
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
        if (!_hasPose) return;

        Vector3 p;
        Quaternion q;
        lock (_lock)
        {
            p = _pLatest;
            q = _qLatest;
        }

        if (applyRelativePose)
        {
            if (!_hasOrigin)
            {
                _p0 = p;
                _q0 = q;
                _hasOrigin = true;
                DebugHUD.Log("[PhonePoseReceiver] Origin captured (first packet).");
            }

            Vector3 dp = (p - _p0) * positionGain;

            Quaternion dq = q * Quaternion.Inverse(_q0);

            Vector3 desiredPos = _targetStartPos + dp;
            Quaternion desiredRot = dq * _targetStartRot;

            float dt = Mathf.Max(Time.deltaTime, 1e-4f);
            float aPos = 1f - Mathf.Exp(-posLerp * dt);
            float aRot = 1f - Mathf.Exp(-rotLerp * dt);

            target.position = Vector3.Lerp(target.position, desiredPos, aPos);
            target.rotation = Quaternion.Slerp(target.rotation, desiredRot, aRot);
        }
        else
        {
            // Absolute apply (not recommended for Phase 0)
            Vector3 desiredPos = p * positionGain;
            Quaternion desiredRot = q;

            float dt = Mathf.Max(Time.deltaTime, 1e-4f);
            float aPos = 1f - Mathf.Exp(-posLerp * dt);
            float aRot = 1f - Mathf.Exp(-rotLerp * dt);

            target.position = Vector3.Lerp(target.position, desiredPos, aPos);
            target.rotation = Quaternion.Slerp(target.rotation, desiredRot, aRot);
        }

        if (Time.unscaledTime >= _nextLogTime)
        {
            _nextLogTime = Time.unscaledTime + 1f;
            int c = _pktCount;
            _pktCount = 0;
            /* DebugHUD.Log($"[PhonePoseReceiver] pkts/sec ~ {c}"); */
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

                Vector3 p = new Vector3(pkt.px, pkt.py, pkt.pz);
                Quaternion q = new Quaternion(pkt.qx, pkt.qy, pkt.qz, pkt.qw);

                bool mvis = pkt.mvis;
                Vector3 mp = new Vector3(pkt.mx, pkt.my, pkt.mz);
                Quaternion mq = new Quaternion(pkt.mqx, pkt.mqy, pkt.mqz, pkt.mqw);

                lock (_lock)
                {
                    _pLatest = p;
                    _qLatest = q;
                    _hasPose = true;

                    _hasMarker = mvis;
                    if (mvis)
                    {
                        _mLatest = mp;
                        _mqLatest = mq;
                    }
                }

                _pktCount++;
            }
            catch (SocketException)
            {
                // timeout; loop continues
            }
            catch (Exception e)
            {
                DebugHUD.Log($"[PhonePoseReceiver] RX error: {e.Message}");
            }
        }
    }

    [ContextMenu("Reset Origin")]
    public void ResetOrigin()
    {
        _hasOrigin = false;
        _targetStartPos = target.position;
        _targetStartRot = target.rotation;
        Debug.Log("[PhonePoseReceiver] Origin reset.");
    }
}
