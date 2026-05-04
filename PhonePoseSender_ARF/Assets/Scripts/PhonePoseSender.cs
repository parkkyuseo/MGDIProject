using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class PhonePoseSender : MonoBehaviour
{
    [Header("Target (HoloLens / Receiver)")]
    [SerializeField] private string targetIp = "192.168.0.100";
    [SerializeField] private int targetPort = 5005;

    [Header("Source")]
    [SerializeField] private Transform arCamera; // XR Origin 아래 AR Camera

    [Header("Send Rate")]
    [SerializeField] private int sendHz = 60;

    private UdpClient _udp;
    private IPEndPoint _endPoint;
    private float _nextSendTime;

    [SerializeField] private PhoneMarkerTracker markerTracker;

    [Serializable]
    private struct PosePacket
    {
        public double t;     // Time.realtimeSinceStartup
        public float px, py, pz;
        public float qx, qy, qz, qw;

        // Phase 1 추가 (marker)
        public bool mvis;
        public string mname;
        public float mx, my, mz;
        public float mqx, mqy, mqz, mqw;

        // NEW: tap toggles grab state
        public bool grab;
    }

    [SerializeField] private float tapMaxDuration = 0.25f;
    [SerializeField] private float tapMaxMovePixels = 25f;

    private bool _grabToggled = false;
    private bool _touchDown = false;

    private bool _touchTracking;
    private float _touchStartTime;
    private Vector2 _touchStartPos;

    void Awake()
    {
        _udp = new UdpClient();
        _endPoint = new IPEndPoint(IPAddress.Parse(targetIp), targetPort);

        if (arCamera == null)
        {
            var cam = Camera.main;
            if (cam != null) arCamera = cam.transform;
        }
        if (markerTracker == null)
              markerTracker = FindFirstObjectByType<PhoneMarkerTracker>();
    }

    void OnDestroy()
    {
        _udp?.Close();
        _udp = null;
    }

    void Update()
    {

        // 1) Touch toggle MUST run every frame (do NOT gate by sendHz)
        if (Input.touchCount == 1)
        {
            Touch t0 = Input.GetTouch(0);

            if (t0.phase == TouchPhase.Began)
            {
                _touchTracking = true;
                _touchStartTime = Time.unscaledTime;
                _touchStartPos = t0.position;
            }
            else if (_touchTracking && (t0.phase == TouchPhase.Moved || t0.phase == TouchPhase.Stationary))
            {
                if ((t0.position - _touchStartPos).magnitude > tapMaxMovePixels)
                    _touchTracking = false;
            }
            else if (_touchTracking && (t0.phase == TouchPhase.Ended || t0.phase == TouchPhase.Canceled))
            {
                float dt = Time.unscaledTime - _touchStartTime;
                float move = (t0.position - _touchStartPos).magnitude;

                if (dt <= tapMaxDuration && move <= tapMaxMovePixels)
                {
                    _grabToggled = !_grabToggled;
                    Debug.Log($"[PhoneTX] grab toggled -> {_grabToggled}");
                }

                _touchTracking = false;
            }
        }
        else
        {
            _touchTracking = false;
        }

        // 2) Sending can be rate-limited
        if (arCamera == null) return;

        float interval = Mathf.Max(1f / Mathf.Max(sendHz, 1), 0.001f);
        if (Time.unscaledTime < _nextSendTime) return;
        _nextSendTime = Time.unscaledTime + interval;

        Vector3 p = arCamera.position;
        Quaternion q = arCamera.rotation;

        bool mvis = markerTracker != null && markerTracker.MarkerVisible;
        Vector3 mp = mvis ? markerTracker.MarkerPosition : Vector3.zero;
        Quaternion mq = mvis ? markerTracker.MarkerRotation : Quaternion.identity;
        string mname = (mvis && markerTracker != null) ? markerTracker.MarkerName : "";

        var pkt = new PosePacket
        {
            t = Time.realtimeSinceStartupAsDouble,
            px = p.x, py = p.y, pz = p.z,
            qx = q.x, qy = q.y, qz = q.z, qw = q.w,

            mvis = mvis,
            mname = mname,
            mx = mp.x, my = mp.y, mz = mp.z,
            mqx = mq.x, mqy = mq.y, mqz = mq.z, mqw = mq.w,

            grab = _grabToggled
        };

        string json = JsonUtility.ToJson(pkt);
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        _udp.Send(bytes, bytes.Length, _endPoint);
    }

    /* void OnGUI()
     * {
     *     GUI.Label(new Rect(20, 20, 800, 40),
     *         $"grab={_grabToggled} touchCount={Input.touchCount} phase={(Input.touchCount>0 ? Input.GetTouch(0).phase.ToString() : "-")}");
     * } */
}
