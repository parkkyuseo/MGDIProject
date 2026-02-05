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

    [Serializable]
    private struct PosePacket
    {
        public double t;     // Time.realtimeSinceStartup
        public float px, py, pz;
        public float qx, qy, qz, qw;
    }

    void Awake()
    {
        _udp = new UdpClient();
        _endPoint = new IPEndPoint(IPAddress.Parse(targetIp), targetPort);

        if (arCamera == null)
        {
            var cam = Camera.main;
            if (cam != null) arCamera = cam.transform;
        }
    }

    void OnDestroy()
    {
        _udp?.Close();
        _udp = null;
    }

    void Update()
    {
        if (arCamera == null) return;

        float interval = Mathf.Max(1f / Mathf.Max(sendHz, 1), 0.001f);
        if (Time.unscaledTime < _nextSendTime) return;
        _nextSendTime = Time.unscaledTime + interval;

        Vector3 p = arCamera.position;
        Quaternion q = arCamera.rotation;

        var pkt = new PosePacket
        {
            t = Time.realtimeSinceStartupAsDouble,
            px = p.x, py = p.y, pz = p.z,
            qx = q.x, qy = q.y, qz = q.z, qw = q.w
        };

        string json = JsonUtility.ToJson(pkt);
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        _udp.Send(bytes, bytes.Length, _endPoint);
    }
}
