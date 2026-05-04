using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class PhonePoseSenderfromMac : MonoBehaviour
{
    [Header("Target (HoloLens / Receiver)")]
    [SerializeField] private string targetIp = "192.168.0.100";
    [SerializeField] private int targetPort = 5555;

    [Header("Source")]
    [SerializeField] private Transform arCamera;
    [SerializeField] private PhoneMarkerTrackerfromMac markerTracker;

    [Header("QR Calibration")]
    [SerializeField] private bool autoCalibrateQrOnFirstMarker = true;
    [SerializeField] private bool includeMarkerPoseInPacket = true;
    [SerializeField] private bool includeQrDeltaInPacket = true;

    [Header("Send Rate")]
    [SerializeField] private int sendHz = 60;

    [Header("Tap/DoubleTap thresholds")]
    [SerializeField] private float tapMaxDuration = 0.25f;
    [SerializeField] private float tapMaxMovePixels = 25f;
    [SerializeField] private float doubleTapMaxGap = 0.35f;
    [SerializeField] private float tripleTapMaxGap = 0.35f;
    [SerializeField] private float tripleTapMaxMovePixels = 35f;

    [Header("Analog drag (virtual joystick)")]
    [SerializeField] private float dragDeadzonePixels = 12f;
    [SerializeField] private float dragMaxRadiusPixels = 140f;
    [SerializeField] private bool invertY = true;

    [Header("iOS / DPI scaling")]
    [SerializeField] private bool useDpiScaling = true;
    [SerializeField] private float referenceDpi = 160f;

    private UdpClient _udp;
    private IPEndPoint _endPoint;
    private float _nextSendTime;

    private bool _touchTracking;
    private float _touchStartTime;
    private Vector2 _touchStartPos;
    private bool _dragEverThisTouch;

    private bool _pendingSingleTap;
    private float _pendingSingleTapExpire;

    private float _lastTapTime = -999f;
    private Vector2 _lastTapPos;

    private bool _grabToggled = false;

    private int _modeToggleId = 0;
    private int _tripleTapId = 0;
    private bool _pendingModeToggle = false;
    private float _pendingModeToggleExpire = -999f;

    private float _ax = 0f;
    private float _ay = 0f;
    private bool _dragActive = false;

    // ★ 추가: 이번 터치가 더블탭으로 확정되었는지
    private bool _doubleTapConfirmedThisTouch = false;
    private bool _awaitingTripleTap = false;
    private float _awaitingTripleTapExpire = -999f;
    private Vector2 _doubleTapPos;

    private bool _hasQrCalibration = false;
    private Pose _qrWorldPose = new Pose(Vector3.zero, Quaternion.identity);
    private Pose _qrPhone0Pose = new Pose(Vector3.zero, Quaternion.identity);

    [Serializable]
    private struct PosePacket
    {
        public double t;
        public float px, py, pz;
        public float qx, qy, qz, qw;

        public bool mvis;
        public string mname;
        public float mx, my, mz;
        public float mqx, mqy, mqz, mqw;

        public bool qrCalibrated;
        public float dx_qr, dy_qr, dz_qr;
        public float dqx_qr, dqy_qr, dqz_qr, dqw_qr;

        public bool hold;
        public bool toggle;
        public int modeToggleId;
        public int tripleTapId;

        public float ax;
        public float ay;
        public bool drag;
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

        if (markerTracker == null)
            markerTracker = FindObjectOfType<PhoneMarkerTrackerfromMac>();
    }

    void OnDestroy()
    {
        _udp?.Close();
        _udp = null;
    }

    private float GetPxScale()
    {
        if (!useDpiScaling) return 1f;

        float dpi = Screen.dpi;
        if (dpi <= 0f) dpi = referenceDpi;
        if (referenceDpi <= 1e-3f) return 1f;

        return Mathf.Clamp(dpi / referenceDpi, 1f, 4f);
    }

    private void CommitPendingModeToggle()
    {
        if (!_pendingModeToggle) return;

        _pendingModeToggle = false;
        _modeToggleId++;
        Debug.Log($"[PhoneTX] modeToggleId -> {_modeToggleId}");
    }

    void Update()
    {
        float pxScale = GetPxScale();
        float tapMovePx = tapMaxMovePixels * pxScale;
        float dragDeadzonePx = dragDeadzonePixels * pxScale;
        float dragMaxRadiusPx = dragMaxRadiusPixels * pxScale;
        float tripleTapMovePx = tripleTapMaxMovePixels * pxScale;

        bool hold = (Input.touchCount > 0);

        _ax = 0f; _ay = 0f; _dragActive = false;

        if (_pendingModeToggle && Input.touchCount == 0 && Time.unscaledTime > _pendingModeToggleExpire)
        {
            _awaitingTripleTap = false;
            CommitPendingModeToggle();
        }

        if (_awaitingTripleTap && Input.touchCount > 1)
        {
            _awaitingTripleTap = false;
            CommitPendingModeToggle();
        }

        if (Input.touchCount == 1)
        {
            Touch t0 = Input.GetTouch(0);
            float now = Time.unscaledTime;

            if (t0.phase == TouchPhase.Began)
            {
                _touchTracking = true;
                _touchStartTime = now;
                _touchStartPos = t0.position;
                _dragEverThisTouch = false;

                // ★ 이번 터치 시작 시 초기화
                _doubleTapConfirmedThisTouch = false;

                bool closeInTime = (now - _lastTapTime) <= doubleTapMaxGap;
                bool closeInSpace = (t0.position - _lastTapPos).sqrMagnitude <= (tapMovePx * tapMovePx);

                if (closeInTime && closeInSpace)
                {
                    _pendingSingleTap = false; // 1st tap 싱글 후보 취소

                    // 이번 터치는 더블탭의 2nd tap 후보로 잡혔으므로 싱글탭 후보에서 제외.
                    _doubleTapConfirmedThisTouch = true;

                    // 다음 연쇄 매칭 방지(3연타에서 엉키는 것 줄이기)
                    _lastTapTime = -999f;
                }
            }

            if (_touchTracking && (t0.phase == TouchPhase.Moved || t0.phase == TouchPhase.Stationary))
            {
                Vector2 delta = t0.position - _touchStartPos;
                float mag = delta.magnitude;

                if (mag >= dragDeadzonePx)
                {
                    _dragEverThisTouch = true;

                    Vector2 clamped = Vector2.ClampMagnitude(delta, dragMaxRadiusPx);
                    Vector2 norm = clamped / Mathf.Max(dragMaxRadiusPx, 1e-3f);

                    _ax = Mathf.Clamp(norm.x, -1f, 1f);
                    _ay = Mathf.Clamp(norm.y, -1f, 1f);
                    if (invertY) _ay = -_ay;

                    _dragActive = true;

                    // 더블탭 이후 드래그가 시작되면 트리플탭 후보는 즉시 무효화.
                    if (_awaitingTripleTap && !_doubleTapConfirmedThisTouch)
                    {
                        _awaitingTripleTap = false;
                        CommitPendingModeToggle();
                    }
                }
            }

            if (_touchTracking && (t0.phase == TouchPhase.Ended || t0.phase == TouchPhase.Canceled))
            {
                float dt = now - _touchStartTime;
                float move = (t0.position - _touchStartPos).magnitude;
                bool isTapCandidate =
                    !_dragEverThisTouch &&
                    dt <= tapMaxDuration &&
                    move <= tapMovePx;

                if (_doubleTapConfirmedThisTouch)
                {
                    if (isTapCandidate)
                    {
                        // 2nd tap이 정상 종료되면 더블탭은 잠시 보류하고, 3rd tap이 오면 트리플로 승격.
                        _pendingModeToggle = true;
                        _pendingModeToggleExpire = now + tripleTapMaxGap;
                        _awaitingTripleTap = true;
                        _awaitingTripleTapExpire = _pendingModeToggleExpire;
                        _doubleTapPos = t0.position;
                    }
                    else
                    {
                        _pendingModeToggle = false;
                        _awaitingTripleTap = false;
                    }
                }
                else if (_awaitingTripleTap)
                {
                    bool withinGap = now <= _awaitingTripleTapExpire;
                    bool closeToDoubleTap =
                        (t0.position - _doubleTapPos).sqrMagnitude <= (tripleTapMovePx * tripleTapMovePx);

                    if (withinGap && closeToDoubleTap && isTapCandidate)
                    {
                        _pendingModeToggle = false;
                        _tripleTapId++;
                        _awaitingTripleTap = false;
                        Debug.Log($"[PhoneTX] tripleTapId -> {_tripleTapId}");

                        // 트리플로 확정된 터치가 싱글탭으로 다시 들어가지 않게 차단.
                        _doubleTapConfirmedThisTouch = true;
                    }
                    else
                    {
                        // 3번째 터치가 끝났으면 성공/실패와 관계없이 트리플 대기는 종료.
                        _awaitingTripleTap = false;
                        CommitPendingModeToggle();
                    }
                }

                // ★ 더블탭으로 확정된 터치는 싱글탭 후보 생성 금지
                if (!_doubleTapConfirmedThisTouch)
                {
                    if (isTapCandidate)
                    {
                        _pendingSingleTap = true;
                        _pendingSingleTapExpire = now + doubleTapMaxGap;

                        _lastTapTime = now;
                        _lastTapPos = t0.position;
                    }
                }

                _touchTracking = false;
            }
        }
        else
        {
            _touchTracking = false;
        }

        if (_pendingSingleTap && Input.touchCount == 0 && Time.unscaledTime >= _pendingSingleTapExpire)
        {
            _pendingSingleTap = false;
            _grabToggled = !_grabToggled;
            Debug.Log($"[PhoneTX] toggle(state) -> {_grabToggled}");
        }

        if (arCamera == null) return;

        float interval = Mathf.Max(1f / Mathf.Max(sendHz, 1), 0.001f);
        if (Time.unscaledTime < _nextSendTime) return;
        _nextSendTime = Time.unscaledTime + interval;

        Vector3 p = arCamera.position;
        Quaternion q = arCamera.rotation;
        Pose phoneWorldPose = new Pose(p, q);

        bool markerVisible = includeMarkerPoseInPacket && markerTracker != null && markerTracker.MarkerVisible;
        Vector3 markerPosition = markerVisible ? markerTracker.MarkerPosition : Vector3.zero;
        Quaternion markerRotation = markerVisible ? markerTracker.MarkerRotation : Quaternion.identity;

        if (autoCalibrateQrOnFirstMarker && !_hasQrCalibration && markerTracker != null && markerTracker.MarkerVisible)
        {
            CalibrateQr(markerTracker.MarkerPosition, markerTracker.MarkerRotation, phoneWorldPose);
        }

        bool hasQrDelta = includeQrDeltaInPacket && _hasQrCalibration;
        Vector3 qrDeltaPosition = Vector3.zero;
        Quaternion qrDeltaRotation = Quaternion.identity;
        if (hasQrDelta)
        {
            Pose qrPhoneNow = MakeRelativePose(_qrWorldPose, phoneWorldPose);
            qrDeltaPosition = qrPhoneNow.position - _qrPhone0Pose.position;
            qrDeltaRotation = qrPhoneNow.rotation * Quaternion.Inverse(_qrPhone0Pose.rotation);
        }

        var pkt = new PosePacket
        {
            t = Time.realtimeSinceStartupAsDouble,
            px = p.x, py = p.y, pz = p.z,
            qx = q.x, qy = q.y, qz = q.z, qw = q.w,

            mvis = markerVisible,
            mname = markerVisible && markerTracker != null ? markerTracker.MarkerName : "",
            mx = markerPosition.x, my = markerPosition.y, mz = markerPosition.z,
            mqx = markerRotation.x, mqy = markerRotation.y, mqz = markerRotation.z, mqw = markerRotation.w,

            qrCalibrated = hasQrDelta,
            dx_qr = qrDeltaPosition.x, dy_qr = qrDeltaPosition.y, dz_qr = qrDeltaPosition.z,
            dqx_qr = qrDeltaRotation.x, dqy_qr = qrDeltaRotation.y, dqz_qr = qrDeltaRotation.z, dqw_qr = qrDeltaRotation.w,

            hold = hold,
            toggle = _grabToggled,
            modeToggleId = _modeToggleId,
            tripleTapId = _tripleTapId,

            ax = _ax,
            ay = _ay,
            drag = _dragActive
        };

        string json = JsonUtility.ToJson(pkt);
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        _udp.Send(bytes, bytes.Length, _endPoint);
    }

    [ContextMenu("Reset QR Calibration")]
    public void ResetQrCalibration()
    {
        _hasQrCalibration = false;
        _qrWorldPose = new Pose(Vector3.zero, Quaternion.identity);
        _qrPhone0Pose = new Pose(Vector3.zero, Quaternion.identity);
        Debug.Log("[PhoneTX] QR calibration reset.");
    }

    [ContextMenu("Calibrate QR Now")]
    public void CalibrateQrNow()
    {
        if (arCamera == null || markerTracker == null || !markerTracker.MarkerVisible)
        {
            Debug.LogWarning("[PhoneTX] Cannot calibrate QR: camera or visible marker is missing.");
            return;
        }

        CalibrateQr(
            markerTracker.MarkerPosition,
            markerTracker.MarkerRotation,
            new Pose(arCamera.position, arCamera.rotation));
    }

    private void CalibrateQr(Vector3 markerPosition, Quaternion markerRotation, Pose phoneWorldPose)
    {
        _qrWorldPose = new Pose(markerPosition, markerRotation);
        _qrPhone0Pose = MakeRelativePose(_qrWorldPose, phoneWorldPose);
        _hasQrCalibration = true;
        Debug.Log("[PhoneTX] QR calibration captured.");
    }

    private static Pose MakeRelativePose(Pose parent, Pose child)
    {
        Quaternion invParentRot = Quaternion.Inverse(parent.rotation);
        return new Pose(
            invParentRot * (child.position - parent.position),
            invParentRot * child.rotation
        );
    }
}
