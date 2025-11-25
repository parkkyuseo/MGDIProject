// ARMarkerAnchorCapture_OpenXR.cs (minimal logs, voice trigger only)
// - OpenXR ARMarkerManager 사용
// - QR(마커) 잠기면 HUD에 1줄 로그만, 자동 촬영 없음(음성만)
// - PhotoCapture 1장 → R_HC / t_HC / K / anchor(R_HA,t_HA) 포함 JSON + PNG를 TCP 전송

using System;
using System.Collections;
using System.Globalization;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.Windows.WebCam;
using UnityEngine.Windows.Speech;           // KeywordRecognizer
using Microsoft.MixedReality.OpenXR;       // ARMarkerManager, ARMarker, ARMarkersChangedEventArgs

public class ARMarkerAnchorCapture_OpenXR : MonoBehaviour
{
    [Header("Refs")]
    public ARMarkerManager markerManager;   // ARMarkerSystem 오브젝트의 컴포넌트 지정

    [Header("Capture")]
    public int targetWidth = 1280;
    public int targetHeight = 720;
    public string filePrefix = "hlcap";

    [Header("PC TCP")]
    public string pcIp = "192.168.1.50";
    public int pcPort = 19610;

    [Header("Thumbnail")]
    public bool includeThumbnail = false;
    [Range(10, 95)] public int jpegQuality = 70;
    public int thumbWidth = 640;

    private bool _anchorLocked = false;
    private bool _capturedOnce = false;
    private Pose _anchorPoseWorld;              // 앵커(A)의 월드 Pose
    private PhotoCapture _photo;
    private KeywordRecognizer _kr;
    private readonly string[] _keywords = new[] { "capture", "캡처", "촬영" };

    void Awake()
    {
        if (markerManager == null) markerManager = GetComponent<ARMarkerManager>();
    }

    void OnEnable()
    {
        if (markerManager == null) markerManager = GetComponent<ARMarkerManager>();
        markerManager.markersChanged += OnMarkersChanged;

        // 음성 인식 시작 (앵커 잠긴 뒤에만 촬영 허용)
        try
        {
            _kr = new KeywordRecognizer(_keywords, ConfidenceLevel.Medium);
            _kr.OnPhraseRecognized += (args) => { TryVoiceCapture(); };
            _kr.Start();
            Log("Voice ready: say 'capture' / '캡처' / '촬영'");
        }
        catch (Exception ex)
        {
            Log("Voice init failed: " + ex.Message);
        }
    }

    void OnDisable()
    {
        markerManager.markersChanged -= OnMarkersChanged;
        if (_kr != null) { if (_kr.IsRunning) _kr.Stop(); _kr.Dispose(); _kr = null; }
    }

    // OpenXR: markersChanged(added/updated/removed)
    private void OnMarkersChanged(ARMarkersChangedEventArgs args)
    {
        if (!_anchorLocked && args.added != null && args.added.Count > 0)
        {
            var m = args.added[0];
            var t = m.transform;                       // 생성된 마커 GO의 transform
            _anchorPoseWorld = new Pose(t.position, t.rotation);
            _anchorLocked = true;
            Log("[Anchor] locked");
        }
    }

    private void TryVoiceCapture()
    {
        if (!_anchorLocked) { Log("QR not locked yet"); return; }
        if (_capturedOnce) { Log("Already captured"); return; }
        StartCoroutine(CaptureAndSendOnce());
    }

    IEnumerator CaptureAndSendOnce()
    {
        if (_capturedOnce) yield break;
        _capturedOnce = true;

        var camParams = new CameraParameters(WebCamMode.PhotoMode)
        {
            cameraResolutionWidth = targetWidth,
            cameraResolutionHeight = targetHeight,
            pixelFormat = CapturePixelFormat.BGRA32
        };

        bool created = false;
        PhotoCapture.CreateAsync(false, c => { _photo = c; created = true; });
        while (!created) yield return null;

        bool started = false;
        _photo.StartPhotoModeAsync(camParams, res => { started = res.success; });
        while (!started) yield return null;

        Matrix4x4 camToWorld = Matrix4x4.identity;
        Matrix4x4 proj = Matrix4x4.identity;
        Texture2D tex = new Texture2D(targetWidth, targetHeight, TextureFormat.BGRA32, false);

        bool done = false;
        _photo.TakePhotoAsync((PhotoCapture.PhotoCaptureResult r, PhotoCaptureFrame frame) =>
        {
            if (!r.success) { Log("TakePhoto failed"); done = true; return; }

            if (!frame.TryGetCameraToWorldMatrix(out camToWorld))
            {
                Log("TryGetCameraToWorldMatrix failed");
                camToWorld = Matrix4x4.identity;
            }

            if (!frame.TryGetProjectionMatrix(out proj))
            {
                Log("TryGetProjectionMatrix failed");
                proj = Matrix4x4.identity;
            }

            frame.UploadImageDataToTexture(tex);
            done = true;
        });
        while (!done) yield return null;

        bool stopped = false;
        _photo.StopPhotoModeAsync(res => { stopped = true; });
        while (!stopped) yield return null;
        _photo.Dispose(); _photo = null;

        // --- Camera->World (R_HC, t_HC) ---
        // row-major flatten (r00 r01 r02 r10 r11 r12 r20 r21 r22)
        float[] R_HC9 = new float[9];
        {
            Matrix4x4 Rm = camToWorld;
            Rm.SetColumn(3, new Vector4(0, 0, 0, 1)); // remove translation
            int k = 0;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    R_HC9[k++] = Rm[c, r];  // Unity col-major -> row-major
        }
        Vector3 tc = camToWorld.GetColumn(3);
        float[] t_HC = new float[] { tc.x, tc.y, tc.z };

        // --- Projection -> K (row-major flatten) ---
        float w = targetWidth, h = targetHeight;
        float p00 = proj[0, 0], p11 = proj[1, 1], p02 = proj[0, 2], p12 = proj[1, 2];
        float fx = Mathf.Abs(p00) * w / 2f;
        float fy = Mathf.Abs(p11) * h / 2f;
        float cx = (1f - p02) * w / 2f;
        float cy = (1f + p12) * h / 2f;
        float[] K9 = new float[9]{
            fx, 0f, cx,
            0f, fy, cy,
            0f, 0f, 1f
        };


        // --- Anchor(A)->World(H) (R_HA, t_HA) ---
        var R_HA = new float[3][] { new float[3], new float[3], new float[3] };
        var t_HA = new float[] { _anchorPoseWorld.position.x, _anchorPoseWorld.position.y, _anchorPoseWorld.position.z };
        Matrix4x4 Ra = Matrix4x4.Rotate(_anchorPoseWorld.rotation);
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) R_HA[r][c] = Ra[c, r];

        // --- Optional thumbnail ---
        string thumbB64 = null;
        if (includeThumbnail)
        {
            Texture2D thumb = MakeThumbnail(tex, thumbWidth);
            byte[] jpg = thumb.EncodeToJPG(jpegQuality);
            thumbB64 = Convert.ToBase64String(jpg);
        }

        // --- Encode PNG & JSON ---
        byte[] pngBytes = tex.EncodeToPNG();
        string imageName = $"{filePrefix}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.png";

        var payload = new RootEnvelope
        {
            type = "anchor_capture_openxr",
            data = new PosePayload
            {
                image = imageName,
                width = targetWidth,
                height = targetHeight,
                R_HC = R_HC9,
                t_HC = t_HC,
                K = K9,
                dist = new float[] { },
                anchor = new AnchorBlock { seen = _anchorLocked, id = "openxr", R_HA = R_HA, t_HA = t_HA }
            },
            thumbnail_jpeg_b64 = thumbB64
        };
        string json = JsonUtility.ToJson(payload);

        // --- TCP send (len-prefixed) ---
        try
        {
            using (var client = new TcpClient())
            {
                client.SendTimeout = 2000; client.ReceiveTimeout = 2000;
                client.Connect(pcIp, pcPort);
                using (var ns = client.GetStream())
                {
                    byte[] j = Encoding.UTF8.GetBytes(json);
                    byte[] p = pngBytes;
                    byte[] jl = BitConverter.GetBytes(j.Length);
                    byte[] pl = BitConverter.GetBytes(p.Length);
                    if (!BitConverter.IsLittleEndian) { Array.Reverse(jl); Array.Reverse(pl); }
                    ns.Write(jl, 0, 4); ns.Write(j, 0, j.Length);
                    ns.Write(pl, 0, 4); ns.Write(p, 0, p.Length);
                    ns.Flush();
                }
            }
            Log($"TCP sent: json={json.Length}B, png={pngBytes.Length}B");
        }
        catch (Exception ex)
        {
            Log("TCP send failed: " + ex.Message);
        }
    }

    Texture2D MakeThumbnail(Texture2D src, int targetW)
    {
        float scale = (float)targetW / src.width;
        int tw = targetW, th = Mathf.RoundToInt(src.height * scale);
        RenderTexture rt = RenderTexture.GetTemporary(tw, th);
        Graphics.Blit(src, rt);
        var prev = RenderTexture.active; RenderTexture.active = rt;
        Texture2D tex = new Texture2D(tw, th, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, tw, th), 0, 0); tex.Apply();
        RenderTexture.active = prev; RenderTexture.ReleaseTemporary(rt);
        return tex;
    }

    // JSON payload types
    [Serializable] public class AnchorBlock { public bool seen; public string id; public float[][] R_HA; public float[] t_HA; }
    [Serializable]
    public class PosePayload
    {
        public string image; public int width, height;
        //public float[][] R_HC; public float[] t_HC;
        //public float[][] K; 
        public float[] R_HC;  // length 9 (row-major)
        public float[] t_HC;  // length 3
        public float[] K;     // length 9 (row-major)

        public float[] dist;
        public AnchorBlock anchor;
    }
    [Serializable] public class RootEnvelope { public string type; public PosePayload data; public string thumbnail_jpeg_b64; }

    // minimal logger (필요한 것만)
    void Log(string msg) { Debug.Log("[ARMarker] " + msg); try { DebugHUD.Log("[ARMarker] " + msg); } catch { } }
}
