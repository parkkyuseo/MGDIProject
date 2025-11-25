// HLCapturePoseAndPhoto.cs (Pictures 전용 저장, DebugHUD, final)
// - Voice("capture","캡처","촬영") 또는 Timer로 촬영
// - PNG + pose.json을 Pictures\HLCapture ONLY 저장
// - UDP로 pose JSON(필수) + 옵션 썸네일(base64) 전송
// - float + jagged 배열(JSON 직렬화 호환), PhotoCapture.PhotoCaptureResult 사용
// - 모든 상태/오류는 DebugHUD.Log(...)로 표시

using System;
using System.IO;
using System.Text;
using System.Collections;
using UnityEngine;
using UnityEngine.Windows.WebCam;   // PhotoCapture
using UnityEngine.Windows.Speech;   // KeywordRecognizer
using System.Net.Sockets;
using System.Globalization;

#if ENABLE_WINMD_SUPPORT
using Windows.Storage;
// using Windows.Storage.Streams; // 제거 (모호성 방지)
using System.Threading.Tasks;
#endif

public class HLCapturePoseAndPhoto : MonoBehaviour
{
    public enum CaptureMode { VoiceOnly, TimerAuto }

    [Header("Capture Mode")]
    public CaptureMode mode = CaptureMode.VoiceOnly;
    public float startDelaySec = 3f;
    public float intervalSec = 2f;
    public int shots = 1;

    [Header("Camera Settings")]
    public int targetWidth = 1280;
    public int targetHeight = 720;
    public string filePrefix = "hlcap";

    [Header("UDP to PC")]
    public string pcIp = "";
    public int pcPort = 19561;
    public bool sendThumbnail = false;
    [Range(10, 95)]
    public int jpegQuality = 70;
    public int thumbWidth = 640;

    private PhotoCapture _photo;
    private bool _busy;
    private KeywordRecognizer _kr;
    private readonly string[] _keywords = new string[] { "capture", "캡처", "촬영" };
    private int _shotsTaken = 0;
    private int _sentCount = 0;

    void Start()
    {
        Log("Started. Saving ONLY to Pictures\\HLCapture (no LocalState).");
        SetupVoice();
        if (mode == CaptureMode.TimerAuto) StartCoroutine(TimerRoutine());
        else Log("Voice mode: say 'capture' / '캡처' / '촬영'");
    }

    void OnDestroy()
    {
        if (_kr != null && _kr.IsRunning) _kr.Stop();
        if (_kr != null) _kr.Dispose();
    }

    void SetupVoice()
    {
        try
        {
            _kr = new KeywordRecognizer(_keywords, ConfidenceLevel.Medium);
            _kr.OnPhraseRecognized += (args) =>
            {
                Log($"Voice: {args.text}");
                TryStartOneShot();
            };
            _kr.Start();
            Log("Voice recognizer started.");
        }
        catch (Exception ex) { Log($"Voice setup failed: {ex.Message}"); }
    }

    IEnumerator TimerRoutine()
    {
        Log($"Timer mode: wait {startDelaySec:0.0}s then every {intervalSec:0.0}s, total {shots} shots.");
        yield return new WaitForSeconds(startDelaySec);
        while (_shotsTaken < shots)
        {
            TryStartOneShot();
            yield return new WaitForSeconds(intervalSec);
        }
        Log("Timer mode complete.");
    }

    public void TryStartOneShot()
    {
        if (_busy) return;
        StartCoroutine(CaptureOne());
    }

    IEnumerator CaptureOne()
    {
        _busy = true;
        _shotsTaken++;
        Log($"Capturing... ({_shotsTaken})");

        var camParams = new CameraParameters(WebCamMode.PhotoMode)
        {
            cameraResolutionWidth = targetWidth,
            cameraResolutionHeight = targetHeight,
            pixelFormat = CapturePixelFormat.BGRA32
        };

        bool created = false;
        PhotoCapture.CreateAsync(false, capture => { _photo = capture; created = true; });
        while (!created) yield return null;

        bool started = false;
        _photo.StartPhotoModeAsync(camParams, result => { started = result.success; });
        while (!started) yield return null;

        Matrix4x4 camToWorld = Matrix4x4.identity;
        Matrix4x4 proj = Matrix4x4.identity;
        Texture2D tex = new Texture2D(targetWidth, targetHeight, TextureFormat.BGRA32, false);

        bool done = false;
        _photo.TakePhotoAsync((PhotoCapture.PhotoCaptureResult res, PhotoCaptureFrame frame) =>
        {
            if (!res.success) { Log("TakePhoto failed."); done = true; return; }
            if (!frame.TryGetCameraToWorldMatrix(out camToWorld))
            { Log("TryGetCameraToWorldMatrix failed. Using identity."); camToWorld = Matrix4x4.identity; }
            if (!frame.TryGetProjectionMatrix(out proj))
            { Log("TryGetProjectionMatrix failed. Using identity."); proj = Matrix4x4.identity; }
            frame.UploadImageDataToTexture(tex);
            done = true;
        });
        while (!done) yield return null;

        bool stopped = false;
        _photo.StopPhotoModeAsync(res => { stopped = true; });
        while (!stopped) yield return null;
        _photo.Dispose(); _photo = null;

        string ts = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        string baseName = $"{filePrefix}_{ts}";

        // PNG bytes
        byte[] pngBytes = null;
        try { pngBytes = tex.EncodeToPNG(); }
        catch (Exception ex) { Log($"PNG encode failed: {ex.Message}"); }

        // Pose (float + jagged)
        Matrix4x4 m = camToWorld;
        Vector3 t = m.GetColumn(3);
        Matrix4x4 Rm = m; Rm.SetColumn(3, new Vector4(0, 0, 0, 1));

        var R_HC = new float[3][];
        for (int r = 0; r < 3; r++)
        {
            R_HC[r] = new float[3];
            for (int c = 0; c < 3; c++) R_HC[r][c] = (float)Rm[c, r];
        }
        var t_HC = new float[] { t.x, t.y, t.z };

        float w = targetWidth, h = targetHeight;
        float p00 = proj[0, 0], p11 = proj[1, 1], p02 = proj[0, 2], p12 = proj[1, 2];
        float fx = Mathf.Abs(p00) * w / 2f;
        float fy = Mathf.Abs(p11) * h / 2f;
        float cx = (1f - p02) * w / 2f;
        float cy = (1f + p12) * h / 2f;

        var K = new float[3][]
        {
            new float[]{ fx, 0f, cx },
            new float[]{ 0f, fy, cy },
            new float[]{ 0f, 0f, 1f }
        };

        var pose = new PosePayload
        {
            image = baseName + ".png",
            width = targetWidth,
            height = targetHeight,
            R_HC = R_HC,
            t_HC = t_HC,
            K = K,
            dist = new float[] { }
        };

        string json = "";
        try { json = JsonUtility.ToJson(new Wrapper<PosePayload>(pose), true); }
        catch (Exception ex) { Log($"POSE json build failed: {ex.Message}"); }

#if ENABLE_WINMD_SUPPORT
        if (pngBytes != null && !string.IsNullOrEmpty(json))
            _ = SaveToPicturesAsync(pngBytes, json, baseName);
        else
            Log("Skip saving to Pictures (no png/json).");
#else
        Log("WINMD not enabled; cannot save to PicturesLibrary on this build target.");
#endif

        // UDP 전송
        if (!string.IsNullOrEmpty(pcIp) && pcPort > 0)
        {
            try
            {
                var payload = new UdpPayload { data = pose };
                if (sendThumbnail)
                {
                    Texture2D thumb = MakeThumbnail(tex, thumbWidth);
                    byte[] jpg = thumb.EncodeToJPG(jpegQuality);
                    payload.thumbnail_jpeg_b64 = Convert.ToBase64String(jpg);
                }
                string udpJson = JsonUtility.ToJson(payload);
                using (var udp = new UdpClient())
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(udpJson);
                    udp.Send(bytes, bytes.Length, pcIp, pcPort);
                }
                _sentCount++;
                Log($"UDP sent to {pcIp}:{pcPort}  (#{_sentCount})");
            }
            catch (Exception ex) { Log($"UDP send failed: {ex.Message}"); }
        }
        else Log("UDP disabled (pcIp or pcPort not set).");

        _busy = false;
    }

#if ENABLE_WINMD_SUPPORT
    private async Task SaveToPicturesAsync(byte[] pngBytes, string jsonText, string baseName)
    {
        try
        {
            StorageFolder pics = KnownFolders.PicturesLibrary;
            StorageFolder folder = await pics.CreateFolderAsync("HLCapture",
                CreationCollisionOption.OpenIfExists);

            StorageFile pngFile = await folder.CreateFileAsync(baseName + ".png",
                CreationCollisionOption.ReplaceExisting);
            await FileIO.WriteBytesAsync(pngFile, pngBytes);

            StorageFile jsonFile = await folder.CreateFileAsync(baseName + "_pose.json",
                CreationCollisionOption.ReplaceExisting);

            // 🔧 모호성 해결: Windows.Storage.Streams.UnicodeEncoding.Utf8 로 완전수식
            await FileIO.WriteTextAsync(jsonFile, jsonText, Windows.Storage.Streams.UnicodeEncoding.Utf8);

            Log($"Saved: Pictures\\HLCapture\\{baseName}.png (+_pose.json)");
        }
        catch (Exception ex) { Log($"SaveToPicturesAsync failed: {ex.Message}"); }
    }
#endif

    Texture2D MakeThumbnail(Texture2D src, int targetW)
    {
        float scale = (float)targetW / src.width;
        int tw = targetW;
        int th = Mathf.RoundToInt(src.height * scale);
        RenderTexture rt = RenderTexture.GetTemporary(tw, th);
        Graphics.Blit(src, rt);
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(tw, th, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, tw, th), 0, 0);
        tex.Apply();
        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);
        return tex;
    }

    [Serializable]
    public class PosePayload
    {
        public string image;
        public int width, height;
        public float[][] R_HC;   // 3x3 (row-major)
        public float[] t_HC;    // 3
        public float[][] K;      // 3x3
        public float[] dist;    // optional
    }

    [Serializable]
    public class UdpPayload
    {
        public PosePayload data;
        public string thumbnail_jpeg_b64; // optional
    }

    [Serializable]
    public class Wrapper<T> { public T data; public Wrapper(T x) { data = x; } }

    void Log(string msg)
    {
        Debug.Log("[HLCapturePoseAndPhoto] " + msg);
        try { DebugHUD.Log("[HLCap] " + msg); } catch { }
    }
}
