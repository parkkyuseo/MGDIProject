// StartupMenu.cs
// - 앱 시작 시 'Calibration' / 'Runtime' 중 사용자가 선택할 수 있는 라우터
// - UI 버튼 OnClick, 음성 명령, 키보드(에디터)까지 지원
// - HoloLens 배포 시 Microphone capability 켜면 음성도 동작

using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Windows.Speech;   // 음성(선택)
using System;

public class StartupMenu : MonoBehaviour
{
    [Header("Scene Names")]
    public string calibrationSceneName = "CalibrationScene";
    public string runtimeSceneName = "RuntimeScene";

    [Header("Voice Options")]
    public bool enableVoice = true;
    public float autoGoAfterSec = 0f;  // 0이면 자동 없음, >0이면 n초 후 runtime으로 자동 이동
    private KeywordRecognizer _kr;
    private string[] _keywords = new[] { "calibrate", "calibration", "runtime", "캘리브레이션", "런타임" };

    private bool _routed = false;

    void Start()
    {
        // 음성 인식(선택)
        if (enableVoice)
        {
            try
            {
                _kr = new KeywordRecognizer(_keywords, ConfidenceLevel.Medium);
                _kr.OnPhraseRecognized += (args) =>
                {
                    if (_routed) return;
                    string s = args.text.ToLower();
                    if (s.Contains("calib") || s.Contains("캘리"))
                        GoCalibration();
                    else
                        GoRuntime();
                };
                _kr.Start();
                Log("Voice ready: say 'calibrate/캘리브레이션' or 'runtime/런타임'");
            }
            catch (Exception ex)
            {
                Log("Voice init failed: " + ex.Message);
            }
        }

        // 자동 이동(선택)
        if (autoGoAfterSec > 0f)
            Invoke(nameof(GoRuntime), autoGoAfterSec);
    }

    void Update()
    {
#if UNITY_EDITOR
        // 에디터 테스트용 단축키
        if (!_routed && Input.GetKeyDown(KeyCode.C)) GoCalibration();
        if (!_routed && Input.GetKeyDown(KeyCode.R)) GoRuntime();
#endif
    }

    // --- UI 버튼에서 연결할 메서드 ---
    public void OnClickCalibration() { GoCalibration(); }
    public void OnClickRuntime() { GoRuntime(); }

    private void GoCalibration()
    {
        if (_routed) return;
        _routed = true;
        StopVoice();
        Log("Loading " + calibrationSceneName);
        SceneManager.LoadScene(calibrationSceneName);
    }

    private void GoRuntime()
    {
        if (_routed) return;
        _routed = true;
        StopVoice();
        Log("Loading " + runtimeSceneName);
        SceneManager.LoadScene(runtimeSceneName);
    }

    void StopVoice()
    {
        try { if (_kr != null && _kr.IsRunning) _kr.Stop(); } catch { }
        try { if (_kr != null) _kr.Dispose(); } catch { }
        _kr = null;
    }

    void Log(string m)
    {
        Debug.Log("[StartupMenu] " + m);
        try { DebugHUD.Log("[Start] " + m); } catch { }
    }
}
