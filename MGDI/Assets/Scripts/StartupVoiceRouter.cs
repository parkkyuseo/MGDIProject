using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Windows.Speech;
using System;

public class StartupVoiceRouter : MonoBehaviour
{
    [Header("Scene Names")]
    public string calibrationSceneName = "CalibrationScene";
    public string runtimeSceneName = "RuntimeScene";

    [Header("Voice")]
    public bool enableVoice = true;
    public float autoGoAfterSec = 0f; // 0이면 자동 없음. >0이면 n초 뒤 runtime으로 자동 이동.

    private KeywordRecognizer _kr;
    private string[] _kws = new[] { "calibrate", "calibration", "runtime", "런타임", "캘리브레이션" };
    private bool _routed;

    void Start()
    {
        if (enableVoice)
        {
            try
            {
                _kr = new KeywordRecognizer(_kws, ConfidenceLevel.Medium);
                _kr.OnPhraseRecognized += (a) =>
                {
                    if (_routed) return;
                    var s = a.text.ToLower();
                    if (s.Contains("calib") || s.Contains("캘리")) Go(calibrationSceneName);
                    else Go(runtimeSceneName);
                };
                _kr.Start();
                Log("Say 'calibrate/캘리브레이션' or 'runtime/런타임'");
            }
            catch (Exception ex) { Log("Voice init failed: " + ex.Message); }
        }
        if (autoGoAfterSec > 0f) Invoke(nameof(GoRuntime), autoGoAfterSec);
    }

    public void GoRuntime() => Go(runtimeSceneName);
    public void GoCalibration() => Go(calibrationSceneName);

    void Go(string scene)
    {
        if (_routed) return;
        _routed = true;
        try { if (_kr != null && _kr.IsRunning) _kr.Stop(); } catch { }
        try { if (_kr != null) _kr.Dispose(); } catch { }
        SceneManager.LoadScene(scene);
    }

    void Log(string m) { Debug.Log("[StartupVoiceRouter] " + m); try { DebugHUD.Log("[Start] " + m); } catch { } }
}
