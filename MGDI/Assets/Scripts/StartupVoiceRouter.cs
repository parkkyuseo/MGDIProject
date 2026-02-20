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
    [Tooltip("Legacy option. Auto runtime transition is disabled in this flow.")]
    public float autoGoAfterSec = 0f;

    [Header("Legacy Runtime Route")]
    [Tooltip("If false, runtime transition from this router is blocked. ParticipantIdGate handles runtime transition.")]
    public bool allowLegacyRuntimeRouting = false;

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
                    else if ((s.Contains("runtime") || s.Contains("런타임")) && allowLegacyRuntimeRouting) Go(runtimeSceneName);
                    else if (s.Contains("runtime") || s.Contains("런타임")) Log("Runtime voice route is disabled.");
                };
                _kr.Start();
                Log("Say 'calibrate/캘리브레이션'");
            }
            catch (Exception ex) { Log("Voice init failed: " + ex.Message); }
        }
    }

    public void GoRuntime()
    {
        if (!allowLegacyRuntimeRouting)
        {
            Log("Runtime route is disabled. Use ParticipantIdGate Continue.");
            return;
        }

        Go(runtimeSceneName);
    }

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
