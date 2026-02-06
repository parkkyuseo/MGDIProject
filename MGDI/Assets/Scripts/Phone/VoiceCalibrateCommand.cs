using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Windows.Speech;

public class VoiceCalibrateCommand : MonoBehaviour
{
    [Header("Target")]
    [SerializeField] private PhoneWorldAlignmentManager alignManager;

    [Header("Keywords")]
    [SerializeField] private string calibrateKeyword = "calibrate";
    [SerializeField] private string recenterKeyword = "recenter";
    [SerializeField] private string clearKeyword = "clear alignment";

    private KeywordRecognizer _recognizer;
    private Dictionary<string, Action> _actions;

    void Start()
    {
        if (alignManager == null)
            alignManager = FindFirstObjectByType<PhoneWorldAlignmentManager>();

        _actions = new Dictionary<string, Action>(StringComparer.OrdinalIgnoreCase);

        if (!string.IsNullOrWhiteSpace(calibrateKeyword))
            _actions[calibrateKeyword] = TryCalibrate;

        if (!string.IsNullOrWhiteSpace(recenterKeyword))
            _actions[recenterKeyword] = TryRecenter;

        if (!string.IsNullOrWhiteSpace(clearKeyword))
            _actions[clearKeyword] = TryClear;

        if (_actions.Count == 0)
        {
            Debug.LogWarning("[VoiceCalibrateCommand] No keywords configured.");
            return;
        }

        _recognizer = new KeywordRecognizer(new List<string>(_actions.Keys).ToArray(), ConfidenceLevel.Medium);
        _recognizer.OnPhraseRecognized += OnPhraseRecognized;
        _recognizer.Start();

        Debug.Log("[VoiceCalibrateCommand] Listening: " + string.Join(", ", _actions.Keys));
    }

    void OnDestroy()
    {
        if (_recognizer != null)
        {
            _recognizer.OnPhraseRecognized -= OnPhraseRecognized;
            if (_recognizer.IsRunning) _recognizer.Stop();
            _recognizer.Dispose();
            _recognizer = null;
        }
    }

    private void OnPhraseRecognized(PhraseRecognizedEventArgs args)
    {
        if (_actions != null && _actions.TryGetValue(args.text, out var act))
            act?.Invoke();
    }

    private void TryCalibrate()
    {
        if (alignManager == null)
        {
            Debug.LogWarning("[VoiceCalibrateCommand] alignManager missing.");
            return;
        }
        alignManager.CalibrateNow();
    }

    private void TryRecenter()
    {
        if (alignManager == null)
        {
            Debug.LogWarning("[VoiceCalibrateCommand] alignManager missing.");
            return;
        }
        alignManager.RecenterNow();
    }

    private void TryClear()
    {
        if (alignManager == null)
        {
            Debug.LogWarning("[VoiceCalibrateCommand] alignManager missing.");
            return;
        }
        alignManager.ClearAlignment();
    }
}
