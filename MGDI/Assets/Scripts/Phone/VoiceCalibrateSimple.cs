using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Windows.Speech;

public class VoiceCalibrateSimple : MonoBehaviour
{
    [SerializeField] private PhoneWorldAlignmentManager align;

    private KeywordRecognizer _rec;
    private Dictionary<string, Action> _map;

    void Start()
    {
        if (align == null)
            align = FindFirstObjectByType<PhoneWorldAlignmentManager>();

        _map = new Dictionary<string, Action>(StringComparer.OrdinalIgnoreCase)
        {
            { "calibrate", () => align?.CalibrateNow() },
            { "clear", () => align?.ClearAlignment() },
        };

        _rec = new KeywordRecognizer(new List<string>(_map.Keys).ToArray(), ConfidenceLevel.Medium);
        _rec.OnPhraseRecognized += e =>
        {
            if (_map.TryGetValue(e.text, out var act)) act?.Invoke();
        };
        _rec.Start();

        Debug.Log("[VoiceCalibrateSimple] Listening: calibrate, clear");
    }

    void OnDestroy()
    {
        if (_rec != null)
        {
            _rec.Stop();
            _rec.Dispose();
            _rec = null;
        }
    }
}
