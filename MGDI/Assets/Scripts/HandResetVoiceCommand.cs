using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Windows.Speech;

/// <summary>
/// HoloLens용 음성 명령으로 RemoteHandRuntime의 ContextClearAndRearm()을 호출하는 스크립트.
/// "reset" 또는 "리셋"이라고 말하면 양쪽 손을 리셋한다.
/// </summary>
public class HandResetVoiceCommand : MonoBehaviour
{
    [Header("Hands to reset")]
    [Tooltip("왼손 RemoteHandRuntime (없으면 비워두기)")]
    public RemoteHandRuntime leftHand;

    [Tooltip("오른손 RemoteHandRuntime (없으면 비워두기)")]
    public RemoteHandRuntime rightHand;

    [Header("Voice keywords")]
    [Tooltip("리셋에 사용할 키워드들 (소문자로 비교). 예: \"reset\", \"리셋\"")]
    public string[] keywords = new[] { "reset", "리셋" };

    KeywordRecognizer _recognizer;
    Dictionary<string, Action> _actions;

    void Start()
    {
        // 키워드 설정이 없으면 아무 것도 안 함
        if (keywords == null || keywords.Length == 0)
        {
            Debug.LogWarning("[HandResetVoiceCommand] No keywords set.");
            return;
        }

        // 키워드 -> 액션 매핑
        _actions = new Dictionary<string, Action>();

        foreach (var word in keywords)
        {
            if (string.IsNullOrWhiteSpace(word))
                continue;

            var key = word.Trim().ToLowerInvariant();

            if (_actions.ContainsKey(key))
                continue; // 중복 방지

            _actions[key] = OnResetCommand;
        }

        if (_actions.Count == 0)
        {
            Debug.LogWarning("[HandResetVoiceCommand] No valid keywords after filtering.");
            return;
        }

        // KeywordRecognizer 생성
        _recognizer = new KeywordRecognizer(_actions.Keys.ToArray());
        _recognizer.OnPhraseRecognized += OnPhraseRecognized;
        _recognizer.Start();

        Debug.Log("[HandResetVoiceCommand] KeywordRecognizer started. Keywords: " +
                  string.Join(", ", _actions.Keys));
    }

    void OnDestroy()
    {
        if (_recognizer != null)
        {
            if (_recognizer.IsRunning)
                _recognizer.Stop();

            _recognizer.OnPhraseRecognized -= OnPhraseRecognized;
            _recognizer.Dispose();
            _recognizer = null;
        }
    }

    void OnPhraseRecognized(PhraseRecognizedEventArgs args)
    {
        var spoken = args.text.Trim().ToLowerInvariant();
        Debug.Log("[HandResetVoiceCommand] Heard: " + spoken);

        if (_actions != null && _actions.TryGetValue(spoken, out var action))
        {
            action?.Invoke();
        }
    }

    void OnResetCommand()
    {
        Debug.Log("[HandResetVoiceCommand] Reset command recognized. Clearing hands.");

        if (leftHand != null)
        {
            leftHand.ContextClearAndRearm();
        }

        if (rightHand != null)
        {
            rightHand.ContextClearAndRearm();
        }
    }
}
