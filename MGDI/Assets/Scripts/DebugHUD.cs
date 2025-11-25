using System.Collections.Generic;
using System.Text;
using TMPro;
using UnityEngine;

/// <summary>
/// Simple on-device text HUD. Call DebugHUD.Log("...") from anywhere.
/// Attach this to the same GameObject that has the TMP_Text component.
/// </summary>
public class DebugHUD : MonoBehaviour
{
    public static DebugHUD Instance;

    [Header("UI")]
    public TMP_Text text;              // Assign DebugHUD_Text here
    public int maxLines = 12;

    readonly Queue<string> _lines = new Queue<string>(32);
    readonly StringBuilder _sb = new StringBuilder(1024);

    void Awake()
    {
        Instance = this;
        if (text == null) text = GetComponentInChildren<TMP_Text>(true);
    }

    public static void Log(string message)
    {
        if (Instance == null) return;
        Instance.Enqueue(message);
    }

    void Enqueue(string msg)
    {
        _lines.Enqueue(msg);
        while (_lines.Count > maxLines) _lines.Dequeue();

        _sb.Clear();
        foreach (var line in _lines) _sb.AppendLine(line);
        if (text != null) text.text = _sb.ToString();
    }
}
