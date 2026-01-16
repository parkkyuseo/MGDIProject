using System.Collections;
using TMPro;
using UnityEngine;

public class InstructionHUD : MonoBehaviour
{
    [SerializeField] private TMP_Text instructionText;
    [SerializeField] private float defaultShowSeconds = 1.8f;

    public float DefaultShowSeconds => defaultShowSeconds;

    Coroutine _hideCo;

    public void HideImmediate()
    {
        if (_hideCo != null) StopCoroutine(_hideCo);
        _hideCo = null;

        if (instructionText != null) instructionText.text = "";
        gameObject.SetActive(false);
    }

    public float Show(string text, float? seconds = null)
    {
        if (instructionText == null) return 0f;

        if (_hideCo != null) StopCoroutine(_hideCo);
        _hideCo = null;

        instructionText.text = text ?? "";
        gameObject.SetActive(true);

        float s = seconds ?? defaultShowSeconds;
        _hideCo = StartCoroutine(HideAfter(s));
        return s;
    }

    IEnumerator HideAfter(float s)
    {
        yield return new WaitForSeconds(Mathf.Max(0f, s));
        HideImmediate();
    }
}
