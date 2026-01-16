using System.Collections;
using TMPro;
using UnityEngine;

public class InstructionHUD : MonoBehaviour
{
    [SerializeField] private TMP_Text instructionText;
    [SerializeField] private float defaultShowSeconds = 1.8f;

    Coroutine _hideCo;

    public void HideImmediate()
    {
        if (_hideCo != null) StopCoroutine(_hideCo);
        _hideCo = null;

        if (instructionText != null) instructionText.text = "";
        gameObject.SetActive(false);
    }

    public void Show(string text, float? seconds = null)
    {
        if (instructionText == null) return;

        if (_hideCo != null) StopCoroutine(_hideCo);
        _hideCo = null;

        instructionText.text = text ?? "";
        gameObject.SetActive(true);

        float s = seconds ?? defaultShowSeconds;
        _hideCo = StartCoroutine(HideAfter(s));
    }

    IEnumerator HideAfter(float s)
    {
        yield return new WaitForSeconds(Mathf.Max(0f, s));
        HideImmediate();
    }
}
