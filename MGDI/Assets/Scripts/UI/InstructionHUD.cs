using System.Collections;
using TMPro;
using UnityEngine;

public class InstructionHUD : MonoBehaviour
{
    [SerializeField] private TMP_Text instructionText;
    [SerializeField] private float defaultShowSeconds = 1.8f;
    [SerializeField] private InstructionPanelAutoSizer panelAutoSizer;

    [Header("Transitions")]
    [SerializeField] private bool useTransitions = true;
    [SerializeField] private CanvasGroup canvasGroup;
    [SerializeField] private RectTransform panelRect;
    [SerializeField] private float fadeInSeconds = 0.12f;
    [SerializeField] private float fadeOutSeconds = 0.18f;
    [SerializeField] private float popScaleStart = 0.96f;
    [SerializeField] private float popScaleEnd = 1.00f;

    [Header("Style")]
    [SerializeField] private bool overrideTypography = true;
    [SerializeField] private Color normalTextColor = new Color(0.96f, 0.98f, 1f, 1f);
    [SerializeField] private Color countdownTextColor = new Color(1f, 0.87f, 0.55f, 1f);
    [SerializeField] private float normalFontSize = 24f;
    [SerializeField] private float countdownFontSize = 40f;
    [SerializeField] private FontStyles normalFontStyle = FontStyles.Bold;
    [SerializeField] private FontStyles countdownFontStyle = FontStyles.Bold;

    public float DefaultShowSeconds => defaultShowSeconds;

    Coroutine _hideCo;

    private void Awake()
    {
        if (canvasGroup == null)
            canvasGroup = GetComponentInChildren<CanvasGroup>(true);

        if (panelRect == null)
            panelRect = transform as RectTransform;

        if (panelAutoSizer == null)
            panelAutoSizer = GetComponent<InstructionPanelAutoSizer>();

        if (canvasGroup != null)
            canvasGroup.alpha = 0f;
    }

    public void HideImmediate()
    {
        if (_hideCo != null) StopCoroutine(_hideCo);
        _hideCo = null;

        if (instructionText != null) instructionText.text = "";
        if (canvasGroup != null) canvasGroup.alpha = 0f;
        if (panelRect != null) panelRect.localScale = Vector3.one * popScaleEnd;
        gameObject.SetActive(false);
    }

    public float Show(string text, float? seconds = null)
    {
        if (instructionText == null) return 0f;

        if (_hideCo != null) StopCoroutine(_hideCo);
        _hideCo = null;

        instructionText.text = text ?? "";
        gameObject.SetActive(true);
        ApplyStyleForText(instructionText.text);
        panelAutoSizer?.RefreshNow();

        float s = seconds ?? defaultShowSeconds;
        _hideCo = useTransitions ? StartCoroutine(ShowHideAnimated(s)) : StartCoroutine(HideAfter(s));
        return s;
    }

    IEnumerator ShowHideAnimated(float s)
    {
        if (canvasGroup != null)
            canvasGroup.alpha = 0f;
        if (panelRect != null)
            panelRect.localScale = Vector3.one * Mathf.Max(0.5f, popScaleStart);

        float inDur = Mathf.Max(0.01f, fadeInSeconds);
        float t = 0f;
        while (t < inDur)
        {
            t += Time.unscaledDeltaTime;
            float u = Mathf.Clamp01(t / inDur);
            if (canvasGroup != null) canvasGroup.alpha = u;
            if (panelRect != null)
            {
                float sc = Mathf.Lerp(popScaleStart, popScaleEnd, u);
                panelRect.localScale = Vector3.one * sc;
            }
            yield return null;
        }

        if (float.IsPositiveInfinity(s))
        {
            _hideCo = null;
            yield break;
        }

        float hold = Mathf.Max(0f, s);
        if (hold > 0f)
            yield return new WaitForSeconds(hold);

        float outDur = Mathf.Max(0.01f, fadeOutSeconds);
        t = 0f;
        while (t < outDur)
        {
            t += Time.unscaledDeltaTime;
            float u = Mathf.Clamp01(t / outDur);
            if (canvasGroup != null) canvasGroup.alpha = 1f - u;
            yield return null;
        }

        HideImmediate();
    }

    IEnumerator HideAfter(float s)
    {
        if (float.IsPositiveInfinity(s))
            yield break;

        if (canvasGroup != null) canvasGroup.alpha = 1f;
        if (panelRect != null) panelRect.localScale = Vector3.one * popScaleEnd;
        yield return new WaitForSeconds(Mathf.Max(0f, s));
        HideImmediate();
    }

    private void ApplyStyleForText(string text)
    {
        if (instructionText == null)
            return;

        string trimmed = string.IsNullOrWhiteSpace(text) ? "" : text.Trim();
        bool isCountdown =
            trimmed == "3" ||
            trimmed == "2" ||
            trimmed == "1" ||
            string.Equals(trimmed, "Go", System.StringComparison.OrdinalIgnoreCase);

        instructionText.color = isCountdown ? countdownTextColor : normalTextColor;
        if (overrideTypography)
            instructionText.fontSize = isCountdown ? countdownFontSize : normalFontSize;
        instructionText.fontStyle = isCountdown ? countdownFontStyle : normalFontStyle;
        instructionText.alignment = TextAlignmentOptions.Center;
        instructionText.enableWordWrapping = !isCountdown;
    }
}
