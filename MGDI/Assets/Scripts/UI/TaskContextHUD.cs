using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class TaskContextHUD : MonoBehaviour
{
    [SerializeField] private TMP_Text taskLabelText;
    [SerializeField] private TMP_Text conditionText;
    [SerializeField] private TMP_Text trialText;
    [SerializeField] private TMP_Text practiceText;
    [SerializeField] private TMP_Text errorText;

    [Header("Optional UGUI Fallback")]
    [SerializeField] private Text taskLabelTextUGUI;
    [SerializeField] private Text conditionTextUGUI;
    [SerializeField] private Text trialTextUGUI;
    [SerializeField] private Text practiceTextUGUI;
    [SerializeField] private Text errorTextUGUI;

    [Header("Task Refs (optional auto-find)")]
    [SerializeField] private ToolPlacementTaskManager placementTask;
    [SerializeField] private ToolRotationTaskManager rotationTask;
    [SerializeField] private ToolScalingTaskManager scalingTask;

    [Header("Error Feedback")]
    [SerializeField] private bool showErrorFeedback = true;
    [SerializeField] private float fadeShowRatio = 4.5f;
    [SerializeField] private float fadeHideRatio = 3.0f;
    [SerializeField] private float fadeAlphaMin = 0.0f;
    [SerializeField] private float fadeLerp = 16f;

    [Header("Debug")]
    [SerializeField] private bool logDebugUpdates = false;

    [Header("Professional Look")]
    [SerializeField] private bool useProfessionalLook = true;
    [SerializeField] private bool autoArrangeRows = true;
    [SerializeField] private bool allowRepositioning = false;
    [SerializeField] private bool overridePanelSizing = false;
    [SerializeField] private bool overrideTextSizing = false;
    [SerializeField] private RectTransform panelRect;
    [SerializeField] private Image panelBackground;
    [SerializeField] private Vector2 panelSize = new Vector2(660f, 280f);
    [SerializeField] private Vector2 panelPadding = new Vector2(30f, 22f);
    [SerializeField] private float rowGap = 46f;
    [SerializeField] private float secondaryRowGap = 40f;
    [SerializeField] private Color panelBgColor = new Color(0.06f, 0.11f, 0.17f, 0.86f);
    [SerializeField] private Color taskPlacementColor = new Color(0.46f, 0.87f, 0.62f, 1f);
    [SerializeField] private Color taskRotationColor = new Color(0.44f, 0.72f, 1f, 1f);
    [SerializeField] private Color taskScalingColor = new Color(1f, 0.73f, 0.45f, 1f);
    [SerializeField] private Color taskDefaultColor = new Color(0.85f, 0.93f, 1f, 1f);
    [SerializeField] private Color conditionColor = new Color(0.76f, 0.83f, 0.91f, 0.95f);
    [SerializeField] private Color trialColor = new Color(0.94f, 0.97f, 1f, 1f);
    [SerializeField] private Color practiceColor = new Color(1f, 0.86f, 0.50f, 1f);
    [SerializeField] private Color errorColor = new Color(1f, 0.72f, 0.57f, 1f);
    [SerializeField] private float taskFontSize = 34f;
    [SerializeField] private float lineFontSize = 24f;
    [SerializeField] private float practiceFontSize = 22f;
    [SerializeField] private float errorFontSize = 20f;
    [SerializeField] private TextAlignmentOptions tmpLineAlignment = TextAlignmentOptions.Left;
    [SerializeField] private TextAnchor uguiLineAlignment = TextAnchor.MiddleLeft;

    private bool _warnedTaskLabelMissing;
    private bool _warnedConditionMissing;
    private bool _warnedTrialMissing;
    private bool _warnedPracticeMissing;
    private bool _warnedErrorMissing;

    private float _errorAlphaSmooth;
    private bool _errorInitialized;
    private bool _visualInitialized;

    public void SetVisible(bool visible) => gameObject.SetActive(visible);

    private void Awake()
    {
        ResolveTaskRefs();
        EnsureVisualRefs();
        ApplyProfessionalLook();
    }

    private void OnEnable()
    {
        if (useProfessionalLook)
            ApplyProfessionalLook();
    }

    private void LateUpdate()
    {
        if (useProfessionalLook && !_visualInitialized)
            ApplyProfessionalLook();
        UpdateErrorFeedback();
    }

    public void Clear()
    {
        SetText(taskLabelText, taskLabelTextUGUI, "", ref _warnedTaskLabelMissing, "Task Label");
        SetText(conditionText, conditionTextUGUI, "", ref _warnedConditionMissing, "Condition");
        SetText(trialText, trialTextUGUI, "", ref _warnedTrialMissing, "Trial/Tool");
        SetPracticeText("");
        SetErrorText("");
        SetErrorAlpha(0f);
        _errorAlphaSmooth = 0f;
        _errorInitialized = false;
    }

    public void SetTaskLabel(string taskName)
    {
        string value = taskName ?? "";
        SetText(taskLabelText, taskLabelTextUGUI, value, ref _warnedTaskLabelMissing, "Task Label");
        ApplyTaskAccentColor(value);
        if (logDebugUpdates) Debug.Log("[TaskContextHUD] TaskLabel=" + value);
    }

    public void SetCondition(string condition)
    {
        string value = condition ?? "";
        SetText(conditionText, conditionTextUGUI, value, ref _warnedConditionMissing, "Condition");
        if (logDebugUpdates) Debug.Log("[TaskContextHUD] Condition=" + value);
    }

    public void SetTrial(int current1Based, int total)
    {
        SetTrialText($"Trial {current1Based} / {total}");
    }

    public void SetTrialWithCountdown(int current1Based, int total, float remainingSec)
    {
        string mmss = FormatMMSS(remainingSec);
        SetTrialText($"Trial {current1Based} / {total} - {mmss}");
    }

    public void SetTrialText(string text)
    {
        string value = text ?? "";
        SetText(trialText, trialTextUGUI, value, ref _warnedTrialMissing, "Trial/Tool");
        if (logDebugUpdates) Debug.Log("[TaskContextHUD] TrialText=" + value);
    }

    public void SetPracticeText(string text)
    {
        string value = text ?? "";
        bool visible = !string.IsNullOrWhiteSpace(value);
        string display = visible ? value.ToUpperInvariant() : value;
        SetText(practiceText, practiceTextUGUI, display, ref _warnedPracticeMissing, "Practice");

        if (practiceText != null) practiceText.gameObject.SetActive(visible);
        if (practiceTextUGUI != null) practiceTextUGUI.gameObject.SetActive(visible);

        if (logDebugUpdates) Debug.Log("[TaskContextHUD] PracticeText=" + value);
    }

    public void SetErrorText(string text)
    {
        string value = text ?? "";
        SetText(errorText, errorTextUGUI, value, ref _warnedErrorMissing, "Error");
        if (logDebugUpdates) Debug.Log("[TaskContextHUD] ErrorText=" + value);
    }

    private static string FormatMMSS(float seconds)
    {
        if (seconds < 0f) seconds = 0f;
        int s = Mathf.CeilToInt(seconds);
        int m = s / 60;
        int r = s % 60;
        return $"{m:00}:{r:00}";
    }

    private void SetText(TMP_Text tmp, Text ugui, string value, ref bool warnedMissing, string fieldLabel)
    {
        bool hasAnyTarget = false;

        if (tmp != null)
        {
            tmp.text = value;
            hasAnyTarget = true;
        }

        if (ugui != null)
        {
            ugui.text = value;
            hasAnyTarget = true;
        }

        if (!hasAnyTarget && !warnedMissing)
        {
            warnedMissing = true;
            Debug.LogWarning($"[TaskContextHUD] {fieldLabel} text reference is missing.");
        }
    }

    private void UpdateErrorFeedback()
    {
        if (!showErrorFeedback)
        {
            HideError(0f);
            return;
        }

        ResolveTaskRefs();

        bool hasTrial = TryGetActiveError(out string message, out float ratio);
        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (!hasTrial)
        {
            HideError(dt);
            return;
        }

        float targetAlpha = ComputeTargetAlpha(ratio);
        float t = 1f - Mathf.Exp(-Mathf.Max(0f, fadeLerp) * dt);

        if (!_errorInitialized)
        {
            _errorAlphaSmooth = targetAlpha;
            _errorInitialized = true;
        }
        else
        {
            _errorAlphaSmooth = Mathf.Lerp(_errorAlphaSmooth, targetAlpha, t);
        }

        SetErrorText(message);
        SetErrorAlpha(_errorAlphaSmooth);
    }

    private bool TryGetActiveError(out string message, out float ratio)
    {
        const float eps = 1e-5f;
        message = "";
        ratio = 0f;

        if (placementTask != null && placementTask.IsTrialRunning)
        {
            float errMeters = placementTask.GetActiveErrorMeters();
            float tolMeters = placementTask.ActiveToleranceMeters;
            if (!IsFinite(errMeters) || !IsFinite(tolMeters) || tolMeters <= eps) return false;

            float errCm = errMeters * 100f;
            message = $"Error: {errCm:F1} cm";
            ratio = errMeters / Mathf.Max(eps, tolMeters);
            return true;
        }

        if (rotationTask != null && rotationTask.IsTrialRunning)
        {
            float errDeg = rotationTask.ActiveRotationErrorDeg;
            float tolDeg = rotationTask.RotationToleranceDeg;
            if (!IsFinite(errDeg) || !IsFinite(tolDeg) || tolDeg <= eps) return false;

            message = $"Error: {errDeg:F1}\u00b0";
            ratio = errDeg / Mathf.Max(eps, tolDeg);
            return true;
        }

        if (scalingTask != null && scalingTask.IsTrialRunning)
        {
            float errFactor = scalingTask.ActiveScalingErrorFactor;
            float tolFactor = scalingTask.ScaleFactorTolerance;
            if (!IsFinite(errFactor) || !IsFinite(tolFactor) || tolFactor <= eps) return false;

            float errPct = errFactor * 100f;
            message = $"Error: {errPct:F1}%";
            ratio = errFactor / Mathf.Max(eps, tolFactor);
            return true;
        }

        return false;
    }

    private float ComputeTargetAlpha(float ratio)
    {
        float hide = Mathf.Max(3.0f, fadeHideRatio);
        float show = Mathf.Max(fadeShowRatio, hide + 0.01f);
        if (show < hide)
        {
            float tmp = show;
            show = hide;
            hide = tmp;
        }

        if (ratio <= hide) return 0f;
        if (ratio >= show) return 1f;
        if (Mathf.Abs(show - hide) < 1e-5f) return 1f;

        float t = (ratio - hide) / (show - hide);
        t = Mathf.Clamp01(t);
        float minA = Mathf.Clamp01(fadeAlphaMin);
        return Mathf.Lerp(minA, 1f, t);
    }

    private void HideError(float dt)
    {
        float t = 1f - Mathf.Exp(-Mathf.Max(0f, fadeLerp) * dt);
        if (!_errorInitialized)
        {
            _errorAlphaSmooth = 0f;
            _errorInitialized = true;
        }
        else
        {
            _errorAlphaSmooth = Mathf.Lerp(_errorAlphaSmooth, 0f, t);
        }

        SetErrorAlpha(_errorAlphaSmooth);
        if (_errorAlphaSmooth <= 0.001f)
            SetErrorText("");
    }

    private void SetErrorAlpha(float alpha)
    {
        float a = Mathf.Clamp01(alpha);

        if (errorText != null)
        {
            Color c = errorText.color;
            c.a = a;
            errorText.color = c;
        }

        if (errorTextUGUI != null)
        {
            Color c = errorTextUGUI.color;
            c.a = a;
            errorTextUGUI.color = c;
        }
    }

    private void ResolveTaskRefs()
    {
        if (placementTask == null) placementTask = FindFirstObjectByType<ToolPlacementTaskManager>();
        if (rotationTask == null) rotationTask = FindFirstObjectByType<ToolRotationTaskManager>();
        if (scalingTask == null) scalingTask = FindFirstObjectByType<ToolScalingTaskManager>();
    }

    private void EnsureVisualRefs()
    {
        if (panelRect == null)
            panelRect = transform as RectTransform;

        if (panelBackground == null)
        {
            Image[] images = GetComponentsInChildren<Image>(true);
            for (int i = 0; i < images.Length; i++)
            {
                Image img = images[i];
                if (img == null) continue;
                string n = img.name.ToLowerInvariant();
                if (n.Contains("bg") || n.Contains("background") || n.Contains("panel"))
                {
                    panelBackground = img;
                    break;
                }
            }

            if (panelBackground == null && images.Length > 0)
                panelBackground = images[0];
        }
    }

    private void ApplyProfessionalLook()
    {
        if (!useProfessionalLook)
            return;

        EnsureVisualRefs();

        if (panelBackground != null)
            panelBackground.color = panelBgColor;

        ApplyTmpStyle(taskLabelText, taskFontSize, FontStyles.Bold, taskDefaultColor, tmpLineAlignment, true, overrideTextSizing);
        ApplyTmpStyle(conditionText, lineFontSize, FontStyles.Normal, conditionColor, tmpLineAlignment, false, overrideTextSizing);
        ApplyTmpStyle(trialText, lineFontSize, FontStyles.Bold, trialColor, tmpLineAlignment, false, overrideTextSizing);
        ApplyTmpStyle(practiceText, practiceFontSize, FontStyles.Bold, practiceColor, tmpLineAlignment, false, overrideTextSizing);
        ApplyTmpStyle(errorText, errorFontSize, FontStyles.Bold, errorColor, tmpLineAlignment, false, overrideTextSizing);

        ApplyUguiStyle(taskLabelTextUGUI, taskDefaultColor, uguiLineAlignment, FontStyle.Bold);
        ApplyUguiStyle(conditionTextUGUI, conditionColor, uguiLineAlignment, FontStyle.Normal);
        ApplyUguiStyle(trialTextUGUI, trialColor, uguiLineAlignment, FontStyle.Bold);
        ApplyUguiStyle(practiceTextUGUI, practiceColor, uguiLineAlignment, FontStyle.Bold);
        ApplyUguiStyle(errorTextUGUI, errorColor, uguiLineAlignment, FontStyle.Bold);

        if (autoArrangeRows && allowRepositioning)
            ArrangeRows();

        ApplyTaskAccentColor(taskLabelText != null ? taskLabelText.text : (taskLabelTextUGUI != null ? taskLabelTextUGUI.text : ""));
        _visualInitialized = true;
    }

    private void ArrangeRows()
    {
        if (panelRect == null)
            return;

        if (overridePanelSizing)
            panelRect.sizeDelta = panelSize;

        float contentWidth = Mathf.Max(80f, panelSize.x - panelPadding.x * 2f);
        float top = panelSize.y * 0.5f - panelPadding.y;
        float x = 0f;

        PlaceRect(taskLabelText != null ? taskLabelText.rectTransform : null, x, top - 32f, contentWidth, 64f);
        PlaceRect(taskLabelTextUGUI != null ? taskLabelTextUGUI.rectTransform : null, x, top - 32f, contentWidth, 64f);

        float y2 = top - 32f - rowGap;
        PlaceRect(conditionText != null ? conditionText.rectTransform : null, x, y2, contentWidth, 46f);
        PlaceRect(conditionTextUGUI != null ? conditionTextUGUI.rectTransform : null, x, y2, contentWidth, 46f);

        float y3 = y2 - secondaryRowGap;
        PlaceRect(trialText != null ? trialText.rectTransform : null, x, y3, contentWidth, 44f);
        PlaceRect(trialTextUGUI != null ? trialTextUGUI.rectTransform : null, x, y3, contentWidth, 44f);

        float y4 = y3 - secondaryRowGap;
        PlaceRect(practiceText != null ? practiceText.rectTransform : null, x, y4, contentWidth, 42f);
        PlaceRect(practiceTextUGUI != null ? practiceTextUGUI.rectTransform : null, x, y4, contentWidth, 42f);

        float y5 = y4 - secondaryRowGap;
        PlaceRect(errorText != null ? errorText.rectTransform : null, x, y5, contentWidth, 40f);
        PlaceRect(errorTextUGUI != null ? errorTextUGUI.rectTransform : null, x, y5, contentWidth, 40f);
    }

    private void ApplyTaskAccentColor(string taskLabel)
    {
        if (!useProfessionalLook)
            return;

        Color accent = taskDefaultColor;
        if (!string.IsNullOrWhiteSpace(taskLabel))
        {
            if (taskLabel.IndexOf("placement", System.StringComparison.OrdinalIgnoreCase) >= 0)
                accent = taskPlacementColor;
            else if (taskLabel.IndexOf("rotation", System.StringComparison.OrdinalIgnoreCase) >= 0)
                accent = taskRotationColor;
            else if (taskLabel.IndexOf("scaling", System.StringComparison.OrdinalIgnoreCase) >= 0)
                accent = taskScalingColor;
        }

        if (taskLabelText != null) taskLabelText.color = accent;
        if (taskLabelTextUGUI != null) taskLabelTextUGUI.color = accent;

        if (panelBackground != null)
        {
            Color bg = panelBgColor;
            bg.r = Mathf.Lerp(bg.r, accent.r, 0.08f);
            bg.g = Mathf.Lerp(bg.g, accent.g, 0.08f);
            bg.b = Mathf.Lerp(bg.b, accent.b, 0.08f);
            panelBackground.color = bg;
        }
    }

    private static void PlaceRect(RectTransform rect, float x, float y, float w, float h)
    {
        if (rect == null)
            return;

        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = new Vector2(x, y);
        rect.sizeDelta = new Vector2(w, h);
    }

    private static void ApplyTmpStyle(
        TMP_Text text,
        float fontSize,
        FontStyles style,
        Color color,
        TextAlignmentOptions alignment,
        bool uppercase,
        bool applyFontSize)
    {
        if (text == null)
            return;

        if (applyFontSize)
            text.fontSize = fontSize;
        text.fontStyle = style;
        text.color = color;
        text.alignment = alignment;
        text.enableWordWrapping = true;
        if (uppercase && !string.IsNullOrEmpty(text.text))
            text.text = text.text.ToUpperInvariant();
    }

    private static void ApplyUguiStyle(Text text, Color color, TextAnchor alignment, FontStyle style)
    {
        if (text == null)
            return;

        text.color = color;
        text.alignment = alignment;
        text.fontStyle = style;
    }

    private static bool IsFinite(float v)
    {
        return !float.IsNaN(v) && !float.IsInfinity(v);
    }
}
