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
    [SerializeField] private TMP_Text stopwatchText;

    [Header("Optional UGUI Fallback")]
    [SerializeField] private Text taskLabelTextUGUI;
    [SerializeField] private Text conditionTextUGUI;
    [SerializeField] private Text trialTextUGUI;
    [SerializeField] private Text practiceTextUGUI;
    [SerializeField] private Text errorTextUGUI;
    [SerializeField] private Text stopwatchTextUGUI;

    [Header("Task Refs (optional auto-find)")]
    [SerializeField] private ToolPlacementTaskManager placementTask;
    [SerializeField] private ToolRotationTaskManager rotationTask;
    [SerializeField] private ToolScalingTaskManager scalingTask;

    [Header("Error Feedback")]
    [SerializeField] private bool showErrorFeedback = false;
    [SerializeField] private float fadeShowRatio = 4.5f;
    [SerializeField] private float fadeHideRatio = 3.0f;
    [SerializeField] private float fadeAlphaMin = 0.0f;
    [SerializeField] private float fadeLerp = 16f;

    [Header("Session Stopwatch")]
    [SerializeField] private bool showSessionStopwatch = false;
    [SerializeField] private string stopwatchPrefix = "Elapsed";
    [SerializeField] private bool stopwatchUseUnscaledTime = true;
    [SerializeField] private bool stopwatchAutoStartOnEnable = true;

    [Header("Debug")]
    [SerializeField] private bool logDebugUpdates = false;

    [Header("Visibility")]
    [SerializeField] private GameObject visibilityRoot;
    [SerializeField] private bool snapPanelToVisiblePoseOnShow = true;
    [SerializeField] private bool useCenteredAnchorsWhenVisible = true;
    [SerializeField] private Vector2 visibleAnchoredPosition = new Vector2(70f, 20f);

    [Header("Professional Look")]
    [SerializeField] private bool useProfessionalLook = true;
    [SerializeField] private bool autoArrangeRows = true;
    [SerializeField] private bool allowRepositioning = false;
    [SerializeField] private bool overridePanelSizing = false;
    [SerializeField] private bool autoFitPanelToVisibleContent = true;
    [SerializeField] private bool overrideTextSizing = false;
    [SerializeField] private RectTransform panelRect;
    [SerializeField] private Image panelBackground;
    [SerializeField] private Vector2 panelSize = new Vector2(660f, 280f);
    [SerializeField] private Vector2 minPanelSize = new Vector2(240f, 120f);
    [SerializeField] private Vector2 maxPanelSize = new Vector2(660f, 280f);
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
    [SerializeField] private float stopwatchFontSize = 20f;
    [SerializeField] private float hintRowHeight = 86f;
    [SerializeField] private float expandedMaxPanelHeightWithHint = 380f;
    [SerializeField] private TextAlignmentOptions tmpLineAlignment = TextAlignmentOptions.Center;
    [SerializeField] private TextAlignmentOptions tmpHintAlignment = TextAlignmentOptions.TopLeft;
    [SerializeField] private TextAnchor uguiLineAlignment = TextAnchor.MiddleCenter;
    [SerializeField] private TextAnchor uguiHintAlignment = TextAnchor.UpperLeft;

    private bool _warnedTaskLabelMissing;
    private bool _warnedConditionMissing;
    private bool _warnedTrialMissing;
    private bool _warnedPracticeMissing;
    private bool _warnedErrorMissing;
    private bool _warnedStopwatchMissing;

    private float _errorAlphaSmooth;
    private bool _errorInitialized;
    private bool _visualInitialized;
    private bool _layoutDirty = true;
    private bool _stopwatchRunning;
    private float _stopwatchStartSec;
    private float _stopwatchAccumulatedSec;

    public void SetVisible(bool visible)
    {
        EnsureVisualRefs();

        GameObject root = visibilityRoot != null ? visibilityRoot : gameObject;
        if (root == null)
            return;

        if (!visible)
        {
            root.SetActive(false);
            return;
        }

        root.SetActive(true);

        if (snapPanelToVisiblePoseOnShow && panelRect != null)
        {
            if (useCenteredAnchorsWhenVisible)
            {
                panelRect.anchorMin = new Vector2(0.5f, 0.5f);
                panelRect.anchorMax = new Vector2(0.5f, 0.5f);
                panelRect.pivot = new Vector2(0.5f, 0.5f);
            }

            panelRect.anchoredPosition = visibleAnchoredPosition;
        }

        _layoutDirty = true;
    }

    private void Awake()
    {
        ResolveTaskRefs();
        EnsureVisualRefs();
        ApplyProfessionalLook();
    }

    private void OnEnable()
    {
        if (stopwatchAutoStartOnEnable)
            StartStopwatch();

        UpdateStopwatchDisplay();

        if (useProfessionalLook)
            ApplyProfessionalLook();
    }

    private void LateUpdate()
    {
        if (useProfessionalLook && (!_visualInitialized || _layoutDirty))
            ApplyProfessionalLook();

        UpdateStopwatchDisplay();
        UpdateErrorFeedback();
    }

    public void Clear()
    {
        SetText(taskLabelText, taskLabelTextUGUI, "", ref _warnedTaskLabelMissing, "Task Label");
        SetHintText("");
        SetText(trialText, trialTextUGUI, "", ref _warnedTrialMissing, "Trial/Tool");
        SetPracticeText("");
        SetErrorText("");
        SetStopwatchText("");
        SetErrorAlpha(0f);
        SetHeaderVisibility(false, false);
        _errorAlphaSmooth = 0f;
        _errorInitialized = false;
        _layoutDirty = true;
    }

    public void SetTaskLabel(string taskName)
    {
        string value = taskName ?? "";
        SetText(taskLabelText, taskLabelTextUGUI, value, ref _warnedTaskLabelMissing, "Task Label");
        if (taskLabelText != null) taskLabelText.gameObject.SetActive(false);
        if (taskLabelTextUGUI != null) taskLabelTextUGUI.gameObject.SetActive(false);
        ApplyTaskAccentColor(value);
        _layoutDirty = true;
        if (logDebugUpdates) Debug.Log("[TaskContextHUD] TaskLabel=" + value);
    }

    public void SetCondition(string condition)
    {
        string value = condition ?? "";
        SetText(conditionText, conditionTextUGUI, value, ref _warnedConditionMissing, "Condition");
        if (conditionText != null) conditionText.gameObject.SetActive(false);
        if (conditionTextUGUI != null) conditionTextUGUI.gameObject.SetActive(false);
        _layoutDirty = true;
        if (logDebugUpdates) Debug.Log("[TaskContextHUD] Condition=" + value);
    }

    public void SetHintText(string text)
    {
        string value = text ?? "";
        SetText(conditionText, conditionTextUGUI, value, ref _warnedConditionMissing, "Hint");
        bool visible = !string.IsNullOrWhiteSpace(value);
        if (conditionText != null) conditionText.gameObject.SetActive(visible);
        if (conditionTextUGUI != null) conditionTextUGUI.gameObject.SetActive(visible);
        _layoutDirty = true;
        if (logDebugUpdates) Debug.Log("[TaskContextHUD] HintText=" + value);
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
        bool visible = !string.IsNullOrWhiteSpace(value);
        if (trialText != null) trialText.gameObject.SetActive(visible);
        if (trialTextUGUI != null) trialTextUGUI.gameObject.SetActive(visible);
        _layoutDirty = true;
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

        _layoutDirty = true;
        if (logDebugUpdates) Debug.Log("[TaskContextHUD] PracticeText=" + value);
    }

    public void SetErrorText(string text)
    {
        string value = text ?? "";
        SetText(errorText, errorTextUGUI, value, ref _warnedErrorMissing, "Error");
        bool visible = !string.IsNullOrWhiteSpace(value);
        if (errorText != null) errorText.gameObject.SetActive(visible);
        if (errorTextUGUI != null) errorTextUGUI.gameObject.SetActive(visible);
        _layoutDirty = true;
        if (logDebugUpdates) Debug.Log("[TaskContextHUD] ErrorText=" + value);
    }

    public void StartStopwatch()
    {
        if (_stopwatchRunning)
            return;

        _stopwatchStartSec = GetNowSec();
        _stopwatchRunning = true;
    }

    public void StopStopwatch()
    {
        if (!_stopwatchRunning)
            return;

        float now = GetNowSec();
        _stopwatchAccumulatedSec += Mathf.Max(0f, now - _stopwatchStartSec);
        _stopwatchRunning = false;
    }

    public void ResetStopwatch(bool keepRunningState = false)
    {
        _stopwatchAccumulatedSec = 0f;
        _stopwatchStartSec = GetNowSec();

        if (!keepRunningState)
            _stopwatchRunning = false;
    }

    private static string FormatMMSS(float seconds)
    {
        if (seconds < 0f) seconds = 0f;
        int s = Mathf.CeilToInt(seconds);
        int m = s / 60;
        int r = s % 60;
        return $"{m:00}:{r:00}";
    }

    private void UpdateStopwatchDisplay()
    {
        if (!showSessionStopwatch)
        {
            SetStopwatchText("");
            if (stopwatchText != null) stopwatchText.gameObject.SetActive(false);
            if (stopwatchTextUGUI != null) stopwatchTextUGUI.gameObject.SetActive(false);
            return;
        }

        if (stopwatchText != null) stopwatchText.gameObject.SetActive(true);
        if (stopwatchTextUGUI != null) stopwatchTextUGUI.gameObject.SetActive(true);

        if (!_stopwatchRunning && stopwatchAutoStartOnEnable)
            StartStopwatch();

        float elapsed = _stopwatchAccumulatedSec;
        if (_stopwatchRunning)
            elapsed += Mathf.Max(0f, GetNowSec() - _stopwatchStartSec);

        string prefix = string.IsNullOrWhiteSpace(stopwatchPrefix) ? "Elapsed" : stopwatchPrefix.Trim();
        SetStopwatchText($"{prefix}: {FormatMMSS(elapsed)}");
    }

    private float GetNowSec()
    {
        return stopwatchUseUnscaledTime ? Time.unscaledTime : Time.time;
    }

    private void SetStopwatchText(string text)
    {
        bool warnIfMissing = showSessionStopwatch;
        bool hasAnyTarget = false;

        if (stopwatchText != null)
        {
            stopwatchText.text = text ?? "";
            hasAnyTarget = true;
        }

        if (stopwatchTextUGUI != null)
        {
            stopwatchTextUGUI.text = text ?? "";
            hasAnyTarget = true;
        }

        if (!hasAnyTarget && warnIfMissing && !_warnedStopwatchMissing)
        {
            _warnedStopwatchMissing = true;
            Debug.LogWarning("[TaskContextHUD] Stopwatch text reference is missing.");
        }

        _layoutDirty = true;
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
        if (visibilityRoot == null)
            visibilityRoot = gameObject;

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

    private void SetHeaderVisibility(bool taskVisible, bool conditionVisible)
    {
        if (taskLabelText != null) taskLabelText.gameObject.SetActive(taskVisible);
        if (taskLabelTextUGUI != null) taskLabelTextUGUI.gameObject.SetActive(taskVisible);
        if (conditionText != null) conditionText.gameObject.SetActive(conditionVisible);
        if (conditionTextUGUI != null) conditionTextUGUI.gameObject.SetActive(conditionVisible);
    }

    private void ApplyProfessionalLook()
    {
        if (!useProfessionalLook)
            return;

        EnsureVisualRefs();

        if (panelBackground != null)
            panelBackground.color = panelBgColor;

        if (panelRect != null && panelRect.gameObject.activeInHierarchy)
            panelRect.anchoredPosition = visibleAnchoredPosition;

        ApplyTmpStyle(taskLabelText, taskFontSize, FontStyles.Bold, taskDefaultColor, tmpLineAlignment, true, overrideTextSizing);
        ApplyTmpStyle(conditionText, lineFontSize, FontStyles.Normal, conditionColor, tmpHintAlignment, false, overrideTextSizing);
        ApplyTmpStyle(trialText, lineFontSize, FontStyles.Bold, trialColor, tmpLineAlignment, false, overrideTextSizing);
        ApplyTmpStyle(practiceText, practiceFontSize, FontStyles.Bold, practiceColor, tmpLineAlignment, false, overrideTextSizing);
        ApplyTmpStyle(errorText, errorFontSize, FontStyles.Bold, errorColor, tmpLineAlignment, false, overrideTextSizing);
        ApplyTmpStyle(stopwatchText, stopwatchFontSize, FontStyles.Normal, conditionColor, tmpLineAlignment, false, overrideTextSizing);

        ApplyUguiStyle(taskLabelTextUGUI, taskDefaultColor, uguiLineAlignment, FontStyle.Bold);
        ApplyUguiStyle(conditionTextUGUI, conditionColor, uguiHintAlignment, FontStyle.Normal);
        ApplyUguiStyle(trialTextUGUI, trialColor, uguiLineAlignment, FontStyle.Bold);
        ApplyUguiStyle(practiceTextUGUI, practiceColor, uguiLineAlignment, FontStyle.Bold);
        ApplyUguiStyle(errorTextUGUI, errorColor, uguiLineAlignment, FontStyle.Bold);
        ApplyUguiStyle(stopwatchTextUGUI, conditionColor, uguiLineAlignment, FontStyle.Normal);

        if (autoArrangeRows)
            ArrangeRows();

        ApplyTaskAccentColor(taskLabelText != null ? taskLabelText.text : (taskLabelTextUGUI != null ? taskLabelTextUGUI.text : ""));
        _visualInitialized = true;
        _layoutDirty = false;
    }

    private void ArrangeRows()
    {
        if (panelRect == null)
            return;

        Vector2 effectivePanelSize = panelSize;
        if (autoFitPanelToVisibleContent)
            effectivePanelSize = ComputePanelSizeForVisibleContent();

        if (overridePanelSizing || autoFitPanelToVisibleContent)
            panelRect.sizeDelta = effectivePanelSize;

        float contentWidth = Mathf.Max(80f, effectivePanelSize.x - panelPadding.x * 2f);
        float top = effectivePanelSize.y * 0.5f - panelPadding.y;
        float x = 0f;
        float yCursor = top;

        yCursor = PlaceVisibleRow(taskLabelText != null ? taskLabelText.gameObject.activeSelf : (taskLabelTextUGUI != null && taskLabelTextUGUI.gameObject.activeSelf),
            taskLabelText != null ? taskLabelText.rectTransform : null,
            taskLabelTextUGUI != null ? taskLabelTextUGUI.rectTransform : null,
            x, yCursor, contentWidth, 64f, rowGap);

        yCursor = PlaceVisibleRow(conditionText != null ? conditionText.gameObject.activeSelf : (conditionTextUGUI != null && conditionTextUGUI.gameObject.activeSelf),
            conditionText != null ? conditionText.rectTransform : null,
            conditionTextUGUI != null ? conditionTextUGUI.rectTransform : null,
            x, yCursor, contentWidth, GetEffectiveHintRowHeight(), secondaryRowGap);

        yCursor = PlaceVisibleRow(trialText != null ? trialText.gameObject.activeSelf : (trialTextUGUI != null && trialTextUGUI.gameObject.activeSelf),
            trialText != null ? trialText.rectTransform : null,
            trialTextUGUI != null ? trialTextUGUI.rectTransform : null,
            x, yCursor, contentWidth, 52f, secondaryRowGap);

        yCursor = PlaceVisibleRow(practiceText != null ? practiceText.gameObject.activeSelf : (practiceTextUGUI != null && practiceTextUGUI.gameObject.activeSelf),
            practiceText != null ? practiceText.rectTransform : null,
            practiceTextUGUI != null ? practiceTextUGUI.rectTransform : null,
            x, yCursor, contentWidth, 42f, secondaryRowGap);

        yCursor = PlaceVisibleRow(errorText != null ? errorText.gameObject.activeSelf : (errorTextUGUI != null && errorTextUGUI.gameObject.activeSelf),
            errorText != null ? errorText.rectTransform : null,
            errorTextUGUI != null ? errorTextUGUI.rectTransform : null,
            x, yCursor, contentWidth, 40f, secondaryRowGap);

        PlaceVisibleRow(stopwatchText != null ? stopwatchText.gameObject.activeSelf : (stopwatchTextUGUI != null && stopwatchTextUGUI.gameObject.activeSelf),
            stopwatchText != null ? stopwatchText.rectTransform : null,
            stopwatchTextUGUI != null ? stopwatchTextUGUI.rectTransform : null,
            x, yCursor, contentWidth, 40f, secondaryRowGap);
    }

    private Vector2 ComputePanelSizeForVisibleContent()
    {
        float maxLineWidth = 0f;
        float totalHeight = panelPadding.y * 2f;
        int visibleRowCount = 0;

        AccumulateRow(taskLabelText, taskLabelTextUGUI, 64f, ref maxLineWidth, ref totalHeight, ref visibleRowCount, rowGap);
        AccumulateRow(conditionText, conditionTextUGUI, GetEffectiveHintRowHeight(), ref maxLineWidth, ref totalHeight, ref visibleRowCount, secondaryRowGap);
        AccumulateRow(trialText, trialTextUGUI, 52f, ref maxLineWidth, ref totalHeight, ref visibleRowCount, secondaryRowGap);
        AccumulateRow(practiceText, practiceTextUGUI, 42f, ref maxLineWidth, ref totalHeight, ref visibleRowCount, secondaryRowGap);
        AccumulateRow(errorText, errorTextUGUI, 40f, ref maxLineWidth, ref totalHeight, ref visibleRowCount, secondaryRowGap);
        AccumulateRow(stopwatchText, stopwatchTextUGUI, 40f, ref maxLineWidth, ref totalHeight, ref visibleRowCount, secondaryRowGap);

        if (visibleRowCount == 0)
            return panelSize;

        Vector2 effectiveMaxPanelSize = maxPanelSize;
        if (IsHintVisible())
            effectiveMaxPanelSize.y = Mathf.Max(effectiveMaxPanelSize.y, expandedMaxPanelHeightWithHint);

        float width = Mathf.Clamp(maxLineWidth + panelPadding.x * 2f, minPanelSize.x, effectiveMaxPanelSize.x);
        float height = Mathf.Clamp(totalHeight, minPanelSize.y, effectiveMaxPanelSize.y);
        return new Vector2(width, height);
    }

    private bool IsHintVisible()
    {
        return (conditionText != null && conditionText.gameObject.activeSelf && !string.IsNullOrWhiteSpace(conditionText.text)) ||
               (conditionTextUGUI != null && conditionTextUGUI.gameObject.activeSelf && !string.IsNullOrWhiteSpace(conditionTextUGUI.text));
    }

    private float GetEffectiveHintRowHeight()
    {
        return Mathf.Max(hintRowHeight, 106f);
    }

    private void AccumulateRow(TMP_Text tmp, Text ugui, float rowHeight, ref float maxLineWidth, ref float totalHeight, ref int visibleRowCount, float gapAfter)
    {
        bool visible = (tmp != null && tmp.gameObject.activeSelf && !string.IsNullOrWhiteSpace(tmp.text)) ||
                       (ugui != null && ugui.gameObject.activeSelf && !string.IsNullOrWhiteSpace(ugui.text));
        if (!visible)
            return;

        visibleRowCount++;
        float preferredWidth = 0f;
        if (tmp != null && tmp.gameObject.activeSelf && !string.IsNullOrWhiteSpace(tmp.text))
            preferredWidth = Mathf.Max(preferredWidth, tmp.GetPreferredValues(tmp.text).x);
        if (ugui != null && ugui.gameObject.activeSelf && !string.IsNullOrWhiteSpace(ugui.text))
            preferredWidth = Mathf.Max(preferredWidth, ugui.preferredWidth);

        maxLineWidth = Mathf.Max(maxLineWidth, preferredWidth);
        totalHeight += rowHeight;
        totalHeight += gapAfter;
    }

    private static float PlaceVisibleRow(bool visible, RectTransform tmpRect, RectTransform uguiRect, float x, float yCursor, float width, float height, float gapAfter)
    {
        if (!visible)
            return yCursor;

        float y = yCursor - (height * 0.5f);
        PlaceRect(tmpRect, x, y, width, height);
        PlaceRect(uguiRect, x, y, width, height);
        return y - (height * 0.5f) - gapAfter;
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
        text.richText = true;
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
        text.supportRichText = true;
    }

    private static bool IsFinite(float v)
    {
        return !float.IsNaN(v) && !float.IsInfinity(v);
    }
}
