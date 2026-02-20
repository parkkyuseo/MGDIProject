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

    private bool _warnedTaskLabelMissing;
    private bool _warnedConditionMissing;
    private bool _warnedTrialMissing;
    private bool _warnedPracticeMissing;
    private bool _warnedErrorMissing;

    private float _errorAlphaSmooth;
    private bool _errorInitialized;

    public void SetVisible(bool visible) => gameObject.SetActive(visible);

    private void Awake()
    {
        ResolveTaskRefs();
    }

    private void LateUpdate()
    {
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
        SetText(practiceText, practiceTextUGUI, value, ref _warnedPracticeMissing, "Practice");

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

    private static bool IsFinite(float v)
    {
        return !float.IsNaN(v) && !float.IsInfinity(v);
    }
}
