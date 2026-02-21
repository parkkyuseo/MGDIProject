using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class DwellConfirmHUD : MonoBehaviour
{
    [Header("Sources (assign what you use)")]
    [SerializeField] private ToolPlacementTaskManager placementTask;
    [SerializeField] private ToolRotationTaskManager rotationTask;   // optional
    [SerializeField] private ToolScalingTaskManager scalingTask;     // optional

    [Header("UI")]
    [SerializeField] private Image fillImage;
    [SerializeField] private TMP_Text statusText;
    [SerializeField] private CanvasGroup canvasGroup;

    [Header("Behavior")]
    [Tooltip("If true, show only when eligible. If false, show whenever a task is running and emitting progress.")]
    [SerializeField] private bool showOnlyWhenEligible = true;
    [Tooltip("If true, status text can keep HUD visible even when eligibility is false.")]
    [SerializeField] private bool showStatusWhenIneligible = true;
    [SerializeField] private float noEventHideSeconds = 0.35f;
    [SerializeField] private bool showRemainingInsteadOfProgress = false;

    [Header("Professional Look")]
    [SerializeField] private bool useProfessionalLook = true;
    [SerializeField] private RectTransform rootRect;
    [SerializeField] private float fillLerp = 20f;
    [SerializeField] private float visibilityLerp = 14f;
    [SerializeField] private float colorLerp = 14f;
    [SerializeField] private Color barHiddenColor = new Color(0.35f, 0.45f, 0.58f, 0.20f);
    [SerializeField] private Color barAlignColor = new Color(0.40f, 0.62f, 0.95f, 0.95f);
    [SerializeField] private Color barReleaseColor = new Color(0.96f, 0.76f, 0.38f, 0.98f);
    [SerializeField] private Color barConfirmingColor = new Color(0.44f, 0.90f, 0.62f, 1f);
    [SerializeField] private Color statusDefaultColor = new Color(0.94f, 0.97f, 1f, 1f);
    [SerializeField] private Color statusConfirmingColor = new Color(0.76f, 1f, 0.84f, 1f);
    [SerializeField] private bool pulseWhenConfirming = true;
    [SerializeField] private float pulseScale = 1.04f;
    [SerializeField] private float pulseSpeed = 2.4f;

    private float _lastEventTime = -999f;
    private bool _lastEligible = false;
    private float _lastT01 = 0f;
    private string _lastStatus = "";

    private float _fillSmooth;
    private float _visibleSmooth;
    private bool _hasInitSmoothing;
    private Vector3 _baseScale = Vector3.one;

    private void Awake()
    {
        if (canvasGroup == null)
            canvasGroup = GetComponentInChildren<CanvasGroup>(true);
        if (rootRect == null)
            rootRect = transform as RectTransform;
        if (rootRect != null)
            _baseScale = rootRect.localScale;
    }

    private void OnEnable()
    {
        if (placementTask != null) placementTask.OnConfirmProgress += HandlePlacementProgress;
        if (placementTask != null) placementTask.OnConfirmStatus += HandlePlacementStatus;
        if (rotationTask != null) rotationTask.OnConfirmProgress += HandleRotationProgress;
        if (rotationTask != null) rotationTask.OnConfirmStatus += HandleRotationStatus;
        if (scalingTask != null) scalingTask.OnConfirmProgress += HandleScalingProgress;
        if (scalingTask != null) scalingTask.OnConfirmStatus += HandleScalingStatus;

        SetVisible(false);
        SetFill(0f);
        SetStatus("");
        _lastStatus = "";
        _fillSmooth = 0f;
        _visibleSmooth = 0f;
        _hasInitSmoothing = false;
        if (rootRect != null)
            rootRect.localScale = _baseScale;
    }

    private void OnDisable()
    {
        if (placementTask != null) placementTask.OnConfirmProgress -= HandlePlacementProgress;
        if (placementTask != null) placementTask.OnConfirmStatus -= HandlePlacementStatus;
        if (rotationTask != null) rotationTask.OnConfirmProgress -= HandleRotationProgress;
        if (rotationTask != null) rotationTask.OnConfirmStatus -= HandleRotationStatus;
        if (scalingTask != null) scalingTask.OnConfirmProgress -= HandleScalingProgress;
        if (scalingTask != null) scalingTask.OnConfirmStatus -= HandleScalingStatus;
    }

    private void Update()
    {
        bool hasRecentEvent = Time.time - _lastEventTime <= noEventHideSeconds;
        bool hasStatus = !string.IsNullOrWhiteSpace(_lastStatus);
        bool statusKeepsVisible = showStatusWhenIneligible && hasStatus;
        bool targetShow = hasRecentEvent && (showOnlyWhenEligible ? (_lastEligible || statusKeepsVisible) : true);
        float targetFill = showRemainingInsteadOfProgress ? (1f - _lastT01) : _lastT01;

        if (!useProfessionalLook)
        {
            SetVisible(targetShow);
            SetFill(targetFill);
            return;
        }

        float dt = Mathf.Max(Time.unscaledDeltaTime, 1e-4f);
        float tFill = 1f - Mathf.Exp(-Mathf.Max(0f, fillLerp) * dt);
        float tVis = 1f - Mathf.Exp(-Mathf.Max(0f, visibilityLerp) * dt);
        float tColor = 1f - Mathf.Exp(-Mathf.Max(0f, colorLerp) * dt);

        if (!_hasInitSmoothing)
        {
            _fillSmooth = targetFill;
            _visibleSmooth = targetShow ? 1f : 0f;
            _hasInitSmoothing = true;
        }
        else
        {
            _fillSmooth = Mathf.Lerp(_fillSmooth, targetFill, tFill);
            _visibleSmooth = Mathf.Lerp(_visibleSmooth, targetShow ? 1f : 0f, tVis);
        }

        SetVisibleAlpha(_visibleSmooth);
        SetFill(_fillSmooth);
        UpdateBarAndStatusColor(targetShow, tColor);
        UpdatePulse(targetShow);
    }

    private void HandlePlacementProgress(float t01, bool eligible)
    {
        HandleProgressCommon(t01, eligible);
    }

    private void HandleRotationProgress(float t01, bool eligible)
    {
        HandleProgressCommon(t01, eligible);
    }

    private void HandleScalingProgress(float t01, bool eligible)
    {
        HandleProgressCommon(t01, eligible);
    }

    private void HandlePlacementStatus(string status)
    {
        HandleStatusCommon(status);
    }

    private void HandleRotationStatus(string status)
    {
        HandleStatusCommon(status);
    }

    private void HandleScalingStatus(string status)
    {
        HandleStatusCommon(status);
    }

    private void HandleProgressCommon(float t01, bool eligible)
    {
        _lastEventTime = Time.time;
        _lastEligible = eligible;
        _lastT01 = Mathf.Clamp01(t01);

        float fill = showRemainingInsteadOfProgress ? (1f - _lastT01) : _lastT01;
        SetFill(fill);

        bool show = showOnlyWhenEligible ? eligible : true;
        SetVisible(show);
    }

    private void HandleStatusCommon(string status)
    {
        _lastStatus = status ?? "";
        SetStatus(_lastStatus);
    }

    private void SetVisible(bool on)
    {
        SetVisibleAlpha(on ? 1f : 0f);
    }

    private void SetVisibleAlpha(float alpha)
    {
        if (canvasGroup == null) return;
        float a = Mathf.Clamp01(alpha);
        canvasGroup.alpha = a;
        bool on = a > 0.01f;
        canvasGroup.interactable = on;
        canvasGroup.blocksRaycasts = on;
    }

    private void SetFill(float v01)
    {
        if (fillImage == null) return;
        fillImage.fillAmount = Mathf.Clamp01(v01);
    }

    private void SetStatus(string s)
    {
        if (statusText == null) return;
        statusText.text = s;
    }

    private void UpdateBarAndStatusColor(bool show, float tColor)
    {
        Color targetBar = barHiddenColor;
        Color targetStatus = statusDefaultColor;

        if (show)
        {
            if (ContainsWord(_lastStatus, "confirming"))
            {
                targetBar = barConfirmingColor;
                targetStatus = statusConfirmingColor;
            }
            else if (ContainsWord(_lastStatus, "release"))
            {
                targetBar = barReleaseColor;
            }
            else
            {
                targetBar = barAlignColor;
            }
        }

        if (fillImage != null)
            fillImage.color = Color.Lerp(fillImage.color, targetBar, tColor);

        if (statusText != null)
            statusText.color = Color.Lerp(statusText.color, targetStatus, tColor);
    }

    private void UpdatePulse(bool show)
    {
        if (rootRect == null)
            return;

        if (!pulseWhenConfirming || !show || !ContainsWord(_lastStatus, "confirming"))
        {
            rootRect.localScale = Vector3.Lerp(rootRect.localScale, _baseScale, 1f - Mathf.Exp(-12f * Mathf.Max(Time.unscaledDeltaTime, 1e-4f)));
            return;
        }

        float amp = Mathf.Max(1f, pulseScale);
        float phase = Mathf.Sin(Time.unscaledTime * Mathf.Max(0.1f, pulseSpeed) * Mathf.PI * 2f);
        float sc = Mathf.Lerp(1f, amp, (phase * 0.5f) + 0.5f);
        rootRect.localScale = _baseScale * sc;
    }

    private static bool ContainsWord(string source, string token)
    {
        if (string.IsNullOrEmpty(source) || string.IsNullOrEmpty(token))
            return false;
        return source.IndexOf(token, System.StringComparison.OrdinalIgnoreCase) >= 0;
    }
}
