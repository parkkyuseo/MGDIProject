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
    [SerializeField] private float noEventHideSeconds = 0.35f;
    [SerializeField] private bool showRemainingInsteadOfProgress = false;

    private float _lastEventTime = -999f;
    private bool _lastEligible = false;
    private float _lastT01 = 0f;
    private string _lastStatus = "";

    private void Awake()
    {
        if (canvasGroup == null)
            canvasGroup = GetComponentInChildren<CanvasGroup>(true);
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
        if (Time.time - _lastEventTime > noEventHideSeconds)
        {
            SetVisible(false);
            return;
        }

        bool show = showOnlyWhenEligible ? _lastEligible : true;
        SetVisible(show);

        float fill = showRemainingInsteadOfProgress ? (1f - _lastT01) : _lastT01;
        SetFill(fill);
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
        if (canvasGroup == null) return;
        canvasGroup.alpha = on ? 1f : 0f;
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
}
