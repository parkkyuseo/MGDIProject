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

    [Tooltip("If true, hide the HUD when no progress event has arrived recently.")]
    [SerializeField] private bool hideIfNoRecentEvents = true;

    [SerializeField] private float noEventHideSeconds = 0.35f;

    private float _lastEventTime = -999f;
    private bool _lastEligible = false;
    private float _lastT01 = 0f;

    private void Awake()
    {
        if (canvasGroup == null)
            canvasGroup = GetComponentInChildren<CanvasGroup>(true);
    }

    private void OnEnable()
    {
        if (placementTask != null) placementTask.OnConfirmProgress += HandlePlacementProgress;

        // Optional: later when you add confirm for rotation/scaling, subscribe here too.
        // if (rotationTask != null) rotationTask.OnConfirmProgress += HandleRotationProgress;
        // if (scalingTask != null) scalingTask.OnConfirmProgress += HandleScalingProgress;

        SetVisible(false);
        SetFill(0f);
        SetStatus("");
    }

    private void OnDisable()
    {
        if (placementTask != null) placementTask.OnConfirmProgress -= HandlePlacementProgress;
        // if (rotationTask != null) rotationTask.OnConfirmProgress -= HandleRotationProgress;
        // if (scalingTask != null) scalingTask.OnConfirmProgress -= HandleScalingProgress;
    }

    private void Update()
    {
        if (!hideIfNoRecentEvents) return;

        // If no events recently, hide. This naturally hides when moving to Rotation/Scaling.
        if (Time.time - _lastEventTime > noEventHideSeconds)
        {
            SetVisible(false);
        }
        else
        {
            bool show = showOnlyWhenEligible ? _lastEligible : true;
            SetVisible(show);
            SetFill(_lastT01);
        }
    }

    private void HandlePlacementProgress(float t01, bool eligible)
    {
        _lastEventTime = Time.time;
        _lastEligible = eligible;
        _lastT01 = Mathf.Clamp01(t01);

        if (statusText != null)
            statusText.text = eligible ? "Confirming..." : "Align to target...";
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
