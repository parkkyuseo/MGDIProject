using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Minimal HUD for dwell-confirm progress.
/// Subscribes to ToolPlacementTaskManager.OnConfirmProgress and updates a fill bar.
/// </summary>
public class DwellConfirmHUD : MonoBehaviour
{
    [Header("Source")]
    [SerializeField] private ToolPlacementTaskManager placementTask;

    [Header("UI")]
    [Tooltip("An Image whose Type is Filled.")]
    [SerializeField] private Image fillImage;

    [Tooltip("Optional: background or container that is shown/hidden.")]
    [SerializeField] private GameObject root;

    [Tooltip("Optional: text for status (requires UnityEngine.UI.Text).")]
    [SerializeField] private TMP_Text statusText;

    [Header("Behavior")]
    [Tooltip("If true, the bar is visible only when eligible.")]
    [SerializeField] private bool showOnlyWhenEligible = false;

    [Tooltip("If true, the fill shows remaining (1-t) instead of progress (t).")]
    [SerializeField] private bool showRemainingInsteadOfProgress = false;

    private void OnEnable()
    {
        if (placementTask != null)
            placementTask.OnConfirmProgress += HandleProgress;

        SetVisible(false);
        SetFill(0f);
        SetStatus("");
    }

    private void OnDisable()
    {
        if (placementTask != null)
            placementTask.OnConfirmProgress -= HandleProgress;
    }

    private void HandleProgress(float t01, bool eligible)
    {
        float v = Mathf.Clamp01(t01);
        if (showRemainingInsteadOfProgress)
            v = 1f - v;

        if (showOnlyWhenEligible)
            SetVisible(eligible);
        else
            SetVisible(true);

        SetFill(v);

        if (statusText != null)
            statusText.text = eligible ? "Hold to confirm..." : "Adjust to match target...";
    }

    private void SetVisible(bool on)
    {
        if (root != null) root.SetActive(on);
        else gameObject.SetActive(on);
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
