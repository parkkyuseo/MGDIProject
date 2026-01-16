using TMPro;
using UnityEngine;

public class TaskContextHUD : MonoBehaviour
{
    [Header("UI References (TextMeshPro)")]
    [SerializeField] private TMP_Text taskLabelText;
    [SerializeField] private TMP_Text conditionText;
    [SerializeField] private TMP_Text trialText;

    public void SetVisible(bool visible)
    {
        gameObject.SetActive(visible);
    }

    public void Clear()
    {
        if (taskLabelText != null) taskLabelText.text = "";
        if (conditionText != null) conditionText.text = "";
        if (trialText != null) trialText.text = "";
    }

    public void SetTaskLabel(string taskName)
    {
        if (taskLabelText != null) taskLabelText.text = taskName ?? "";
    }

    public void SetCondition(string condition)
    {
        if (conditionText != null) conditionText.text = condition ?? "";
    }

    public void SetTrial(int current1Based, int total)
    {
        if (trialText != null) trialText.text = $"Trial {current1Based} / {total}";
    }
}
