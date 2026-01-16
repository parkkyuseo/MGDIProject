using TMPro;
using UnityEngine;

public class TaskContextHUD : MonoBehaviour
{
    [Header("UI References (TextMeshPro)")]
    [SerializeField] private TMP_Text taskLabelText;   // e.g., "Rotation Task"
    [SerializeField] private TMP_Text conditionText;   // e.g., "Macro / NearHead"
    [SerializeField] private TMP_Text trialText;       // e.g., "Trial 3 / 20"

    public void SetTaskLabel(string taskName)
    {
        if (taskLabelText != null)
            taskLabelText.text = taskName ?? "";
    }

    public void SetCondition(string condition)
    {
        if (conditionText != null)
            conditionText.text = condition ?? "";
    }

    public void SetTrial(int current1Based, int total)
    {
        if (trialText != null)
            trialText.text = $"Trial {current1Based} / {total}";
    }
}
