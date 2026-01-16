using TMPro;
using UnityEngine;

public class TaskContextHUD : MonoBehaviour
{
    [SerializeField] private TMP_Text taskLabelText;
    [SerializeField] private TMP_Text conditionText;
    [SerializeField] private TMP_Text trialText;

    public void SetVisible(bool visible) => gameObject.SetActive(visible);

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

    public void SetTrialWithCountdown(int current1Based, int total, float remainingSec)
    {
        if (trialText == null) return;

        string mmss = FormatMMSS(remainingSec);
        trialText.text = $"Trial {current1Based} / {total} · {mmss}";
    }

    static string FormatMMSS(float seconds)
    {
        if (seconds < 0f) seconds = 0f;
        int s = Mathf.CeilToInt(seconds);
        int m = s / 60;
        int r = s % 60;
        return $"{m:00}:{r:00}";
    }
}
