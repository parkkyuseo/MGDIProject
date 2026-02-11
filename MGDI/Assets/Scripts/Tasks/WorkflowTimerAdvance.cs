using UnityEngine;

public class WorkflowTimerAdvance : MonoBehaviour
{
    [SerializeField] private WorkflowProgressionController workflow;
    [SerializeField] private float intervalSec = 3.0f;
    [SerializeField] private bool runOnStart = true;

    private float _t;

    void Start()
    {
        if (workflow == null)
            workflow = FindFirstObjectByType<WorkflowProgressionController>();

        _t = intervalSec;
    }

    void Update()
    {
        if (!runOnStart || workflow == null) return;

        _t -= Time.unscaledDeltaTime;
        if (_t <= 0f)
        {
            workflow.Advance();
            _t = intervalSec;
        }
    }
}
