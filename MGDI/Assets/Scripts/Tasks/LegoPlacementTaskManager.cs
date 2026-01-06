using UnityEngine;

public class LegoPlacementTaskManager : MonoBehaviour
{
    [Header("References")]
    [Tooltip("The transform that moves (use LegoBlockRoot).")]
    [SerializeField] private Transform blockRoot;

    [Tooltip("The target slot root transform (e.g., TargetSlot/LegoBlockRoot).")]
    [SerializeField] private Transform targetSlotRoot;

    [Header("Trial Parameters")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Tooltip("Target translation distance range (meters). Sampled on XZ plane.")]
    [SerializeField] private float targetDistMin = 0.20f;
    [SerializeField] private float targetDistMax = 0.40f;

    [Tooltip("Tolerance = max(0.05 * targetDistance, minTolMeters).")]
    [SerializeField] private float minTolMeters = 0.005f;

    [Tooltip("If true, keep targetSlot Y equal to its current Y (ignore block Y).")]
    [SerializeField] private bool keepTargetSlotY = true;

    [Header("Optional: Trial Count")]
    [SerializeField] private int totalTrials = 20;

    // Runtime state
    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;

    private Vector3 startPos;
    private Vector3 targetPos;
    private float targetDistance;
    private float tolerance;

    private bool trialRunning = false;

    // --- Public controls (can be wired to UI/voice later) ---
    [ContextMenu("Start / Restart Block")]
    public void StartBlock()
    {
        trialIndex = 0;
        BeginNextTrial();
    }

    private void Start()
    {
        // Optional auto-start:
        // BeginNextTrial();
    }

    private void Update()
    {
        if (!trialRunning) return;

        trialTimer += Time.deltaTime;

        // Timeout?
        if (trialTimer >= trialTimeoutSeconds)
        {
            EndTrial(success: false, timedOut: true);
            return;
        }

        // Evaluate translation error
        float err = Vector3.Distance(blockRoot.position, targetSlotRoot.position);

        if (err <= tolerance)
        {
            dwellTimer += Time.deltaTime;
            if (dwellTimer >= dwellSeconds)
            {
                EndTrial(success: true, timedOut: false);
            }
        }
        else
        {
            dwellTimer = 0f;
        }
    }

    private void BeginNextTrial()
    {
        if (blockRoot == null || targetSlotRoot == null)
        {
            Debug.LogError("[LegoPlacementTaskManager] Missing references: blockRoot or targetSlotRoot.");
            trialRunning = false;
            return;
        }

        if (totalTrials > 0 && trialIndex >= totalTrials)
        {
            trialRunning = false;
            Debug.Log("[LegoPlacementTaskManager] Block finished.");
            return;
        }

        // Record start pose (keep as-is; do not force y right now)
        startPos = blockRoot.position;

        // Sample target pose on XZ plane around start
        Vector2 dir2 = Random.insideUnitCircle.normalized;
        if (dir2.sqrMagnitude < 1e-6f) dir2 = Vector2.right;

        float dist = Random.Range(targetDistMin, targetDistMax);
        Vector3 offset = new Vector3(dir2.x, 0f, dir2.y) * dist;

        float y = keepTargetSlotY ? targetSlotRoot.position.y : startPos.y;
        targetPos = new Vector3(startPos.x + offset.x, y, startPos.z + offset.z);

        // Apply target to slot (slot is the visual guide)
        targetSlotRoot.position = targetPos;

        // Compute tolerance
        targetDistance = Vector3.Distance(startPos, targetPos);
        tolerance = Mathf.Max(0.05f * targetDistance, minTolMeters);

        // Reset timers
        trialTimer = 0f;
        dwellTimer = 0f;

        trialRunning = true;

        Debug.Log($"[LegoPlacementTaskManager] Trial {trialIndex + 1}/{totalTrials} " +
                  $"targetDist={targetDistance:F3}m tol={tolerance:F3}m");
    }

    private void EndTrial(bool success, bool timedOut)
    {
        trialRunning = false;

        // --- Log (placeholder) ---
        float finalErr = Vector3.Distance(blockRoot.position, targetSlotRoot.position);
        Debug.Log($"[LegoPlacementTaskManager] Trial {trialIndex + 1} End. " +
                  $"success={success} timedOut={timedOut} time={trialTimer:F2}s finalErr={finalErr:F3}m");

        if (success)
        {
            SnapBlockToTarget();
            OnTrialSuccess();
        }
        else
        {
            OnTrialFail(timedOut);
        }

        trialIndex++;

        // Small delay could be added later; for now, start immediately
        BeginNextTrial();
    }

    private void SnapBlockToTarget()
    {
        // Translation task: snap position only. Keep current rotation/scale.
        Vector3 p = targetSlotRoot.position;
        blockRoot.position = new Vector3(p.x, blockRoot.position.y, p.z); // keep block y as-is for now
    }

    // Hooks for later: audio/UI/logging integration
    private void OnTrialSuccess()
    {
        // TODO: play snap sound, show star, write to CSV
    }

    private void OnTrialFail(bool timedOut)
    {
        // TODO: show X, write to CSV
    }
}
