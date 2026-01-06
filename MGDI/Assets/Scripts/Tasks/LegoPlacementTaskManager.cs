using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Windows.Speech;

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

    [Tooltip("Tolerance = max(0.05 * targetDistance, minTolMeters).")]
    [SerializeField] private float minTolMeters = 0.005f;

    [Tooltip("If true, keep targetSlot Y equal to its current Y (ignore sampled Y offset).")]
    [SerializeField] private bool keepTargetSlotY = true;

    [Header("Target Offset Range (meters) - sampled per axis")]
    [Tooltip("Offset added to startPos.x")]
    [SerializeField] private Vector2 offsetXRange = new Vector2(-0.35f, 0.35f);

    [Tooltip("Offset added to startPos.y (used only when keepTargetSlotY is false)")]
    [SerializeField] private Vector2 offsetYRange = new Vector2(0.00f, 0.00f);

    [Tooltip("Offset added to startPos.z")]
    [SerializeField] private Vector2 offsetZRange = new Vector2(0.30f, 0.60f);

    [Header("Min distance constraint (prevents too-close targets)")]
    [Tooltip("Minimum planar (XZ) offset magnitude in meters.")]
    [SerializeField] private float minPlanarOffsetMeters = 0.30f;

    [Tooltip("Max resampling attempts to satisfy minPlanarOffsetMeters.")]
    [SerializeField] private int maxResampleAttempts = 20;

    [Header("Optional: Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = true;
    [SerializeField] private string startKeyword = "start";
    [SerializeField] private string restartKeyword = "restart";
    [SerializeField] private bool autoStartInEditor = false;

    // Runtime state
    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;

    private Vector3 startPos;
    private Vector3 targetPos;
    private float targetDistance;
    private float tolerance;

    private bool trialRunning = false;

    // Voice
    private KeywordRecognizer keywordRecognizer;
    private Dictionary<string, System.Action> keywordActions;

    // --- Public controls (can be wired to UI later) ---
    public void StartBlock()
    {
        trialIndex = 0;
        BeginNextTrial();
    }

    private void Start()
    {
        if (enableVoiceStart)
            SetupVoiceCommands();

        if (autoStartInEditor && Application.isEditor)
            StartBlock();
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

        // Sample offset per axis with min-planar-distance constraint
        Vector3 offset = SampleOffsetWithMinPlanarDistance();

        float y = keepTargetSlotY ? targetSlotRoot.position.y : (startPos.y + offset.y);
        targetPos = new Vector3(startPos.x + offset.x, y, startPos.z + offset.z);

        // Apply target to slot (slot is the visual guide)
        targetSlotRoot.position = targetPos;

        // Compute tolerance (based on target distance)
        targetDistance = Vector3.Distance(startPos, targetPos);
        tolerance = Mathf.Max(0.05f * targetDistance, minTolMeters);

        // Reset timers
        trialTimer = 0f;
        dwellTimer = 0f;

        trialRunning = true;

        Debug.Log($"[LegoPlacementTaskManager] Trial {trialIndex + 1}/{totalTrials} " +
                  $"targetDist={targetDistance:F3}m tol={tolerance:F3}m " +
                  $"offset=({offset.x:F3},{offset.y:F3},{offset.z:F3})");
    }

    private Vector3 SampleOffsetWithMinPlanarDistance()
    {
        Vector3 offset = Vector3.zero;

        for (int attempt = 0; attempt < maxResampleAttempts; attempt++)
        {
            float ox = Random.Range(offsetXRange.x, offsetXRange.y);
            float oy = Random.Range(offsetYRange.x, offsetYRange.y);
            float oz = Random.Range(offsetZRange.x, offsetZRange.y);

            offset = new Vector3(ox, oy, oz);

            // Only constrain planar distance in XZ (Placement task)
            Vector2 planar = new Vector2(offset.x, offset.z);
            if (planar.magnitude >= minPlanarOffsetMeters)
                return offset;
        }

        // If resampling fails, return the last sample but warn
        Debug.LogWarning("[LegoPlacementTaskManager] Could not satisfy minPlanarOffsetMeters after resampling. Using last sample.");
        return offset;
    }

    private void EndTrial(bool success, bool timedOut)
    {
        trialRunning = false;

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

        // For now, start next trial immediately
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

    private void SetupVoiceCommands()
    {
        if (keywordRecognizer != null) return;

        keywordActions = new Dictionary<string, System.Action>
        {
            { startKeyword.ToLower(), () => StartBlock() },
            { restartKeyword.ToLower(), () => StartBlock() }
        };

        keywordRecognizer = new KeywordRecognizer(keywordActions.Keys.ToArray());
        keywordRecognizer.OnPhraseRecognized += args =>
        {
            string key = args.text.ToLower();
            if (keywordActions.TryGetValue(key, out var action))
                action.Invoke();
        };
        keywordRecognizer.Start();

        Debug.Log($"[LegoPlacementTaskManager] Voice commands enabled: '{startKeyword}', '{restartKeyword}'");
    }

    private void OnDisable()
    {
        if (keywordRecognizer != null)
        {
            keywordRecognizer.Stop();
            keywordRecognizer.Dispose();
            keywordRecognizer = null;
        }
    }
}
