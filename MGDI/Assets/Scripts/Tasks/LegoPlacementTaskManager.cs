using System.Collections;
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

    [Tooltip("Optional: a component on the grabbed object or grab controller that supports ForceRelease() and SetFollowHeldRotation(bool).")]
    [SerializeField] private MonoBehaviour grabReleaseComponent;

    [Header("Grab behavior per task")]
    [Tooltip("If true, lock rotation follow during placement trials (translation-only).")]
    [SerializeField] private bool lockRotationDuringPlacement = true;

    [Tooltip("If true, restore rotation follow after each trial (useful when later adding rotation tasks).")]
    [SerializeField] private bool restoreRotationFollowAfterTrial = true;

    [Header("Feedback (Audio/UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;

    [Tooltip("Enable/disable star object for success feedback.")]
    [SerializeField] private GameObject starUI;

    [Tooltip("Enable/disable X object for failure feedback.")]
    [SerializeField] private GameObject xUI;

    [Tooltip("How long to show star or X (seconds).")]
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Trial Parameters")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Tooltip("Tolerance = max(0.05 * targetDistance, minTolMeters).")]
    [SerializeField] private float minTolMeters = 0.005f;

    [Tooltip("If true, keep targetSlot Y equal to its current Y (ignore sampled Y offset).")]
    [SerializeField] private bool keepTargetSlotY = true;

    [Header("Target Offset Range (meters) - sampled per axis")]
    [SerializeField] private Vector2 offsetXRange = new Vector2(-0.35f, 0.35f);
    [SerializeField] private Vector2 offsetYRange = new Vector2(0.00f, 0.00f);
    [SerializeField] private Vector2 offsetZRange = new Vector2(0.30f, 0.60f);

    [Header("Min distance constraint (prevents too-close targets)")]
    [SerializeField] private float minPlanarOffsetMeters = 0.30f;
    [SerializeField] private int maxResampleAttempts = 20;

    [Header("Snap + Inter-trial timing")]
    [Tooltip("If true, snap block to target when success.")]
    [SerializeField] private bool snapOnSuccess = true;

    [Tooltip("Additional wait after snapping (seconds) before resetting and next trial.")]
    [SerializeField] private float postSnapHoldSeconds = 0.35f;

    [Tooltip("If true, block is reset to startPos after success/timeout before next trial.")]
    [SerializeField] private bool resetBlockToStartAfterTrial = true;

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

    private Vector3 startPos;     // start position for CURRENT trial
    private Vector3 targetPos;
    private float targetDistance;
    private float tolerance;

    private bool trialRunning = false;
    private bool inTransition = false;

    // Voice
    private KeywordRecognizer keywordRecognizer;
    private Dictionary<string, System.Action> keywordActions;

    // --- Public controls ---
    public void StartBlock()
    {
        StopAllCoroutines();
        inTransition = false;
        trialRunning = false;

        trialIndex = 0;
        BeginNextTrial();
    }

    private void Start()
    {
        if (enableVoiceStart)
            SetupVoiceCommands();

        HideFeedbackUI();

        if (autoStartInEditor && Application.isEditor)
            StartBlock();
    }

    private void Update()
    {
        if (!trialRunning || inTransition) return;

        trialTimer += Time.deltaTime;

        if (trialTimer >= trialTimeoutSeconds)
        {
            StartCoroutine(EndTrialRoutine(success: false, timedOut: true));
            return;
        }

        float err = Vector3.Distance(blockRoot.position, targetSlotRoot.position);

        if (err <= tolerance)
        {
            dwellTimer += Time.deltaTime;
            if (dwellTimer >= dwellSeconds)
            {
                StartCoroutine(EndTrialRoutine(success: true, timedOut: false));
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

        // Placement trials: lock rotation-follow so incidental hand/camera rotation doesn't rotate the object.
        if (lockRotationDuringPlacement)
        {
            SetGrabberFollowRotation(false);
        }

        // Record current block position as the startPos for THIS trial.
        startPos = blockRoot.position;

        // Sample target offset (axis ranges + min planar distance)
        Vector3 offset = SampleOffsetWithMinPlanarDistance();

        float y = keepTargetSlotY ? targetSlotRoot.position.y : (startPos.y + offset.y);
        targetPos = new Vector3(startPos.x + offset.x, y, startPos.z + offset.z);

        targetSlotRoot.position = targetPos;

        targetDistance = Vector3.Distance(startPos, targetPos);
        tolerance = Mathf.Max(0.05f * targetDistance, minTolMeters);

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        Debug.Log($"[LegoPlacementTaskManager] Trial {trialIndex + 1}/{totalTrials} " +
                  $"targetDist={targetDistance:F3}m tol={tolerance:F3}m " +
                  $"start=({startPos.x:F3},{startPos.y:F3},{startPos.z:F3}) " +
                  $"target=({targetPos.x:F3},{targetPos.y:F3},{targetPos.z:F3})");
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (inTransition) yield break;
        inTransition = true;
        trialRunning = false;

        float finalErr = Vector3.Distance(blockRoot.position, targetSlotRoot.position);
        Debug.Log($"[LegoPlacementTaskManager] Trial {trialIndex + 1} End. " +
                  $"success={success} timedOut={timedOut} time={trialTimer:F2}s finalErr={finalErr:F3}m");

        HideFeedbackUI();

        if (success)
        {
            // Release first so grabber does not fight the snap/reset.
            ForceReleaseIfPossible();

            if (snapOnSuccess)
            {
                SnapBlockToTarget();
            }

            PlaySnapSound();
            ShowStar();

            // Hold a short moment so the user perceives "clicked in"
            yield return new WaitForSeconds(Mathf.Max(feedbackShowSeconds, postSnapHoldSeconds));
        }
        else
        {
            ShowX();
            yield return new WaitForSeconds(feedbackShowSeconds);
        }

        HideFeedbackUI();

        // Restore rotation-follow (optional; useful if later tasks require rotation)
        if (restoreRotationFollowAfterTrial)
        {
            // Safe to call even if not grabbed
            SetGrabberFollowRotation(true);
        }

        // Reset block position back to startPos (per your request)
        if (resetBlockToStartAfterTrial)
        {
            ForceReleaseIfPossible(); // extra safety
            ResetBlockToStart();
        }

        trialIndex++;

        BeginNextTrial();
    }

    private void SnapBlockToTarget()
    {
        // Translation task: snap position only. Keep current rotation/scale.
        Vector3 p = targetSlotRoot.position;
        blockRoot.position = new Vector3(p.x, blockRoot.position.y, p.z);
    }

    private void ResetBlockToStart()
    {
        blockRoot.position = startPos;
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

            Vector2 planar = new Vector2(offset.x, offset.z);
            if (planar.magnitude >= minPlanarOffsetMeters)
                return offset;
        }

        Debug.LogWarning("[LegoPlacementTaskManager] Could not satisfy minPlanarOffsetMeters after resampling. Using last sample.");
        return offset;
    }

    private void ForceReleaseIfPossible()
    {
        if (grabReleaseComponent != null)
        {
            grabReleaseComponent.SendMessage("ForceRelease", SendMessageOptions.DontRequireReceiver);
        }
    }

    private void SetGrabberFollowRotation(bool follow)
    {
        if (grabReleaseComponent != null)
        {
            // ProxyHandGrabber has: public void SetFollowHeldRotation(bool value)
            grabReleaseComponent.SendMessage("SetFollowHeldRotation", follow, SendMessageOptions.DontRequireReceiver);
        }
    }

    private void PlaySnapSound()
    {
        if (audioSource == null || snapClip == null) return;
        audioSource.PlayOneShot(snapClip);
    }

    private void ShowStar()
    {
        if (starUI != null) starUI.SetActive(true);
        if (xUI != null) xUI.SetActive(false);
    }

    private void ShowX()
    {
        if (xUI != null) xUI.SetActive(true);
        if (starUI != null) starUI.SetActive(false);
    }

    private void HideFeedbackUI()
    {
        if (starUI != null) starUI.SetActive(false);
        if (xUI != null) xUI.SetActive(false);
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
