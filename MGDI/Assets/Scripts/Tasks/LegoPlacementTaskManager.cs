using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Windows.Speech;

public class LegoPlacementTaskManager : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private Transform blockRoot;
    [SerializeField] private Transform targetSlotRoot;

    [Tooltip("Grab controller (e.g., ProxyHandGrabber) supporting ForceRelease() and SetFollowHeldRotation(bool).")]
    [SerializeField] private MonoBehaviour grabReleaseComponent;

    [Tooltip("Reference frame for target sampling (e.g., empty object at table center).")]
    [SerializeField] private Transform referenceFrame;

    [Header("Grab behavior per task")]
    [SerializeField] private bool lockRotationDuringPlacement = true;
    [SerializeField] private bool restoreRotationFollowAfterTrial = true;

    [Header("Feedback (Audio / UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Target Visual")]
    [SerializeField] private Transform targetSlotVisual;
    [SerializeField] private bool matchTargetVisualRotationToBlock = true;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Tooltip("Tolerance = max(0.05 * targetDistance, minTolMeters).")]
    [SerializeField] private float minTolMeters = 0.005f;

    [Header("Snap / Reset")]
    [SerializeField] private bool snapOnSuccess = true;
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetBlockToStartAfterTrial = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = true;
    [SerializeField] private string startKeyword = "start";
    [SerializeField] private string restartKeyword = "restart";
    [SerializeField] private bool autoStartInEditor = false;

    [Header("Target Sampling (referenceFrame local space)")]
    [Tooltip("Local X range (meters).")]
    [SerializeField] private Vector2 localXRange = new Vector2(-0.35f, 0.35f);

    [Tooltip("Local Y range (meters). Can be non-zero if table is disabled.")]
    [SerializeField] private Vector2 localYRange = new Vector2(0.00f, 0.15f);

    [Tooltip("Local Z range (meters). Strongly constrains reachability.")]
    [SerializeField] private Vector2 localZRange = new Vector2(0.25f, 0.55f);

    [Header("Forbidden Center Region")]
    [Tooltip("Half-width of forbidden center strip on X (meters). abs(x) < value is rejected.")]
    [SerializeField] private float centerNoGoHalfWidthX = 0.12f;

    [Header("Distance Constraints")]
    [Tooltip("Minimum planar (XZ) distance from start (meters).")]
    [SerializeField] private float minPlanarOffsetMeters = 0.30f;

    [Tooltip("Max resampling attempts.")]
    [SerializeField] private int maxResampleAttempts = 40;

    // Runtime
    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;

    private Vector3 startPos;
    private Vector3 targetPos;
    private float tolerance;

    private bool trialRunning = false;
    private bool inTransition = false;

    // Voice
    private KeywordRecognizer keywordRecognizer;
    private Dictionary<string, System.Action> keywordActions;

    // ---------------- Public ----------------
    public void StartBlock()
    {
        StopAllCoroutines();
        inTransition = false;
        trialRunning = false;
        trialIndex = 0;
        BeginNextTrial();
    }

    void Start()
    {
        if (enableVoiceStart)
            SetupVoiceCommands();

        HideFeedbackUI();

        if (autoStartInEditor && Application.isEditor)
            StartBlock();
    }

    void Update()
    {
        if (!trialRunning || inTransition) return;

        trialTimer += Time.deltaTime;

        if (trialTimer >= trialTimeoutSeconds)
        {
            StartCoroutine(EndTrialRoutine(false, true));
            return;
        }

        float err = Vector3.Distance(blockRoot.position, targetSlotRoot.position);

        if (err <= tolerance)
        {
            dwellTimer += Time.deltaTime;
            if (dwellTimer >= dwellSeconds)
                StartCoroutine(EndTrialRoutine(true, false));
        }
        else
        {
            dwellTimer = 0f;
        }
    }

    // ---------------- Trial Flow ----------------
    private void BeginNextTrial()
    {
        if (blockRoot == null || targetSlotRoot == null || referenceFrame == null)
        {
            Debug.LogError("[LegoPlacementTaskManager] Missing references.");
            return;
        }

        if (totalTrials > 0 && trialIndex >= totalTrials)
        {
            Debug.Log("[LegoPlacementTaskManager] Block finished.");
            trialRunning = false;
            return;
        }

        if (lockRotationDuringPlacement)
            SetGrabberFollowRotation(false);

        startPos = blockRoot.position;

        Vector3 localOffset = SampleLocalOffset();
        Vector3 worldTarget = referenceFrame.TransformPoint(localOffset);

        targetPos = worldTarget;
        targetSlotRoot.position = targetPos;

        float targetDistance = Vector3.Distance(startPos, targetPos);
        tolerance = Mathf.Max(0.05f * targetDistance, minTolMeters);

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        Debug.Log($"[LegoPlacementTaskManager] Trial {trialIndex + 1}/{totalTrials} " +
                  $"localOffset=({localOffset.x:F2},{localOffset.y:F2},{localOffset.z:F2})");
    }

    private Vector3 SampleLocalOffset()
    {
        for (int i = 0; i < maxResampleAttempts; i++)
        {
            float lx = Random.Range(localXRange.x, localXRange.y);
            float ly = Random.Range(localYRange.x, localYRange.y);
            float lz = Random.Range(localZRange.x, localZRange.y);

            // 중앙 금지 스트립 (X만)
            if (Mathf.Abs(lx) < centerNoGoHalfWidthX)
                continue;

            // 최소 평면 거리 (XZ)
            float planar = Mathf.Sqrt(lx * lx + lz * lz);
            if (planar < minPlanarOffsetMeters)
                continue;

            return new Vector3(lx, ly, lz);
        }

        Debug.LogWarning("[LegoPlacementTaskManager] Sampling failed; using fallback.");
        return new Vector3(
            Mathf.Sign(localXRange.y) * centerNoGoHalfWidthX,
            localYRange.x,
            localZRange.x
        );
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (inTransition) yield break;
        inTransition = true;
        trialRunning = false;

        HideFeedbackUI();

        if (success)
        {
            ForceReleaseIfPossible();

            if (snapOnSuccess)
                SnapBlockToTarget();

            PlaySnapSound();
            ShowStar();

            yield return new WaitForSeconds(Mathf.Max(feedbackShowSeconds, postSnapHoldSeconds));
        }
        else
        {
            ShowX();
            yield return new WaitForSeconds(feedbackShowSeconds);
        }

        HideFeedbackUI();

        if (restoreRotationFollowAfterTrial)
            SetGrabberFollowRotation(true);

        if (resetBlockToStartAfterTrial)
        {
            ForceReleaseIfPossible();
            blockRoot.position = startPos;
        }

        trialIndex++;
        BeginNextTrial();
    }

    // ---------------- Helpers ----------------
    private void SnapBlockToTarget()
    {
        blockRoot.position = targetSlotRoot.position;
    }

    private void ForceReleaseIfPossible()
    {
        if (grabReleaseComponent != null)
            grabReleaseComponent.SendMessage("ForceRelease", SendMessageOptions.DontRequireReceiver);
    }

    private void SetGrabberFollowRotation(bool follow)
    {
        if (grabReleaseComponent != null)
            grabReleaseComponent.SendMessage("SetFollowHeldRotation", follow, SendMessageOptions.DontRequireReceiver);
    }

    private void PlaySnapSound()
    {
        if (audioSource != null && snapClip != null)
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

    // ---------------- Voice ----------------
    private void SetupVoiceCommands()
    {
        if (keywordRecognizer != null) return;

        keywordActions = new Dictionary<string, System.Action>
        {
            { startKeyword.ToLower(), StartBlock },
            { restartKeyword.ToLower(), StartBlock }
        };

        keywordRecognizer = new KeywordRecognizer(keywordActions.Keys.ToArray());
        keywordRecognizer.OnPhraseRecognized += args =>
        {
            if (keywordActions.TryGetValue(args.text.ToLower(), out var action))
                action.Invoke();
        };
        keywordRecognizer.Start();
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
