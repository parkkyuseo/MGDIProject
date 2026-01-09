using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Windows.Speech;

public class LegoRotationTaskManager : MonoBehaviour
{
    [Header("References")]
    [Tooltip("The transform that rotates (use LegoBlockRoot).")]
    [SerializeField] private Transform blockRoot;

    [Tooltip("The target slot root transform (EMPTY root). Used for position anchoring.")]
    [SerializeField] private Transform targetSlotRoot;

    [Tooltip("Target visual transform (EMPTY visual parent). Used for yaw reference.")]
    [SerializeField] private Transform targetSlotVisual;

    [Tooltip("Grab controller (e.g., ProxyHandGrabber) supporting ForceRelease() and SetFollowHeldRotation(bool).")]
    [SerializeField] private MonoBehaviour grabReleaseComponent;

    [Header("Feedback (Audio / UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Header("Yaw Success Threshold")]
    [Tooltip("Success if absolute yaw error <= this threshold (degrees).")]
    [SerializeField] private float yawToleranceDeg = 10f;

    [Header("Inter-trial behavior")]
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetBlockYawAfterTrial = true;
    [SerializeField] private bool resetBlockPositionAfterTrial = false;

    [Header("Target yaw sampling")]
    [Tooltip("Allowed yaw offsets (degrees). Sampled per trial.")]
    [SerializeField] private float yawMinDeg = 30f;
    [SerializeField] private float yawMaxDeg = 90f;

    [Tooltip("If true, randomly choose left/right (negative/positive) offset.")]
    [SerializeField] private bool randomizeYawSign = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = true;
    [SerializeField] private string startKeyword = "start";
    [SerializeField] private string restartKeyword = "restart";
    [SerializeField] private bool autoStartInEditor = false;

    // Runtime
    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;

    private Vector3 startPos;
    private float startYawDeg;

    private bool trialRunning = false;
    private bool inTransition = false;

    // Voice
    private KeywordRecognizer keywordRecognizer;
    private Dictionary<string, System.Action> keywordActions;

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
        HideFeedbackUI();

        if (enableVoiceStart)
            SetupVoiceCommands();

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

        float yawErr = ComputeYawErrorDeg();
        if (yawErr <= yawToleranceDeg)
        {
            dwellTimer += Time.deltaTime;
            if (dwellTimer >= dwellSeconds)
                StartCoroutine(EndTrialRoutine(success: true, timedOut: false));
        }
        else
        {
            dwellTimer = 0f;
        }

        // Optional debug
        // if (Time.frameCount % 30 == 0) Debug.Log($"yawErr={yawErr:F2} tol={yawToleranceDeg:F1}");
    }

    private void BeginNextTrial()
    {
        if (blockRoot == null || targetSlotRoot == null || targetSlotVisual == null)
        {
            Debug.LogError("[LegoRotationTaskManager] Missing references.");
            trialRunning = false;
            return;
        }

        if (totalTrials > 0 && trialIndex >= totalTrials)
        {
            Debug.Log("[LegoRotationTaskManager] Block finished.");
            trialRunning = false;
            return;
        }

        // Rotation task: allow rotation follow
        SetGrabberFollowRotation(true);

        // Record start pose
        startPos = blockRoot.position;
        startYawDeg = blockRoot.eulerAngles.y;

        // Anchor target position to current block position (pure rotation task)
        targetSlotRoot.position = startPos;

        // Sample target yaw offset and apply to target visual
        float offset = Random.Range(yawMinDeg, yawMaxDeg);
        if (randomizeYawSign && Random.value < 0.5f) offset = -offset;

        // Target yaw = start yaw + offset
        float targetYaw = startYawDeg + offset;

        // Apply yaw-only rotation to the visual (keep pitch/roll at 0 for clarity)
        targetSlotVisual.rotation = Quaternion.Euler(0f, targetYaw, 0f);

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        Debug.Log($"[LegoRotationTaskManager] Trial {trialIndex + 1}/{totalTrials} targetYawOffset={offset:F1}deg tol={yawToleranceDeg:F1}deg");
    }

    private float ComputeYawErrorDeg()
    {
        float blockYaw = blockRoot.eulerAngles.y;
        float targetYaw = targetSlotVisual.eulerAngles.y;
        return Mathf.Abs(Mathf.DeltaAngle(blockYaw, targetYaw));
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

            // Snap rotation (yaw-only) to target
            SnapYawToTarget();

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

        // Optional reset after each trial
        ForceReleaseIfPossible();
        if (resetBlockYawAfterTrial)
            ResetBlockYaw();
        if (resetBlockPositionAfterTrial)
            blockRoot.position = startPos;

        trialIndex++;
        BeginNextTrial();
    }

    private void SnapYawToTarget()
    {
        float targetYaw = targetSlotVisual.eulerAngles.y;
        Vector3 e = blockRoot.eulerAngles;
        e.x = 0f; // optional: keep upright
        e.z = 0f;
        e.y = targetYaw;
        blockRoot.rotation = Quaternion.Euler(e);
    }

    private void ResetBlockYaw()
    {
        Vector3 e = blockRoot.eulerAngles;
        e.x = 0f;
        e.z = 0f;
        e.y = startYawDeg;
        blockRoot.rotation = Quaternion.Euler(e);
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

        Debug.Log($"[LegoRotationTaskManager] Voice commands enabled: '{startKeyword}', '{restartKeyword}'");
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
