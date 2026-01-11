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

    [Tooltip("ProxyHandGrabber (or similar) that supports ForceRelease() and SetFollowHeldRotation(bool).")]
    [SerializeField] private MonoBehaviour grabReleaseComponent;

    [Tooltip("ProxyHandGrabber instance (for gating rotation only when holding).")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Tooltip("RemoteHandRuntime that provides TwistDegrees / TwistReady.")]
    [SerializeField] private RemoteHandRuntime remoteHand;

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
    [SerializeField] private float yawToleranceDeg = 10f;

    [Header("Twist → Yaw mapping")]
    [Tooltip("Yaw change per 1 degree of twist. 1.0 = 1:1, 2.0 = 2x.")]
    [SerializeField] private float twistToYawGain = 1.5f;

    [Tooltip("If true, invert twist sign for yaw mapping.")]
    [SerializeField] private bool invertTwistToYaw = false;

    [Tooltip("Clamp yaw speed (deg/sec) to prevent jumps.")]
    [SerializeField] private float yawMaxDegPerSec = 240f;

    [Tooltip("Extra smoothing for yaw command (0=no extra smoothing, 1=very slow).")]
    [Range(0f, 1f)]
    [SerializeField] private float yawLerp = 0.15f;

    [Header("Rotation gating")]
    [Tooltip("If true, block yaw is driven only while grabber is holding something (more natural).")]
    [SerializeField] private bool rotateOnlyWhenHolding = true;

    [Tooltip("If true, rotation is driven only when grabber is holding THIS block rigidbody.")]
    [SerializeField] private bool requireHoldingThisBlock = true;

    [Header("Inter-trial behavior")]
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetBlockYawAfterTrial = true;

    [Header("Target yaw sampling")]
    [SerializeField] private float yawMinDeg = 30f;
    [SerializeField] private float yawMaxDeg = 90f;
    [SerializeField] private bool randomizeYawSign = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = false; // 보통 FlowController가 담당
    [SerializeField] private string startKeyword = "start rotation";
    [SerializeField] private string restartKeyword = "restart rotation";
    [SerializeField] private bool autoStartInEditor = false;

    // Runtime
    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;

    private Vector3 startPos;
    private float startYawDeg;

    private float twistBaselineDeg;
    private float yawCmdDeg;
    private bool yawCmdInit = false;

    private bool trialRunning = false;
    private bool inTransition = false;

    // Cached block rigidbody (for holding check)
    private Rigidbody _blockBody;

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

        // cache block rigidbody if available
        if (blockRoot != null)
            _blockBody = blockRoot.GetComponentInParent<Rigidbody>() ?? blockRoot.GetComponentInChildren<Rigidbody>();

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

        // 1) Drive yaw from twist ONLY when allowed
        if (ShouldDriveRotationThisFrame())
        {
            UpdateYawFromTwist();
        }
        else
        {
            // If rotation is gated off, do not accumulate dwell
            dwellTimer = 0f;
        }

        // 2) Evaluate yaw error
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
    }

    private bool ShouldDriveRotationThisFrame()
    {
        if (!rotateOnlyWhenHolding) return true;

        if (grabber == null) return false;
        if (!grabber.IsHolding) return false;

        if (!requireHoldingThisBlock) return true;

        // Need block body and grabber held body
        if (_blockBody == null) return false;
        if (grabber.HeldBody == null) return false;

        return grabber.HeldBody == _blockBody;
    }

    private void BeginNextTrial()
    {
        if (blockRoot == null || targetSlotRoot == null || targetSlotVisual == null)
        {
            Debug.LogError("[LegoRotationTaskManager] Missing references.");
            trialRunning = false;
            return;
        }

        if (remoteHand == null)
        {
            Debug.LogError("[LegoRotationTaskManager] remoteHand is null (RemoteHandRuntime).");
            trialRunning = false;
            return;
        }

        if (totalTrials > 0 && trialIndex >= totalTrials)
        {
            Debug.Log("[LegoRotationTaskManager] Block finished.");
            trialRunning = false;
            return;
        }

        // Rotation task: we drive yaw ourselves from twist, so we do NOT follow full hand rotation.
        SetGrabberFollowRotation(false);

        // Record start pose
        startPos = blockRoot.position;
        startYawDeg = blockRoot.eulerAngles.y;

        // Anchor target position to current block position (pure rotation task)
        targetSlotRoot.position = startPos;

        // Sample target yaw offset and apply to target visual
        float offset = Random.Range(yawMinDeg, yawMaxDeg);
        if (randomizeYawSign && Random.value < 0.5f) offset = -offset;

        float targetYaw = startYawDeg + offset;
        targetSlotVisual.rotation = Quaternion.Euler(0f, targetYaw, 0f);

        // Capture twist baseline (trial-specific)
        twistBaselineDeg = GetTwistSafe();
        yawCmdDeg = startYawDeg;
        yawCmdInit = true;

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        Debug.Log($"[LegoRotationTaskManager] Trial {trialIndex + 1}/{totalTrials} targetYawOffset={offset:F1}deg tol={yawToleranceDeg:F1}deg");
    }

    private float GetTwistSafe()
    {
        if (remoteHand == null) return 0f;
        return remoteHand.TwistDegrees; // best effort (TwistReady may still be false early)
    }

    private void UpdateYawFromTwist()
    {
        float twistNow = GetTwistSafe();
        float dTwist = twistNow - twistBaselineDeg; // degrees

        float sign = invertTwistToYaw ? -1f : 1f;

        // Desired yaw
        float desiredYaw = startYawDeg + dTwist * twistToYawGain * sign;

        // Smooth + clamp speed
        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);

        if (!yawCmdInit)
        {
            yawCmdDeg = desiredYaw;
            yawCmdInit = true;
        }

        // Clamp yaw speed (deg/sec)
        float maxStep = Mathf.Max(10f, yawMaxDegPerSec) * dt;
        float step = Mathf.DeltaAngle(yawCmdDeg, desiredYaw);
        step = Mathf.Clamp(step, -maxStep, maxStep);

        float yawNext = yawCmdDeg + step;

        // Extra LPF
        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(yawLerp), dt * 60f);
        yawCmdDeg = Mathf.LerpAngle(yawCmdDeg, yawNext, k);

        // Apply yaw-only rotation (keep upright)
        blockRoot.rotation = Quaternion.Euler(0f, yawCmdDeg, 0f);
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
            // Snap yaw exactly
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

        if (resetBlockYawAfterTrial)
            ResetBlockYaw();

        trialIndex++;
        BeginNextTrial();
    }

    private void SnapYawToTarget()
    {
        float targetYaw = targetSlotVisual.eulerAngles.y;
        blockRoot.rotation = Quaternion.Euler(0f, targetYaw, 0f);
        yawCmdDeg = targetYaw;
    }

    private void ResetBlockYaw()
    {
        blockRoot.rotation = Quaternion.Euler(0f, startYawDeg, 0f);
        yawCmdDeg = startYawDeg;
    }

    private void SetGrabberFollowRotation(bool follow)
    {
        if (grabReleaseComponent != null)
            grabReleaseComponent.SendMessage("SetFollowHeldRotation", follow, SendMessageOptions.DontRequireReceiver);
    }

    private void ForceReleaseIfPossible()
    {
        if (grabReleaseComponent != null)
            grabReleaseComponent.SendMessage("ForceRelease", SendMessageOptions.DontRequireReceiver);
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
            string key = args.text.ToLower();
            if (keywordActions.TryGetValue(key, out var action))
                action.Invoke();
        };
        keywordRecognizer.Start();

        Debug.Log($"[LegoRotationTaskManager] Voice enabled: '{startKeyword}', '{restartKeyword}'");
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
