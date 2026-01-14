using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Windows.Speech;

public class LegoRotationTaskManager : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private Transform blockRoot;
    [SerializeField] private Transform targetSlotRoot;    // EMPTY root for target pose
    [SerializeField] private Transform targetSlotVisual;  // EMPTY visual parent (yaw reference)

    [Tooltip("ProxyHandGrabber (or similar) that supports ForceRelease() and SetHeldRotationMode(...).")]
    [SerializeField] private MonoBehaviour grabReleaseComponent;

    [Tooltip("ProxyHandGrabber instance (for gating rotation only when holding, and for hand-based reset).")]
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
    [SerializeField] private float twistToYawGain = 1.5f;
    [SerializeField] private bool invertTwistToYaw = false;
    [SerializeField] private float yawMaxDegPerSec = 240f;
    [Range(0f, 1f)]
    [SerializeField] private float yawLerp = 0.15f;

    [Header("Rotation gating")]
    [SerializeField] private bool rotateOnlyWhenHolding = true;

    [Tooltip("If true, drive rotation only when the grabber is holding THIS block rigidbody.")]
    [SerializeField] private bool requireHoldingThisBlock = true;

    [Header("Target position policy")]
    [Tooltip("If true, target is kept at fixedTargetPose.position each trial (Rotation task assumes Targets is detached to world by FlowController).")]
    [SerializeField] private bool lockTargetPosition = true;

    [Tooltip("REQUIRED for rotation task fixed target. Place this object under StudyRuntime or scene root (NOT under workspaceAnchor).")]
    [SerializeField] private Transform fixedTargetPose;

    [Header("Inter-trial behavior")]
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetBlockYawAfterTrial = true;

    [Header("B) Require re-grab every trial (hand-based reset)")]
    [SerializeField] private bool resetBlockAfterTrial = true;
    [SerializeField] private bool resetNearHand = true;
    [SerializeField] private Transform resetHandAnchorOverride;

    [Header("Reset offset ranges (meters, randomized once per trial end)")]
    [SerializeField] private Vector2 resetForwardRange = new Vector2(0.08f, 0.12f);
    [SerializeField] private Vector2 resetUpRange = new Vector2(-0.03f, -0.01f);
    [SerializeField] private Vector2 resetRightRange = new Vector2(-0.01f, 0.01f);
    [SerializeField] private float resetAttachMargin = 0.02f;
    [SerializeField] private bool logResetSample = false;

    [SerializeField] private Transform fixedBlockStartPose;
    [SerializeField] private bool resetBlockToTrialStartPos = true;

    [Header("Target yaw sampling")]
    [SerializeField] private float yawMinDeg = 30f;
    [SerializeField] private float yawMaxDeg = 90f;
    [SerializeField] private bool randomizeYawSign = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = false;
    [SerializeField] private string startKeyword = "start rotation";
    [SerializeField] private string restartKeyword = "restart rotation";
    [SerializeField] private bool autoStartInEditor = false;

    [Header("Grabber rotation policy")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode rotationTrialGrabberMode = ProxyHandGrabber.HeldRotationMode.ExternalControl;
    [SerializeField] private ProxyHandGrabber.HeldRotationMode restoreGrabberModeOnDisable = ProxyHandGrabber.HeldRotationMode.LockAtGrab;

    [SerializeField] private Transform targetPivot;

    // Runtime
    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;

    private float startYawDeg;
    private float twistBaselineDeg;
    private float yawCmdDeg;
    private bool yawCmdInit = false;

    private bool trialRunning = false;
    private bool inTransition = false;

    private Rigidbody _blockBody;
    private Vector3 _trialStartBlockPosWorld;

    private bool _prevEffectiveDriving = false;

    // Voice
    private KeywordRecognizer keywordRecognizer;
    private Dictionary<string, System.Action> keywordActions;

    public void StartBlock()
    {
        StopAllCoroutines();
        inTransition = false;
        trialRunning = false;
        trialIndex = 0;

        _prevEffectiveDriving = false;

        EnsureBlockBody();

        BeginNextTrial();
    }

    private void Start()
    {
        HideFeedbackUI();
        EnsureBlockBody();

        if (enableVoiceStart)
            SetupVoiceCommands();

        if (autoStartInEditor && Application.isEditor)
            StartBlock();
    }

    private void EnsureBlockBody()
    {
        if (_blockBody != null) return;
        if (blockRoot == null) return;

        _blockBody = blockRoot.GetComponent<Rigidbody>();
        if (_blockBody == null)
            _blockBody = blockRoot.GetComponentInChildren<Rigidbody>(true);
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

        bool drive = ShouldDriveRotationThisFrame();
        bool twistReady = (remoteHand != null && remoteHand.TwistReady);
        bool effectiveDriving = drive && twistReady;

        if (effectiveDriving)
        {
            if (!_prevEffectiveDriving)
                RebaselineTwistBaselineToCurrentYaw();

            UpdateYawFromTwist();
        }
        else
        {
            dwellTimer = 0f;
        }

        _prevEffectiveDriving = effectiveDriving;

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

        EnsureBlockBody();
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

        // Rotation task: RotationTM controls rotation; grabber must NOT override rotation.
        SetGrabberRotationMode(rotationTrialGrabberMode);

        // record start pos (fallback reset)
        _trialStartBlockPosWorld = blockRoot.position;

        // Target fixed position (assumes Targets is detached to world by FlowController)
        if (lockTargetPosition)
        {
            if (fixedTargetPose == null)
            {
                Debug.LogError("[LegoRotationTaskManager] fixedTargetPose is required when lockTargetPosition is true.");
                trialRunning = false;
                return;
            }
            targetSlotRoot.position = fixedTargetPose.position;
        }
        else
        {
            targetSlotRoot.position = blockRoot.position;
        }

        // Record start yaw and baseline twist (TwistReady일 때만 유효)
        startYawDeg = blockRoot.eulerAngles.y;
        twistBaselineDeg = remoteHand.TwistReady ? remoteHand.TwistDegrees : 0f;
        yawCmdDeg = startYawDeg;
        yawCmdInit = true;

        _prevEffectiveDriving = false;

        // Sample target yaw offset and apply to target visual (yaw-only)
        float offset = Random.Range(yawMinDeg, yawMaxDeg);
        if (randomizeYawSign && Random.value < 0.5f) offset = -offset;

        float targetYaw = startYawDeg + offset;
        // targetSlotVisual.rotation = Quaternion.Euler(0f, targetYaw, 0f);
        targetPivot.rotation = Quaternion.Euler(0f, targetYaw, 0f);

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        Debug.Log($"[LegoRotationTaskManager] Trial {trialIndex + 1}/{totalTrials} " +
                  $"targetYawOffset={offset:F1}deg tol={yawToleranceDeg:F1}deg");
    }

    private void RebaselineTwistBaselineToCurrentYaw()
    {
        if (remoteHand == null || blockRoot == null) return;
        if (!remoteHand.TwistReady) return;

        float twistNow = remoteHand.TwistDegrees;
        float blockYawNow = blockRoot.eulerAngles.y;

        float sign = invertTwistToYaw ? -1f : 1f;
        float denom = twistToYawGain * sign;
        if (Mathf.Abs(denom) < 1e-5f)
            denom = (denom >= 0f ? 1e-5f : -1e-5f);

        float yawDelta = Mathf.DeltaAngle(startYawDeg, blockYawNow);
        twistBaselineDeg = twistNow - (yawDelta / denom);

        yawCmdDeg = blockYawNow;
        yawCmdInit = true;
    }

    private void UpdateYawFromTwist()
    {
        if (remoteHand == null || blockRoot == null) return;
        if (!remoteHand.TwistReady) return;

        float twistNow = remoteHand.TwistDegrees;
        float dTwist = Mathf.DeltaAngle(twistBaselineDeg, twistNow);

        float sign = invertTwistToYaw ? -1f : 1f;
        float desiredYaw = startYawDeg + dTwist * twistToYawGain * sign;

        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);

        if (!yawCmdInit)
        {
            yawCmdDeg = desiredYaw;
            yawCmdInit = true;
        }

        float maxStep = Mathf.Max(10f, yawMaxDegPerSec) * dt;
        float step = Mathf.DeltaAngle(yawCmdDeg, desiredYaw);
        step = Mathf.Clamp(step, -maxStep, maxStep);
        float yawNext = yawCmdDeg + step;

        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(yawLerp), dt * 60f);
        yawCmdDeg = Mathf.LerpAngle(yawCmdDeg, yawNext, k);

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

        if (resetBlockAfterTrial)
        {
            ForceReleaseIfPossible();

            if (resetNearHand)
            {
                Transform hand = GetResetHandAnchor();
                if (hand != null)
                {
                    float f = SampleRange(resetForwardRange);
                    float u = SampleRange(resetUpRange);
                    float r = SampleRange(resetRightRange);

                    float minForward = f;
                    if (grabber != null)
                        minForward = Mathf.Max(minForward, grabber.attachDistance + Mathf.Max(0f, resetAttachMargin));

                    Vector3 pos =
                        hand.position +
                        hand.forward * minForward +
                        hand.up * u +
                        hand.right * r;

                    blockRoot.position = pos;

                    if (logResetSample)
                        Debug.Log($"[RotationTM] Reset sample f={minForward:F3} u={u:F3} r={r:F3}");
                }
                else
                {
                    if (resetBlockToTrialStartPos)
                        blockRoot.position = _trialStartBlockPosWorld;
                    else if (fixedBlockStartPose != null)
                        blockRoot.position = fixedBlockStartPose.position;
                }
            }
            else
            {
                if (fixedBlockStartPose != null)
                    blockRoot.position = fixedBlockStartPose.position;
                else if (resetBlockToTrialStartPos)
                    blockRoot.position = _trialStartBlockPosWorld;
            }
        }

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

    private void SetGrabberRotationMode(ProxyHandGrabber.HeldRotationMode mode)
    {
        if (grabber != null)
        {
            grabber.SetHeldRotationMode(mode);
            return;
        }

        if (grabReleaseComponent != null)
            grabReleaseComponent.SendMessage("SetHeldRotationMode", mode, SendMessageOptions.DontRequireReceiver);
    }

    private void ForceReleaseIfPossible()
    {
        if (grabber != null)
        {
            grabber.ForceRelease();
            return;
        }

        if (grabReleaseComponent != null)
            grabReleaseComponent.SendMessage("ForceRelease", SendMessageOptions.DontRequireReceiver);
    }

    private static float SampleRange(Vector2 r)
    {
        float a = Mathf.Min(r.x, r.y);
        float b = Mathf.Max(r.x, r.y);
        return Random.Range(a, b);
    }

    private Transform GetResetHandAnchor()
    {
        if (resetHandAnchorOverride != null) return resetHandAnchorOverride;
        if (grabber != null && grabber.grabAnchor != null) return grabber.grabAnchor;
        return null;
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
    }

    private void OnDisable()
    {
        SetGrabberRotationMode(restoreGrabberModeOnDisable);

        if (keywordRecognizer != null)
        {
            keywordRecognizer.Stop();
            keywordRecognizer.Dispose();
            keywordRecognizer = null;
        }
    }
}
