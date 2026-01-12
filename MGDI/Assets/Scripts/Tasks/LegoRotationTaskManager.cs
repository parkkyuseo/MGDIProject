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
    [Tooltip("If true, target position is fixed (does not change across trials).")]
    [SerializeField] private bool lockTargetPosition = true;

    [Tooltip("If lockTargetPosition is true, use this fixed world position. If not set, first trial captures current targetSlotRoot position.")]
    [SerializeField] private Transform fixedTargetPose; // optional anchor transform

    [Header("Inter-trial behavior")]
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetBlockYawAfterTrial = true;

    [Header("B) Require re-grab every trial (hand-based reset)")]
    [Tooltip("If true, force release and reposition the block at the end of every trial (success/fail).")]
    [SerializeField] private bool resetBlockAfterTrial = true;

    [Tooltip("If true, reset near the hand (grabAnchor). If false, reset to a fixed pose or the recorded trial-start position.")]
    [SerializeField] private bool resetNearHand = true;

    [Tooltip("Optional override if grabber is null. If set, this transform is used as the 'hand' for reset.")]
    [SerializeField] private Transform resetHandAnchorOverride;

    [Tooltip("Additional distance from the hand when resetting (meters). Should be > grabber.attachDistance to require re-grab.")]
    [SerializeField] private float resetHandForwardMeters = 0.09f;

    [Tooltip("Additional vertical offset when resetting near hand (meters). Negative moves block down.")]
    [SerializeField] private float resetHandUpMeters = -0.02f;

    [Tooltip("Additional right offset when resetting near hand (meters).")]
    [SerializeField] private float resetHandRightMeters = 0.00f;

    [Tooltip("If resetNearHand is false and this is assigned, reset block to this transform position.")]
    [SerializeField] private Transform fixedBlockStartPose;

    [Tooltip("If resetNearHand is false and fixedBlockStartPose is not assigned, reset block to trial-start position.")]
    [SerializeField] private bool resetBlockToTrialStartPos = true;

    [Header("Target yaw sampling")]
    [SerializeField] private float yawMinDeg = 30f;
    [SerializeField] private float yawMaxDeg = 90f;
    [SerializeField] private bool randomizeYawSign = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = false; // FlowController 권장
    [SerializeField] private string startKeyword = "start rotation";
    [SerializeField] private string restartKeyword = "restart rotation";
    [SerializeField] private bool autoStartInEditor = false;

    [Header("Grabber rotation policy")]
    [Tooltip("During rotation trials, the task manager controls block rotation and the grabber must not override rotation.")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode rotationTrialGrabberMode = ProxyHandGrabber.HeldRotationMode.ExternalControl;

    [Tooltip("When this manager stops/gets disabled, restore grabber to this rotation mode.")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode restoreGrabberModeOnDisable = ProxyHandGrabber.HeldRotationMode.LockAtGrab;

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

    // Used when resetNearHand is false
    private Vector3 _trialStartBlockPosWorld;

    // Fixed target position state
    private bool _fixedTargetCaptured = false;
    private Vector3 _fixedTargetPosWorld;

    // Voice
    private KeywordRecognizer keywordRecognizer;
    private Dictionary<string, System.Action> keywordActions;

    public void StartBlock()
    {
        StopAllCoroutines();
        inTransition = false;
        trialRunning = false;
        trialIndex = 0;

        EnsureBlockBody();
        EnsureFixedTarget();

        BeginNextTrial();
    }

    private void Start()
    {
        HideFeedbackUI();

        EnsureBlockBody();
        EnsureFixedTarget();

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

    private void EnsureFixedTarget()
    {
        if (!lockTargetPosition) return;
        if (_fixedTargetCaptured) return;

        if (fixedTargetPose != null)
        {
            _fixedTargetPosWorld = fixedTargetPose.position;
            _fixedTargetCaptured = true;
            return;
        }

        if (targetSlotRoot != null)
        {
            _fixedTargetPosWorld = targetSlotRoot.position;
            _fixedTargetCaptured = true;
        }
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

        if (ShouldDriveRotationThisFrame())
        {
            UpdateYawFromTwist();
        }
        else
        {
            dwellTimer = 0f;
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

        // Record trial-start block pos (fallback reset path)
        _trialStartBlockPosWorld = blockRoot.position;

        // Target position policy
        if (lockTargetPosition)
        {
            EnsureFixedTarget();
            targetSlotRoot.position = _fixedTargetPosWorld;
        }
        else
        {
            targetSlotRoot.position = blockRoot.position;
        }

        // Record start yaw and baseline twist
        startYawDeg = blockRoot.eulerAngles.y;
        twistBaselineDeg = remoteHand.TwistDegrees;
        yawCmdDeg = startYawDeg;
        yawCmdInit = true;

        // Sample target yaw offset and apply to target visual (yaw-only)
        float offset = Random.Range(yawMinDeg, yawMaxDeg);
        if (randomizeYawSign && Random.value < 0.5f) offset = -offset;

        float targetYaw = startYawDeg + offset;
        targetSlotVisual.rotation = Quaternion.Euler(0f, targetYaw, 0f);

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        Debug.Log($"[LegoRotationTaskManager] Trial {trialIndex + 1}/{totalTrials} " +
                  $"targetYawOffset={offset:F1}deg tol={yawToleranceDeg:F1}deg " +
                  $"targetPosFixed={lockTargetPosition}");
    }

    private void UpdateYawFromTwist()
    {
        float twistNow = remoteHand.TwistDegrees;
        float dTwist = twistNow - twistBaselineDeg;

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
            // 1) Release first so the block is not parented to the hand when moved.
            ForceReleaseIfPossible();

            // 2) Reposition block
            if (resetNearHand)
            {
                Transform hand = null;

                if (resetHandAnchorOverride != null) hand = resetHandAnchorOverride;
                else if (grabber != null && grabber.grabAnchor != null) hand = grabber.grabAnchor;

                if (hand != null)
                {
                    // Ensure the reset distance is larger than attachDistance so the user must re-grab.
                    float minDist = resetHandForwardMeters;
                    if (grabber != null)
                        minDist = Mathf.Max(minDist, grabber.attachDistance + 0.02f);

                    Vector3 pos =
                        hand.position +
                        hand.forward * minDist +
                        hand.up * resetHandUpMeters +
                        hand.right * resetHandRightMeters;

                    blockRoot.position = pos;
                }
                else
                {
                    // Fallback if no hand anchor is available
                    blockRoot.position = _trialStartBlockPosWorld;
                }
            }
            else
            {
                if (fixedBlockStartPose != null)
                {
                    blockRoot.position = fixedBlockStartPose.position;
                }
                else if (resetBlockToTrialStartPos)
                {
                    blockRoot.position = _trialStartBlockPosWorld;
                }
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
