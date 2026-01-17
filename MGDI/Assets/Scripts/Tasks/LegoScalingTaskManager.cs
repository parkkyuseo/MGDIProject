using System;
using System.Collections;
using UnityEngine;
using Random = UnityEngine.Random;

public class LegoScalingTaskManager : MonoBehaviour
{
    // Fired when the scaling block finishes (all trials complete).
    public event Action OnBlockFinished;
    public event System.Action<int, int> OnTrialChanged; // (current1Based, total)

    [Header("References")]
    [Tooltip("The transform that scales (use LegoBlockRoot).")]
    [SerializeField] private Transform blockRoot;

    [Tooltip("Optional target visual that shows the desired scale (set its localScale).")]
    [SerializeField] private Transform targetPivot;

    [Tooltip("ProxyHandGrabber instance (preferred). Used for gating and force release.")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Tooltip("RemoteHandRuntime that provides remote joint transforms (smoothed).")]
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

    [Header("Scale Success Threshold")]
    [Tooltip("Success if abs(currentUniformScale - targetScale) <= tolerance.")]
    [SerializeField] private float scaleTolerance = 0.05f;

    [Header("Target Scale Sampling")]
    [SerializeField] private float targetScaleMin = 0.70f;
    [SerializeField] private float targetScaleMax = 1.60f;

    [Header("Control: wrist diagonal movement -> scale")]
    [Tooltip("Axis = normalize(cam.up + cam.forward). If false, uses world (Vector3.up + Vector3.forward).")]
    [SerializeField] private bool axisFromCamera = true;

    [Tooltip("Scale mapping gain for exp(gain * deltaMeters). Typical 3~6.")]
    [SerializeField] private float moveToScaleGain = 4.0f;

    [Tooltip("Ignore very small axis movement (meters).")]
    [SerializeField] private float moveDeadZoneMeters = 0.002f;

    [Tooltip("Max scale change speed (scale units per second) for safety. 0 = no limit.")]
    [SerializeField] private float maxScaleRatePerSec = 3.0f;

    [Range(0f, 1f)]
    [Tooltip("Extra smoothing on commanded scale (0=no smoothing, 1=very slow).")]
    [SerializeField] private float scaleLerp = 0.15f;

    [Header("Scale clamp (safety)")]
    [SerializeField] private float minScale = 0.50f;
    [SerializeField] private float maxScale = 2.00f;

    [Header("Scaling gating")]
    [SerializeField] private bool scaleOnlyWhenHolding = true;

    [Tooltip("If true, drive scaling only when the grabber is holding THIS block rigidbody.")]
    [SerializeField] private bool requireHoldingThisBlock = true;

    [Header("Inter-trial behavior")]
    [SerializeField] private bool snapOnSuccess = true;
    [SerializeField] private float postSnapHoldSeconds = 0.35f;

    [SerializeField] private bool resetScaleAfterTrial = true;
    [SerializeField] private bool forceReleaseAfterTrial = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    // Runtime
    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;

    private bool trialRunning = false;
    private bool inTransition = false;

    private float _targetScale = 1f;
    private Vector3 _trialStartLocalScale = Vector3.one;

    // drive state
    private bool _prevEffectiveDriving = false;
    private Vector3 _wristPrev;
    private bool _haveWristPrev = false;

    // command smoothing
    private float _scaleCmd = 1f;
    private bool _scaleCmdInit = false;

    private Rigidbody _blockBody;

    public bool IsTrialRunning => trialRunning && !inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => trialIndex + 1;

    public void StartBlock()
    {
        StopAllCoroutines();
        inTransition = false;
        trialRunning = false;
        trialIndex = 0;

        _prevEffectiveDriving = false;
        _haveWristPrev = false;
        _scaleCmdInit = false;

        EnsureBlockBody();
        HideFeedbackUI();

        BeginNextTrial();
    }

    private void Start()
    {
        HideFeedbackUI();
        EnsureBlockBody();
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

        bool drive = ShouldDriveScalingThisFrame();
        bool ready = HasWrist();
        bool effectiveDriving = drive && ready;

        if (effectiveDriving)
        {
            if (!_prevEffectiveDriving)
                RebaselineWrist();

            UpdateScaleFromWristDiagonal();
        }
        else
        {
            dwellTimer = 0f;
        }

        _prevEffectiveDriving = effectiveDriving;

        float err = Mathf.Abs(GetUniformScale(blockRoot.localScale) - _targetScale);
        if (err <= scaleTolerance)
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

    private bool ShouldDriveScalingThisFrame()
    {
        if (!scaleOnlyWhenHolding) return true;

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
        if (blockRoot == null)
        {
            Debug.LogError("[LegoScalingTaskManager] Missing reference: blockRoot.");
            FinishBlock();
            return;
        }

        if (remoteHand == null)
        {
            Debug.LogError("[LegoScalingTaskManager] remoteHand is null (RemoteHandRuntime).");
            FinishBlock();
            return;
        }

        if (totalTrials > 0 && trialIndex >= totalTrials)
        {
            Debug.Log("[LegoScalingTaskManager] Block finished.");
            FinishBlock();
            return;
        }

        _trialStartLocalScale = blockRoot.localScale;

        _targetScale = Random.Range(Mathf.Min(targetScaleMin, targetScaleMax), Mathf.Max(targetScaleMin, targetScaleMax));
        if (targetPivot != null)
            targetPivot.localScale = Vector3.one * _targetScale;

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        _prevEffectiveDriving = false;
        _haveWristPrev = false;
        _scaleCmdInit = false;

        OnTrialChanged?.Invoke(trialIndex + 1, totalTrials);

        Debug.Log($"[LegoScalingTaskManager] Trial {trialIndex + 1}/{totalTrials} targetScale={_targetScale:F2} tol={scaleTolerance:F3}");
    }

    private bool HasWrist()
    {
        return remoteHand != null &&
               remoteHand.remoteByIndex != null &&
               remoteHand.remoteByIndex.Length > 0 &&
               remoteHand.remoteByIndex[0] != null;
    }

    private Vector3 GetAxis()
    {
        Vector3 axis = Vector3.up + Vector3.forward;

        if (axisFromCamera && Camera.main != null)
        {
            Transform cam = Camera.main.transform;
            axis = cam.up + cam.forward;
        }

        if (axis.sqrMagnitude < 1e-8f) axis = Vector3.up;
        return axis.normalized;
    }

    private void RebaselineWrist()
    {
        Vector3 w = remoteHand.remoteByIndex[0].position;
        _wristPrev = w;
        _haveWristPrev = true;

        float s0 = Mathf.Clamp(GetUniformScale(blockRoot.localScale), minScale, maxScale);
        _scaleCmd = s0;
        _scaleCmdInit = true;
    }

    private void UpdateScaleFromWristDiagonal()
    {
        if (!HasWrist()) return;

        Vector3 w = remoteHand.remoteByIndex[0].position;

        if (!_haveWristPrev)
        {
            _wristPrev = w;
            _haveWristPrev = true;
            return;
        }

        Vector3 dp = w - _wristPrev;
        _wristPrev = w;

        Vector3 axis = GetAxis();
        float delta = Vector3.Dot(dp, axis); // meters along axis

        if (Mathf.Abs(delta) < Mathf.Max(0f, moveDeadZoneMeters))
            return;

        float s0 = _scaleCmdInit ? _scaleCmd : GetUniformScale(blockRoot.localScale);
        s0 = Mathf.Clamp(s0, minScale, maxScale);

        float desired = s0 * Mathf.Exp(moveToScaleGain * delta);
        desired = Mathf.Clamp(desired, minScale, maxScale);

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        if (maxScaleRatePerSec > 0f)
        {
            float maxStep = maxScaleRatePerSec * dt;
            float dS = desired - s0;
            dS = Mathf.Clamp(dS, -maxStep, maxStep);
            desired = s0 + dS;
        }

        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(scaleLerp), dt * 60f);
        _scaleCmd = Mathf.Lerp(s0, desired, k);

        blockRoot.localScale = Vector3.one * _scaleCmd;
        _scaleCmdInit = true;
    }

    private static float GetUniformScale(Vector3 s)
    {
        return (s.x + s.y + s.z) / 3f;
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (inTransition) yield break;
        inTransition = true;
        trialRunning = false;

        HideFeedbackUI();

        if (success)
        {
            if (forceReleaseAfterTrial) ForceReleaseIfPossible();

            if (snapOnSuccess)
            {
                blockRoot.localScale = Vector3.one * _targetScale;
                _scaleCmd = _targetScale;
                _scaleCmdInit = true;
            }

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

        if (resetScaleAfterTrial)
            blockRoot.localScale = _trialStartLocalScale;

        if (forceReleaseAfterTrial) ForceReleaseIfPossible();

        trialIndex++;
        inTransition = false;
        BeginNextTrial();
    }

    private void ForceReleaseIfPossible()
    {
        if (grabber != null)
            grabber.ForceRelease();
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

    private void FinishBlock()
    {
        trialRunning = false;
        inTransition = false;
        HideFeedbackUI();

        try { OnBlockFinished?.Invoke(); } catch { /* ignore */ }
    }
}
