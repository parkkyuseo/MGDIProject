using System;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.OpenXR;

public class QRWorkspaceLock_OpenXR : MonoBehaviour
{
    public enum MarkerYawAxisMode
    {
        AutoBestHorizontal = 0,
        Forward = 1,
        Right = 2,
        Up = 3
    }

    public bool IsWorkspaceReady { get; private set; } = false;
    public event Action OnWorkspaceReady;

    [Header("Refs")]
    public ARMarkerManager markerManager;
    public Transform workshopEnvironment; // WorkshopEnvironment root

    [Header("Lock")]
    public bool lockOnce = true;
    public bool applyRotation = true;
    public bool zeroOutPitchRoll = true;
    public Vector3 localOffset = Vector3.zero;

    [Header("Stabilize (sample & average before locking)")]
    [Tooltip("Samples are collected for this duration, then averaged.")]
    public float settleSeconds = 0.45f;

    [Tooltip("Minimum samples required to lock.")]
    public int minSamples = 10;

    [Tooltip("If true, uses UPDATED markers first. Added markers are used as fallback.")]
    public bool preferUpdatedSamples = true;

    [Tooltip("If true, lock is allowed only after reaching minSamples.")]
    public bool requireMinSamplesForLock = false;

    [Tooltip("If true, timeout lock with fewer samples is allowed when requireMinSamplesForLock is false.")]
    public bool allowTimeoutLockWithFewSamples = true;

    [Header("Optional")]
    public RemoteHandRuntime remoteHandRuntime;

    [Header("Yaw Override")]
    public bool forceYawToCamera = true;
    [Tooltip("If true, marker-derived yaw is preferred even when forceYawToCamera is enabled.")]
    public bool preferMarkerYawWhenAvailable = true;
    [Tooltip("Marker axis used to derive yaw. Auto picks the most horizontal axis each sample.")]
    public MarkerYawAxisMode markerYawAxisMode = MarkerYawAxisMode.Right;
    public float yawOffsetDeg = 0f;

    private bool _locked = false;
    private bool _workspaceReadyFired = false;

    // Sampling state
    private bool _sampling = false;
    private float _sampleEndTime = -1f;
    private readonly List<Vector3> _posSamples = new List<Vector3>(128);
    private readonly List<float> _yawSamplesDeg = new List<float>(128);

    void Awake()
    {
        if (markerManager == null) markerManager = GetComponent<ARMarkerManager>();

        if (workshopEnvironment == null)
        {
            var go = GameObject.Find("WorkshopEnvironment");
            if (go != null) workshopEnvironment = go.transform;
        }
    }

    void OnEnable()
    {
        if (markerManager == null) markerManager = GetComponent<ARMarkerManager>();
        if (markerManager != null) markerManager.markersChanged += OnMarkersChanged;
    }

    void OnDisable()
    {
        if (markerManager != null) markerManager.markersChanged -= OnMarkersChanged;
    }

    void Update()
    {
        // Complete lock even if marker callbacks pause after initial detection.
        if (!_sampling) return;
        if (_locked && lockOnce) return;
        if (workshopEnvironment == null) return;

        TryFinishSamplingWindow();
    }

    private void OnMarkersChanged(ARMarkersChangedEventArgs args)
    {
        if (_locked && lockOnce) return;
        if (workshopEnvironment == null) return;

        if (!_sampling)
        {
            bool hasAny =
                (args.updated != null && args.updated.Count > 0) ||
                (args.added != null && args.added.Count > 0);

            if (hasAny)
                StartSampling();
        }

        if (!_sampling) return;

        if (preferUpdatedSamples)
        {
            if (!TrySampleFromList(args.updated))
                TrySampleFromList(args.added);
        }
        else
        {
            if (!TrySampleFromList(args.updated))
                TrySampleFromList(args.added);
        }

        // Early lock when enough samples arrive before timeout.
        TryFinishSamplingWindow();
    }

    private void StartSampling()
    {
        _sampling = true;
        _sampleEndTime = Time.unscaledTime + Mathf.Max(0.05f, settleSeconds);
        _posSamples.Clear();
        _yawSamplesDeg.Clear();
        Log("[Workspace] sampling...");
    }

    private void TryFinishSamplingWindow()
    {
        int needed = Mathf.Max(1, minSamples);
        bool enoughSamples = _posSamples.Count >= needed;
        bool timedOut = Time.unscaledTime >= _sampleEndTime;

        bool canLock = false;
        if (enoughSamples)
        {
            canLock = true;
        }
        else if (timedOut)
        {
            if (!requireMinSamplesForLock && allowTimeoutLockWithFewSamples && _posSamples.Count > 0)
                canLock = true;
        }

        if (!canLock)
            return;

        ApplyAveragedWorkspacePose();
        _locked = true;
        _sampling = false;

        Log($"[Workspace] locked (averaged) samples={_posSamples.Count}");
        HandleRemoteHandAfterWorkspaceJump();
        NotifyWorkspaceReadyOnce();
    }

    private bool TrySampleFromList(IReadOnlyList<ARMarker> list)
    {
        if (list == null || list.Count == 0) return false;

        var marker = list[0];
        if (marker == null) return false;

        Vector3 pos = marker.transform.position;
        Quaternion rot = marker.transform.rotation;

        float yawDeg = ExtractYawDeg(rot);
        _posSamples.Add(pos);
        _yawSamplesDeg.Add(yawDeg);

        return true;
    }

    private float ExtractYawDeg(Quaternion rot)
    {
        if (!applyRotation)
            return workshopEnvironment != null ? workshopEnvironment.rotation.eulerAngles.y : 0f;

        if (!TryExtractMarkerYawDeg(rot, out float yawDeg))
            return workshopEnvironment != null ? workshopEnvironment.rotation.eulerAngles.y : 0f;

        return yawDeg;
    }

    private bool TryExtractMarkerYawDeg(Quaternion rot, out float yawDeg)
    {
        switch (markerYawAxisMode)
        {
            case MarkerYawAxisMode.Forward:
                return TryExtractYawFromAxis(rot * Vector3.forward, out yawDeg);
            case MarkerYawAxisMode.Right:
                return TryExtractYawFromAxis(rot * Vector3.right, out yawDeg);
            case MarkerYawAxisMode.Up:
                return TryExtractYawFromAxis(rot * Vector3.up, out yawDeg);
            default:
                return TryExtractBestHorizontalYaw(rot, out yawDeg);
        }
    }

    private static bool TryExtractBestHorizontalYaw(Quaternion rot, out float yawDeg)
    {
        Vector3[] candidates =
        {
            rot * Vector3.forward,
            rot * Vector3.right,
            rot * Vector3.up
        };

        Vector3 best = Vector3.zero;
        float bestMag = 0f;

        for (int i = 0; i < candidates.Length; i++)
        {
            Vector3 p = candidates[i];
            p.y = 0f;
            float mag = p.sqrMagnitude;
            if (mag > bestMag)
            {
                best = p;
                bestMag = mag;
            }
        }

        if (bestMag < 1e-6f)
        {
            yawDeg = 0f;
            return false;
        }

        return TryExtractYawFromAxis(best, out yawDeg);
    }

    private static bool TryExtractYawFromAxis(Vector3 axisWorld, out float yawDeg)
    {
        axisWorld.y = 0f;
        if (axisWorld.sqrMagnitude < 1e-6f)
        {
            yawDeg = 0f;
            return false;
        }

        axisWorld.Normalize();
        yawDeg = Mathf.Atan2(axisWorld.x, axisWorld.z) * Mathf.Rad2Deg;
        return true;
    }

    private void ApplyAveragedWorkspacePose()
    {
        Vector3 posAvg = Vector3.zero;
        for (int i = 0; i < _posSamples.Count; i++)
            posAvg += _posSamples[i];
        posAvg /= Mathf.Max(1, _posSamples.Count);

        float sumSin = 0f;
        float sumCos = 0f;
        for (int i = 0; i < _yawSamplesDeg.Count; i++)
        {
            float rad = _yawSamplesDeg[i] * Mathf.Deg2Rad;
            sumSin += Mathf.Sin(rad);
            sumCos += Mathf.Cos(rad);
        }

        float meanRad = Mathf.Atan2(sumSin, sumCos);
        float yawDeg = meanRad * Mathf.Rad2Deg;

        bool useCameraYaw = applyRotation && forceYawToCamera && Camera.main != null;
        bool allowCameraYawOverride = useCameraYaw && !preferMarkerYawWhenAvailable;

        if (allowCameraYawOverride)
        {
            Vector3 fwd = -Camera.main.transform.forward;
            fwd.y = 0f;
            if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
            fwd.Normalize();
            yawDeg = Mathf.Atan2(fwd.x, fwd.z) * Mathf.Rad2Deg;
        }

        yawDeg += yawOffsetDeg;

        Quaternion rotAvg = workshopEnvironment.rotation;
        if (applyRotation)
            rotAvg = Quaternion.Euler(0f, yawDeg, 0f);

        Vector3 worldOffset = rotAvg * localOffset;
        workshopEnvironment.SetPositionAndRotation(posAvg + worldOffset, rotAvg);
    }

    private void HandleRemoteHandAfterWorkspaceJump()
    {
        if (remoteHandRuntime == null) return;

        bool canRecaptureNow = (remoteHandRuntime.SampleId > 0) && (remoteHandRuntime.rWrist != null);

        if (canRecaptureNow)
            remoteHandRuntime.ContextRecaptureNow();
        else
            remoteHandRuntime.ContextClearAndRearm();
    }

    private void NotifyWorkspaceReadyOnce()
    {
        if (_workspaceReadyFired) return;

        _workspaceReadyFired = true;
        IsWorkspaceReady = true;
        try { OnWorkspaceReady?.Invoke(); } catch { }
    }

    void Log(string msg)
    {
        string line = "[QRWorkspace] " + msg;
        Debug.Log(line);
        try { DebugHUD.Log(line); } catch { }
    }
}
