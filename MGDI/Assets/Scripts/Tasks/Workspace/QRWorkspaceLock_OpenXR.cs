using System;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.OpenXR;

public class QRWorkspaceLock_OpenXR : MonoBehaviour
{
    [Header("Refs")]
    public ARMarkerManager markerManager;
    public Transform workshopEnvironment; // WorkshopEnvironment 루트

    [Header("Lock")]
    public bool lockOnce = true;
    public bool applyRotation = true;
    public bool zeroOutPitchRoll = true;  // 테이블 수평 가정이면 추천
    public Vector3 localOffset = Vector3.zero; // QR 중심에서 오프셋(미터)

    [Header("Stabilize (sample & average before locking)")]
    [Tooltip("Samples are collected for this duration, then averaged.")]
    public float settleSeconds = 0.45f;

    [Tooltip("Minimum samples required to lock.")]
    public int minSamples = 10;

    [Tooltip("If true, uses only UPDATED markers for samples (more stable than ADDED).")]
    public bool preferUpdatedSamples = true;

    [Header("Optional")]
    public RemoteHandRuntime remoteHandRuntime;

    [Header("Yaw Override")]
    public bool forceYawToCamera = true;     // 켜면 QR yaw 무시하고 카메라 yaw로 고정
    public float yawOffsetDeg = 0f;          // 필요하면 몇 도 보정(±)

    private bool _locked = false;

    // sampling state
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

    private void OnMarkersChanged(ARMarkersChangedEventArgs args)
    {
        if (_locked && lockOnce) return;
        if (workshopEnvironment == null) return;

        // Sampling flow:
        // 1) When a marker is seen, start sampling window.
        // 2) Keep accumulating samples from UPDATED (preferred) or ADDED+UPDATED.
        // 3) When window ends and enough samples exist, lock once using averaged pose.
        if (!_sampling)
        {
            // trigger sampling only when any marker is visible
            if ((args.updated != null && args.updated.Count > 0) || (args.added != null && args.added.Count > 0))
                StartSampling();
        }

        if (!_sampling) return;

        // collect samples
        if (preferUpdatedSamples)
        {
            TrySampleFromList(args.updated);
        }
        else
        {
            // if updated is empty early, allow added too
            if (!TrySampleFromList(args.updated))
                TrySampleFromList(args.added);
        }

        // finish sampling -> lock
/*         if (Time.unscaledTime >= _sampleEndTime && _posSamples.Count >= minSamples)
 *         {
 *             ApplyAveragedWorkspacePose();
 *             _locked = true;
 *             _sampling = false;
 *
 *             Log("[Workspace] locked (averaged)");
 *
 *             HandleRemoteHandAfterWorkspaceJump();
 *         } */
        // 기존: 시간이 끝나고 + 샘플 충분할 때만 lock
        // above
        // 개선: 시간이 끝났거나, 샘플이 충분하면 즉시 lock
        if ((_posSamples.Count >= minSamples) || (Time.unscaledTime >= _sampleEndTime))
        {
            if (_posSamples.Count >= 1) // 최소 1개는 있어야
            {
                ApplyAveragedWorkspacePose();
                _locked = true;
                _sampling = false;
                Log("[Workspace] locked (averaged)");
                HandleRemoteHandAfterWorkspaceJump();
            }
        }
    }

    private void StartSampling()
    {
        _sampling = true;
        _sampleEndTime = Time.unscaledTime + Mathf.Max(0.05f, settleSeconds);
        _posSamples.Clear();
        _yawSamplesDeg.Clear();
        Log("[Workspace] sampling...");
    }

    private bool TrySampleFromList(IReadOnlyList<ARMarker> list)
    {
        if (list == null || list.Count == 0) return false;

        // Single QR assumption: use the first marker in the list
        var m = list[0];
        if (m == null) return false;

        Vector3 pos = m.transform.position;
        Quaternion rot = m.transform.rotation;

        float yawDeg = ExtractYawDeg(rot);
        _posSamples.Add(pos);
        _yawSamplesDeg.Add(yawDeg);

        return true;
    }

    private float ExtractYawDeg(Quaternion rot)
    {
        if (!applyRotation) return workshopEnvironment != null ? workshopEnvironment.rotation.eulerAngles.y : 0f;

        if (!zeroOutPitchRoll)
        {
            // full rotation requested; still store yaw from full rot for averaging yaw-only lock
            Vector3 fwd = rot * Vector3.forward;
            fwd.y = 0f;
            if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
            return Mathf.Atan2(fwd.x, fwd.z) * Mathf.Rad2Deg;
        }

        // yaw-only from marker forward projected onto horizontal plane
        Vector3 f = rot * Vector3.forward;
        f.y = 0f;
        if (f.sqrMagnitude < 1e-6f) f = Vector3.forward;
        f.Normalize();
        return Mathf.Atan2(f.x, f.z) * Mathf.Rad2Deg;
    }

    private void ApplyAveragedWorkspacePose()
    {
        // average position
        Vector3 posAvg = Vector3.zero;
        for (int i = 0; i < _posSamples.Count; i++) posAvg += _posSamples[i];
        posAvg /= Mathf.Max(1, _posSamples.Count);

        // circular mean yaw (handles wrap-around)
        float sumSin = 0f, sumCos = 0f;
        for (int i = 0; i < _yawSamplesDeg.Count; i++)
        {
            float rad = _yawSamplesDeg[i] * Mathf.Deg2Rad;
            sumSin += Mathf.Sin(rad);
            sumCos += Mathf.Cos(rad);
        }

        float meanRad = Mathf.Atan2(sumSin, sumCos);
        float meanYawDeg = meanRad * Mathf.Rad2Deg;

        // choose yaw
        float yawDeg = meanYawDeg;

        if (applyRotation && forceYawToCamera && Camera.main != null)
        {
            Vector3 fwd = Camera.main.transform.forward;
            fwd.y = 0f;
            if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
            fwd.Normalize();
            yawDeg = Mathf.Atan2(fwd.x, fwd.z) * Mathf.Rad2Deg;
        }

        yawDeg += yawOffsetDeg;

        Quaternion rotAvg = workshopEnvironment.rotation;
        if (applyRotation)
        {
            // yaw-only (pitch/roll removed)
            rotAvg = Quaternion.Euler(0f, yawDeg, 0f);
        }

        Vector3 worldOffset = rotAvg * localOffset;
        workshopEnvironment.SetPositionAndRotation(posAvg + worldOffset, rotAvg);
    }
/*     private void ApplyAveragedWorkspacePose()
 *     {
 *         // average position
 *         Vector3 posAvg = Vector3.zero;
 *         for (int i = 0; i < _posSamples.Count; i++) posAvg += _posSamples[i];
 *         posAvg /= Mathf.Max(1, _posSamples.Count);
 *
 *         // circular mean yaw (handles wrap-around)
 *         float sumSin = 0f, sumCos = 0f;
 *         for (int i = 0; i < _yawSamplesDeg.Count; i++)
 *         {
 *             float rad = _yawSamplesDeg[i] * Mathf.Deg2Rad;
 *             sumSin += Mathf.Sin(rad);
 *             sumCos += Mathf.Cos(rad);
 *         }
 *
 *         float meanRad = Mathf.Atan2(sumSin, sumCos);
 *         float meanYawDeg = meanRad * Mathf.Rad2Deg;
 *
 *         Quaternion rotAvg = workshopEnvironment.rotation;
 *         if (applyRotation)
 *         {
 *             // yaw-only lock (pitch/roll removed)
 *             rotAvg = Quaternion.Euler(0f, meanYawDeg, 0f);
 *         }
 *
 *         Vector3 worldOffset = rotAvg * localOffset;
 *         workshopEnvironment.SetPositionAndRotation(posAvg + worldOffset, rotAvg);
 *     } */

    private void HandleRemoteHandAfterWorkspaceJump()
    {
        if (remoteHandRuntime == null) return;

        bool canRecaptureNow = (remoteHandRuntime.SampleId > 0) && (remoteHandRuntime.rWrist != null);

        if (canRecaptureNow)
            remoteHandRuntime.ContextRecaptureNow();
        else
            remoteHandRuntime.ContextClearAndRearm();
    }

    void Log(string msg)
    {
        Debug.Log("[QRWorkspace] " + msg);
        try { DebugHUD.Log("[QRWorkspace] " + msg); } catch { }
    }
}
