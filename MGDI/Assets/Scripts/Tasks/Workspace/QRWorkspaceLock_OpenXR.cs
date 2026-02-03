using UnityEngine;
using Microsoft.MixedReality.OpenXR; // ARMarkerManager, ARMarker, ARMarkersChangedEventArgs

public class QRWorkspaceLock_OpenXR : MonoBehaviour
{
    [Header("Refs")]
    public ARMarkerManager markerManager;
    public Transform workshopEnvironment; // WorkshopEnvironment 루트

    [Header("Lock")]
    public bool lockOnce = true;
    public bool applyRotation = true;
    public bool zeroOutPitchRoll = true;  // 테이블이 수평이라고 가정하면 추천
    public Vector3 localOffset = Vector3.zero; // QR 중심에서 오프셋(미터)

    [Header("Optional")]
    public RemoteHandRuntime remoteHandRuntime;

    private bool _locked = false;

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
        markerManager.markersChanged += OnMarkersChanged;
    }

    void OnDisable()
    {
        if (markerManager != null) markerManager.markersChanged -= OnMarkersChanged;
    }

    private void OnMarkersChanged(ARMarkersChangedEventArgs args)
    {
        if (_locked && lockOnce) return;
        if (workshopEnvironment == null) return;

        // added가 없을 수도 있어서 updated도 같이 봄
        if (TryLockFromList(args.added)) return;
        TryLockFromList(args.updated);
    }

    private bool TryLockFromList(System.Collections.Generic.IReadOnlyList<ARMarker> list)
    {
        if (list == null || list.Count == 0) return false;

        // QR이 하나만 있다는 전제: 첫 번째로 들어온 마커를 사용
        var m = list[0];
        if (m == null) return false;

        ApplyWorkspacePose(m.transform);
        _locked = true;

        Log("[Workspace] locked (first QR)");

        // ✅ RemoteHandRuntime: WorkspaceAnchor 기반 리베이스는 제거됐으므로,
        // QR로 워크스페이스가 점프하면 "오프셋 재캡처/스무딩 리셋"으로 대응.
        HandleRemoteHandAfterWorkspaceJump();

        return true;
    }

    private void HandleRemoteHandAfterWorkspaceJump()
    {
        if (remoteHandRuntime == null) return;

        // RemoteHandRuntime이 이미 샘플을 받았고 rWrist가 있으면 즉시 오프셋 재캡처
        // (QR lock으로 리그가 이동/회전했으니, 손 정렬을 다시 맞추기 위함)
        bool canRecaptureNow = (remoteHandRuntime.SampleId > 0) && (remoteHandRuntime.rWrist != null);

        if (canRecaptureNow)
        {
            remoteHandRuntime.ContextRecaptureNow();
        }
        else
        {
            // 샘플이 아직 없거나 rWrist가 없으면, 다음 샘플에서 자동 캡처되도록 리셋
            remoteHandRuntime.ContextClearAndRearm();
        }
    }

    private void ApplyWorkspacePose(Transform markerTf)
    {
        Vector3 pos = markerTf.position;
        Quaternion rot = markerTf.rotation;

        if (applyRotation)
        {
            if (zeroOutPitchRoll)
            {
                // yaw만 남기기 (월드 up 기준)
                Vector3 fwd = rot * Vector3.forward;
                fwd.y = 0f;
                if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
                rot = Quaternion.LookRotation(fwd.normalized, Vector3.up);
            }
        }
        else
        {
            rot = workshopEnvironment.rotation;
        }

        Vector3 worldOffset = rot * localOffset;
        workshopEnvironment.SetPositionAndRotation(pos + worldOffset, rot);
    }

    void Log(string msg)
    {
        Debug.Log("[QRWorkspace] " + msg);
        try { DebugHUD.Log("[QRWorkspace] " + msg); } catch { }
    }
}
