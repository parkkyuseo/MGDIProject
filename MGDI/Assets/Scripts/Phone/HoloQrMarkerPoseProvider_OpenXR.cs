using UnityEngine;
using Microsoft.MixedReality.OpenXR;

public class HoloQrMarkerPoseProvider_OpenXR : MonoBehaviour
{
    [SerializeField] private ARMarkerManager markerManager;

    public bool MarkerVisible { get; private set; }
    public Vector3 MarkerPosition { get; private set; }
    public Quaternion MarkerRotation { get; private set; }

    void OnEnable()
    {
        if (markerManager != null)
            markerManager.markersChanged += OnMarkersChanged;
    }

    void OnDisable()
    {
        if (markerManager != null)
            markerManager.markersChanged -= OnMarkersChanged;
    }

    private void OnMarkersChanged(ARMarkersChangedEventArgs args)
    {
        // 가장 단순: tracking 중인 첫 마커를 사용(마커 1개 전제)
        foreach (var m in args.added)
            TryUse(m);

        foreach (var m in args.updated)
            TryUse(m);

        if (args.removed.Count > 0)
            MarkerVisible = false;
    }

    private void TryUse(ARMarker m)
    {
        if (m == null) return;

        // Unity transform은 월드 좌표계 기준 포즈
        MarkerVisible = true;
        MarkerPosition = m.transform.position;
        MarkerRotation = m.transform.rotation;
    }
}
