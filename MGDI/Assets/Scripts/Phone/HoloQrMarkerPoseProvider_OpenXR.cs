using UnityEngine;
using Microsoft.MixedReality.OpenXR;

public class HoloQrMarkerPoseProvider_OpenXR : MonoBehaviour
{
    [SerializeField] private ARMarkerManager markerManager;

    public bool MarkerVisible { get; private set; }
    public Pose MarkerPose { get; private set; }

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
        foreach (var m in args.added)  TryUse(m);
        foreach (var m in args.updated) TryUse(m);

        if (args.removed.Count > 0)
            MarkerVisible = false;
    }

    private void TryUse(ARMarker m)
    {
        if (m == null) return;

        MarkerVisible = true;
        MarkerPose = new Pose(m.transform.position, m.transform.rotation);
    }
}
