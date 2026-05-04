using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class PhoneMarkerTrackerfromMac : MonoBehaviour
{
    [SerializeField] private ARTrackedImageManager trackedImageManager;

    public bool MarkerVisible { get; private set; }
    public string MarkerName { get; private set; } = "";
    public Vector3 MarkerPosition { get; private set; }
    public Quaternion MarkerRotation { get; private set; }

    void Awake()
    {
        if (trackedImageManager == null)
            trackedImageManager = GetComponent<ARTrackedImageManager>();
    }

    void OnEnable()
    {
        if (trackedImageManager != null)
            trackedImageManager.trackedImagesChanged += OnTrackedImagesChanged;
    }

    void OnDisable()
    {
        if (trackedImageManager != null)
            trackedImageManager.trackedImagesChanged -= OnTrackedImagesChanged;
    }

    private void OnTrackedImagesChanged(ARTrackedImagesChangedEventArgs args)
    {
        // added/updated 둘 다 처리
        Process(args.added);
        Process(args.updated);

        // removed 처리(필요하면)
        if (args.removed.Count > 0)
        {
            // 제거되면 보이는 마커가 없는 걸로 처리
            MarkerVisible = false;
            MarkerName = "";
        }
    }

    private void Process(System.Collections.Generic.List<ARTrackedImage> list)
    {
        foreach (var img in list)
        {
            bool tracking = img.trackingState == TrackingState.Tracking;
            if (!tracking) continue;

            MarkerVisible = true;
            MarkerName = img.referenceImage.name;
            MarkerPosition = img.transform.position;
            MarkerRotation = img.transform.rotation;
            return; // QR 1개만 쓰는 전제
        }

        // 업데이트 리스트에 tracking 중인 게 없으면 visible=false로 내림(보수적)
        MarkerVisible = false;
    }
}
