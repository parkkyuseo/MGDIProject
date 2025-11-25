using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Microsoft.MixedReality.OpenXR;
using System.Threading.Tasks;

public class AnchorOnLockPersistSafe : MonoBehaviour
{
    [Header("Required")]
    public ARMarkerManager markerManager;   // ARMarkerManager 직접 참조
    public ARAnchorManager anchorManager;   // ARAnchorManager 직접 참조

    [Header("Persist Settings")]
    public string persistedName = "LabA_QR1";
    const string KeyPersistedName = "PersistedAnchorName";

    bool _done;

    void OnEnable()
    {
#if WINDOWS_UWP
        if (!markerManager || !anchorManager) { Debug.Log("Assign ARMarkerManager/ARAnchorManager."); return; }

        // 이미 저장되어 있으면 스킵
        var saved = PlayerPrefs.GetString(KeyPersistedName, "");
        if (!string.IsNullOrEmpty(saved)) { Debug.Log($"Already saved: '{saved}'. Skip."); _done = true; return; }

        markerManager.markersChanged += OnMarkersChanged;   // 강타입 구독
        Debug.Log("Waiting QR lock... (will persist anchor once)");
#endif
    }

    void OnDisable()
    {
#if WINDOWS_UWP
        if (markerManager) markerManager.markersChanged -= OnMarkersChanged;
#endif
    }

#if WINDOWS_UWP
    // 강타입 이벤트 핸들러 (Action<ARMarkersChangedEventArgs>)
    void OnMarkersChanged(ARMarkersChangedEventArgs e)
    {
        if (_done || e.added == null || e.added.Count == 0) return;

        var m = e.added[0];

        // Unity API는 메인 스레드에서만!
        UnityEngine.WSA.Application.InvokeOnAppThread(async () =>
        {
            if (_done || m == null) return;

            // 1) 마커 포즈로 앵커 생성 (ARFoundation 표준)
            var pose = new Pose(m.transform.position, m.transform.rotation);
#if ARFOUNDATION_6_OR_NEWER
            var addRes = await anchorManager.TryAddAnchorAsync(pose);
            if (!addRes.success || addRes.value == null) { Debug.Log("TryAddAnchorAsync failed."); return; }
            var anchor = addRes.value;
#else
            var anchor = anchorManager.AddAnchor(pose);
            if (anchor == null) { Debug.Log("AddAnchor failed."); return; }
#endif

            // 2) 앵커 스토어 로드 & 퍼시스트
            var store = await XRAnchorStore.LoadAnchorStoreAsync(anchorManager.subsystem);
            store.UnpersistAnchor(persistedName); // 덮어쓰기 용
            if (!store.TryPersistAnchor(anchor.trackableId, persistedName))
            {
                Debug.Log($"Persist failed: {persistedName}");
                return;
            }

            PlayerPrefs.SetString(KeyPersistedName, persistedName);
            PlayerPrefs.Save();

            _done = true;
            Debug.Log($"[Anchor persisted] name='{persistedName}'");
        }, false); // false = 블로킹 안함
    }
#endif
}
