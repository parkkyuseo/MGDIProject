using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Microsoft.MixedReality.OpenXR;
using System.Threading.Tasks;

public class PersistAnchorLoaderSafe : MonoBehaviour
{
    [Header("Required")]
    public ARAnchorManager anchorManager;

    public static bool AnchorReady;

    const string KeyPersistedName = "PersistedAnchorName";

    bool _loadRequested;
    bool _loaded;
    TrackableId _expectedId = TrackableId.invalidId; // 필터링용

    void OnEnable()
    {
        if (anchorManager != null)
            anchorManager.anchorsChanged += OnAnchorsChanged;  // UWP 가드 불필요 (핸들러 항상 존재)
    }

    void OnDisable()
    {
        if (anchorManager != null)
            anchorManager.anchorsChanged -= OnAnchorsChanged;
        AnchorRuntime.AnchorReady = false;
    }

    async void Start()
    {
        var name = PlayerPrefs.GetString(KeyPersistedName, "");
        if (string.IsNullOrEmpty(name))
        {
            Log("[AnchorLoad] No persisted name. Run calibration (persist) first.");
            return;
        }

#if WINDOWS_UWP
        if (anchorManager == null || anchorManager.subsystem == null)
        {
            Log("[AnchorLoad] ARAnchorManager / XRAnchorSubsystem not ready.");
            return;
        }

        // XR 세션 초기화 전에는 null이 올 수 있음 → 문서 권고대로 세션 초기화 후 재시도 필요
        var store = await XRAnchorStore.LoadAnchorStoreAsync(anchorManager.subsystem);
        if (store == null)
        {
            Log("[AnchorLoad] XRAnchorStore not available yet. Try again later (XR session not initialized).");
            return;
        }

        _loadRequested = true;

        // UWP 메인 스레드에서 호출
        UnityEngine.WSA.Application.InvokeOnAppThread(() =>
        {
            try
            {
                // LoadAnchor는 다음 프레임에 앵커를 생성하고,
                // 그 앵커가 가질 TrackableId를 미리 반환한다.
                _expectedId = store.LoadAnchor(name);
                Log($"[AnchorLoad] Requested load '{name}', expected id={_expectedId}");
            }
            catch (System.Exception ex)
            {
                Log("[AnchorLoad] LoadAnchor exception: " + ex.Message);
            }
        }, false);
#else
        Log("[AnchorLoad] UWP-only API.");
#endif
    }

    // 중요: 이 핸들러는 항상 컴파일되어야 한다 (플랫폼 가드 금지)
    void OnAnchorsChanged(ARAnchorsChangedEventArgs e)
    {
        if (!_loadRequested || _loaded) return;
        if (e.added == null || e.added.Count == 0) return;

        // 예상 TrackableId가 있으면 그것만 집는다. 없으면 첫 번째를 사용.
        ARAnchor found = null;
        if (_expectedId != TrackableId.invalidId)
        {
            foreach (var a in e.added)
            {
                if (a.trackableId == _expectedId) { found = a; break; }
            }
        }
        if (found == null) found = e.added[0];

        AnchorReady = true; DebugHUD.Log("[Anchor] ready");

        _loaded = true;
        AnchorRuntime.AnchorReady = true;
        Log($"[AnchorLoad] Anchor instantiated (id={found.trackableId}).");
        // TODO: 필요한 콘텐츠를 found.transform에 붙이면 됨
    }

    void Log(string s)
    {
        Debug.Log(s);
        //try { DebugHUD.Log(s); } catch { }
    }
}
