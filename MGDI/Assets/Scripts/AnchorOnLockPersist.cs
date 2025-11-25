using UnityEngine;
using System;
using System.Reflection;
using System.Threading.Tasks;

public class AnchorOnLockPersist : MonoBehaviour 
{
    [Header("AR Marker System (required)")]
    // 씬의 ARMarkerManager가 붙어 있는 오브젝트를 드래그해 주세요
    public GameObject arMarkerSystem;

    [Header("Persist Settings")]
    public string persistedName = "LabA_QR1";       // 저장 키(앵커 이름)
    const string KeyPersistedName = "PersistedAnchorName";

    bool _done;  // 한 번만 실행

    void OnEnable()
    {
#if WINDOWS_UWP
        if (!arMarkerSystem)
        {
            Log("Assign ARMarkerSystem GameObject (with ARMarkerManager).");
            return;
        }
        var mm = arMarkerSystem.GetComponent("Microsoft.MixedReality.OpenXR.ARMarkerManager");
        if (mm == null) { Log("ARMarkerManager component not found."); return; }

        // 이미 저장되어 있으면 스킵
        var saved = PlayerPrefs.GetString(KeyPersistedName, "");
        if (!string.IsNullOrEmpty(saved))
        {
            Log($"Already saved persisted anchor name in PlayerPrefs: '{saved}'. Skip.");
            _done = true;
            return;
        }

        // markersChanged 구독 (reflection)
        var evt = mm.GetType().GetEvent("markersChanged");
        if (evt == null) { Log("markersChanged event not found."); return; }
        // 델리게이트 생성
        var handler = Delegate.CreateDelegate(evt.EventHandlerType, this,
            typeof(AnchorOnLockPersist).GetMethod(nameof(OnMarkersChangedWrapper),
            BindingFlags.NonPublic | BindingFlags.Instance));
        evt.AddEventHandler(mm, handler);

        Log("Waiting QR lock... (will persist anchor once)");
#else
        Log("UWP only. Build UWP/IL2CPP/ARM64 on device.");
#endif
    }

#if WINDOWS_UWP
    // 이벤트 콜백(Reflection용 시그니처: void(object,args))
    void OnMarkersChangedWrapper(object args)
    {
        if (_done || args == null) return;

        var addedProp = args.GetType().GetProperty("added");
        var added = addedProp?.GetValue(args) as System.Collections.IList;
        if (added == null || added.Count == 0) return;

        // 첫 마커(잠금) 사용
        var marker = added[0];
        var trProp = marker.GetType().GetProperty("transform");
        var tr = trProp?.GetValue(marker) as Transform;
        if (tr == null) return;

        // 한 번만 실행
        _ = CreateAndPersistAt(tr.position, tr.rotation);
        _done = true;
    }

    static async Task<object> AwaitTaskObject(object taskObj)
    {
        var task = taskObj as System.Threading.Tasks.Task;
        if (task == null) return null;
        await task;
        var t = taskObj.GetType();
        if (t.IsGenericType)
        {
            var prop = t.GetProperty("Result");
            return prop?.GetValue(taskObj);
        }
        return null;
    }

    public async Task CreateAndPersistAt(Vector3 pos, Quaternion rot)
    {
        try
        {
            const string asm = "Microsoft.MixedReality.OpenXR";
            var tSpatialAnchor = Type.GetType($"Microsoft.MixedReality.OpenXR.SpatialAnchor, {asm}");
            var tPersistStore  = Type.GetType($"Microsoft.MixedReality.OpenXR.PersistedAnchorStore, {asm}");
            if (tSpatialAnchor == null || tPersistStore == null)
            {
                Log("OpenXR types not found. Check 'Microsoft Mixed Reality OpenXR Plugin' + UWP settings.");
                return;
            }

            // SpatialAnchor.CreateFromPose(Vector3, Quaternion)
            var mCreate = tSpatialAnchor.GetMethod("CreateFromPose",
                BindingFlags.Public | BindingFlags.Static,
                null, new Type[]{ typeof(Vector3), typeof(Quaternion) }, null);
            if (mCreate == null) { Log("CreateFromPose not found."); return; }
            var anchor = mCreate.Invoke(null, new object[]{ pos, rot });
            if (anchor == null) { Log("CreateFromPose failed."); return; }

            // PersistedAnchorStore.LoadAsync()
            var mLoadAsync = tPersistStore.GetMethod("LoadAsync", BindingFlags.Public | BindingFlags.Static);
            var taskStore = mLoadAsync.Invoke(null, null);
            var store = await AwaitTaskObject(taskStore);
            if (store == null) { Log("PersistedAnchorStore.LoadAsync() failed."); return; }

            // Unpersist (덮어쓰기)
            var mUnpersist = store.GetType().GetMethod("UnpersistAnchorAsync", new Type[]{ typeof(string) });
            if (mUnpersist != null)
                await AwaitTaskObject(mUnpersist.Invoke(store, new object[]{ persistedName }));

            // TryPersistAnchorAsync(string, SpatialAnchor)
            var mPersist = store.GetType().GetMethod("TryPersistAnchorAsync", new Type[]{ typeof(string), tSpatialAnchor });
            if (mPersist == null) { Log("TryPersistAnchorAsync not found."); return; }
            var tPersist = mPersist.Invoke(store, new object[]{ persistedName, anchor });
            var okObj = await AwaitTaskObject(tPersist);
            bool ok = (okObj is bool b) ? b : false;
            if (!ok) { Log($"Persist failed: {persistedName}"); return; }

            // 이름 저장
            PlayerPrefs.SetString(KeyPersistedName, persistedName);
            PlayerPrefs.Save();

            Log($"[Anchor persisted] name='{persistedName}'  (saved to PlayerPrefs)");
        }
        catch (Exception ex) { Log("Persist exception: " + ex.Message); }
    }
#endif

    void Log(string s) { Debug.Log("[AnchorOnLockPersist] " + s); try { DebugHUD.Log(s); } catch { } }
}
