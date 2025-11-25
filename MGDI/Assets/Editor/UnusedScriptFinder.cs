#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public class UnusedScriptFinder : EditorWindow
{
    [MenuItem("Tools/Find Unused MonoBehaviours")]
    public static void ShowWindow()
    {
        GetWindow<UnusedScriptFinder>("Unused Scripts");
    }

    Vector2 _scroll;
    List<MonoScript> _unusedScripts = new List<MonoScript>();
    bool _includeEditorScripts = false;

    void OnGUI()
    {
        EditorGUILayout.LabelField("Unused MonoBehaviour Finder", EditorStyles.boldLabel);
        EditorGUILayout.Space();

        _includeEditorScripts = EditorGUILayout.Toggle("Include Editor scripts", _includeEditorScripts);

        if (GUILayout.Button("Scan Project"))
        {
            ScanProject();
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField($"Results: {_unusedScripts.Count} scripts", EditorStyles.boldLabel);

        _scroll = EditorGUILayout.BeginScrollView(_scroll);
        foreach (var script in _unusedScripts)
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.ObjectField(script, typeof(MonoScript), false);

            if (GUILayout.Button("Select", GUILayout.Width(60)))
            {
                Selection.activeObject = script;
            }

            EditorGUILayout.EndHorizontal();
        }
        EditorGUILayout.EndScrollView();
    }

    void ScanProject()
    {
        _unusedScripts.Clear();

        // 1) 프로젝트 내 모든 MonoScript 찾기 (이 부분은 그대로 OK)
        string[] guids = AssetDatabase.FindAssets("t:MonoScript", new[] {"Assets/Scripts"});
        List<MonoScript> allScripts = new List<MonoScript>();
        foreach (string guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            var ms = AssetDatabase.LoadAssetAtPath<MonoScript>(path);
            if (ms == null) continue;

            var t = ms.GetClass();
            if (t == null) continue;

            // MonoBehaviour만 대상 (ScriptableObject/EditorWindow 등 제외)
            if (!typeof(MonoBehaviour).IsAssignableFrom(t)) continue;
            if (t.IsAbstract) continue;

            // Editor 전용 타입 제외 (원하면 토글로 포함 가능)
            if (!_includeEditorScripts && IsEditorScript(ms, t)) continue;

            allScripts.Add(ms);
        }

        // 2) "이 스크립트를 가진 컴포넌트가 씬/프리팹 어디에도 붙어 있는지" 탐색
        HashSet<Type> usedTypes = new HashSet<Type>();

        // 2-1) Assets 아래 모든 프리팹 검색 (Packages 제외)
        string[] prefabGuids = AssetDatabase.FindAssets("t:Prefab", new[] { "Assets" });
        for (int i = 0; i < prefabGuids.Length; i++)
        {
            string path = AssetDatabase.GUIDToAssetPath(prefabGuids[i]);
            var go = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (!go) continue;

            var comps = go.GetComponentsInChildren<MonoBehaviour>(true);
            foreach (var c in comps)
            {
                if (!c) continue;
                usedTypes.Add(c.GetType());
            }
        }

        // 2-2) Assets 아래 모든 씬 에셋 검색 (Packages/ 씬은 스킵)
        string[] sceneGuids = AssetDatabase.FindAssets("t:Scene", new[] { "Assets" });
        foreach (string sg in sceneGuids)
        {
            string scenePath = AssetDatabase.GUIDToAssetPath(sg);

            // 혹시 모르니 OpenScene 실패는 무시
            try
            {
                var scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Additive);

                var roots = scene.GetRootGameObjects();
                foreach (var root in roots)
                {
                    var comps = root.GetComponentsInChildren<MonoBehaviour>(true);
                    foreach (var c in comps)
                    {
                        if (!c) continue;
                        usedTypes.Add(c.GetType());
                    }
                }

                EditorSceneManager.CloseScene(scene, true);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[UnusedScriptFinder] 씬을 여는 데 실패: {scenePath} ({e.Message})");
            }
        }

        // 3) 사용되지 않은 타입 찾기
        foreach (var ms in allScripts)
        {
            var t = ms.GetClass();
            if (t == null) continue;

            if (!usedTypes.Contains(t))
            {
                _unusedScripts.Add(ms);
            }
        }

        _unusedScripts = _unusedScripts.OrderBy(s => s.name).ToList();
        Debug.Log($"[UnusedScriptFinder] Found {_unusedScripts.Count} unused MonoBehaviour scripts (Assets/ 기준 Scene+Prefab).");
    }


    bool IsEditorScript(MonoScript ms, Type t)
    {
        // 1) 경로 상에 "/Editor/" 폴더가 포함되어 있으면 Editor 전용일 가능성 큼
        string path = AssetDatabase.GetAssetPath(ms);
        if (path.Contains("/Editor/")) return true;

        // 2) UnityEditor 관련 타입들 제외
        return typeof(UnityEditor.Editor).IsAssignableFrom(t)
               || typeof(EditorWindow).IsAssignableFrom(t);
    }
}
#endif
