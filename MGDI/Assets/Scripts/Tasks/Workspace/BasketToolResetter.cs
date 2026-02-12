using System.Collections.Generic;
using UnityEngine;

public class BasketToolResetter : MonoBehaviour
{
    [Header("Roots")]
    [SerializeField] private Transform toolsRuntimeRoot;   // Tools_Dynamic/ToolsRuntime
    [SerializeField] private Transform toolSpawnPointsRoot; // Props_Static/.../ToolSpawnPoints

    [Header("Options")]
    [SerializeField] private bool resetRotation = true;
    [SerializeField] private bool resetScale = true;
    [SerializeField] private bool setKinematicWhileReset = true;

    private Dictionary<string, Transform> _spawnById;

    public void RebuildSpawnMap()
    {
        _spawnById = new Dictionary<string, Transform>();

        if (toolSpawnPointsRoot == null) return;

        // SpawnPoint에도 ToolId를 붙여두는 걸 추천 (없으면 이름 매칭으로 fallback 가능)
        var ids = toolSpawnPointsRoot.GetComponentsInChildren<ToolId>(true);
        foreach (var tid in ids)
        {
            if (tid == null || string.IsNullOrEmpty(tid.id)) continue;
            _spawnById[tid.id] = tid.transform;
        }
    }

    public void ResetAllToolsToBasket()
    {
        if (toolsRuntimeRoot == null || toolSpawnPointsRoot == null) return;

        if (_spawnById == null) RebuildSpawnMap();
        if (_spawnById == null || _spawnById.Count == 0)
        {
            Debug.LogWarning("[BasketToolResetter] No spawn map. Add ToolId to spawn points or call RebuildSpawnMap.");
            return;
        }

        var tools = toolsRuntimeRoot.GetComponentsInChildren<ToolId>(true);
        foreach (var tid in tools)
        {
            if (tid == null || string.IsNullOrEmpty(tid.id)) continue;

            if (!_spawnById.TryGetValue(tid.id, out var sp)) continue;

            var toolTf = tid.transform;

            Rigidbody rb = toolTf.GetComponent<Rigidbody>();
            bool hadRb = (rb != null);

            if (hadRb && setKinematicWhileReset)
            {
                rb.velocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                rb.isKinematic = true;
            }

            toolTf.position = sp.position;
            if (resetRotation) toolTf.rotation = sp.rotation;

            if (resetScale)
            {
                // baseline scale을 유지하려면 ToolScalingTaskManager가 startLocalScale로 복구하도록 했으니,
                // 여기서는 그냥 현재 localScale 유지해도 되는데, 깔끔히 1로 돌리고 싶으면 ToolId별 baseline을 저장하는 구조가 필요함.
                // 지금은 "유지"가 안전.
            }

            if (hadRb && setKinematicWhileReset)
                rb.isKinematic = true; // 계속 잡기용이면 kinematic 유지해도 됨
        }

        Debug.Log("[BasketToolResetter] ResetAllToolsToBasket done.");
    }
}
