using UnityEngine;

public class MatchToolAndGhostPose : MonoBehaviour
{
    [Header("Assign these roots")]
    public Transform toolsDynamicRoot;   // ContentRoot/Tools_Dynamic
    public Transform slotsTargetsRoot;   // ContentRoot/Slots_Targets

    [Header("Options")]
    public bool matchRotation = true;
    public bool matchPosition = false; // 보통 false (위치까지 같게 하면 겹침)
    public bool matchScale = false;

    // ToolId를 기준으로 Tool과 Ghost를 매칭
    [ContextMenu("Copy Tool -> Ghost (Pose)")]
    public void CopyToolToGhost()
    {
        if (toolsDynamicRoot == null || slotsTargetsRoot == null)
        {
            Debug.LogError("[MatchPose] Roots not assigned.");
            return;
        }

        var toolIds = toolsDynamicRoot.GetComponentsInChildren<ToolId>(true);
        foreach (var t in toolIds)
        {
            if (t == null || string.IsNullOrEmpty(t.id)) continue;
            var toolTf = t.transform;

            // 같은 id를 가진 ghost 찾기
            ToolId ghostId = FindGhostById(t.id);
            if (ghostId == null) continue;

            var ghostTf = ghostId.transform;

            if (matchRotation) ghostTf.rotation = toolTf.rotation;
            if (matchPosition) ghostTf.position = toolTf.position;
            if (matchScale) ghostTf.localScale = toolTf.localScale;
        }

        Debug.Log("[MatchPose] Tool -> Ghost copied.");
    }

    [ContextMenu("Copy Ghost -> Tool (Pose)")]
    public void CopyGhostToTool()
    {
        if (toolsDynamicRoot == null || slotsTargetsRoot == null)
        {
            Debug.LogError("[MatchPose] Roots not assigned.");
            return;
        }

        var ghostIds = slotsTargetsRoot.GetComponentsInChildren<ToolId>(true);
        foreach (var g in ghostIds)
        {
            if (g == null || string.IsNullOrEmpty(g.id)) continue;
            var ghostTf = g.transform;

            // 같은 id를 가진 tool 찾기
            ToolId toolId = FindToolById(g.id);
            if (toolId == null) continue;

            var toolTf = toolId.transform;

            if (matchRotation) toolTf.rotation = ghostTf.rotation;
            if (matchPosition) toolTf.position = ghostTf.position;
            if (matchScale) toolTf.localScale = ghostTf.localScale;
        }

        Debug.Log("[MatchPose] Ghost -> Tool copied.");
    }

    private ToolId FindGhostById(string id)
    {
        var ids = slotsTargetsRoot.GetComponentsInChildren<ToolId>(true);
        for (int i = 0; i < ids.Length; i++)
            if (ids[i] != null && ids[i].id == id) return ids[i];
        return null;
    }

    private ToolId FindToolById(string id)
    {
        var ids = toolsDynamicRoot.GetComponentsInChildren<ToolId>(true);
        for (int i = 0; i < ids.Length; i++)
            if (ids[i] != null && ids[i].id == id) return ids[i];
        return null;
    }
}
