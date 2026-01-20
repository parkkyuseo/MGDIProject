using UnityEngine;

public class HandVisualModeSwitcher : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private StudyFlowController flow;
    [SerializeField] private GameObject realHandVisualRoot;  // 실제 손 따라오는 손(원본)
    [SerializeField] private GameObject microHandVisualRoot; // 블록 따라가는 손(복제본)

    void Update()
    {
        if (flow == null) return;

        bool micro = (flow.currentTechnique == StudyFlowController.Technique.Micro);

        if (realHandVisualRoot != null)
            realHandVisualRoot.SetActive(!micro);

        if (microHandVisualRoot != null)
            microHandVisualRoot.SetActive(micro);
    }
}
