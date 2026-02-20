using System;
using TMPro;
using UnityEngine;

public class InstructionPanelAutoSizer : MonoBehaviour
{
    [SerializeField] private TMP_Text instructionText;
    [SerializeField] private RectTransform targetRect;

    [SerializeField] private Vector2 padding = new Vector2(60f, 40f);
    [SerializeField] private Vector2 minSize = new Vector2(400f, 140f);
    [SerializeField] private Vector2 maxSize = new Vector2(1200f, 500f);
    [SerializeField] private float resizeLerp = 18f;

    [Header("Countdown Style")]
    [SerializeField] private bool useCountdownStyle = true;
    [SerializeField] private Vector2 paddingCountdown = new Vector2(20f, 14f);
    [SerializeField] private Vector2 minSizeCountdown = new Vector2(120f, 90f);

    [Header("Position Style")]
    [SerializeField] private bool usePositionStyle = true;
    [SerializeField] private Vector2 anchoredPosNormal = new Vector2(-80f, 0f);
    [SerializeField] private Vector2 anchoredPosCountdown = new Vector2(0f, 0f);
    [SerializeField] private float positionLerp = 18f;

    private void Reset()
    {
        AutoAssignReferences();
    }

    private void Awake()
    {
        AutoAssignReferences();
    }

    private void OnValidate()
    {
        padding.x = Mathf.Max(0f, padding.x);
        padding.y = Mathf.Max(0f, padding.y);
        paddingCountdown.x = Mathf.Max(0f, paddingCountdown.x);
        paddingCountdown.y = Mathf.Max(0f, paddingCountdown.y);

        minSize.x = Mathf.Max(0f, minSize.x);
        minSize.y = Mathf.Max(0f, minSize.y);
        minSizeCountdown.x = Mathf.Max(0f, minSizeCountdown.x);
        minSizeCountdown.y = Mathf.Max(0f, minSizeCountdown.y);

        float requiredMinX = Mathf.Max(minSize.x, minSizeCountdown.x);
        float requiredMinY = Mathf.Max(minSize.y, minSizeCountdown.y);
        maxSize.x = Mathf.Max(requiredMinX, maxSize.x);
        maxSize.y = Mathf.Max(requiredMinY, maxSize.y);

        resizeLerp = Mathf.Max(0f, resizeLerp);
        positionLerp = Mathf.Max(0f, positionLerp);

        if (!Application.isPlaying)
            AutoAssignReferences();
    }

    private void LateUpdate()
    {
        if (instructionText == null || targetRect == null) return;
        if (!instructionText.gameObject.activeInHierarchy) return;
        if (!targetRect.gameObject.activeInHierarchy) return;

        string text = instructionText.text;
        if (string.IsNullOrEmpty(text)) return;

        string trimmed = text.Trim();
        if (trimmed.Length == 0) return;

        bool isCountdown = (trimmed == "3" ||
                            trimmed == "2" ||
                            trimmed == "1" ||
                            string.Equals(trimmed, "Go", StringComparison.OrdinalIgnoreCase));

        Vector2 paddingToUse = padding;
        Vector2 minSizeToUse = minSize;

        if (useCountdownStyle && isCountdown)
        {
            paddingToUse = paddingCountdown;
            minSizeToUse = minSizeCountdown;
        }

        float maxWidthConstraint = Mathf.Max(50f, maxSize.x - paddingToUse.x);
        Vector2 preferred = instructionText.GetPreferredValues(text, maxWidthConstraint, 0f);

        float targetW = Mathf.Clamp(preferred.x + paddingToUse.x, minSizeToUse.x, maxSize.x);
        float targetH = Mathf.Clamp(preferred.y + paddingToUse.y, minSizeToUse.y, maxSize.y);
        Vector2 desired = new Vector2(targetW, targetH);

        float dt = Mathf.Max(Time.unscaledDeltaTime, 1e-4f);

        if (resizeLerp <= 0f)
        {
            targetRect.sizeDelta = desired;
        }
        else
        {
            float tSize = 1f - Mathf.Exp(-resizeLerp * dt);
            targetRect.sizeDelta = Vector2.Lerp(targetRect.sizeDelta, desired, tSize);
        }

        if (usePositionStyle)
        {
            Vector2 targetAnchoredPos = isCountdown ? anchoredPosCountdown : anchoredPosNormal;

            if (positionLerp <= 0f)
            {
                targetRect.anchoredPosition = targetAnchoredPos;
            }
            else
            {
                float tPos = 1f - Mathf.Exp(-positionLerp * dt);
                targetRect.anchoredPosition = Vector2.Lerp(targetRect.anchoredPosition, targetAnchoredPos, tPos);
            }
        }
    }

    [ContextMenu("Auto Assign References")]
    private void AutoAssignReferences()
    {
        if (targetRect == null)
        {
            Transform panelBg = transform.Find("Panel_BG");
            if (panelBg != null)
                targetRect = panelBg as RectTransform;

            if (targetRect == null)
                targetRect = GetComponent<RectTransform>();
        }

        if (instructionText == null)
            instructionText = GetComponentInChildren<TMP_Text>(true);
    }
}
