using System;
using TMPro;
using UnityEngine;

public class InstructionPanelAutoSizer : MonoBehaviour
{
    [SerializeField] private TMP_Text instructionText;
    [SerializeField] private RectTransform targetRect;
    [SerializeField] private RectTransform contentRect;

    [SerializeField] private Vector2 padding = new Vector2(60f, 40f);
    [SerializeField] private Vector2 minSize = new Vector2(400f, 140f);
    [SerializeField] private Vector2 maxSize = new Vector2(1200f, 500f);
    [SerializeField] private float resizeLerp = 18f;
    [SerializeField, Range(0.5f, 1.2f)] private float panelSizeScale = 0.78f;
    [SerializeField] private Vector2 minAbsoluteSizeNormal = new Vector2(220f, 90f);
    [SerializeField] private Vector2 minAbsoluteSizeCountdown = new Vector2(110f, 74f);
    [SerializeField] private bool scalePaddingWithPanel = true;
    [SerializeField] private bool autoScaleTextWithPanel = true;
    [SerializeField, Range(0.5f, 1f)] private float minTextScale = 0.72f;
    [SerializeField] private bool autoSizeContentRect = true;
    [SerializeField] private bool centerContentRect = true;
    [SerializeField] private bool centerInstructionText = true;
    [SerializeField] private bool constrainNormalLineWidth = true;
    [SerializeField] private float normalMaxLineWidth = 430f;

    [Header("Countdown Style")]
    [SerializeField] private bool useCountdownStyle = true;
    [SerializeField] private Vector2 paddingCountdown = new Vector2(20f, 14f);
    [SerializeField] private Vector2 minSizeCountdown = new Vector2(120f, 90f);

    [Header("Position Style")]
    [SerializeField] private bool usePositionStyle = true;
    [SerializeField] private Vector2 anchoredPosNormal = new Vector2(-80f, 0f);
    [SerializeField] private Vector2 anchoredPosCountdown = new Vector2(0f, 0f);
    [SerializeField] private Vector2 countdownExtraOffset = new Vector2(20f, 0f);
    [SerializeField] private Vector2 globalPositionOffset = new Vector2(-70f, 0f);
    [SerializeField] private bool applyGlobalOffsetToCountdown = false;
    [SerializeField] private float positionLerp = 18f;

    [Header("Embedded Example Reserve")]
    [SerializeField] private bool reserveSpaceForEmbeddedExample = true;
    [SerializeField] private float embeddedExampleReservedWidth = 170f;
    [SerializeField] private float embeddedExampleReservedGap = 12f;
    [SerializeField] private float embeddedExampleReservedHeight = 120f;

    private bool _hasLastCountdownMode;
    private bool _lastCountdownMode;
    private float _baseFontNormal;
    private float _baseFontCountdown;
    private string _lastAppliedText;
    private bool _embeddedExampleActive;

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
        ApplySizing(smooth: true);
    }

    public void RefreshNow()
    {
        ApplySizing(smooth: false);
    }

    public void SetEmbeddedExampleLayoutState(bool active, float reservedWidth, float reservedGap, float reservedHeight)
    {
        _embeddedExampleActive = active;
        embeddedExampleReservedWidth = Mathf.Max(0f, reservedWidth);
        embeddedExampleReservedGap = Mathf.Max(0f, reservedGap);
        embeddedExampleReservedHeight = Mathf.Max(0f, reservedHeight);
    }

    private void ApplySizing(bool smooth)
    {
        if (instructionText == null || targetRect == null) return;
        if (!instructionText.gameObject.activeInHierarchy) return;
        if (!targetRect.gameObject.activeInHierarchy) return;

        string text = instructionText.text;
        if (string.IsNullOrEmpty(text)) return;

        string trimmed = text.Trim();
        if (trimmed.Length == 0) return;

        bool isCountdown = IsCountdownText(trimmed);

        float scale = Mathf.Clamp(panelSizeScale, 0.5f, 1.2f);

        if (autoScaleTextWithPanel)
            ApplyTextScale(isCountdown, scale);

        Vector2 paddingToUse = padding;
        Vector2 minSizeToUse = minSize;

        if (useCountdownStyle && isCountdown)
        {
            paddingToUse = paddingCountdown;
            minSizeToUse = minSizeCountdown;
        }

        Vector2 paddingEffective = scalePaddingWithPanel ? (paddingToUse * scale) : paddingToUse;

        float maxW = Mathf.Max(100f, maxSize.x * scale);
        float maxH = Mathf.Max(80f, maxSize.y * scale);
        float minW;
        float minH;
        if (isCountdown)
        {
            minW = Mathf.Max(minSizeToUse.x * scale, minAbsoluteSizeCountdown.x * scale);
            minH = Mathf.Max(minSizeToUse.y * scale, minAbsoluteSizeCountdown.y * scale);
        }
        else
        {
            minW = Mathf.Max(minAbsoluteSizeNormal.x * scale, 1f);
            minH = Mathf.Max(minAbsoluteSizeNormal.y * scale, 1f);
        }
        minW = Mathf.Min(minW, maxW);
        minH = Mathf.Min(minH, maxH);

        float maxWidthConstraint = Mathf.Max(50f, maxW - paddingEffective.x);
        float embeddedReserve = 0f;
        if (!isCountdown && reserveSpaceForEmbeddedExample && _embeddedExampleActive)
            embeddedReserve = Mathf.Max(0f, embeddedExampleReservedWidth + embeddedExampleReservedGap);

        maxWidthConstraint = Mathf.Max(50f, maxWidthConstraint - embeddedReserve);
        if (!isCountdown && constrainNormalLineWidth)
            maxWidthConstraint = Mathf.Min(maxWidthConstraint, Mathf.Max(120f, normalMaxLineWidth * scale));

        Vector2 preferred = instructionText.GetPreferredValues(text, maxWidthConstraint, 0f);

        float targetW = Mathf.Clamp(preferred.x + paddingEffective.x + embeddedReserve, minW, maxW);
        float embeddedHeightReserve = 0f;
        if (!isCountdown && reserveSpaceForEmbeddedExample && _embeddedExampleActive)
            embeddedHeightReserve = Mathf.Max(0f, embeddedExampleReservedHeight + paddingEffective.y);

        float targetH = Mathf.Clamp(
            Mathf.Max(preferred.y + paddingEffective.y, embeddedHeightReserve),
            minH,
            maxH);
        Vector2 desired = new Vector2(targetW, targetH);

        float dt = Mathf.Max(Time.unscaledDeltaTime, 1e-4f);
        bool textChanged = !string.Equals(_lastAppliedText, text, StringComparison.Ordinal);
        _lastAppliedText = text;
        bool useSmooth = smooth && !textChanged;

        if (!useSmooth || resizeLerp <= 0f)
        {
            targetRect.sizeDelta = desired;
        }
        else
        {
            float tSize = 1f - Mathf.Exp(-resizeLerp * dt);
            targetRect.sizeDelta = Vector2.Lerp(targetRect.sizeDelta, desired, tSize);
        }

        if (autoSizeContentRect)
        {
            ResolveContentRectForCurrentHierarchy();
            if (contentRect != null)
                ResizeContentRect(paddingEffective);
        }

        if (usePositionStyle)
        {
            Vector2 targetAnchoredPos = isCountdown
                ? (anchoredPosCountdown + countdownExtraOffset)
                : anchoredPosNormal;

            if (!isCountdown || applyGlobalOffsetToCountdown)
                targetAnchoredPos += globalPositionOffset;

            if (!useSmooth || positionLerp <= 0f)
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

    private static bool IsCountdownText(string trimmed)
    {
        if (string.Equals(trimmed, "Go", StringComparison.OrdinalIgnoreCase))
            return true;

        if (int.TryParse(trimmed, out int sec))
            return sec >= 1 && sec <= 20;

        return false;
    }

    [ContextMenu("Auto Assign References")]
    private void AutoAssignReferences()
    {
        if (targetRect == null)
        {
            Transform panelBg = transform.Find("Panel_BG");
            if (panelBg == null)
                panelBg = transform.Find("PanelBG");
            if (panelBg != null)
                targetRect = panelBg as RectTransform;

            if (targetRect == null)
                targetRect = GetComponent<RectTransform>();
        }

        if (instructionText == null)
            instructionText = GetComponentInChildren<TMP_Text>(true);

        ResolveContentRectForCurrentHierarchy();

        if (centerInstructionText && instructionText != null)
        {
            instructionText.alignment = TextAlignmentOptions.Center;
            instructionText.enableWordWrapping = true;
        }
    }

    private void ApplyTextScale(bool isCountdown, float panelScale)
    {
        if (instructionText == null)
            return;

        if (!_hasLastCountdownMode || _lastCountdownMode != isCountdown)
        {
            if (isCountdown)
                _baseFontCountdown = Mathf.Max(1f, instructionText.fontSize);
            else
                _baseFontNormal = Mathf.Max(1f, instructionText.fontSize);

            _lastCountdownMode = isCountdown;
            _hasLastCountdownMode = true;
        }

        float baseFont = isCountdown ? _baseFontCountdown : _baseFontNormal;
        if (baseFont <= 0f)
            baseFont = Mathf.Max(1f, instructionText.fontSize);

        float t = Mathf.InverseLerp(0.5f, 1f, panelScale);
        float textScale = Mathf.Lerp(minTextScale, 1f, t);
        instructionText.fontSize = baseFont * textScale;
    }

    private void ResizeContentRect(Vector2 paddingEffective)
    {
        if (targetRect == null || contentRect == null)
            return;

        if (contentRect == targetRect)
            return;

        float innerW = Mathf.Max(20f, targetRect.sizeDelta.x - paddingEffective.x);
        float innerH = Mathf.Max(20f, targetRect.sizeDelta.y - paddingEffective.y);

        if (centerContentRect)
        {
            contentRect.anchorMin = new Vector2(0.5f, 0.5f);
            contentRect.anchorMax = new Vector2(0.5f, 0.5f);
            contentRect.pivot = new Vector2(0.5f, 0.5f);
            contentRect.anchoredPosition = Vector2.zero;
        }

        contentRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, innerW);
        contentRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, innerH);
    }

    private void ResolveContentRectForCurrentHierarchy()
    {
        if (instructionText == null || instructionText.rectTransform == null)
            return;

        if (contentRect == instructionText.rectTransform)
        {
            contentRect = null;
            return;
        }

        if (contentRect != null && contentRect != targetRect)
            return;

        RectTransform textRect = instructionText.rectTransform;
        RectTransform parentRect = textRect.parent as RectTransform;

        if (parentRect != null && parentRect != targetRect)
        {
            contentRect = parentRect;
            return;
        }

        // If the TMP text is directly under the panel background, resizing the text
        // rect itself conflicts with the embedded-image layout offsets. In that case,
        // keep the text rect untouched and size only the panel background.
        contentRect = null;
    }
}
