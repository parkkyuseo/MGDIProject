using System.Collections;
using System.Reflection;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class InstructionHUD : MonoBehaviour
{
    [SerializeField] private TMP_Text instructionText;
    [SerializeField] private float defaultShowSeconds = 1.8f;
    [SerializeField] private InstructionPanelAutoSizer panelAutoSizer;

    [Header("Example Image")]
    [SerializeField] private Image exampleImage;
    [SerializeField] private bool hideExampleOnCountdown = true;
    [SerializeField] private bool preserveExampleAspect = true;
    [SerializeField] private bool autoLayoutExampleImage = true;
    [SerializeField] private bool embedExampleInsidePanel = true;
    [SerializeField] private bool embedExampleOnRight = false;
    [SerializeField, Range(0.2f, 0.5f)] private float embeddedExampleWidthRatio = 0.22f;
    [SerializeField] private float embeddedExampleMinWidth = 78f;
    [SerializeField] private float embeddedExampleMaxWidth = 110f;
    [SerializeField] private Vector2 embeddedExamplePanelPadding = new Vector2(16f, 12f);
    [SerializeField] private float embeddedExampleTextGap = 14f;
    [SerializeField] private float embeddedExampleExtraTextInset = 12f;
    [SerializeField] private Vector2 embeddedExampleOffset = new Vector2(8f, 0f);
    [SerializeField] private bool placeExampleBesideInstructionText = false;
    [SerializeField] private Vector2 exampleImageAnchoredPos = new Vector2(-34f, -32f);
    [SerializeField] private Vector2 exampleImageSize = new Vector2(150f, 108f);
    [SerializeField] private float exampleGapFromText = 18f;
    [SerializeField] private Vector2 exampleImageTextOffset = new Vector2(0f, -8f);
    [SerializeField] private bool clampExampleInsidePanel = false;

    [Header("Transitions")]
    [SerializeField] private bool useTransitions = true;
    [SerializeField] private CanvasGroup canvasGroup;
    [SerializeField] private RectTransform panelRect;
    [SerializeField] private float fadeInSeconds = 0.12f;
    [SerializeField] private float fadeOutSeconds = 0.18f;
    [SerializeField] private float popScaleStart = 0.96f;
    [SerializeField] private float popScaleEnd = 1.00f;

    [Header("Panel Shape")]
    [SerializeField] private bool useRoundedPanelBackground = true;
    [SerializeField] private float roundedPanelCornerRadius = 10f;
    [SerializeField] private int roundedPanelCornerSegments = 7;
    [SerializeField] private int roundedPanelSpriteSize = 96;

    [Header("Style")]
    [SerializeField] private bool overrideTypography = true;
    [SerializeField] private Color normalTextColor = new Color(0.06f, 0.08f, 0.09f, 1f);
    [SerializeField] private Color countdownTextColor = new Color(0.06f, 0.08f, 0.09f, 1f);
    [SerializeField] private float normalFontSize = 20f;
    [SerializeField] private float countdownFontSize = 44f;
    [SerializeField] private FontStyles normalFontStyle = FontStyles.Bold;
    [SerializeField] private FontStyles countdownFontStyle = FontStyles.Bold;

    [Header("Overlay Announcement")]
    [SerializeField] private TMP_Text overlayText;
    [SerializeField] private bool autoCreateOverlayText = true;
    [SerializeField] private Vector2 overlayAnchoredPosition = new Vector2(0f, -10f);
    [SerializeField] private Vector2 overlaySize = new Vector2(560f, 64f);
    [SerializeField] private Color overlayTextColor = new Color(0.9f, 0.05f, 0.02f, 1f);
    [SerializeField] private float overlayFontSize = 30f;
    [SerializeField] private FontStyles overlayFontStyle = FontStyles.Bold;
    [SerializeField] private bool speakOverlayText = true;

    [Header("Speech")]
    [SerializeField] private bool enableSpeech = true;
    [SerializeField] private bool speakInstructionText = true;
    [SerializeField] private bool speakCountdownText = true;
    [SerializeField] private bool stopPreviousSpeechOnShow = true;
    [SerializeField] private float minSpeakIntervalSeconds = 0.03f;
    [Tooltip("Optional. If empty, TextToSpeech is auto-resolved from this object or children.")]
    [SerializeField] private Component textToSpeechComponent;

    [Header("Speech Duration Sync")]
    [SerializeField] private bool syncPanelDurationToSpeech = true;
    [Tooltip("Used when speech is shorter than requested show time: panel remains at most this much longer.")]
    [SerializeField] private float maxExtraAfterSpeechWhenShorter = 2f;
    [Tooltip("Used when speech is longer than requested show time: extra hold after speech ends.")]
    [SerializeField] private float extraAfterSpeechWhenLonger = 0.15f;
    [SerializeField] private bool waitForSpeechCompletionBeforeHide = true;
    [SerializeField] private float maxSpeechCompletionWaitSeconds = 12f;
    [SerializeField] private float estimatedSpeechCharsPerSecond = 13f;
    [SerializeField] private float estimatedSpeechWordsPerSecond = 2.5f;

    public float DefaultShowSeconds => defaultShowSeconds;

    Coroutine _hideCo;
    Coroutine _overlayHideCo;
    Component _resolvedTextToSpeech;
    AudioSource _resolvedSpeechAudioSource;
    bool _warnedNoTextToSpeech;
    bool _warnedSpeakMethodMissing;
    bool _ttsResolveAttempted;
    float _lastSpeakTime = -999f;
    RectTransform _instructionTextRect;
    RectTransform _exampleImageRect;
    Transform _exampleImageOriginalParent;
    Vector2 _textDefaultAnchorMin;
    Vector2 _textDefaultAnchorMax;
    Vector2 _textDefaultOffsetMin;
    Vector2 _textDefaultOffsetMax;
    Vector2 _textDefaultPivot;
    Vector2 _textDefaultAnchoredPosition;
    Vector2 _textDefaultSizeDelta;
    TextAlignmentOptions _textDefaultAlignment;
    Vector2 _exampleDefaultAnchorMin;
    Vector2 _exampleDefaultAnchorMax;
    Vector2 _exampleDefaultPivot;
    Vector2 _exampleDefaultAnchoredPosition;
    Vector2 _exampleDefaultSizeDelta;
    RectTransform _overlayTextRect;
    bool _layoutDefaultsCached;

    private void Awake()
    {
        if (canvasGroup == null)
            canvasGroup = GetComponentInChildren<CanvasGroup>(true);

        if (panelRect == null)
            panelRect = transform as RectTransform;

        if (panelAutoSizer == null)
            panelAutoSizer = GetComponent<InstructionPanelAutoSizer>();

        CacheLayoutDefaults();
        EnsureRoundedPanelBackground();

        if (canvasGroup != null)
            canvasGroup.alpha = 0f;

        HideExample();
        ResolveTextToSpeech();
    }

    private void EnsureRoundedPanelBackground()
    {
        if (!useRoundedPanelBackground)
            return;

        Transform panelBg = transform.Find("Panel_BG");
        if (panelBg == null)
            return;

        Image squareImage = panelBg.GetComponent<Image>();
        if (squareImage != null)
        {
            squareImage.sprite = BuildRoundedSprite(
                Mathf.Max(16, roundedPanelSpriteSize),
                Mathf.Max(1f, roundedPanelCornerRadius));
            squareImage.type = Image.Type.Sliced;
            squareImage.enabled = true;
        }

        RoundedRectGraphic rounded = panelBg.GetComponent<RoundedRectGraphic>();
        if (rounded != null)
        {
            rounded.enabled = false;
        }
    }

    private static Sprite BuildRoundedSprite(int size, float radiusPx)
    {
        int texSize = Mathf.Clamp(size, 16, 512);
        float radius = Mathf.Clamp(radiusPx, 1f, texSize * 0.5f - 1f);

        Texture2D tex = new Texture2D(texSize, texSize, TextureFormat.ARGB32, false);
        tex.name = "InstructionHUD_RoundedPanel";
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;

        Color32 clear = new Color32(255, 255, 255, 0);
        Color32 solid = new Color32(255, 255, 255, 255);

        float min = 0.5f;
        float max = texSize - 0.5f;
        float left = min + radius;
        float right = max - radius;
        float bottom = min + radius;
        float top = max - radius;

        for (int y = 0; y < texSize; y++)
        {
            for (int x = 0; x < texSize; x++)
            {
                float px = x + 0.5f;
                float py = y + 0.5f;

                bool insideCore = (px >= left && px <= right) || (py >= bottom && py <= top);
                if (insideCore)
                {
                    tex.SetPixel(x, y, solid);
                    continue;
                }

                float cx = px < left ? left : right;
                float cy = py < bottom ? bottom : top;
                float dx = px - cx;
                float dy = py - cy;
                bool insideCorner = (dx * dx + dy * dy) <= (radius * radius);
                tex.SetPixel(x, y, insideCorner ? solid : clear);
            }
        }

        tex.Apply(false, false);

        Vector4 border = new Vector4(radius, radius, radius, radius);
        Rect rect = new Rect(0f, 0f, texSize, texSize);
        return Sprite.Create(tex, rect, new Vector2(0.5f, 0.5f), 100f, 0u, SpriteMeshType.FullRect, border);
    }

    public void HideImmediate()
    {
        if (_hideCo != null) StopCoroutine(_hideCo);
        _hideCo = null;
        if (_overlayHideCo != null) StopCoroutine(_overlayHideCo);
        _overlayHideCo = null;

        if (instructionText != null) instructionText.text = "";
        HideOverlayImmediate();
        HideExample();
        if (canvasGroup != null) canvasGroup.alpha = 0f;
        if (panelRect != null) panelRect.localScale = Vector3.one * popScaleEnd;
        gameObject.SetActive(false);
    }

    public float Show(string text, float? seconds = null)
    {
        if (instructionText == null) return 0f;

        if (_hideCo != null) StopCoroutine(_hideCo);
        _hideCo = null;

        if (exampleImage != null && exampleImage.sprite == null)
            HideExample();

        instructionText.text = text ?? "";
        gameObject.SetActive(true);
        ApplyStyleForText(instructionText.text);
        ApplyExampleImageLayout();
        panelAutoSizer?.RefreshNow();
        ApplyExampleImageLayout();
        TrySpeak(instructionText.text);

        float requestedSeconds = seconds ?? defaultShowSeconds;
        float s = ResolveDisplaySeconds(instructionText.text, requestedSeconds);
        _hideCo = useTransitions ? StartCoroutine(ShowHideAnimated(s)) : StartCoroutine(HideAfter(s));
        return s;
    }

    public float ShowOverlay(string text, float? seconds = null, bool speak = true)
    {
        if (string.IsNullOrWhiteSpace(text))
            return 0f;

        TMP_Text target = ResolveOverlayText();
        if (target == null)
            return 0f;

        if (_overlayHideCo != null) StopCoroutine(_overlayHideCo);
        _overlayHideCo = null;

        target.text = text;
        ApplyOverlayStyle(target);
        target.gameObject.SetActive(true);
        target.transform.SetAsLastSibling();

        gameObject.SetActive(true);
        if (canvasGroup != null && canvasGroup.alpha <= 0f)
            canvasGroup.alpha = 1f;
        if (panelRect != null)
            panelRect.localScale = Vector3.one * popScaleEnd;

        if (speak && speakOverlayText)
            TrySpeak(text);

        float requestedSeconds = seconds ?? defaultShowSeconds;
        float s = Mathf.Max(0f, requestedSeconds);
        _overlayHideCo = StartCoroutine(HideOverlayAfter(s));
        return s;
    }

    public void HideOverlayImmediate()
    {
        if (overlayText == null)
            return;

        overlayText.text = "";
        overlayText.gameObject.SetActive(false);
    }

    public void ShowExample(Sprite sprite)
    {
        if (exampleImage == null)
            return;

        if (sprite == null)
        {
            HideExample();
            return;
        }

        exampleImage.sprite = sprite;
        exampleImage.preserveAspect = preserveExampleAspect;
        ApplyExampleImageLayout();
        exampleImage.enabled = true;
        exampleImage.gameObject.SetActive(true);
        LayoutRebuilder.ForceRebuildLayoutImmediate(exampleImage.rectTransform);
    }

    public void HideExample()
    {
        if (exampleImage == null)
            return;

        exampleImage.enabled = false;
        exampleImage.gameObject.SetActive(false);
        exampleImage.sprite = null;
        ApplyExampleImageLayout();
    }

    private void ApplyExampleImageLayout()
    {
        if (!autoLayoutExampleImage || exampleImage == null)
            return;

        CacheLayoutDefaults();

        RectTransform rect = exampleImage.rectTransform;
        if (rect == null)
            return;

        bool isCountdown = instructionText != null && IsCountdownText(instructionText.text);
        bool exampleVisible = exampleImage.gameObject.activeSelf &&
                              exampleImage.enabled &&
                              exampleImage.sprite != null;

        bool useEmbedded = embedExampleInsidePanel &&
                           !isCountdown &&
                           exampleVisible &&
                           panelRect != null &&
                           ResolvePanelBackgroundRect() != null &&
                           _instructionTextRect != null;

        if (panelAutoSizer != null)
        {
            float reserveWidth = Mathf.Clamp(
                Mathf.Max(embeddedExampleMaxWidth, exampleImageSize.x),
                embeddedExampleMinWidth,
                embeddedExampleMaxWidth);
            float reserveHeight = Mathf.Max(exampleImageSize.y, ComputeEmbeddedExampleHeightHint(reserveWidth));
            panelAutoSizer.SetEmbeddedExampleLayoutState(useEmbedded, reserveWidth, embeddedExampleTextGap, reserveHeight);
        }

        if (useEmbedded)
        {
            ApplyEmbeddedExampleLayout();
            return;
        }

        RestoreDefaultTextLayout();

        if (!placeExampleBesideInstructionText || instructionText == null)
        {
            RestoreExampleImageParentAndDefaults();
            rect.anchorMin = new Vector2(1f, 1f);
            rect.anchorMax = new Vector2(1f, 1f);
            rect.pivot = new Vector2(1f, 1f);
            rect.sizeDelta = exampleImageSize;
            rect.anchoredPosition = exampleImageAnchoredPos;
            return;
        }

        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.sizeDelta = exampleImageSize;

        RectTransform textRect = instructionText.rectTransform;
        RectTransform imageParentRect = rect.parent as RectTransform;
        if (textRect == null || imageParentRect == null)
        {
            rect.anchoredPosition = exampleImageAnchoredPos;
            return;
        }

        Vector3[] textCorners = new Vector3[4];
        textRect.GetWorldCorners(textCorners);
        Camera uiCamera = ResolveUiCamera(imageParentRect);

        Vector2 topRightLocal;
        Vector2 bottomRightLocal;
        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(
                imageParentRect,
                RectTransformUtility.WorldToScreenPoint(uiCamera, textCorners[2]),
                uiCamera,
                out topRightLocal) ||
            !RectTransformUtility.ScreenPointToLocalPointInRectangle(
                imageParentRect,
                RectTransformUtility.WorldToScreenPoint(uiCamera, textCorners[3]),
                uiCamera,
                out bottomRightLocal))
        {
            rect.anchoredPosition = exampleImageAnchoredPos;
            return;
        }

        float halfWidth = rect.sizeDelta.x * 0.5f;
        Vector2 rightMidLocal = (topRightLocal + bottomRightLocal) * 0.5f;
        Vector2 anchored = rightMidLocal + new Vector2(exampleGapFromText + halfWidth, 0f) + exampleImageTextOffset;

        if (clampExampleInsidePanel)
        {
            Rect panelBounds = imageParentRect.rect;
            float minX = panelBounds.xMin + halfWidth;
            float maxX = panelBounds.xMax - halfWidth;
            float halfHeight = rect.sizeDelta.y * 0.5f;
            float minY = panelBounds.yMin + halfHeight;
            float maxY = panelBounds.yMax - halfHeight;
            anchored.x = Mathf.Clamp(anchored.x, minX, maxX);
            anchored.y = Mathf.Clamp(anchored.y, minY, maxY);
        }

        rect.anchoredPosition = anchored;
    }

    private void CacheLayoutDefaults()
    {
        if (_layoutDefaultsCached)
            return;

        _instructionTextRect = instructionText != null ? instructionText.rectTransform : null;
        _exampleImageRect = exampleImage != null ? exampleImage.rectTransform : null;

        if (_instructionTextRect != null)
        {
            _textDefaultAnchorMin = _instructionTextRect.anchorMin;
            _textDefaultAnchorMax = _instructionTextRect.anchorMax;
            _textDefaultOffsetMin = _instructionTextRect.offsetMin;
            _textDefaultOffsetMax = _instructionTextRect.offsetMax;
            _textDefaultPivot = _instructionTextRect.pivot;
            _textDefaultAnchoredPosition = _instructionTextRect.anchoredPosition;
            _textDefaultSizeDelta = _instructionTextRect.sizeDelta;
            _textDefaultAlignment = instructionText.alignment;
        }

        if (_exampleImageRect != null)
        {
            _exampleImageOriginalParent = _exampleImageRect.parent;
            _exampleDefaultAnchorMin = _exampleImageRect.anchorMin;
            _exampleDefaultAnchorMax = _exampleImageRect.anchorMax;
            _exampleDefaultPivot = _exampleImageRect.pivot;
            _exampleDefaultAnchoredPosition = _exampleImageRect.anchoredPosition;
            _exampleDefaultSizeDelta = _exampleImageRect.sizeDelta;
        }

        _layoutDefaultsCached = true;
    }

    private RectTransform ResolvePanelBackgroundRect()
    {
        Transform panelBg = transform.Find("Panel_BG");
        if (panelBg == null)
            panelBg = transform.Find("PanelBG");

        return panelBg as RectTransform;
    }

    private void ApplyEmbeddedExampleLayout()
    {
        RectTransform panelBgRect = ResolvePanelBackgroundRect();
        if (panelBgRect == null || _instructionTextRect == null || _exampleImageRect == null)
            return;

        if (_exampleImageRect.parent != panelBgRect)
            _exampleImageRect.SetParent(panelBgRect, false);

        float panelWidth = panelBgRect.rect.width > 1f ? panelBgRect.rect.width : panelBgRect.sizeDelta.x;
        float panelHeight = panelBgRect.rect.height > 1f ? panelBgRect.rect.height : panelBgRect.sizeDelta.y;

        float maxBoxWidth = Mathf.Clamp(panelWidth * embeddedExampleWidthRatio, embeddedExampleMinWidth, embeddedExampleMaxWidth);
        float maxBoxHeight = Mathf.Max(48f, panelHeight - embeddedExamplePanelPadding.y * 2f);
        Vector2 fittedImageSize = ComputeEmbeddedImageSize(maxBoxWidth, maxBoxHeight);

        _instructionTextRect.anchorMin = new Vector2(0f, 0f);
        _instructionTextRect.anchorMax = new Vector2(1f, 1f);
        _instructionTextRect.pivot = new Vector2(0.5f, 0.5f);
        instructionText.alignment = TextAlignmentOptions.MidlineLeft;

        float leftPad = embeddedExamplePanelPadding.x;
        float rightPad = embeddedExamplePanelPadding.x;
        float topPad = embeddedExamplePanelPadding.y;
        float bottomPad = embeddedExamplePanelPadding.y;
        float reserve = fittedImageSize.x + embeddedExampleTextGap + Mathf.Max(0f, embeddedExampleExtraTextInset);

        if (embedExampleOnRight)
        {
            _instructionTextRect.offsetMin = new Vector2(leftPad, bottomPad);
            _instructionTextRect.offsetMax = new Vector2(-(rightPad + reserve), -topPad);
        }
        else
        {
            _instructionTextRect.offsetMin = new Vector2(leftPad + reserve, bottomPad);
            _instructionTextRect.offsetMax = new Vector2(-rightPad, -topPad);
        }

        _exampleImageRect.anchorMin = new Vector2(embedExampleOnRight ? 1f : 0f, 0.5f);
        _exampleImageRect.anchorMax = new Vector2(embedExampleOnRight ? 1f : 0f, 0.5f);
        _exampleImageRect.pivot = new Vector2(embedExampleOnRight ? 1f : 0f, 0.5f);
        _exampleImageRect.sizeDelta = fittedImageSize;

        float x = embedExampleOnRight
            ? -rightPad
            : leftPad;
        _exampleImageRect.anchoredPosition = new Vector2(x, 0f) + embeddedExampleOffset;
        LayoutRebuilder.ForceRebuildLayoutImmediate(_instructionTextRect);
    }

    private Vector2 ComputeEmbeddedImageSize(float maxBoxWidth, float maxBoxHeight)
    {
        if (exampleImage == null || exampleImage.sprite == null || !preserveExampleAspect)
            return new Vector2(maxBoxWidth, Mathf.Min(exampleImageSize.y, maxBoxHeight));

        Rect spriteRect = exampleImage.sprite.rect;
        float spriteWidth = Mathf.Max(1f, spriteRect.width);
        float spriteHeight = Mathf.Max(1f, spriteRect.height);
        float aspect = spriteWidth / spriteHeight;

        float width = maxBoxWidth;
        float height = width / aspect;
        if (height > maxBoxHeight)
        {
            height = maxBoxHeight;
            width = height * aspect;
        }

        return new Vector2(width, height);
    }

    private float ComputeEmbeddedExampleHeightHint(float widthHint)
    {
        if (exampleImage == null || exampleImage.sprite == null || !preserveExampleAspect)
            return exampleImageSize.y;

        Rect spriteRect = exampleImage.sprite.rect;
        float spriteWidth = Mathf.Max(1f, spriteRect.width);
        float spriteHeight = Mathf.Max(1f, spriteRect.height);
        float aspect = spriteWidth / spriteHeight;
        return Mathf.Max(1f, widthHint / Mathf.Max(0.01f, aspect));
    }

    private void RestoreDefaultTextLayout()
    {
        if (!_layoutDefaultsCached || _instructionTextRect == null)
            return;

        _instructionTextRect.anchorMin = _textDefaultAnchorMin;
        _instructionTextRect.anchorMax = _textDefaultAnchorMax;
        _instructionTextRect.pivot = _textDefaultPivot;
        _instructionTextRect.sizeDelta = _textDefaultSizeDelta;
        _instructionTextRect.anchoredPosition = _textDefaultAnchoredPosition;
        _instructionTextRect.offsetMin = _textDefaultOffsetMin;
        _instructionTextRect.offsetMax = _textDefaultOffsetMax;
    }

    private void RestoreExampleImageParentAndDefaults()
    {
        if (!_layoutDefaultsCached || _exampleImageRect == null)
            return;

        if (_exampleImageOriginalParent != null && _exampleImageRect.parent != _exampleImageOriginalParent)
            _exampleImageRect.SetParent(_exampleImageOriginalParent, false);

        _exampleImageRect.anchorMin = _exampleDefaultAnchorMin;
        _exampleImageRect.anchorMax = _exampleDefaultAnchorMax;
        _exampleImageRect.pivot = _exampleDefaultPivot;
        _exampleImageRect.sizeDelta = _exampleDefaultSizeDelta;
        _exampleImageRect.anchoredPosition = _exampleDefaultAnchoredPosition;
    }

    private static Camera ResolveUiCamera(RectTransform rect)
    {
        if (rect == null)
            return Camera.main;

        Canvas canvas = rect.GetComponentInParent<Canvas>();
        if (canvas == null)
            return Camera.main;

        if (canvas.renderMode == RenderMode.ScreenSpaceOverlay)
            return null;

        if (canvas.worldCamera != null)
            return canvas.worldCamera;

        return Camera.main;
    }

    public IEnumerator WaitForSpeechCompletionRoutine()
    {
        yield return WaitForSpeechCompletionIfNeeded();
    }

    public IEnumerator WaitForTaskGate(string text, float fallbackSeconds, float speechActivationGraceSeconds = 0.25f)
    {
        float fallback = Mathf.Max(0f, fallbackSeconds);
        if (!CanSynchronizeTaskGateToSpeech(text))
        {
            if (fallback > 0f)
                yield return new WaitForSeconds(fallback);
            yield break;
        }

        float grace = Mathf.Max(0f, speechActivationGraceSeconds);
        float elapsed = 0f;
        while (grace > 0f && !IsSpeechActive())
        {
            if (elapsed >= grace)
                break;

            yield return null;
            elapsed += Time.unscaledDeltaTime;
        }

        if (IsSpeechActive())
        {
            yield return WaitForSpeechCompletionIfNeeded();
            yield break;
        }

        float estimatedSpeechSeconds = EstimateSpeechDurationSeconds(PrepareTextForSpeech(text));
        if (estimatedSpeechSeconds > 0f)
            yield return new WaitForSeconds(estimatedSpeechSeconds);
    }

    IEnumerator ShowHideAnimated(float s)
    {
        if (canvasGroup != null)
            canvasGroup.alpha = 0f;
        if (panelRect != null)
            panelRect.localScale = Vector3.one * Mathf.Max(0.5f, popScaleStart);

        float inDur = Mathf.Max(0.01f, fadeInSeconds);
        float t = 0f;
        while (t < inDur)
        {
            t += Time.unscaledDeltaTime;
            float u = Mathf.Clamp01(t / inDur);
            if (canvasGroup != null) canvasGroup.alpha = u;
            if (panelRect != null)
            {
                float sc = Mathf.Lerp(popScaleStart, popScaleEnd, u);
                panelRect.localScale = Vector3.one * sc;
            }
            yield return null;
        }

        if (float.IsPositiveInfinity(s))
        {
            _hideCo = null;
            yield break;
        }

        float hold = Mathf.Max(0f, s);
        if (hold > 0f)
            yield return new WaitForSeconds(hold);
        yield return WaitForSpeechCompletionIfNeeded();

        float outDur = Mathf.Max(0.01f, fadeOutSeconds);
        t = 0f;
        while (t < outDur)
        {
            t += Time.unscaledDeltaTime;
            float u = Mathf.Clamp01(t / outDur);
            if (canvasGroup != null) canvasGroup.alpha = 1f - u;
            yield return null;
        }

        HideImmediate();
    }

    IEnumerator HideAfter(float s)
    {
        if (float.IsPositiveInfinity(s))
            yield break;

        if (canvasGroup != null) canvasGroup.alpha = 1f;
        if (panelRect != null) panelRect.localScale = Vector3.one * popScaleEnd;
        yield return new WaitForSeconds(Mathf.Max(0f, s));
        yield return WaitForSpeechCompletionIfNeeded();
        HideImmediate();
    }

    IEnumerator HideOverlayAfter(float s)
    {
        if (float.IsPositiveInfinity(s))
        {
            _overlayHideCo = null;
            yield break;
        }

        yield return new WaitForSeconds(Mathf.Max(0f, s));
        HideOverlayImmediate();
        _overlayHideCo = null;
    }

    private void ApplyStyleForText(string text)
    {
        if (instructionText == null)
            return;

        bool isCountdown = IsCountdownText(text);

        if (isCountdown && hideExampleOnCountdown)
            HideExample();

        instructionText.color = isCountdown ? countdownTextColor : normalTextColor;
        if (overrideTypography)
            instructionText.fontSize = isCountdown ? countdownFontSize : normalFontSize;
        instructionText.fontStyle = isCountdown ? countdownFontStyle : normalFontStyle;
        instructionText.alignment = TextAlignmentOptions.Center;
        instructionText.enableWordWrapping = !isCountdown;
        instructionText.overflowMode = TextOverflowModes.Overflow;
    }

    private TMP_Text ResolveOverlayText()
    {
        if (overlayText != null)
        {
            ConfigureOverlayText(overlayText);
            return overlayText;
        }

        if (!autoCreateOverlayText || instructionText == null)
            return null;

        Transform parent = panelRect != null ? panelRect : instructionText.transform.parent;
        if (parent == null)
            parent = transform;

        GameObject go = new GameObject("InstructionHUD_OverlayText", typeof(RectTransform), typeof(CanvasRenderer), typeof(TextMeshProUGUI));
        go.transform.SetParent(parent, false);
        overlayText = go.GetComponent<TMP_Text>();

        if (instructionText.font != null)
            overlayText.font = instructionText.font;
        overlayText.material = instructionText.material;

        ConfigureOverlayText(overlayText);
        overlayText.gameObject.SetActive(false);
        return overlayText;
    }

    private void ConfigureOverlayText(TMP_Text target)
    {
        if (target == null)
            return;

        _overlayTextRect = target.rectTransform;
        if (_overlayTextRect != null)
        {
            _overlayTextRect.anchorMin = new Vector2(0.5f, 1f);
            _overlayTextRect.anchorMax = new Vector2(0.5f, 1f);
            _overlayTextRect.pivot = new Vector2(0.5f, 1f);
            _overlayTextRect.anchoredPosition = overlayAnchoredPosition;
            _overlayTextRect.sizeDelta = overlaySize;
        }

        target.raycastTarget = false;
        ApplyOverlayStyle(target);
    }

    private void ApplyOverlayStyle(TMP_Text target)
    {
        if (target == null)
            return;

        target.color = overlayTextColor;
        target.fontSize = overlayFontSize;
        target.fontStyle = overlayFontStyle;
        target.alignment = TextAlignmentOptions.Center;
        target.enableWordWrapping = false;
        target.overflowMode = TextOverflowModes.Overflow;
    }

    private static bool IsCountdownText(string text)
    {
        string trimmed = string.IsNullOrWhiteSpace(text) ? "" : text.Trim();
        if (string.Equals(trimmed, "Go", System.StringComparison.OrdinalIgnoreCase))
            return true;

        if (int.TryParse(trimmed, out int sec))
            return sec >= 1 && sec <= 20;

        return false;
    }

    private void TrySpeak(string text)
    {
        if (!enableSpeech)
            return;

        string speechText = PrepareTextForSpeech(text);
        if (string.IsNullOrWhiteSpace(speechText))
            return;

        float now = Time.unscaledTime;
        if (now - _lastSpeakTime < Mathf.Max(0f, minSpeakIntervalSeconds))
            return;

        bool isCountdown = IsCountdownText(text);
        if (isCountdown && !speakCountdownText)
            return;
        if (!isCountdown && !speakInstructionText)
            return;

        if (_resolvedTextToSpeech == null)
            ResolveTextToSpeech();

        if (_resolvedTextToSpeech == null)
            return;

        if (stopPreviousSpeechOnShow)
            InvokeIfExists(_resolvedTextToSpeech, "StopSpeaking");

        if (!InvokeWithString(_resolvedTextToSpeech, "StartSpeaking", speechText) &&
            !InvokeWithString(_resolvedTextToSpeech, "Speak", speechText))
        {
            if (!_warnedSpeakMethodMissing)
            {
                Debug.LogWarning("[InstructionHUD] TextToSpeech method not found. Expected StartSpeaking(string) or Speak(string).");
                _warnedSpeakMethodMissing = true;
            }
            return;
        }

        _lastSpeakTime = now;
    }

    private float ResolveDisplaySeconds(string text, float requestedSeconds)
    {
        if (float.IsPositiveInfinity(requestedSeconds))
            return requestedSeconds;

        float requested = Mathf.Max(0f, requestedSeconds);
        if (!syncPanelDurationToSpeech)
            return requested;

        if (!CanSpeakText(text))
            return requested;

        float speechSec = EstimateSpeechDurationSeconds(PrepareTextForSpeech(text));
        if (speechSec <= 0f)
            return requested;

        if (speechSec >= requested)
            return speechSec + Mathf.Max(0f, extraAfterSpeechWhenLonger);

        float remaining = Mathf.Max(0f, requested - speechSec);
        float extra = Mathf.Min(Mathf.Max(0f, maxExtraAfterSpeechWhenShorter), remaining);
        return speechSec + extra;
    }

    private bool CanSpeakText(string text)
    {
        if (!enableSpeech)
            return false;
        if (string.IsNullOrWhiteSpace(text))
            return false;

        bool isCountdown = IsCountdownText(text);
        if (isCountdown && !speakCountdownText)
            return false;
        if (!isCountdown && !speakInstructionText)
            return false;

        if (_resolvedTextToSpeech == null)
            ResolveTextToSpeech();

        return _resolvedTextToSpeech != null;
    }

    private bool CanSynchronizeTaskGateToSpeech(string text)
    {
        if (!syncPanelDurationToSpeech || !waitForSpeechCompletionBeforeHide)
            return false;

        return CanSpeakText(text);
    }

    private float EstimateSpeechDurationSeconds(string text)
    {
        text = PrepareTextForSpeech(text);
        if (string.IsNullOrWhiteSpace(text))
            return 0f;

        string trimmed = text.Trim();
        if (IsCountdownText(trimmed))
            return string.Equals(trimmed, "Go", System.StringComparison.OrdinalIgnoreCase) ? 0.45f : 0.55f;

        int nonWsChars = 0;
        int words = 0;
        bool inWord = false;
        int commas = 0;
        int sentenceStops = 0;
        int lineBreaks = 0;

        for (int i = 0; i < trimmed.Length; i++)
        {
            char ch = trimmed[i];
            if (!char.IsWhiteSpace(ch))
                nonWsChars++;

            bool wordChar = !char.IsWhiteSpace(ch);
            if (wordChar && !inWord)
            {
                words++;
                inWord = true;
            }
            else if (!wordChar)
            {
                inWord = false;
            }

            if (ch == ',' || ch == ';' || ch == ':')
                commas++;
            if (ch == '.' || ch == '!' || ch == '?')
                sentenceStops++;
            if (ch == '\n')
                lineBreaks++;
        }

        float cps = Mathf.Max(1f, estimatedSpeechCharsPerSecond);
        float wps = Mathf.Max(0.5f, estimatedSpeechWordsPerSecond);
        float byChars = nonWsChars / cps;
        float byWords = words / wps;
        float baseDur = Mathf.Max(byChars, byWords);
        float pauseDur = (commas * 0.08f) + (sentenceStops * 0.16f) + (lineBreaks * 0.12f);

        return Mathf.Max(0.35f, baseDur + pauseDur);
    }

    private static string PrepareTextForSpeech(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        System.Text.StringBuilder sb = new System.Text.StringBuilder(text.Length);
        bool prevWasSpace = false;

        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            bool isLineBreak = ch == '\r' || ch == '\n';
            bool isWhitespace = ch == '\t' || char.IsWhiteSpace(ch);

            if (isLineBreak)
            {
                string separator = ChooseSpeechSeparator(text, i);
                for (int s = 0; s < separator.Length; s++)
                    sb.Append(separator[s]);
                prevWasSpace = separator.Length > 0 && char.IsWhiteSpace(separator[separator.Length - 1]);
                continue;
            }

            if (isWhitespace)
            {
                if (!prevWasSpace)
                {
                    sb.Append(' ');
                    prevWasSpace = true;
                }
                continue;
            }

            sb.Append(ch);
            prevWasSpace = false;
        }

        return sb.ToString().Trim();
    }

    private static string ChooseSpeechSeparator(string text, int breakIndex)
    {
        char prev = FindPreviousMeaningfulChar(text, breakIndex - 1);
        char next = FindNextMeaningfulChar(text, breakIndex + 1);

        if (prev == '\0' || next == '\0')
            return " ";

        if (prev == '.' || prev == '!' || prev == '?')
            return " ";

        if (prev == ',' || prev == ';' || prev == ':')
            return " ";

        if (char.IsUpper(next))
            return ". ";

        return " ";
    }

    private static char FindPreviousMeaningfulChar(string text, int startIndex)
    {
        for (int i = startIndex; i >= 0; i--)
        {
            char ch = text[i];
            if (!char.IsWhiteSpace(ch) && ch != '\r' && ch != '\n')
                return ch;
        }

        return '\0';
    }

    private static char FindNextMeaningfulChar(string text, int startIndex)
    {
        for (int i = startIndex; i < text.Length; i++)
        {
            char ch = text[i];
            if (!char.IsWhiteSpace(ch) && ch != '\r' && ch != '\n')
                return ch;
        }

        return '\0';
    }

    private void ResolveTextToSpeech()
    {
        if (_resolvedTextToSpeech != null)
            return;
        if (_ttsResolveAttempted)
            return;
        _ttsResolveAttempted = true;

        if (textToSpeechComponent != null)
        {
            _resolvedTextToSpeech = textToSpeechComponent;
            _resolvedSpeechAudioSource = ResolveSpeechAudioSource(_resolvedTextToSpeech);
            return;
        }

        System.Type ttsType = FindTypeByName("Microsoft.MixedReality.Toolkit.Audio.TextToSpeech");
        if (ttsType == null)
            ttsType = FindTypeByName("TextToSpeech");

        if (ttsType != null)
        {
            _resolvedTextToSpeech = GetComponent(ttsType) as Component;
            if (_resolvedTextToSpeech == null)
                _resolvedTextToSpeech = GetComponentInChildren(ttsType, true) as Component;
            if (_resolvedTextToSpeech != null)
                _resolvedSpeechAudioSource = ResolveSpeechAudioSource(_resolvedTextToSpeech);
        }

        if (_resolvedTextToSpeech == null && !_warnedNoTextToSpeech && enableSpeech)
        {
            Debug.LogWarning("[InstructionHUD] TextToSpeech component was not found. Speech output is disabled until one is assigned.");
            _warnedNoTextToSpeech = true;
        }
    }

    private IEnumerator WaitForSpeechCompletionIfNeeded()
    {
        if (!syncPanelDurationToSpeech || !waitForSpeechCompletionBeforeHide || !enableSpeech)
            yield break;

        float timeout = Mathf.Max(0f, maxSpeechCompletionWaitSeconds);
        float elapsed = 0f;
        while (IsSpeechActive())
        {
            if (timeout > 0f && elapsed >= timeout)
                yield break;

            yield return null;
            elapsed += Time.unscaledDeltaTime;
        }
    }

    private bool IsSpeechActive()
    {
        if (_resolvedTextToSpeech == null)
            return false;

        if (TryInvokeBool(_resolvedTextToSpeech, "IsSpeaking", out bool speakingByMethod))
            return speakingByMethod;

        if (TryGetBoolMember(_resolvedTextToSpeech, "IsSpeaking", out bool speakingByMember))
            return speakingByMember;

        if (_resolvedSpeechAudioSource == null)
            _resolvedSpeechAudioSource = ResolveSpeechAudioSource(_resolvedTextToSpeech);

        return _resolvedSpeechAudioSource != null && _resolvedSpeechAudioSource.isPlaying;
    }

    private static bool TryInvokeBool(Component target, string methodName, out bool value)
    {
        value = false;
        if (target == null)
            return false;

        MethodInfo method = target.GetType().GetMethod(
            methodName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            types: System.Type.EmptyTypes,
            modifiers: null
        );
        if (method == null || method.ReturnType != typeof(bool))
            return false;

        value = (bool)method.Invoke(target, null);
        return true;
    }

    private static bool TryGetBoolMember(Component target, string memberName, out bool value)
    {
        value = false;
        if (target == null)
            return false;

        BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        PropertyInfo prop = target.GetType().GetProperty(memberName, flags);
        if (prop != null && prop.PropertyType == typeof(bool) && prop.GetIndexParameters().Length == 0)
        {
            object v = prop.GetValue(target, null);
            if (v is bool b)
            {
                value = b;
                return true;
            }
        }

        FieldInfo field = target.GetType().GetField(memberName, flags);
        if (field != null && field.FieldType == typeof(bool))
        {
            object v = field.GetValue(target);
            if (v is bool b)
            {
                value = b;
                return true;
            }
        }

        return false;
    }

    private static AudioSource ResolveSpeechAudioSource(Component target)
    {
        if (target == null)
            return null;

        BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        string[] preferredNames = { "AudioSource", "audioSource", "audioSrc", "source" };

        for (int i = 0; i < preferredNames.Length; i++)
        {
            string n = preferredNames[i];
            PropertyInfo p = target.GetType().GetProperty(n, flags);
            if (p != null && typeof(AudioSource).IsAssignableFrom(p.PropertyType) && p.GetIndexParameters().Length == 0)
            {
                object v = p.GetValue(target, null);
                if (v is AudioSource a1)
                    return a1;
            }

            FieldInfo f = target.GetType().GetField(n, flags);
            if (f != null && typeof(AudioSource).IsAssignableFrom(f.FieldType))
            {
                object v = f.GetValue(target);
                if (v is AudioSource a2)
                    return a2;
            }
        }

        PropertyInfo[] props = target.GetType().GetProperties(flags);
        for (int i = 0; i < props.Length; i++)
        {
            PropertyInfo p = props[i];
            if (!typeof(AudioSource).IsAssignableFrom(p.PropertyType) || p.GetIndexParameters().Length != 0)
                continue;

            object v = p.GetValue(target, null);
            if (v is AudioSource a)
                return a;
        }

        FieldInfo[] fields = target.GetType().GetFields(flags);
        for (int i = 0; i < fields.Length; i++)
        {
            FieldInfo f = fields[i];
            if (!typeof(AudioSource).IsAssignableFrom(f.FieldType))
                continue;

            object v = f.GetValue(target);
            if (v is AudioSource a)
                return a;
        }

        return null;
    }

    private static bool InvokeWithString(Component target, string methodName, string arg)
    {
        if (target == null)
            return false;

        MethodInfo method = target.GetType().GetMethod(
            methodName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(string) },
            modifiers: null
        );

        if (method == null)
            return false;

        method.Invoke(target, new object[] { arg });
        return true;
    }

    private static void InvokeIfExists(Component target, string methodName)
    {
        if (target == null)
            return;

        MethodInfo method = target.GetType().GetMethod(
            methodName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            types: System.Type.EmptyTypes,
            modifiers: null
        );

        if (method != null)
            method.Invoke(target, null);
    }

    private static System.Type FindTypeByName(string typeName)
    {
        if (string.IsNullOrWhiteSpace(typeName))
            return null;

        var assemblies = System.AppDomain.CurrentDomain.GetAssemblies();
        for (int i = 0; i < assemblies.Length; i++)
        {
            var t = assemblies[i].GetType(typeName, false);
            if (t != null)
                return t;
        }

        for (int i = 0; i < assemblies.Length; i++)
        {
            System.Type[] types;
            try
            {
                types = assemblies[i].GetTypes();
            }
            catch (ReflectionTypeLoadException ex)
            {
                types = ex.Types;
            }

            if (types == null)
                continue;

            for (int j = 0; j < types.Length; j++)
            {
                if (types[j] != null && types[j].Name == typeName)
                    return types[j];
            }
        }

        return null;
    }
}
