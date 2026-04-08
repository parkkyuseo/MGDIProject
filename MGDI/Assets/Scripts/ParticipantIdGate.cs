using System;
using System.Collections;
using System.IO;
using System.Reflection;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Collections.Generic;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;

public class ParticipantIdGate : MonoBehaviour
{
    [Header("Flow")]
    [SerializeField] private string runtimeSceneName = "Runtime";
    [SerializeField] private string emptyIdFallback = "TEST";

    [Header("Text (TMP preferred)")]
    [SerializeField] private TMP_Text instructionText;
    [SerializeField] private TMP_Text idDisplayText;
    [SerializeField] private Text instructionTextUGUI;
    [SerializeField] private Text idDisplayTextUGUI;

    [Header("Buttons")]
    [SerializeField] private Button editButton;
    [SerializeField] private Button continueButton;

    [Header("Labels")]
    [SerializeField] private string instructionMessage = "Enter your assigned Participant ID";
    [SerializeField] private string idPrefix = "Participant ID: ";
    [SerializeField] private string idTypingPrefix = "Typing ID: ";

    [Header("Participant ID Format")]
    [SerializeField] private string participantIdPrefix = "P";
    [SerializeField] private bool digitsOnlyAfterPrefix = true;
    [SerializeField] private bool startWithPrefixTemplate = true;

    [Header("Behavior")]
    [SerializeField] private bool allowEmptyContinue = true;
    [SerializeField] private bool autoContinueOnKeyboardDone = true;
    [SerializeField] private bool autoOpenKeyboardOnStart = true;
    [SerializeField] private float autoOpenKeyboardDelaySeconds = 0.15f;
    [SerializeField] private bool hideButtonsInAutoKeyboardFlow = true;

    [Header("Keyboard Hint")]
    [SerializeField] private bool showKeyboardOpenHint = false;
    [SerializeField] private string keyboardOpenHint = "";

    [Header("Button Feedback")]
    [SerializeField] private bool useButtonFeedback = true;
    [SerializeField] private float buttonFeedbackScale = 1.06f;
    [SerializeField] private float buttonFeedbackDuration = 0.10f;
    [SerializeField] private Color buttonFlashColor = new Color(0.78f, 0.95f, 1f, 1f);
    [SerializeField] private bool useInstructionStatusFeedback = true;
    [SerializeField] private float statusMessageSeconds = 0.75f;
    [SerializeField] private string editPressedStatus = "Opening keyboard...";
    [SerializeField] private string continuePressedStatus = "Continuing...";

    [Header("Optional Click Audio")]
    [SerializeField] private AudioSource feedbackAudioSource;
    [SerializeField] private AudioClip buttonClickSfx;
    [SerializeField] [Range(0f, 1f)] private float buttonClickVolume = 0.8f;

    [Header("Auto UI Layout")]
    [SerializeField] private bool autoLayoutUi = true;
    [SerializeField] private Vector2 instructionAnchoredPos = new Vector2(0f, 150f);
    [SerializeField] private Vector2 instructionSize = new Vector2(920f, 120f);
    [SerializeField] private Vector2 idDisplayAnchoredPos = new Vector2(0f, 52f);
    [SerializeField] private Vector2 idDisplaySize = new Vector2(780f, 72f);
    [SerializeField] private Vector2 editButtonAnchoredPos = new Vector2(-150f, -58f);
    [SerializeField] private Vector2 continueButtonAnchoredPos = new Vector2(150f, -58f);
    [SerializeField] private Vector2 buttonSize = new Vector2(220f, 68f);

    [Header("Button Hit Area Tuning")]
    [SerializeField] private Vector2 buttonGraphicRaycastPadding = new Vector2(24f, 16f);
    [SerializeField] private Vector2 buttonHitZonePadding = new Vector2(34f, 24f);
    [SerializeField] private Vector2 directPressRectPadding = new Vector2(28f, 20f);
    [SerializeField] private bool disablePassiveTextRaycast = true;
    [SerializeField] private bool disableButtonLabelRaycast = true;

    [Header("Spawn In Front (Once)")]
    [SerializeField] private bool placeInFrontOnStart = true;
    [SerializeField] private bool detachFromParentOnStart = true;
    [SerializeField] private float spawnDistanceMeters = 1.2f;
    [SerializeField] private float verticalOffsetMeters = -0.05f;
    [SerializeField] private bool faceTowardCameraOnStart = true;
    [SerializeField] private bool invertFacingForWorldCanvas = true;
    [SerializeField] private bool followCameraDuringSpawn = false;
    [SerializeField] private float followCameraSecondsOnStart = 2.0f;
    [SerializeField] private int cameraFindMaxFrames = 120;

    [Header("Debug")]
    [SerializeField] private bool logDebug = false;

    [Header("Direct Finger Press Fallback")]
    [SerializeField] private bool enableDirectFingerPress = true;
    [SerializeField] private bool includePointerRayInDirectPress = false;
    [SerializeField] private float directPressDepthMeters = 0.10f;
    [SerializeField] private float directPressHoldSeconds = 0.04f;
    [SerializeField] private float directPressCooldownSeconds = 0.2f;

    public const string ParticipantPrefKey = "participant_id";
    public const string ParticipantFileName = "participant.txt";

    private string _participantId = "";
    private bool _warnedInstructionMissing;
    private bool _warnedIdDisplayMissing;

    private TouchScreenKeyboard _touchKeyboard;

    private object _mrtkKeyboard;
    private PropertyInfo _mrtkTextProperty;
    private PropertyInfo _mrtkVisibleProperty;
    private bool _mrtkWasVisible;
    private bool _warnedNoEventSystem;
    private bool _warnedNoMrtkInputType;
    private bool _isKeyboardActive;
    private bool _isLoadingRuntimeScene;
    private Coroutine _statusMessageRoutine;
    private readonly Dictionary<Button, Coroutine> _buttonFeedbackRoutineByButton = new Dictionary<Button, Coroutine>();

    private readonly Dictionary<Button, FingerTouchState> _fingerTouchStateByButton = new Dictionary<Button, FingerTouchState>();

    private struct FingerTouchState
    {
        public bool anyInside;
        public bool leftInside;
        public bool rightInside;
        public bool invokedWhileInside;
        public float insideTime;
        public float cooldown;
    }

    private void Awake()
    {
        if (editButton != null)
            editButton.onClick.AddListener(OnEditButton);

        if (continueButton != null)
            continueButton.onClick.AddListener(OnContinueButton);
    }

    private void Start()
    {
        ApplyAutoUiLayout();
        ApplyHitAreaTuning();
        EnsureUiInputCompatibility();
        EnsureCanvasRaycastCompatibility();

        if (placeInFrontOnStart)
            StartCoroutine(PlaceInFrontWhenCameraReady());

        if (startWithPrefixTemplate)
        {
            _participantId = NormalizeParticipantInput(participantIdPrefix);
        }
        else
        {
            _participantId = NormalizeParticipantInput(LoadSavedParticipantId());
            if (string.IsNullOrEmpty(_participantId))
                _participantId = NormalizeParticipantInput(participantIdPrefix);
        }

        RefreshInstructionText();
        RefreshIdDisplay();
        UpdateContinueInteractable();
        ApplyAutoKeyboardFlowState();

        if (autoOpenKeyboardOnStart)
            StartCoroutine(AutoOpenKeyboardAfterDelay());
    }

    private void Update()
    {
        PollTouchKeyboard();
        PollMrtkKeyboard();

        if (enableDirectFingerPress)
            UpdateDirectFingerPress();
    }

    private void OnDestroy()
    {
        if (editButton != null)
            editButton.onClick.RemoveListener(OnEditButton);

        if (continueButton != null)
            continueButton.onClick.RemoveListener(OnContinueButton);
    }

    public void OnEditButton()
    {
        OpenKeyboard(showStatus: true);
    }

    private void OpenKeyboard(bool showStatus)
    {
        if (editButton != null && editButton.gameObject.activeInHierarchy)
            PlayButtonFeedback(editButton);
        if (showStatus)
            ShowTransientInstructionStatus(editPressedStatus, statusMessageSeconds);

        if (logDebug)
            Debug.Log("[ParticipantIdGate] Edit button pressed.");

        bool openedByMrtk = TryOpenMrtkKeyboard();
        if (logDebug)
            Debug.Log($"[ParticipantIdGate] MRTK keyboard open result: {openedByMrtk}.");

        _isKeyboardActive = openedByMrtk;
        RefreshInstructionText();
        RefreshIdDisplay();

        if (openedByMrtk)
            return;

        OpenTouchKeyboard();
        _isKeyboardActive = _touchKeyboard != null;
        RefreshInstructionText();
        RefreshIdDisplay();

        if (logDebug)
            Debug.Log($"[ParticipantIdGate] Touch keyboard open result: {_touchKeyboard != null}.");
    }

    public void OnContinueButton()
    {
        if (_isLoadingRuntimeScene)
            return;

        if (continueButton != null && continueButton.gameObject.activeInHierarchy)
            PlayButtonFeedback(continueButton);
        ShowTransientInstructionStatus(continuePressedStatus, statusMessageSeconds);

        string id = NormalizeParticipantInput(_participantId);
        if (IsPrefixOnlyId(id))
            id = "";
        if (string.IsNullOrEmpty(id))
            id = NormalizeBasicId(emptyIdFallback);
        if (string.IsNullOrEmpty(id))
            id = "TEST";

        _participantId = id;
        SaveParticipantId(_participantId);
        _isKeyboardActive = false;
        RefreshInstructionText();
        RefreshIdDisplay();

        string sceneToLoad = ResolveRuntimeSceneName();
        if (string.IsNullOrEmpty(sceneToLoad))
        {
            Debug.LogError("[ParticipantIdGate] Runtime scene name is empty.");
            return;
        }

        if (!Application.CanStreamedLevelBeLoaded(sceneToLoad))
        {
            Debug.LogError($"[ParticipantIdGate] Scene '{sceneToLoad}' is not in Build Settings.");
            return;
        }

        if (logDebug)
            Debug.Log($"[ParticipantIdGate] Loading scene '{sceneToLoad}' with participant_id='{_participantId}'.");

        _isLoadingRuntimeScene = true;
        StartCoroutine(LoadSceneAfterFeedback(sceneToLoad));
    }

    private string ResolveRuntimeSceneName()
    {
        if (!string.IsNullOrWhiteSpace(runtimeSceneName))
        {
            string configured = runtimeSceneName.Trim();
            if (Application.CanStreamedLevelBeLoaded(configured))
                return configured;
        }

        if (Application.CanStreamedLevelBeLoaded("Runtime"))
            return "Runtime";

        if (Application.CanStreamedLevelBeLoaded("RuntimeScene"))
            return "RuntimeScene";

        return "";
    }

    private void PollTouchKeyboard()
    {
        if (_touchKeyboard == null)
            return;

        _isKeyboardActive = true;

        string live = _touchKeyboard.text;
        if (live != null)
            SetParticipantIdFromInput(live);

        if (_touchKeyboard.status == TouchScreenKeyboard.Status.Done)
        {
            _touchKeyboard = null;
            _isKeyboardActive = false;
            RefreshInstructionText();
            RefreshIdDisplay();
            if (autoContinueOnKeyboardDone)
                OnContinueButton();
            return;
        }

        if (_touchKeyboard.status == TouchScreenKeyboard.Status.Canceled ||
            _touchKeyboard.status == TouchScreenKeyboard.Status.LostFocus)
        {
            _touchKeyboard = null;
            _isKeyboardActive = false;
            RefreshInstructionText();
            RefreshIdDisplay();
        }
    }

    private void OpenTouchKeyboard()
    {
        _touchKeyboard = TouchScreenKeyboard.Open(
            text: _participantId ?? "",
            keyboardType: TouchScreenKeyboardType.Default,
            autocorrection: false,
            multiline: false,
            secure: false,
            alert: false,
            textPlaceholder: "Participant ID");

        _isKeyboardActive = _touchKeyboard != null;
        RefreshInstructionText();
        RefreshIdDisplay();

        if (_touchKeyboard == null && logDebug)
            Debug.Log("[ParticipantIdGate] TouchScreenKeyboard open failed.");
    }

    private bool TryOpenMrtkKeyboard()
    {
        Type keyboardType = FindMrtkKeyboardType();
        if (keyboardType == null)
            return false;

        object keyboard = GetOrCreateMrtkKeyboard(keyboardType);
        if (keyboard == null)
            return false;

        _mrtkKeyboard = keyboard;
        _mrtkTextProperty = keyboardType.GetProperty("Text", BindingFlags.Public | BindingFlags.Instance);
        _mrtkVisibleProperty = FindFirstBoolProperty(
            keyboardType,
            "Visible",
            "IsVisible",
            "IsOpen",
            "IsKeyboardVisible");

        if (_mrtkTextProperty != null && _mrtkTextProperty.CanWrite)
            _mrtkTextProperty.SetValue(_mrtkKeyboard, _participantId ?? "");

        if (!InvokeMrtkKeyboardOpen(keyboardType, _mrtkKeyboard, _participantId ?? ""))
        {
            _mrtkKeyboard = null;
            _mrtkTextProperty = null;
            _mrtkVisibleProperty = null;
            return false;
        }

        _mrtkWasVisible = IsMrtkKeyboardVisible();
        return true;
    }

    private void PollMrtkKeyboard()
    {
        if (_mrtkKeyboard == null)
            return;

        if (_mrtkTextProperty != null && _mrtkTextProperty.CanRead)
        {
            object textObj = _mrtkTextProperty.GetValue(_mrtkKeyboard);
            if (textObj is string liveText)
                SetParticipantIdFromInput(liveText);
        }

        if (_mrtkVisibleProperty == null)
        {
            _isKeyboardActive = true;
            return;
        }

        bool visible = IsMrtkKeyboardVisible();
        _isKeyboardActive = visible;
        if (_mrtkWasVisible && !visible)
        {
            _mrtkKeyboard = null;
            _mrtkTextProperty = null;
            _mrtkVisibleProperty = null;
            _mrtkWasVisible = false;
            _isKeyboardActive = false;
            RefreshInstructionText();
            RefreshIdDisplay();
            if (autoContinueOnKeyboardDone)
                OnContinueButton();
            return;
        }

        _mrtkWasVisible = visible;
    }

    private bool IsMrtkKeyboardVisible()
    {
        if (_mrtkKeyboard == null || _mrtkVisibleProperty == null || !_mrtkVisibleProperty.CanRead)
            return false;

        object v = _mrtkVisibleProperty.GetValue(_mrtkKeyboard);
        return v is bool b && b;
    }

    private static Type FindMrtkKeyboardType()
    {
        string[] typeNames =
        {
            "Microsoft.MixedReality.Toolkit.Experimental.UI.MixedRealityKeyboard",
            "Microsoft.MixedReality.Toolkit.Experimental.UI.NonNativeKeyboard",
            "Microsoft.MixedReality.Toolkit.UI.NonNativeKeyboard"
        };

        Assembly[] assemblies = AppDomain.CurrentDomain.GetAssemblies();
        for (int i = 0; i < assemblies.Length; i++)
        {
            for (int j = 0; j < typeNames.Length; j++)
            {
                Type t = assemblies[i].GetType(typeNames[j], throwOnError: false);
                if (t != null)
                    return t;
            }
        }

        return null;
    }

    private static object GetOrCreateMrtkKeyboard(Type keyboardType)
    {
        if (keyboardType == null) return null;

        string fullName = keyboardType.FullName ?? "";
        bool isNonNativeKeyboard =
            fullName.Contains("NonNativeKeyboard", StringComparison.Ordinal);

        PropertyInfo instanceProperty = keyboardType.GetProperty("Instance", BindingFlags.Public | BindingFlags.Static);
        if (instanceProperty != null && instanceProperty.CanRead)
        {
            object staticInstance = instanceProperty.GetValue(null, null);
            if (staticInstance != null)
                return staticInstance;
        }

        UnityEngine.Object existing = UnityEngine.Object.FindObjectOfType(keyboardType);
        if (existing != null)
            return existing;

        // NonNativeKeyboard requires a configured prefab hierarchy.
        // Skip auto-creation and fall back to system keyboard path.
        if (isNonNativeKeyboard)
            return null;

        GameObject go = new GameObject(keyboardType.Name);
        Component created = go.AddComponent(keyboardType);
        return created;
    }

    private static bool InvokeMrtkKeyboardOpen(Type keyboardType, object keyboard, string seedText)
    {
        if (keyboardType == null || keyboard == null)
            return false;

        string[] methodNames = { "PresentKeyboard", "Open", "ShowKeyboard", "Show" };

        for (int i = 0; i < methodNames.Length; i++)
        {
            MethodInfo withStringBool = keyboardType.GetMethod(
                methodNames[i],
                BindingFlags.Public | BindingFlags.Instance,
                binder: null,
                types: new[] { typeof(string), typeof(bool) },
                modifiers: null);

            if (withStringBool != null)
            {
                withStringBool.Invoke(keyboard, new object[] { seedText ?? "", false });
                return true;
            }

            MethodInfo withString = keyboardType.GetMethod(
                methodNames[i],
                BindingFlags.Public | BindingFlags.Instance,
                binder: null,
                types: new[] { typeof(string) },
                modifiers: null);

            if (withString != null)
            {
                withString.Invoke(keyboard, new object[] { seedText ?? "" });
                return true;
            }

            MethodInfo noArg = keyboardType.GetMethod(
                methodNames[i],
                BindingFlags.Public | BindingFlags.Instance,
                binder: null,
                types: Type.EmptyTypes,
                modifiers: null);

            if (noArg != null)
            {
                noArg.Invoke(keyboard, null);
                return true;
            }
        }

        return false;
    }

    private static PropertyInfo FindFirstBoolProperty(Type type, params string[] candidateNames)
    {
        if (type == null || candidateNames == null)
            return null;

        for (int i = 0; i < candidateNames.Length; i++)
        {
            PropertyInfo p = type.GetProperty(candidateNames[i], BindingFlags.Public | BindingFlags.Instance);
            if (p != null && p.PropertyType == typeof(bool))
                return p;
        }

        return null;
    }

    private void SetParticipantIdFromInput(string rawText)
    {
        _participantId = NormalizeParticipantInput(rawText);
        RefreshIdDisplay();
        UpdateContinueInteractable();
    }

    private void RefreshInstructionText()
    {
        string message = instructionMessage ?? "";
        if (_isKeyboardActive && showKeyboardOpenHint && !string.IsNullOrWhiteSpace(keyboardOpenHint))
            message = message + "\n" + keyboardOpenHint.Trim();

        SetText(instructionText, instructionTextUGUI, message, ref _warnedInstructionMissing, "Instruction Text");
    }

    private void RefreshIdDisplay()
    {
        string prefix = _isKeyboardActive ? idTypingPrefix : idPrefix;
        if (string.IsNullOrEmpty(prefix))
            prefix = idPrefix;

        string label = string.IsNullOrEmpty(_participantId) ? "-" : _participantId;
        SetText(idDisplayText, idDisplayTextUGUI, prefix + label, ref _warnedIdDisplayMissing, "ID Display Text");
    }

    private void UpdateContinueInteractable()
    {
        if (continueButton == null)
            return;

        continueButton.interactable = allowEmptyContinue || !string.IsNullOrEmpty(_participantId);
    }

    private void ApplyAutoKeyboardFlowState()
    {
        bool hideButtons = autoOpenKeyboardOnStart && hideButtonsInAutoKeyboardFlow;

        if (editButton != null)
            editButton.gameObject.SetActive(!hideButtons);

        if (continueButton != null)
            continueButton.gameObject.SetActive(!hideButtons);
    }

    private IEnumerator AutoOpenKeyboardAfterDelay()
    {
        float wait = Mathf.Max(0f, autoOpenKeyboardDelaySeconds);
        if (wait > 0f)
            yield return new WaitForSecondsRealtime(wait);

        if (_isLoadingRuntimeScene || _isKeyboardActive)
            yield break;

        OpenKeyboard(showStatus: false);
    }

    private string NormalizeParticipantInput(string raw)
    {
        string prefix = NormalizeBasicId(participantIdPrefix);
        string value = NormalizeBasicId(raw);

        if (string.IsNullOrEmpty(prefix))
            return value;

        if (string.IsNullOrEmpty(value))
            return prefix;

        if (value.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        {
            value = prefix + value.Substring(prefix.Length);
        }
        else
        {
            value = prefix + value;
        }

        if (!digitsOnlyAfterPrefix)
            return value;

        string suffix = value.Length > prefix.Length ? value.Substring(prefix.Length) : "";
        var sb = new StringBuilder(suffix.Length);
        for (int i = 0; i < suffix.Length; i++)
        {
            char c = suffix[i];
            if (char.IsDigit(c))
                sb.Append(c);
        }

        return prefix + sb.ToString();
    }

    private bool IsPrefixOnlyId(string id)
    {
        string normalized = NormalizeBasicId(id);
        if (string.IsNullOrEmpty(normalized))
            return true;

        string prefix = NormalizeBasicId(participantIdPrefix);
        if (string.IsNullOrEmpty(prefix))
            return false;

        if (!digitsOnlyAfterPrefix)
            return false;

        return string.Equals(normalized, prefix, StringComparison.OrdinalIgnoreCase);
    }

    private static string NormalizeBasicId(string id)
    {
        return string.IsNullOrWhiteSpace(id) ? "" : id.Trim();
    }

    private static void SetText(TMP_Text tmp, Text ugui, string value, ref bool warnedMissing, string label)
    {
        bool hasTarget = false;

        if (tmp != null)
        {
            tmp.text = value ?? "";
            hasTarget = true;
        }

        if (ugui != null)
        {
            ugui.text = value ?? "";
            hasTarget = true;
        }

        if (!hasTarget && !warnedMissing)
        {
            warnedMissing = true;
            Debug.LogWarning($"[ParticipantIdGate] {label} reference is missing.");
        }
    }

    private void ApplyAutoUiLayout()
    {
        if (!autoLayoutUi)
            return;

        ConfigureCenteredRect(GetInstructionRect(), instructionAnchoredPos, instructionSize);
        ConfigureCenteredRect(GetIdDisplayRect(), idDisplayAnchoredPos, idDisplaySize);
        ConfigureCenteredRect(editButton != null ? editButton.transform as RectTransform : null, editButtonAnchoredPos, buttonSize);
        ConfigureCenteredRect(continueButton != null ? continueButton.transform as RectTransform : null, continueButtonAnchoredPos, buttonSize);

        if (instructionText != null)
        {
            instructionText.alignment = TextAlignmentOptions.Center;
            instructionText.enableWordWrapping = true;
        }

        if (idDisplayText != null)
        {
            idDisplayText.alignment = TextAlignmentOptions.Center;
            idDisplayText.enableWordWrapping = false;
        }

        if (instructionTextUGUI != null)
            instructionTextUGUI.alignment = TextAnchor.MiddleCenter;

        if (idDisplayTextUGUI != null)
            idDisplayTextUGUI.alignment = TextAnchor.MiddleCenter;
    }

    private RectTransform GetInstructionRect()
    {
        if (instructionText != null)
            return instructionText.rectTransform;
        if (instructionTextUGUI != null)
            return instructionTextUGUI.rectTransform;
        return null;
    }

    private RectTransform GetIdDisplayRect()
    {
        if (idDisplayText != null)
            return idDisplayText.rectTransform;
        if (idDisplayTextUGUI != null)
            return idDisplayTextUGUI.rectTransform;
        return null;
    }

    private static void ConfigureCenteredRect(RectTransform rect, Vector2 anchoredPos, Vector2 size)
    {
        if (rect == null)
            return;

        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = anchoredPos;
        rect.sizeDelta = size;
    }

    private void ApplyHitAreaTuning()
    {
        if (disablePassiveTextRaycast)
        {
            SetGraphicRaycastTarget(instructionText, false);
            SetGraphicRaycastTarget(idDisplayText, false);
            SetGraphicRaycastTarget(instructionTextUGUI, false);
            SetGraphicRaycastTarget(idDisplayTextUGUI, false);
        }

        ConfigureButtonHitArea(editButton);
        ConfigureButtonHitArea(continueButton);
    }

    private void ConfigureButtonHitArea(Button button)
    {
        if (button == null)
            return;

        if (button.targetGraphic != null)
        {
            Vector4 padding = new Vector4(
                Mathf.Max(0f, buttonGraphicRaycastPadding.x),
                Mathf.Max(0f, buttonGraphicRaycastPadding.y),
                Mathf.Max(0f, buttonGraphicRaycastPadding.x),
                Mathf.Max(0f, buttonGraphicRaycastPadding.y));

            button.targetGraphic.raycastPadding = padding;
        }

        if (!disableButtonLabelRaycast)
        {
            EnsureExpandedHitZone(button);
            return;
        }

        Graphic[] graphics = button.GetComponentsInChildren<Graphic>(true);
        for (int i = 0; i < graphics.Length; i++)
        {
            Graphic graphic = graphics[i];
            if (graphic == null || graphic == button.targetGraphic)
                continue;

            graphic.raycastTarget = false;
        }

        EnsureExpandedHitZone(button);
    }

    private static void SetGraphicRaycastTarget(Graphic graphic, bool enabled)
    {
        if (graphic == null)
            return;

        graphic.raycastTarget = enabled;
    }

    private void EnsureExpandedHitZone(Button button)
    {
        if (button == null)
            return;

        RectTransform buttonRect = button.transform as RectTransform;
        if (buttonRect == null)
            return;

        const string hitZoneName = "__ExpandedHitZone";
        Transform existing = button.transform.Find(hitZoneName);
        GameObject hitZoneObject = existing != null ? existing.gameObject : new GameObject(hitZoneName, typeof(RectTransform), typeof(Image));
        if (existing == null)
        {
            hitZoneObject.transform.SetParent(button.transform, false);
            hitZoneObject.transform.SetAsLastSibling();
        }

        RectTransform hitZoneRect = hitZoneObject.transform as RectTransform;
        if (hitZoneRect == null)
            return;

        float padX = Mathf.Max(0f, buttonHitZonePadding.x);
        float padY = Mathf.Max(0f, buttonHitZonePadding.y);

        hitZoneRect.anchorMin = new Vector2(0.5f, 0.5f);
        hitZoneRect.anchorMax = new Vector2(0.5f, 0.5f);
        hitZoneRect.pivot = new Vector2(0.5f, 0.5f);
        hitZoneRect.anchoredPosition = Vector2.zero;
        hitZoneRect.sizeDelta = buttonRect.rect.size + new Vector2(padX * 2f, padY * 2f);

        Image hitZoneImage = hitZoneObject.GetComponent<Image>();
        if (hitZoneImage == null)
            return;

        hitZoneImage.color = new Color(1f, 1f, 1f, 0.001f);
        hitZoneImage.raycastTarget = true;
        hitZoneImage.maskable = false;
    }

    private void PlayButtonFeedback(Button button)
    {
        if (button == null)
            return;

        if (feedbackAudioSource != null && buttonClickSfx != null)
            feedbackAudioSource.PlayOneShot(buttonClickSfx, buttonClickVolume);

        if (!useButtonFeedback)
            return;

        if (_buttonFeedbackRoutineByButton.TryGetValue(button, out Coroutine active) && active != null)
            StopCoroutine(active);

        Coroutine routine = StartCoroutine(CoButtonFeedback(button));
        _buttonFeedbackRoutineByButton[button] = routine;
    }

    private IEnumerator CoButtonFeedback(Button button)
    {
        if (button == null)
            yield break;

        RectTransform rect = button.transform as RectTransform;
        if (rect == null)
            yield break;

        Graphic graphic = button.targetGraphic;
        Vector3 baseScale = rect.localScale;
        Color baseColor = graphic != null ? graphic.color : Color.white;
        float duration = Mathf.Max(0.04f, buttonFeedbackDuration);
        float half = duration * 0.5f;
        Vector3 peakScale = baseScale * Mathf.Max(1f, buttonFeedbackScale);

        float t = 0f;
        while (t < duration)
        {
            t += Time.unscaledDeltaTime;
            if (t <= half)
            {
                float u = Mathf.Clamp01(t / Mathf.Max(0.001f, half));
                rect.localScale = Vector3.Lerp(baseScale, peakScale, u);
                if (graphic != null)
                    graphic.color = Color.Lerp(baseColor, buttonFlashColor, u);
            }
            else
            {
                float u = Mathf.Clamp01((t - half) / Mathf.Max(0.001f, half));
                rect.localScale = Vector3.Lerp(peakScale, baseScale, u);
                if (graphic != null)
                    graphic.color = Color.Lerp(buttonFlashColor, baseColor, u);
            }

            yield return null;
        }

        rect.localScale = baseScale;
        if (graphic != null)
            graphic.color = baseColor;

        _buttonFeedbackRoutineByButton[button] = null;
    }

    private void ShowTransientInstructionStatus(string message, float seconds)
    {
        if (!useInstructionStatusFeedback || string.IsNullOrWhiteSpace(message))
            return;

        if (_statusMessageRoutine != null)
            StopCoroutine(_statusMessageRoutine);

        _statusMessageRoutine = StartCoroutine(CoShowTransientInstructionStatus(message, seconds));
    }

    private IEnumerator CoShowTransientInstructionStatus(string message, float seconds)
    {
        SetText(instructionText, instructionTextUGUI, message, ref _warnedInstructionMissing, "Instruction Text");
        yield return new WaitForSecondsRealtime(Mathf.Max(0.05f, seconds));
        _statusMessageRoutine = null;
        RefreshInstructionText();
    }

    private IEnumerator LoadSceneAfterFeedback(string sceneName)
    {
        float delay = useButtonFeedback ? Mathf.Max(0.04f, buttonFeedbackDuration * 0.75f) : 0f;
        if (delay > 0f)
            yield return new WaitForSecondsRealtime(delay);

        SceneManager.LoadScene(sceneName);
    }

    private static string GetParticipantFilePath()
    {
        return Path.Combine(Application.persistentDataPath, ParticipantFileName);
    }

    private void EnsureUiInputCompatibility()
    {
        EventSystem eventSystem = EventSystem.current;
        if (eventSystem == null)
            eventSystem = UnityEngine.Object.FindObjectOfType<EventSystem>();

        if (eventSystem == null)
        {
            GameObject go = new GameObject("EventSystem");
            eventSystem = go.AddComponent<EventSystem>();
            go.AddComponent<StandaloneInputModule>();

            if (!_warnedNoEventSystem)
            {
                _warnedNoEventSystem = true;
                Debug.LogWarning("[ParticipantIdGate] EventSystem was missing and has been created.");
            }
        }

        Type mrtkInputModuleType = FindTypeByName("Microsoft.MixedReality.Toolkit.Input.MixedRealityInputModule");
        if (mrtkInputModuleType != null)
        {
            Component mrtkModule = eventSystem.GetComponent(mrtkInputModuleType);
            if (mrtkModule == null)
                mrtkModule = eventSystem.gameObject.AddComponent(mrtkInputModuleType);

            if (mrtkModule is BaseInputModule baseInputModule)
                baseInputModule.enabled = true;

            StandaloneInputModule[] standaloneModules = eventSystem.GetComponents<StandaloneInputModule>();
            for (int i = 0; i < standaloneModules.Length; i++)
            {
                StandaloneInputModule module = standaloneModules[i];
                if (module == null)
                    continue;

                // Keep MRTK module enabled, disable only non-MRTK standalone modules.
                if (module.GetType() == mrtkInputModuleType)
                    continue;

                module.enabled = false;
            }
        }
        else if (!_warnedNoMrtkInputType)
        {
            _warnedNoMrtkInputType = true;
            Debug.LogWarning("[ParticipantIdGate] MRTK MixedRealityInputModule type was not found. UI click may be limited on device.");
        }

        RemoveButtonNearTouchable(editButton);
        RemoveButtonNearTouchable(continueButton);
    }

    private static void RemoveButtonNearTouchable(Button button)
    {
        if (button == null)
            return;

        Type nearTouchableType = FindTypeByName("Microsoft.MixedReality.Toolkit.Input.NearInteractionTouchableUnityUI");
        if (nearTouchableType == null)
            return;

        Component nearTouchable = button.gameObject.GetComponent(nearTouchableType);
        if (nearTouchable != null)
            UnityEngine.Object.Destroy(nearTouchable);
    }

    private void EnsureCanvasRaycastCompatibility()
    {
        var canvases = new HashSet<Canvas>();

        if (instructionText != null && instructionText.canvas != null)
            canvases.Add(instructionText.canvas);
        if (idDisplayText != null && idDisplayText.canvas != null)
            canvases.Add(idDisplayText.canvas);
        if (instructionTextUGUI != null)
        {
            Canvas c = instructionTextUGUI.GetComponentInParent<Canvas>(true);
            if (c != null) canvases.Add(c);
        }
        if (idDisplayTextUGUI != null)
        {
            Canvas c = idDisplayTextUGUI.GetComponentInParent<Canvas>(true);
            if (c != null) canvases.Add(c);
        }
        if (editButton != null)
        {
            Canvas c = editButton.GetComponentInParent<Canvas>(true);
            if (c != null) canvases.Add(c);
        }
        if (continueButton != null)
        {
            Canvas c = continueButton.GetComponentInParent<Canvas>(true);
            if (c != null) canvases.Add(c);
        }

        Type canvasUtilityType = FindTypeByName("Microsoft.MixedReality.Toolkit.Input.Utilities.CanvasUtility");
        Type nearTouchableType = FindTypeByName("Microsoft.MixedReality.Toolkit.Input.NearInteractionTouchableUnityUI");
        Camera cam = canvasUtilityType == null ? ResolveCamera() : null;

        foreach (Canvas canvas in canvases)
        {
            if (canvas == null)
                continue;

            GraphicRaycaster raycaster = canvas.GetComponent<GraphicRaycaster>();
            if (raycaster != null)
            {
                // Accept hits regardless of facing to avoid one-sided panel misses.
                raycaster.ignoreReversedGraphics = false;
            }

            if (canvasUtilityType != null)
            {
                if (canvas.renderMode == RenderMode.WorldSpace)
                    canvas.worldCamera = null;

                if (canvas.GetComponent(canvasUtilityType) == null)
                    canvas.gameObject.AddComponent(canvasUtilityType);
            }
            else if (canvas.renderMode == RenderMode.WorldSpace && cam != null)
            {
                canvas.worldCamera = cam;
            }

            if (nearTouchableType != null && canvas.GetComponentInChildren(nearTouchableType, true) == null)
            {
                canvas.gameObject.AddComponent(nearTouchableType);
            }
        }
    }

    private static Type FindTypeByName(string fullName)
    {
        if (string.IsNullOrEmpty(fullName))
            return null;

        Assembly[] assemblies = AppDomain.CurrentDomain.GetAssemblies();
        for (int i = 0; i < assemblies.Length; i++)
        {
            Type t = assemblies[i].GetType(fullName, throwOnError: false);
            if (t != null)
                return t;
        }

        return null;
    }

    private IEnumerator PlaceInFrontWhenCameraReady()
    {
        int maxFrames = Mathf.Max(1, cameraFindMaxFrames);
        Camera cam = null;

        for (int i = 0; i < maxFrames; i++)
        {
            cam = ResolveCamera();
            if (cam != null)
                break;

            yield return null;
        }

        if (cam == null)
        {
            Debug.LogWarning("[ParticipantIdGate] Main Camera not found. Start panel placement was skipped.");
            yield break;
        }

        if (detachFromParentOnStart && transform.parent != null)
            transform.SetParent(null, true);

        float followSec = followCameraDuringSpawn ? Mathf.Max(0f, followCameraSecondsOnStart) : 0f;
        float endTime = Time.unscaledTime + followSec;

        do
        {
            Camera resolved = ResolveCamera();
            if (resolved != null)
                cam = resolved;

            PlaceFromCamera(cam);
            yield return null;
        }
        while (Time.unscaledTime < endTime);

        PlaceFromCamera(cam);
    }

    private void PlaceFromCamera(Camera cam)
    {
        if (cam == null)
            return;

        float d = Mathf.Max(0.3f, spawnDistanceMeters);
        Vector3 pos = cam.transform.position + cam.transform.forward * d + cam.transform.up * verticalOffsetMeters;
        transform.position = pos;

        if (!faceTowardCameraOnStart)
            return;

        Vector3 toCamera = cam.transform.position - pos;
        if (toCamera.sqrMagnitude < 1e-6f)
            return;

        Vector3 forward = toCamera.normalized;
        if (invertFacingForWorldCanvas)
            forward = -forward;

        transform.rotation = Quaternion.LookRotation(forward, cam.transform.up);
    }

    private static Camera ResolveCamera()
    {
        if (Camera.main != null)
            return Camera.main;

        Camera[] cameras = UnityEngine.Object.FindObjectsOfType<Camera>(includeInactive: false);
        for (int i = 0; i < cameras.Length; i++)
        {
            Camera c = cameras[i];
            if (c != null && c.enabled && c.gameObject.activeInHierarchy)
                return c;
        }

        return null;
    }

    private string LoadSavedParticipantId()
    {
        string id = NormalizeBasicId(PlayerPrefs.GetString(ParticipantPrefKey, ""));
        if (!string.IsNullOrEmpty(id))
            return id;

        try
        {
            string filePath = GetParticipantFilePath();
            if (!File.Exists(filePath))
                return "";

            string fromFile = NormalizeBasicId(File.ReadAllText(filePath, Encoding.UTF8));
            return fromFile;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[ParticipantIdGate] Failed to read participant file: {e.Message}");
            return "";
        }
    }

    private static void SaveParticipantId(string id)
    {
        string normalized = NormalizeBasicId(id);
        PlayerPrefs.SetString(ParticipantPrefKey, normalized);
        PlayerPrefs.Save();

        try
        {
            File.WriteAllText(GetParticipantFilePath(), normalized, Encoding.UTF8);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[ParticipantIdGate] Failed to write participant file: {e.Message}");
        }
    }

    private void UpdateDirectFingerPress()
    {
        float dt = Mathf.Max(Time.unscaledDeltaTime, 1e-4f);
        ProcessDirectFingerPress(editButton, dt);
        ProcessDirectFingerPress(continueButton, dt);
    }

    private void ProcessDirectFingerPress(Button button, float dt)
    {
        if (button == null || !button.gameObject.activeInHierarchy || !button.interactable)
            return;

        RectTransform rect = button.transform as RectTransform;
        if (rect == null)
            return;

        if (!_fingerTouchStateByButton.TryGetValue(button, out FingerTouchState state))
            state = default;

        state.cooldown = Mathf.Max(0f, state.cooldown - dt);

        bool leftInside = IsFingerInsideButton(rect, Handedness.Left);
        bool rightInside = IsFingerInsideButton(rect, Handedness.Right);

        bool pointerInside = includePointerRayInDirectPress && IsAnyHandPointerInsideButton(rect);
        bool anyInside = leftInside || rightInside || pointerInside;
        bool pressedNow = anyInside && !state.anyInside;
        bool releasedNow = !anyInside && state.anyInside;

        if (anyInside)
        {
            state.insideTime = state.anyInside ? state.insideTime + dt : dt;
        }
        else
        {
            state.insideTime = 0f;
        }

        float holdSeconds = Mathf.Max(0.01f, directPressHoldSeconds);
        bool holdTriggered = anyInside && !state.invokedWhileInside && state.insideTime >= holdSeconds;

        if ((pressedNow || holdTriggered) && state.cooldown <= 0f)
        {
            button.onClick?.Invoke();
            state.cooldown = Mathf.Max(0.05f, directPressCooldownSeconds);
            state.invokedWhileInside = true;

            if (logDebug)
                Debug.Log($"[ParticipantIdGate] Direct finger press invoked: {button.gameObject.name} (pressedNow={pressedNow}, holdTriggered={holdTriggered})");
        }
        else if (releasedNow)
        {
            state.invokedWhileInside = false;
        }

        state.anyInside = anyInside;
        state.leftInside = leftInside;
        state.rightInside = rightInside;
        _fingerTouchStateByButton[button] = state;
    }

    private bool IsFingerInsideButton(RectTransform rect, Handedness hand)
    {
        if (!HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexTip, hand, out MixedRealityPose pose))
            return false;

        Vector3 local = rect.InverseTransformPoint(pose.Position);
        Rect r = GetExpandedButtonRect(rect.rect);

        bool inRect =
            local.x >= r.xMin && local.x <= r.xMax &&
            local.y >= r.yMin && local.y <= r.yMax;

        float depth = Mathf.Max(0.005f, directPressDepthMeters);
        bool inDepth = local.z >= -depth && local.z <= depth;

        return inRect && inDepth;
    }

    private bool IsAnyHandPointerInsideButton(RectTransform rect)
    {
        IMixedRealityInputSystem inputSystem = CoreServices.InputSystem;
        if (inputSystem == null)
            return false;

        IEnumerable<IMixedRealityController> controllers = inputSystem.DetectedControllers;
        if (controllers == null)
            return false;

        foreach (IMixedRealityController controller in controllers)
        {
            if (controller == null)
                continue;

            if (controller.ControllerHandedness == Handedness.None)
                continue;

            IMixedRealityInputSource inputSource = controller.InputSource;
            if (inputSource == null || inputSource.Pointers == null)
                continue;

            IMixedRealityPointer[] pointers = inputSource.Pointers;
            for (int j = 0; j < pointers.Length; j++)
            {
                IMixedRealityPointer pointer = pointers[j];
                if (pointer == null)
                    continue;

                if (IsWorldPointInsideButton(rect, pointer.Position))
                    return true;
            }
        }

        return false;
    }

    private bool IsWorldPointInsideButton(RectTransform rect, Vector3 worldPoint)
    {
        Vector3 local = rect.InverseTransformPoint(worldPoint);
        Rect r = GetExpandedButtonRect(rect.rect);

        bool inRect =
            local.x >= r.xMin && local.x <= r.xMax &&
            local.y >= r.yMin && local.y <= r.yMax;

        float depth = Mathf.Max(0.005f, directPressDepthMeters);
        bool inDepth = local.z >= -depth && local.z <= depth;
        return inRect && inDepth;
    }

    private Rect GetExpandedButtonRect(Rect rect)
    {
        float padX = Mathf.Max(0f, directPressRectPadding.x);
        float padY = Mathf.Max(0f, directPressRectPadding.y);
        return Rect.MinMaxRect(rect.xMin - padX, rect.yMin - padY, rect.xMax + padX, rect.yMax + padY);
    }
}
