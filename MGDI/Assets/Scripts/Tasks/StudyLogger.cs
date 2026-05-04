using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class StudyLogger : MonoBehaviour
{
    private const string ParticipantPrefKey = "participant_id";
    private const string ParticipantFileName = "participant.txt";

    [Header("Logging")]
    [SerializeField] private bool enableLogging = true;
    public bool LoggingEnabled { get; set; } = true;
    [SerializeField] private string participantIdFallback = "TEST";

    [Header("Refs")]
    [SerializeField] private StudyFlowController_V2 flow;
    [SerializeField] private ToolPlacementTaskManager placementTask;
    [SerializeField] private ToolRotationTaskManager rotationTask;
    [SerializeField] private ToolScalingTaskManager scalingTask;
    [SerializeField] private PhoneInputRouter router;
    [SerializeField] private PhonePoseStreamReceiver phonePoseReceiver;

    [Header("Effort Source (Macro)")]
    [Tooltip("If unset, active tool transform is used while a trial is running.")]
    [SerializeField] private Transform macroEffortTransform;

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

    [Header("Mirror To Laptop (Optional)")]
    [SerializeField] private bool enableMirrorSend = false;
    [SerializeField] private string mirrorHost = "10.138.130.118";
    [SerializeField] private int mirrorPort = 19620;
    [SerializeField] private bool mirrorRequireAck = true;
    [SerializeField] private int mirrorConnectTimeoutMs = 1200;
    [SerializeField] private int mirrorAckTimeoutMs = 1500;
    [SerializeField] private float mirrorRetryIntervalSeconds = 1.5f;
    [SerializeField] private int mirrorMaxSendsPerUpdate = 1;
    [SerializeField] private int mirrorQueueHardLimit = 5000;
    [SerializeField] private bool mirrorSendOnBackgroundThread = true;

    private enum TaskKind
    {
        None = 0,
        Placement = 1,
        Rotation = 2,
        Scaling = 3
    }

    private enum TechniqueKind
    {
        Unknown = 0,
        Macro = 1,
        Micro = 2
    }

    [Serializable]
    private class MirrorRowEnvelope
    {
        public string type = "trial_row";
        public string protocol = "study_logger_mirror_v1";
        public string row_id;
        public string session_timestamp;
        public string participant_id;
        public string csv_header;
        public string csv_row;
        public string csv_path;
        public long created_unix_ms;
    }

    [Serializable]
    private class MirrorAckEnvelope
    {
        public string type;
        public string row_id;
        public bool ok;
        public string status;
        public string error;
    }

    private const string CsvHeader =
        "session_timestamp,participant_id,task,technique,hand_location,condition_label,condition_index,condition_total,condition_order,condition_sequence_index,condition_sequence_total,tool_id," +
        "completion_time_s," +
        "translation_error_cm,rotation_error_deg,scaling_error_pct," +
        "time_to_first_within_tol_s,eligible_breaks," +
        "micro_axis_active_duration_s,micro_axis_integral," +
        "macro_path_length_m,phone_path_length_m," +
        "mode_switch_count";

    private StreamWriter _writer;
    private string _sessionTimestamp;
    private string _participantId = "TEST";
    private string _filePath;
    private PhoneInputRouter _subscribedRouter;

    private bool _prevPlacementRunning;
    private bool _prevRotationRunning;
    private bool _prevScalingRunning;

    private bool _trialActive;
    private TaskKind _trialTask = TaskKind.None;
    private TechniqueKind _trialTechnique = TechniqueKind.Unknown;
    private string _trialHandLocation = "Unknown";
    private string _trialConditionLabel = "Unknown";
    private int _trialConditionIndex = -1;
    private int _trialConditionTotal = -1;
    private string _trialConditionOrder = string.Empty;
    private int _trialConditionSequenceIndex = -1;
    private int _trialConditionSequenceTotal = -1;
    private string _trialToolId = "Unknown";
    private float _trialStartTime;

    private float _firstWithinTolTime = -1f;
    private bool _hasEnteredWithinTol;
    private bool _wasWithinTol;
    private int _eligibleBreaks;

    private float _microAxisActiveDuration;
    private float _microAxisIntegral;
    private float _macroPathLength;
    private double _trialPhonePathLengthBaselineMeters;
    private bool _hasTrialPhonePathLengthBaseline;
    private int _modeSwitchCount;

    private bool _hasPrevMacroPos;
    private Vector3 _prevMacroPos;

    private bool _warnedMissingWriter;
    private bool _warnedMissingMicroRouter;
    private bool _warnedMissingMacroEffortTransform;
    private bool _warnedMissingPhonePoseReceiver;
    private bool _warnedMissingTasks;
    private bool _warnedMirrorConfig;
    private bool EffectiveLoggingEnabled => enableLogging && LoggingEnabled;

    private readonly List<MirrorRowEnvelope> _mirrorQueue = new List<MirrorRowEnvelope>();
    private bool _mirrorOutboxLoaded;
    private string _mirrorOutboxPath;
    private float _nextMirrorAttemptAt;
    private readonly object _mirrorSendStateLock = new object();
    private bool _mirrorSendInFlight;
    private bool _mirrorSendResultReady;
    private bool _mirrorSendResultOk;
    private string _mirrorSendResultRowId;

    private void Awake()
    {
        ResolveRefs();
        _participantId = ResolveParticipantId();
        EnsureMirrorOutboxLoaded();
        if (EffectiveLoggingEnabled)
            InitCsv();
    }

    private void OnEnable()
    {
        if (!enableLogging) return;
        ResolveRefs();
        SubscribeRouter();
    }

    private void OnDisable()
    {
        UnsubscribeRouter();
        SaveMirrorOutbox();
    }

    private void OnDestroy()
    {
        SaveMirrorOutbox();
        CloseWriter();
    }

    private void Update()
    {
        if (!enableLogging)
        {
            if (_writer != null)
                CloseWriter();

            UnsubscribeRouter();
            _trialActive = false;
            ProcessMirrorQueue();
            return;
        }

        if (_writer == null && EffectiveLoggingEnabled)
            InitCsv();

        ResolveRefs();
        SubscribeRouter();

        bool placementRunning = placementTask != null && placementTask.IsTrialRunning;
        bool rotationRunning = rotationTask != null && rotationTask.IsTrialRunning;
        bool scalingRunning = scalingTask != null && scalingTask.IsTrialRunning;

        if (!_warnedMissingTasks && placementTask == null && rotationTask == null && scalingTask == null)
        {
            _warnedMissingTasks = true;
            Debug.LogWarning("[StudyLogger] No task manager references were found. Trial logging is inactive.");
        }

        if (!_trialActive)
        {
            TryStartTrialFromEdges(
                placementRunning && !_prevPlacementRunning,
                rotationRunning && !_prevRotationRunning,
                scalingRunning && !_prevScalingRunning);
        }
        else
        {
            if (IsCurrentTrialStillRunning())
            {
                UpdateRunningTrialMetrics(Mathf.Max(Time.deltaTime, 0f));
            }
            else
            {
                EndTrialAndWriteRow();
                TryStartTrialFromEdges(
                    placementRunning && !_prevPlacementRunning,
                    rotationRunning && !_prevRotationRunning,
                    scalingRunning && !_prevScalingRunning);
            }
        }

        _prevPlacementRunning = placementRunning;
        _prevRotationRunning = rotationRunning;
        _prevScalingRunning = scalingRunning;

        ProcessMirrorQueue();
    }

    private void ResolveRefs()
    {
        if (flow == null) flow = FindFirstObjectByType<StudyFlowController_V2>();
        if (placementTask == null) placementTask = FindFirstObjectByType<ToolPlacementTaskManager>();
        if (rotationTask == null) rotationTask = FindFirstObjectByType<ToolRotationTaskManager>();
        if (scalingTask == null) scalingTask = FindFirstObjectByType<ToolScalingTaskManager>();

        if (router == null)
        {
            router = FindFirstObjectByType<PhoneInputRouter>();
        }

        if (phonePoseReceiver == null)
        {
            phonePoseReceiver = FindFirstObjectByType<PhonePoseStreamReceiver>();
        }
    }

    private void SubscribeRouter()
    {
        if (router == null) return;
        if (_subscribedRouter == router) return;

        UnsubscribeRouter();
        router.OnModeToggle += HandleModeToggle;
        _subscribedRouter = router;
    }

    private void UnsubscribeRouter()
    {
        if (_subscribedRouter == null) return;
        _subscribedRouter.OnModeToggle -= HandleModeToggle;
        _subscribedRouter = null;
    }

    private void HandleModeToggle()
    {
        if (!_trialActive) return;
        if (_trialTechnique != TechniqueKind.Micro) return;
        _modeSwitchCount++;
    }

    private void TryStartTrialFromEdges(bool placementRising, bool rotationRising, bool scalingRising)
    {
        if (placementRising)
        {
            BeginTrial(TaskKind.Placement);
            return;
        }

        if (rotationRising)
        {
            BeginTrial(TaskKind.Rotation);
            return;
        }

        if (scalingRising)
        {
            BeginTrial(TaskKind.Scaling);
        }
    }

    private void BeginTrial(TaskKind task)
    {
        _trialActive = true;
        _trialTask = task;
        _trialTechnique = GetCurrentTechnique();
        _trialConditionLabel = GetCurrentConditionLabel();
        _trialHandLocation = ExtractHandLocationFromCondition(_trialConditionLabel);
        _trialConditionIndex = GetCurrentConditionIndex1Based();
        _trialConditionTotal = GetCurrentConditionCount();
        _trialConditionOrder = GetCurrentConditionOrderLabel();
        _trialConditionSequenceIndex = GetCurrentConditionSequenceIndex1Based();
        _trialConditionSequenceTotal = GetCurrentConditionSequenceCount();
        _trialToolId = GetCurrentToolId(task);
        _trialStartTime = Time.time;

        _firstWithinTolTime = -1f;
        _hasEnteredWithinTol = false;
        _wasWithinTol = false;
        _eligibleBreaks = 0;

        _microAxisActiveDuration = 0f;
        _microAxisIntegral = 0f;
        _macroPathLength = 0f;
        _trialPhonePathLengthBaselineMeters = 0.0;
        _hasTrialPhonePathLengthBaseline = false;
        _modeSwitchCount = 0;

        if (phonePoseReceiver != null)
        {
            _trialPhonePathLengthBaselineMeters = phonePoseReceiver.CumulativePathLengthMeters;
            _hasTrialPhonePathLengthBaseline = true;
        }

        _hasPrevMacroPos = false;
        Transform eff = ResolveMacroEffortTransform();
        if (eff != null)
        {
            _prevMacroPos = eff.position;
            _hasPrevMacroPos = true;
        }

        if (logDebug)
        {
            Debug.Log($"[StudyLogger] Trial start task={TaskToCsv(task)} tech={TechniqueToCsv(_trialTechnique)} hand={_trialHandLocation} tool={_trialToolId}");
        }
    }

    private bool IsCurrentTrialStillRunning()
    {
        switch (_trialTask)
        {
            case TaskKind.Placement: return placementTask != null && placementTask.IsTrialRunning;
            case TaskKind.Rotation: return rotationTask != null && rotationTask.IsTrialRunning;
            case TaskKind.Scaling: return scalingTask != null && scalingTask.IsTrialRunning;
            default: return false;
        }
    }

    private void UpdateRunningTrialMetrics(float dt)
    {
        if (TryGetCurrentErrorAndTolerance(_trialTask, out float err, out float tol))
        {
            bool withinTol = err <= tol;
            if (withinTol && !_hasEnteredWithinTol)
            {
                _hasEnteredWithinTol = true;
                _firstWithinTolTime = Mathf.Max(0f, Time.time - _trialStartTime);
            }

            if (_wasWithinTol && !withinTol && _hasEnteredWithinTol)
                _eligibleBreaks++;

            _wasWithinTol = withinTol;
        }

        if (_trialTechnique == TechniqueKind.Micro)
        {
            if (router == null)
            {
                if (!_warnedMissingMicroRouter)
                {
                    _warnedMissingMicroRouter = true;
                    Debug.LogWarning("[StudyLogger] Router reference is missing. Micro effort metrics are skipped.");
                }
                return;
            }

            if (router.AxisActive)
            {
                _microAxisActiveDuration += dt;
                _microAxisIntegral += router.Axis.magnitude * dt;
            }
            return;
        }

        if (_trialTechnique == TechniqueKind.Macro)
        {
            Transform eff = ResolveMacroEffortTransform();
            if (eff == null)
            {
                if (!_warnedMissingMacroEffortTransform)
                {
                    _warnedMissingMacroEffortTransform = true;
                    Debug.LogWarning("[StudyLogger] macroEffortTransform and active tool are both unavailable. Macro path length is skipped.");
                }
                return;
            }

            if (!_hasPrevMacroPos)
            {
                _prevMacroPos = eff.position;
                _hasPrevMacroPos = true;
                return;
            }

            _macroPathLength += Vector3.Distance(_prevMacroPos, eff.position);
            _prevMacroPos = eff.position;
        }
    }

    private void EndTrialAndWriteRow()
    {
        if (!EffectiveLoggingEnabled)
        {
            if (logDebug)
                Debug.Log("[StudyLogger] LoggingEnabled=false. Trial row skipped.");

            _trialActive = false;
            _trialTask = TaskKind.None;
            _trialTechnique = TechniqueKind.Unknown;
            return;
        }

        float completionTime = Mathf.Max(0f, Time.time - _trialStartTime);

        float? translationErrorCm = null;
        float? rotationErrorDeg = null;
        float? scalingErrorPct = null;

        switch (_trialTask)
        {
            case TaskKind.Placement:
            {
                float errM = placementTask != null ? placementTask.GetActiveErrorMeters() : float.NaN;
                if (IsFinite(errM)) translationErrorCm = errM * 100f;
                break;
            }
            case TaskKind.Rotation:
            {
                float errDeg = float.NaN;
                if (rotationTask != null)
                {
                    errDeg = rotationTask.LastSubmittedErrorDeg;
                    if (!IsFinite(errDeg))
                        errDeg = rotationTask.ActiveRotationErrorDeg;
                }
                if (IsFinite(errDeg)) rotationErrorDeg = errDeg;
                break;
            }
            case TaskKind.Scaling:
            {
                float errFactor = scalingTask != null ? scalingTask.ActiveScalingErrorFactor : float.NaN;
                if (IsFinite(errFactor)) scalingErrorPct = errFactor * 100f;
                break;
            }
        }

        float? firstWithin = _hasEnteredWithinTol ? _firstWithinTolTime : (float?)null;
        float? microActiveDur = _trialTechnique == TechniqueKind.Micro ? _microAxisActiveDuration : (float?)null;
        float? microIntegral = _trialTechnique == TechniqueKind.Micro ? _microAxisIntegral : (float?)null;
        float? macroPath = _trialTechnique == TechniqueKind.Macro ? _macroPathLength : (float?)null;
        float? phonePath = null;
        if (_hasTrialPhonePathLengthBaseline && phonePoseReceiver != null)
        {
            double rawPhonePath = phonePoseReceiver.CumulativePathLengthMeters - _trialPhonePathLengthBaselineMeters;
            if (!double.IsNaN(rawPhonePath) && !double.IsInfinity(rawPhonePath))
                phonePath = (float)Math.Max(0.0, rawPhonePath);
        }
        else if (!_warnedMissingPhonePoseReceiver)
        {
            _warnedMissingPhonePoseReceiver = true;
            Debug.LogWarning("[StudyLogger] PhonePoseStreamReceiver reference is missing. Raw phone path length is skipped.");
        }
        int modeSwitch = _trialTechnique == TechniqueKind.Micro ? _modeSwitchCount : 0;

        string row = string.Join(",",
            EscapeCsv(_sessionTimestamp),
            EscapeCsv(_participantId),
            EscapeCsv(TaskToCsv(_trialTask)),
            EscapeCsv(TechniqueToCsv(_trialTechnique)),
            EscapeCsv(_trialHandLocation),
            EscapeCsv(_trialConditionLabel),
            _trialConditionIndex > 0 ? _trialConditionIndex.ToString(CultureInfo.InvariantCulture) : "",
            _trialConditionTotal > 0 ? _trialConditionTotal.ToString(CultureInfo.InvariantCulture) : "",
            EscapeCsv(_trialConditionOrder),
            _trialConditionSequenceIndex > 0 ? _trialConditionSequenceIndex.ToString(CultureInfo.InvariantCulture) : "",
            _trialConditionSequenceTotal > 0 ? _trialConditionSequenceTotal.ToString(CultureInfo.InvariantCulture) : "",
            EscapeCsv(_trialToolId),
            FormatFloat(completionTime),
            FormatNullableFloat(translationErrorCm),
            FormatNullableFloat(rotationErrorDeg),
            FormatNullableFloat(scalingErrorPct),
            FormatNullableFloat(firstWithin),
            _eligibleBreaks.ToString(CultureInfo.InvariantCulture),
            FormatNullableFloat(microActiveDur),
            FormatNullableFloat(microIntegral),
            FormatNullableFloat(macroPath),
            FormatNullableFloat(phonePath),
            modeSwitch.ToString(CultureInfo.InvariantCulture));

        bool wrote = WriteCsvRow(row);
        if (wrote)
            EnqueueMirrorRow(row);

        if (logDebug)
        {
            Debug.Log($"[StudyLogger] Trial end task={TaskToCsv(_trialTask)} tech={TechniqueToCsv(_trialTechnique)} hand={_trialHandLocation} tool={_trialToolId} t={completionTime:F3}s");
        }

        _trialActive = false;
        _trialTask = TaskKind.None;
        _trialTechnique = TechniqueKind.Unknown;
        _trialHandLocation = "Unknown";
        _trialConditionLabel = "Unknown";
        _trialConditionIndex = -1;
        _trialConditionTotal = -1;
        _trialConditionOrder = string.Empty;
        _trialConditionSequenceIndex = -1;
        _trialConditionSequenceTotal = -1;
        _trialToolId = "Unknown";
        _trialPhonePathLengthBaselineMeters = 0.0;
        _hasTrialPhonePathLengthBaseline = false;
    }

    private bool TryGetCurrentErrorAndTolerance(TaskKind task, out float err, out float tol)
    {
        err = float.MaxValue;
        tol = float.MaxValue;

        switch (task)
        {
            case TaskKind.Placement:
                if (placementTask == null) return false;
                err = placementTask.GetActiveErrorMeters();
                tol = placementTask.ActiveToleranceMeters;
                return IsFinite(err) && IsFinite(tol);

            case TaskKind.Rotation:
                if (rotationTask == null) return false;
                err = rotationTask.ActiveRotationErrorDeg;
                tol = rotationTask.RotationToleranceDeg;
                return IsFinite(err) && IsFinite(tol);

            case TaskKind.Scaling:
                if (scalingTask == null) return false;
                err = scalingTask.ActiveScalingErrorFactor;
                tol = scalingTask.ScaleFactorTolerance;
                return IsFinite(err) && IsFinite(tol);

            default:
                return false;
        }
    }

    private TechniqueKind GetCurrentTechnique()
    {
        if (router == null) return TechniqueKind.Unknown;
        return router.CurrentMode == PhoneInputRouter.Mode.Micro ? TechniqueKind.Micro : TechniqueKind.Macro;
    }

    private string GetCurrentConditionLabel()
    {
        if (flow != null)
        {
            string fromFlow = flow.GetConditionLabel();
            if (!string.IsNullOrWhiteSpace(fromFlow))
                return fromFlow.Trim();
        }

        TechniqueKind tech = GetCurrentTechnique();
        string techLabel = TechniqueToCsv(tech);
        if (string.IsNullOrEmpty(techLabel) || techLabel == "Unknown")
            techLabel = "Unknown";
        return $"{techLabel} - Near Head";
    }

    private static string ExtractHandLocationFromCondition(string conditionLabel)
    {
        if (string.IsNullOrWhiteSpace(conditionLabel))
            return "Unknown";

        string lower = conditionLabel.ToLowerInvariant();
        if (lower.Contains("side"))
            return "SideOfBody";
        if (lower.Contains("near"))
            return "NearHead";
        return "Unknown";
    }

    private string GetCurrentToolId(TaskKind task)
    {
        switch (task)
        {
            case TaskKind.Placement:
                if (placementTask != null && !string.IsNullOrWhiteSpace(placementTask.ActiveToolId))
                    return placementTask.ActiveToolId.Trim();
                if (placementTask != null && placementTask.ActiveToolTransform != null)
                    return ExtractToolIdFromTransform(placementTask.ActiveToolTransform);
                break;

            case TaskKind.Rotation:
                if (rotationTask != null && !string.IsNullOrWhiteSpace(rotationTask.ActiveToolId))
                    return rotationTask.ActiveToolId.Trim();
                if (rotationTask != null && rotationTask.ActiveToolTransform != null)
                    return ExtractToolIdFromTransform(rotationTask.ActiveToolTransform);
                break;

            case TaskKind.Scaling:
                if (scalingTask != null && !string.IsNullOrWhiteSpace(scalingTask.ActiveId))
                    return scalingTask.ActiveId.Trim();
                if (scalingTask != null && scalingTask.ActiveToolTransform != null)
                    return ExtractToolIdFromTransform(scalingTask.ActiveToolTransform);
                break;
        }

        return "Unknown";
    }

    private static string ExtractToolIdFromTransform(Transform toolTransform)
    {
        if (toolTransform == null)
            return "Unknown";

        ToolId tid = toolTransform.GetComponent<ToolId>();
        if (tid != null && !string.IsNullOrWhiteSpace(tid.id))
            return tid.id.Trim();

        return string.IsNullOrWhiteSpace(toolTransform.name) ? "Unknown" : toolTransform.name.Trim();
    }

    private int GetCurrentConditionIndex1Based()
    {
        if (flow != null)
            return flow.GetConditionIndex1Based();
        return -1;
    }

    private int GetCurrentConditionCount()
    {
        if (flow != null)
            return flow.GetConditionCount();
        return 0;
    }

    private string GetCurrentConditionOrderLabel()
    {
        if (flow != null)
        {
            string v = flow.GetConditionOrderLabel();
            if (!string.IsNullOrWhiteSpace(v))
                return v.Trim();
        }

        return string.Empty;
    }

    private int GetCurrentConditionSequenceIndex1Based()
    {
        if (flow != null)
            return flow.GetConditionSequenceIndex1Based();
        return -1;
    }

    private int GetCurrentConditionSequenceCount()
    {
        if (flow != null)
            return flow.GetConditionSequenceCount();
        return 0;
    }

    private Transform ResolveMacroEffortTransform()
    {
        if (macroEffortTransform != null)
            return macroEffortTransform;

        switch (_trialTask)
        {
            case TaskKind.Placement:
                if (placementTask != null && placementTask.ActiveToolTransform != null)
                    return placementTask.ActiveToolTransform;
                break;

            case TaskKind.Rotation:
                if (rotationTask != null && rotationTask.ActiveToolTransform != null)
                    return rotationTask.ActiveToolTransform;
                break;

            case TaskKind.Scaling:
                if (scalingTask != null && scalingTask.ActiveToolTransform != null)
                    return scalingTask.ActiveToolTransform;
                break;
        }

        return null;
    }

    private void InitCsv()
    {
        if (!EffectiveLoggingEnabled) return;
        if (_writer != null) return;

        try
        {
            string folder = Path.Combine(Application.persistentDataPath, "StudyLogs");
            Directory.CreateDirectory(folder);

            _participantId = ResolveParticipantId();
            DateTime now = DateTime.Now;
            _sessionTimestamp = now.ToString("yyyy-MM-dd_HH-mm-ss", CultureInfo.InvariantCulture);
            _filePath = GetUniqueLogPath(folder, _participantId, _sessionTimestamp);

            _writer = new StreamWriter(_filePath, append: false, Encoding.UTF8);
            _writer.WriteLine(CsvHeader);
            _writer.Flush();

            if (logDebug)
                Debug.Log($"[StudyLogger] Logging to {_filePath}");
        }
        catch (Exception e)
        {
            _writer = null;
            Debug.LogError($"[StudyLogger] Failed to initialize CSV logger: {e.Message}");
        }
    }

    private static string GetUniqueLogPath(string folder, string participantId, string stamp)
    {
        string safePid = SanitizeForFileName(participantId);
        string baseName = $"Study1_{safePid}_{stamp}.csv";
        string path = Path.Combine(folder, baseName);
        if (!File.Exists(path)) return path;

        for (int i = 1; i < 1000; i++)
        {
            string candidate = Path.Combine(folder, $"Study1_{safePid}_{stamp}_{i:00}.csv");
            if (!File.Exists(candidate))
                return candidate;
        }

        return Path.Combine(folder, $"Study1_{safePid}_{stamp}_{Guid.NewGuid().ToString("N").Substring(0, 8)}.csv");
    }

    private void EnsureMirrorOutboxLoaded()
    {
        if (_mirrorOutboxLoaded)
            return;

        try
        {
            string folder = Path.Combine(Application.persistentDataPath, "StudyLogs");
            Directory.CreateDirectory(folder);
            _mirrorOutboxPath = Path.Combine(folder, "mirror_outbox.jsonl");
            _mirrorOutboxLoaded = true;

            if (!File.Exists(_mirrorOutboxPath))
                return;

            string[] lines = File.ReadAllLines(_mirrorOutboxPath, Encoding.UTF8);
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i];
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                MirrorRowEnvelope env = null;
                try
                {
                    env = JsonUtility.FromJson<MirrorRowEnvelope>(line);
                }
                catch
                {
                    env = null;
                }

                if (env == null || string.IsNullOrEmpty(env.row_id) || string.IsNullOrEmpty(env.csv_row))
                    continue;

                _mirrorQueue.Add(env);
            }

            if (logDebug && _mirrorQueue.Count > 0)
                Debug.Log($"[StudyLogger] Loaded {_mirrorQueue.Count} pending mirror row(s) from outbox.");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[StudyLogger] Failed to load mirror outbox: {e.Message}");
            _mirrorOutboxLoaded = true;
        }
    }

    private void SaveMirrorOutbox()
    {
        if (!_mirrorOutboxLoaded)
            return;

        if (string.IsNullOrEmpty(_mirrorOutboxPath))
            return;

        try
        {
            if (_mirrorQueue.Count <= 0)
            {
                if (File.Exists(_mirrorOutboxPath))
                    File.Delete(_mirrorOutboxPath);
                return;
            }

            var sb = new StringBuilder(_mirrorQueue.Count * 256);
            for (int i = 0; i < _mirrorQueue.Count; i++)
            {
                sb.Append(JsonUtility.ToJson(_mirrorQueue[i]));
                sb.Append('\n');
            }

            string tmpPath = _mirrorOutboxPath + ".tmp";
            File.WriteAllText(tmpPath, sb.ToString(), Encoding.UTF8);
            File.Copy(tmpPath, _mirrorOutboxPath, overwrite: true);
            File.Delete(tmpPath);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[StudyLogger] Failed to save mirror outbox: {e.Message}");
        }
    }

    private void EnqueueMirrorRow(string csvRow)
    {
        if (!enableMirrorSend)
            return;

        EnsureMirrorOutboxLoaded();

        int hardLimit = Mathf.Max(100, mirrorQueueHardLimit);
        if (_mirrorQueue.Count >= hardLimit)
        {
            int removeCount = Mathf.Max(1, _mirrorQueue.Count - hardLimit + 1);
            _mirrorQueue.RemoveRange(0, removeCount);
            Debug.LogWarning($"[StudyLogger] Mirror queue limit reached. Dropped {removeCount} oldest row(s).");
        }

        _mirrorQueue.Add(new MirrorRowEnvelope
        {
            row_id = Guid.NewGuid().ToString("N"),
            session_timestamp = _sessionTimestamp,
            participant_id = _participantId,
            csv_header = CsvHeader,
            csv_row = csvRow,
            csv_path = _filePath,
            created_unix_ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
        });

        SaveMirrorOutbox();
        _nextMirrorAttemptAt = Mathf.Min(_nextMirrorAttemptAt, Time.unscaledTime);
    }

    private void ProcessMirrorQueue()
    {
        if (!enableMirrorSend)
            return;

        EnsureMirrorOutboxLoaded();
        FinalizeMirrorSendResultOnMainThread();

        if (_mirrorQueue.Count == 0)
            return;

        if (IsMirrorSendInFlight())
            return;

        if (Time.unscaledTime < _nextMirrorAttemptAt)
            return;

        if (string.IsNullOrWhiteSpace(mirrorHost) || mirrorPort <= 0 || mirrorPort > 65535)
        {
            if (!_warnedMirrorConfig)
            {
                _warnedMirrorConfig = true;
                Debug.LogWarning("[StudyLogger] Mirror send is enabled but mirrorHost/mirrorPort is invalid.");
            }
            _nextMirrorAttemptAt = Time.unscaledTime + Mathf.Max(0.5f, mirrorRetryIntervalSeconds);
            return;
        }

        int budget = Mathf.Clamp(mirrorMaxSendsPerUpdate, 1, 16);
        for (int i = 0; i < budget && _mirrorQueue.Count > 0; i++)
        {
            MirrorRowEnvelope entry = _mirrorQueue[0];

            if (mirrorSendOnBackgroundThread)
            {
                StartMirrorSendAsync(entry);
                return;
            }

            if (!TrySendMirrorEntry(entry))
            {
                _nextMirrorAttemptAt = Time.unscaledTime + Mathf.Max(0.1f, mirrorRetryIntervalSeconds);
                return;
            }

            _mirrorQueue.RemoveAt(0);
            SaveMirrorOutbox();
        }

        if (_mirrorQueue.Count > 0)
            _nextMirrorAttemptAt = Time.unscaledTime + 0.01f;
    }

    private void StartMirrorSendAsync(MirrorRowEnvelope entry)
    {
        if (entry == null || string.IsNullOrEmpty(entry.row_id))
        {
            _nextMirrorAttemptAt = Time.unscaledTime + Mathf.Max(0.1f, mirrorRetryIntervalSeconds);
            return;
        }

        lock (_mirrorSendStateLock)
        {
            if (_mirrorSendInFlight)
                return;

            _mirrorSendInFlight = true;
            _mirrorSendResultReady = false;
            _mirrorSendResultOk = false;
            _mirrorSendResultRowId = entry.row_id;
        }

        ThreadPool.QueueUserWorkItem(_ =>
        {
            bool ok = false;
            try
            {
                ok = TrySendMirrorEntry(entry);
            }
            catch
            {
                ok = false;
            }

            lock (_mirrorSendStateLock)
            {
                _mirrorSendResultOk = ok;
                _mirrorSendResultReady = true;
                _mirrorSendInFlight = false;
            }
        });
    }

    private bool IsMirrorSendInFlight()
    {
        lock (_mirrorSendStateLock)
            return _mirrorSendInFlight;
    }

    private void FinalizeMirrorSendResultOnMainThread()
    {
        bool ready;
        bool ok;
        string rowId;

        lock (_mirrorSendStateLock)
        {
            ready = _mirrorSendResultReady;
            if (!ready)
                return;

            ok = _mirrorSendResultOk;
            rowId = _mirrorSendResultRowId;
            _mirrorSendResultReady = false;
            _mirrorSendResultRowId = null;
        }

        if (ok)
        {
            int idx = FindMirrorQueueIndexByRowId(rowId);
            if (idx >= 0)
            {
                _mirrorQueue.RemoveAt(idx);
                SaveMirrorOutbox();
            }

            _nextMirrorAttemptAt = Time.unscaledTime + 0.01f;
        }
        else
        {
            _nextMirrorAttemptAt = Time.unscaledTime + Mathf.Max(0.1f, mirrorRetryIntervalSeconds);
        }
    }

    private int FindMirrorQueueIndexByRowId(string rowId)
    {
        if (string.IsNullOrEmpty(rowId))
            return -1;

        for (int i = 0; i < _mirrorQueue.Count; i++)
        {
            MirrorRowEnvelope env = _mirrorQueue[i];
            if (env == null) continue;
            if (string.Equals(env.row_id, rowId, StringComparison.Ordinal))
                return i;
        }

        return -1;
    }

    private bool TrySendMirrorEntry(MirrorRowEnvelope entry)
    {
        if (entry == null)
            return false;

        TcpClient client = null;
        try
        {
            client = new TcpClient();
            int connectTimeout = Mathf.Max(1, mirrorConnectTimeoutMs);
            if (!TryConnectWithTimeout(client, mirrorHost, mirrorPort, connectTimeout))
                return false;

            using (NetworkStream stream = client.GetStream())
            {
                int ackTimeout = Mathf.Max(1, mirrorAckTimeoutMs);
                stream.WriteTimeout = ackTimeout;
                stream.ReadTimeout = ackTimeout;

                string payload = JsonUtility.ToJson(entry) + "\n";
                byte[] bytes = Encoding.UTF8.GetBytes(payload);
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush();

                if (!mirrorRequireAck)
                    return true;

                string ackLine = ReadLineFromStream(stream, ackTimeout);
                if (string.IsNullOrWhiteSpace(ackLine))
                    return false;

                MirrorAckEnvelope ack = null;
                try
                {
                    ack = JsonUtility.FromJson<MirrorAckEnvelope>(ackLine);
                }
                catch
                {
                    ack = null;
                }

                if (ack != null)
                {
                    if (!string.Equals(ack.type, "ack", StringComparison.OrdinalIgnoreCase))
                        return false;

                    if (string.IsNullOrEmpty(ack.row_id) ||
                        !string.Equals(ack.row_id, entry.row_id, StringComparison.Ordinal))
                        return false;

                    return ack.ok ||
                           string.Equals(ack.status, "ok", StringComparison.OrdinalIgnoreCase) ||
                           string.Equals(ack.status, "duplicate", StringComparison.OrdinalIgnoreCase);
                }

                string lower = ackLine.Trim().ToLowerInvariant();
                if (lower == "ok" || lower == "ack")
                    return true;

                return lower.Contains(entry.row_id.ToLowerInvariant()) && lower.Contains("ok");
            }
        }
        catch (Exception e)
        {
            if (logDebug && !mirrorSendOnBackgroundThread)
                Debug.Log($"[StudyLogger] Mirror send failed: {e.Message}");
            return false;
        }
        finally
        {
            if (client != null)
                client.Close();
        }
    }

    private static bool TryConnectWithTimeout(TcpClient client, string host, int port, int timeoutMs)
    {
        IAsyncResult ar = null;
        try
        {
            ar = client.BeginConnect(host, port, null, null);
            if (!ar.AsyncWaitHandle.WaitOne(Mathf.Max(1, timeoutMs)))
                return false;

            client.EndConnect(ar);
            return client.Connected;
        }
        catch
        {
            return false;
        }
        finally
        {
            if (ar != null)
                ar.AsyncWaitHandle.Close();
        }
    }

    private static string ReadLineFromStream(NetworkStream stream, int timeoutMs)
    {
        if (stream == null)
            return null;

        stream.ReadTimeout = Mathf.Max(1, timeoutMs);
        var sb = new StringBuilder(256);
        const int maxChars = 8192;
        int count = 0;

        try
        {
            while (count < maxChars)
            {
                int b = stream.ReadByte();
                if (b < 0)
                    break;

                count++;
                char ch = (char)b;
                if (ch == '\n')
                    break;
                if (ch == '\r')
                    continue;

                sb.Append(ch);
            }
        }
        catch
        {
            return null;
        }

        return sb.ToString();
    }

    private bool WriteCsvRow(string row)
    {
        if (!EffectiveLoggingEnabled) return false;

        if (_writer == null)
        {
            if (!_warnedMissingWriter)
            {
                _warnedMissingWriter = true;
                Debug.LogWarning("[StudyLogger] CSV writer is unavailable. Trial row was not recorded.");
            }
            return false;
        }

        _writer.WriteLine(row);
        _writer.Flush();
        return true;
    }

    private void CloseWriter()
    {
        if (_writer == null) return;
        _writer.Flush();
        _writer.Close();
        _writer = null;
    }

    private static string TaskToCsv(TaskKind task)
    {
        switch (task)
        {
            case TaskKind.Placement: return "Placement";
            case TaskKind.Rotation: return "Rotation";
            case TaskKind.Scaling: return "Scaling";
            default: return "Unknown";
        }
    }

    private static string TechniqueToCsv(TechniqueKind tech)
    {
        switch (tech)
        {
            case TechniqueKind.Macro: return "Macro";
            case TechniqueKind.Micro: return "Micro";
            default: return "Unknown";
        }
    }

    private static string EscapeCsv(string value)
    {
        if (string.IsNullOrEmpty(value)) return "";
        bool mustQuote = value.Contains(",") || value.Contains("\"") || value.Contains("\n") || value.Contains("\r");
        if (!mustQuote) return value;
        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    private static string FormatFloat(float value)
    {
        return value.ToString("0.######", CultureInfo.InvariantCulture);
    }

    private static string FormatNullableFloat(float? value)
    {
        if (!value.HasValue) return "";
        return value.Value.ToString("0.######", CultureInfo.InvariantCulture);
    }

    private static bool IsFinite(float v)
    {
        return !float.IsNaN(v) && !float.IsInfinity(v);
    }

    private string ResolveParticipantId()
    {
        string id = NormalizeParticipantId(PlayerPrefs.GetString(ParticipantPrefKey, ""));
        if (!string.IsNullOrEmpty(id))
            return id;

        try
        {
            string participantFilePath = Path.Combine(Application.persistentDataPath, ParticipantFileName);
            if (File.Exists(participantFilePath))
            {
                id = NormalizeParticipantId(File.ReadAllText(participantFilePath));
                if (!string.IsNullOrEmpty(id))
                    return id;
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[StudyLogger] Failed to read participant file: {e.Message}");
        }

        id = NormalizeParticipantId(participantIdFallback);
        if (!string.IsNullOrEmpty(id))
            return id;

        return "TEST";
    }

    private static string NormalizeParticipantId(string id)
    {
        return string.IsNullOrWhiteSpace(id) ? "" : id.Trim();
    }

    private static string SanitizeForFileName(string value)
    {
        string normalized = NormalizeParticipantId(value);
        if (string.IsNullOrEmpty(normalized))
            return "TEST";

        char[] invalid = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(normalized.Length);
        for (int i = 0; i < normalized.Length; i++)
        {
            char c = normalized[i];
            if (Array.IndexOf(invalid, c) >= 0 || char.IsWhiteSpace(c))
                sb.Append('_');
            else
                sb.Append(c);
        }

        string clean = sb.ToString().Trim('_');
        return string.IsNullOrEmpty(clean) ? "TEST" : clean;
    }
}
