using System;
using System.Globalization;
using System.IO;
using System.Text;
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

    [Header("Effort Source (Macro)")]
    [Tooltip("If unset, active tool transform is used while a trial is running.")]
    [SerializeField] private Transform macroEffortTransform;

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

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

    private const string CsvHeader =
        "session_timestamp,participant_id,task,technique,trial_index," +
        "completion_time_s," +
        "translation_error_cm,rotation_error_deg,scaling_error_pct," +
        "time_to_first_within_tol_s,eligible_breaks," +
        "micro_axis_active_duration_s,micro_axis_integral," +
        "macro_path_length_m," +
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
    private int _trialIndex = -1;
    private float _trialStartTime;

    private float _firstWithinTolTime = -1f;
    private bool _hasEnteredWithinTol;
    private bool _wasWithinTol;
    private int _eligibleBreaks;

    private float _microAxisActiveDuration;
    private float _microAxisIntegral;
    private float _macroPathLength;
    private int _modeSwitchCount;

    private bool _hasPrevMacroPos;
    private Vector3 _prevMacroPos;

    private bool _warnedMissingWriter;
    private bool _warnedMissingMicroRouter;
    private bool _warnedMissingMacroEffortTransform;
    private bool _warnedMissingTasks;
    private bool EffectiveLoggingEnabled => enableLogging && LoggingEnabled;

    private void Awake()
    {
        ResolveRefs();
        _participantId = ResolveParticipantId();
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
    }

    private void OnDestroy()
    {
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
        _trialIndex = GetCurrentTrialIndex(task);
        _trialStartTime = Time.time;

        _firstWithinTolTime = -1f;
        _hasEnteredWithinTol = false;
        _wasWithinTol = false;
        _eligibleBreaks = 0;

        _microAxisActiveDuration = 0f;
        _microAxisIntegral = 0f;
        _macroPathLength = 0f;
        _modeSwitchCount = 0;

        _hasPrevMacroPos = false;
        Transform eff = ResolveMacroEffortTransform();
        if (eff != null)
        {
            _prevMacroPos = eff.position;
            _hasPrevMacroPos = true;
        }

        if (logDebug)
        {
            Debug.Log($"[StudyLogger] Trial start task={TaskToCsv(task)} tech={TechniqueToCsv(_trialTechnique)} idx={_trialIndex}");
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
                float errDeg = rotationTask != null ? rotationTask.ActiveRotationErrorDeg : float.NaN;
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
        int modeSwitch = _trialTechnique == TechniqueKind.Micro ? _modeSwitchCount : 0;

        string row = string.Join(",",
            EscapeCsv(_sessionTimestamp),
            EscapeCsv(_participantId),
            EscapeCsv(TaskToCsv(_trialTask)),
            EscapeCsv(TechniqueToCsv(_trialTechnique)),
            _trialIndex > 0 ? _trialIndex.ToString(CultureInfo.InvariantCulture) : "",
            FormatFloat(completionTime),
            FormatNullableFloat(translationErrorCm),
            FormatNullableFloat(rotationErrorDeg),
            FormatNullableFloat(scalingErrorPct),
            FormatNullableFloat(firstWithin),
            _eligibleBreaks.ToString(CultureInfo.InvariantCulture),
            FormatNullableFloat(microActiveDur),
            FormatNullableFloat(microIntegral),
            FormatNullableFloat(macroPath),
            modeSwitch.ToString(CultureInfo.InvariantCulture));

        WriteCsvRow(row);

        if (logDebug)
        {
            Debug.Log($"[StudyLogger] Trial end task={TaskToCsv(_trialTask)} tech={TechniqueToCsv(_trialTechnique)} idx={_trialIndex} t={completionTime:F3}s");
        }

        _trialActive = false;
        _trialTask = TaskKind.None;
        _trialTechnique = TechniqueKind.Unknown;
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

    private int GetCurrentTrialIndex(TaskKind task)
    {
        switch (task)
        {
            case TaskKind.Placement: return placementTask != null ? placementTask.CurrentTrialIndex1Based : -1;
            case TaskKind.Rotation: return rotationTask != null ? rotationTask.CurrentTrialIndex1Based : -1;
            case TaskKind.Scaling: return scalingTask != null ? scalingTask.CurrentTrialIndex1Based : -1;
            default: return -1;
        }
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

    private void WriteCsvRow(string row)
    {
        if (!EffectiveLoggingEnabled) return;

        if (_writer == null)
        {
            if (!_warnedMissingWriter)
            {
                _warnedMissingWriter = true;
                Debug.LogWarning("[StudyLogger] CSV writer is unavailable. Trial row was not recorded.");
            }
            return;
        }

        _writer.WriteLine(row);
        _writer.Flush();
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
