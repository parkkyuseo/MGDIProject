# Process Audit Bug Report - 2026-05-04

## Scope

- Runtime study flow: condition gate -> practice -> Placement -> Rotation -> Scaling -> next condition.
- Local study logging through `StudyLogger`.
- Optional laptop mirror logging through `StudyLoggerMirrorReceiver.ps1`.
- Runtime scene serialized settings in `Assets/Scenes/RuntimeScene.unity`.

## Summary

Main completed task rows appear to be written to the local HoloLens CSV path immediately: `StudyLogger` creates a CSV under `Application.persistentDataPath/StudyLogs`, writes the header, writes one row when a task manager stops reporting `IsTrialRunning`, and flushes after every row.

Practice rows are intentionally skipped by `StudyFlowController_V2` through `logger.LoggingEnabled = false` during practice. The practice-to-real transition has a `yield return null` gap before real block start, so the logger has a frame to observe the practice trial ending before logging is re-enabled.

The highest-risk issue found is not the primary local CSV write path. It is the optional laptop mirror path, which can duplicate rows if an ACK is lost after the receiver has already appended the row.

## Confirmed Bugs

### BR-001 - Laptop mirror receiver can duplicate task rows on retry

- Severity: High for mirrored data integrity.
- Status: Patched on 2026-05-04 in `Tools/StudyLoggerMirrorReceiver.ps1`.
- Area: `Tools/StudyLoggerMirrorReceiver.ps1`, `Assets/Scripts/Tasks/StudyLogger.cs`.
- Evidence:
  - `StudyLogger` sends a `row_id` and keeps the row in `mirror_outbox.jsonl` until an ACK is received.
  - `StudyLoggerMirrorReceiver.ps1` appends `csv_row` every time it receives an envelope.
  - The receiver does not check whether `row_id` was already received.
  - The mirrored CSV header does not include `row_id`, so duplicate rows in the mirror CSV cannot be reliably deduplicated from the CSV alone.
- Impact:
  - If the receiver writes the row but the ACK is lost, times out, or is not read by HoloLens, the HoloLens retries later.
  - The receiver appends the same trial again.
  - The local HoloLens CSV remains correct, but `StudyMirror_<participant>_<session>.csv` can overcount trials.
- Recommended fix:
  - Include `row_id` in the mirrored CSV output, or add a receiver-side received-row-id index.
  - If a duplicate `row_id` arrives, return OK ACK but do not append the row again.
  - Keep the raw JSONL either deduped or explicitly mark duplicate receipts.
- Patch:
  - The receiver now loads existing `received_rows.jsonl` entries into a `HashSet` at startup.
  - A received envelope without `row_id` is rejected.
  - A duplicate `row_id` receives an OK ACK with status `duplicate` and is not appended to either the mirror CSV or `received_rows.jsonl`.
  - The receiver also supports a stop file / Q / Esc shutdown path so long-running verification can stop cleanly.
- Residual risk:
  - A crash after CSV append but before JSONL append can still leave a very small duplicate window. Closing that completely would require a more transactional write protocol or including `row_id` in the CSV itself.

### BR-002 - `phoneConnectionMaxWaitSeconds` is serialized but not enforced

- Severity: Medium.
- Status: Accepted as intended for the current experiment setup. The operator confirmed that no phone connection means the flow should not proceed.
- Area: `Assets/Scripts/Tasks/ConditionBlockController.cs`.
- Evidence:
  - The field exists with tooltip semantics in the scene and script.
  - Search results show it is only declared/serialized; no timeout logic reads it.
  - `WaitForPhoneConnectionIfNeeded` loops while `token == _conditionEntryToken` until the phone pose becomes fresh long enough.
- Impact:
  - If a nonzero max wait is configured, it is ignored.
  - During condition entry, the app can stay blocked indefinitely when phone data never arrives or becomes stale.
  - Current `RuntimeScene` has `phoneConnectionMaxWaitSeconds: 0`, so the current scene is explicitly infinite-wait, but the field is misleading and unsafe for future runs.
- Recommended fix:
  - Track `startedAt` in `WaitForPhoneConnectionIfNeeded`.
  - If `phoneConnectionMaxWaitSeconds > 0`, proceed with warning or show a clear recoverable error when exceeded.
  - Use `waitingForPhoneConnectionText` while waiting so the participant/operator sees why the flow is blocked.

### BR-003 - Dwell/confirm progress settings are currently dead code

- Severity: Medium if dwell confirmation UI is expected; Low if triple-tap-only submission is intentional.
- Status: Accepted as intended for the current experiment setup. The operator confirmed that triple-tap submission is intended.
- Area: `ToolPlacementTaskManager`, `ToolRotationTaskManager`, `ToolScalingTaskManager`, `DwellConfirmHUD`.
- Evidence:
  - `dwellSeconds`, `confirmDwellSeconds`, `dwellTimer`, and `confirmDwellTimer` are reset but not used to complete trials.
  - `ComputeActiveStability` and `IsConfirmEligible` are present but not used in the active Update paths.
  - Rotation and scaling call `OnConfirmProgress(0f, false)` rather than emitting dwell progress.
  - Current completion path is manual submit through triple tap / voice submit.
- Impact:
  - Inspector values imply automatic dwell confirmation or progress HUD behavior, but changing those values will not affect runtime behavior.
  - `DwellConfirmHUD` will not show useful progress except status messages from blocked triple taps.
- Recommended fix:
  - Decide whether the task flow is triple-tap-only or dwell-confirm based.
  - If triple-tap-only is intended, remove or rename the dwell fields and retire the progress HUD wiring.
  - If dwell confirmation is intended, reconnect the stability/tolerance eligibility code and emit `OnConfirmProgress(t01, eligible)`.

### BR-004 - Failed mirror ACK can be treated as successful

- Severity: High for mirrored backup data integrity.
- Status: Patched on 2026-05-04 in `Assets/Scripts/Tasks/StudyLogger.cs`.
- Area: `Assets/Scripts/Tasks/StudyLogger.cs`.
- Evidence:
  - Before the patch, structured ACK parsing treated `ack.type == "ack"` as successful even when `ack.ok == false` and `ack.status == "error"`.
  - `StudyLogger` removes a mirror queue entry after `TrySendMirrorEntry` returns true.
- Impact:
  - If the laptop receiver rejected a row but returned a structured error ACK, the HoloLens could incorrectly remove that row from `mirror_outbox.jsonl`.
  - The local HoloLens CSV remained correct, but the laptop mirror backup could silently miss the row.
- Patch:
  - Structured ACKs must now have `type == "ack"`.
  - Structured ACKs must include the matching `row_id`.
  - Only `ok == true`, `status == "ok"`, or `status == "duplicate"` are accepted as successful.
  - `status == "error"` with `ok == false` is no longer treated as success.

## Risks / Notes

### R-001 - In-progress trials are not written on app shutdown

`StudyLogger.OnDisable` and `OnDestroy` save the mirror outbox and close the CSV writer, but they do not write a partial row for an active trial. This is probably acceptable if only completed trials should be recorded. If operator aborts should be visible in data, add an explicit aborted-trial row.

### R-002 - Primary local CSV save path looks sound for completed main trials

For completed main trials, the logger edge-detects task start/stop through `IsTrialRunning`, writes one CSV row, flushes immediately, and only enqueues mirror rows after the local write succeeds. I did not find a confirmed local CSV row-loss bug for completed main trials in static review.

### R-003 - Inactive legacy receiver has the same UDP port

`PhonePoseReceiver` and `PhonePoseStreamReceiver` both serialize `listenPort: 5555`, but `PhonePoseReceiver` is inactive in the current RuntimeScene. There is no current port conflict. If that GameObject is reactivated later, one UDP receiver can fail to bind.

## Verification Performed

- Read and traced the runtime flow through:
  - `StudyFlowController_V2`
  - `WorkflowProgressionController`
  - `ConditionBlockController`
  - `ToolPlacementTaskManager`
  - `ToolRotationTaskManager`
  - `ToolScalingTaskManager`
  - `StudyLogger`
  - `StudyLoggerMirrorReceiver.ps1`
- Checked RuntimeScene serialized references/settings for the same components.
- Parsed `Tools/StudyLoggerMirrorReceiver.ps1` with PowerShell parser: syntax OK.
- Verified BR-001 patch with a TCP duplicate test:
  - first receipt returned ACK status `ok`;
  - same `row_id` returned ACK status `duplicate`;
  - mirror CSV stayed at header + one data row;
  - `received_rows.jsonl` stayed at one entry.
- Verified receiver restart behavior with a pre-existing `received_rows.jsonl` entry:
  - startup loaded one received row id;
  - retry of that same `row_id` returned `duplicate`;
  - no extra CSV or JSONL row was appended.
- Verified BR-004 patch by static review of ACK acceptance criteria:
  - `type == "ack"` alone no longer clears the mirror queue entry;
  - failed structured ACKs return false;
  - duplicate ACKs remain successful so the BR-001 receiver patch drains retried rows correctly.
- Attempted `dotnet build MGDI.sln --no-restore`: Unity-generated solution failed without diagnostics.
- Attempted Unity 2022.3.62f1 batch compile: blocked by Unity LicensingClient IPC timeout, return code 199. No Unity C# compile result was available from this environment.
