# Study Logging Fields

## Overview

- One CSV row is written for each completed trial.
- The log file is stored under `Application.persistentDataPath/StudyLogs`.
- Some fields are task-specific. If a field does not apply to the current task or technique, it may be left blank.

## CSV Columns

### Session and participant

- `session_timestamp`
  Session start timestamp used for the log file.

- `participant_id`
  Participant ID used for this session.

### Task and condition

- `task`
  Task type for the row: `Placement`, `Rotation`, or `Scaling`.

- `technique`
  Interaction technique: `Macro` or `Micro`.

- `hand_location`
  Hand-location condition, such as `NearHead` or `SideOfBody`.

- `condition_label`
  Human-readable condition label, for example `Macro - Near Head`.

- `condition_index`
  Current condition index within the condition sequence, using 1-based numbering.

- `condition_total`
  Total number of conditions in the study sequence.

- `condition_order`
  Full condition order label for the current session, written as a single string.

- `condition_sequence_index`
  Selected Latin-square sequence index for this participant, using 1-based numbering.

- `condition_sequence_total`
  Total number of possible Latin-square sequences for the current condition set.

- `tool_id`
  Identifier of the active tool for the row.
  Tool ID from the task setup. Use this instead of an internal trial counter for tool-level analysis.

### Performance

- `completion_time_s`
  Time from trial start to submission, in seconds.

- `translation_error_cm`
  Final placement error at submission, in centimeters.
  Used for placement trials.

- `rotation_error_deg`
  Final rotation error at submission, in degrees.
  Used for rotation trials.

- `scaling_error_pct`
  Final scaling error at submission, in percent.
  Used for scaling trials.

- `time_to_first_within_tol_s`
  Time from trial start until the first moment the tool entered the success tolerance.

- `eligible_breaks`
  Number of times the tool entered the tolerance and then left it again before submission.

### Effort and interaction

- `micro_axis_active_duration_s`
  Total time, in seconds, during which Micro analog input was actively being dragged.

- `micro_axis_integral`
  Accumulated Micro input magnitude over time.
  Computed as `|Axis| * dt` over the trial.

- `macro_path_length_m`
  Estimated Macro movement distance, in meters, based on the tracked Macro effort transform during the trial.
  In the current scene setup, this reflects the movement path of the proxy wrist/hand transform rather than the raw phone pose itself.
  Cumulative path length, not straight-line displacement.

- `phone_path_length_m`
  Total raw phone translation distance, in meters, accumulated from incoming phone pose samples during the trial.
  This applies to both Macro and Micro trials and reflects how much the participant physically moved the phone itself.
  Cumulative path length, not straight-line displacement.

- `mode_switch_count`
  Number of Micro mode switches recorded during the trial, such as axis-mode toggles.

## Notes

- Placement, rotation, and scaling rows share the same CSV header for consistency.
- Only the error column relevant to the current task is expected to contain a value.
- `condition_order` is useful for reconstructing the participant-specific task order during analysis.
- `macro_path_length_m` and `phone_path_length_m` are intentionally different measures.
- `macro_path_length_m` is based on the in-system control transform and may differ from raw phone movement because gains, remapping, offsets, smoothing, and rotation coupling can affect the proxy wrist path.
- `phone_path_length_m` is the better field to use when analyzing how much the participant physically moved the phone in either Macro or Micro.
