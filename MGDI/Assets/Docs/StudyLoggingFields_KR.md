# Study Logging Fields

## 개요

- 완료된 각 trial마다 CSV 한 줄이 기록됩니다.
- 로그 파일은 `Application.persistentDataPath/StudyLogs` 아래에 저장됩니다.
- 일부 필드는 특정 task 또는 technique에만 해당합니다. 현재 task/technique에 적용되지 않는 필드는 비어 있을 수 있습니다.

## CSV 컬럼

### 세션 및 참가자 정보

- `session_timestamp`
  로그 파일에 사용되는 세션 시작 시각입니다.

- `participant_id`
  해당 세션에 사용된 참가자 ID입니다.

### Task 및 condition 정보

- `task`
  해당 row의 task 유형입니다: `Placement`, `Rotation`, `Scaling`.

- `technique`
  상호작용 technique입니다: `Macro` 또는 `Micro`.

- `hand_location`
  손 위치 condition입니다. 예: `NearHead`, `SideOfBody`.

- `condition_label`
  사람이 읽기 쉬운 condition label입니다. 예: `Macro - Near Head`.

- `condition_index`
  현재 condition의 순번입니다. 1부터 시작합니다.

- `condition_total`
  전체 study sequence 안의 총 condition 수입니다.

- `condition_order`
  현재 세션의 전체 condition 순서를 하나의 문자열로 기록한 값입니다.

- `condition_sequence_index`
  현재 참가자에게 선택된 Latin-square sequence의 인덱스입니다. 1부터 시작합니다.

- `condition_sequence_total`
  현재 condition set에서 가능한 전체 Latin-square sequence 수입니다.

- `tool_id`
  해당 row의 active tool 식별자입니다.
  task setup에서 지정된 tool ID이며, 내부 trial 번호보다 tool별 효과를 분석할 때 더 유용합니다.

- `trial_index_global`
  현재 세션에서 기록된 main-task trial의 전체 순번입니다. 1부터 시작합니다.

- `trial_index_in_task`
  현재 task phase 안에서의 tool/trial 순번입니다. 1부터 시작합니다.

- `trial_result`
  trial의 최종 결과입니다: `success`, `timeout`, `failed`, `unknown`.

- `timed_out`
  timeout으로 끝난 trial이면 `1`, timeout 없이 끝났으면 `0`, 알 수 없으면 비어 있습니다.

- `success`
  성공으로 끝난 trial이면 `1`, 실패로 끝났으면 `0`, 알 수 없으면 비어 있습니다.

### 수행 지표

- `completion_time_s`
  trial 시작부터 제출까지 걸린 시간입니다. 단위는 초입니다.

- `translation_error_cm`
  제출 시점의 최종 placement error입니다. 단위는 센티미터입니다.
  placement trial에서 사용됩니다.

- `rotation_error_deg`
  제출 시점의 최종 rotation error입니다. 단위는 도(degree)입니다.
  rotation trial에서 사용됩니다.

- `scaling_error_pct`
  제출 시점의 최종 scaling error입니다. 단위는 퍼센트입니다.
  scaling trial에서 사용됩니다.

- `time_to_first_within_tol_s`
  trial 시작 후 tool이 처음으로 success tolerance 안에 들어가기까지 걸린 시간입니다.

- `eligible_breaks`
  제출 전에 tool이 tolerance 안에 들어갔다가 다시 벗어난 횟수입니다.

- `error_recovery_time_s`
  tool이 처음으로 success tolerance 안에 들어간 뒤, 제출 전까지 tolerance 밖에 있었던 총 시간입니다. 단위는 초입니다.
  한 번도 tolerance 안에 들어가지 못한 trial에서는 비어 있습니다. 처음 들어간 뒤 제출까지 계속 tolerance 안에 머문 경우에는 `0`입니다.

### Effort 및 interaction 지표

- `micro_axis_active_duration_s`
  Micro analog input가 실제로 drag 상태였던 총 시간입니다. 단위는 초입니다.

- `micro_axis_integral`
  시간에 따라 누적된 Micro input magnitude입니다.
  trial 동안 `|Axis| * dt`를 합산해 계산합니다.

- `macro_path_length_m`
  trial 동안 추적된 Macro effort transform 기준의 추정 이동 거리입니다. 단위는 미터입니다.
  현재 scene setup에서는 raw phone pose 자체가 아니라 proxy wrist/hand transform의 이동 경로를 반영합니다.
  이 값은 시작점과 끝점 사이의 직선거리가 아니라 누적 경로 길이입니다.

- `phone_path_length_m`
  trial 동안 들어온 phone pose sample을 기준으로 계산한 raw phone translation의 총 이동 거리입니다. 단위는 미터입니다.
  이 값은 Macro와 Micro 모두에 적용되며, 참가자가 실제로 폰을 얼마나 움직였는지를 반영합니다.
  이 값도 시작점과 끝점 사이의 직선거리가 아니라 누적 경로 길이입니다.

- `mode_switch_count`
  trial 동안 기록된 Micro mode switch 횟수입니다. 예를 들어 axis-mode toggle 같은 이벤트가 여기에 포함됩니다.

## 참고

- Placement, rotation, scaling row는 일관성을 위해 동일한 CSV header를 공유합니다.
- 현재 task에 해당하는 error 컬럼만 값이 들어가는 것이 정상입니다.
- `condition_order`는 참가자별 task 순서를 분석 단계에서 복원할 때 유용합니다.
- `error_recovery_time_s`는 보조적인 안정성/회복 지표입니다. target에 한 번 도달한 뒤 다시 벗어나 수정한 경우와, 처음 도달한 뒤 계속 tolerance 안에 머문 경우를 구분하는 데 사용합니다.
- `macro_path_length_m`와 `phone_path_length_m`는 의도적으로 서로 다른 지표입니다.
- `macro_path_length_m`는 시스템 내부 control transform 기준이므로 gain, remapping, offset, smoothing, rotation coupling 등의 영향으로 raw phone movement와 다를 수 있습니다.
- `phone_path_length_m`는 Macro와 Micro 모두에서 참가자가 실제로 폰을 얼마나 움직였는지 분석할 때 더 적합한 필드입니다.
