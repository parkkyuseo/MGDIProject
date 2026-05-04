# Study Logger Mirror Receiver

File:

- `Tools/StudyLoggerMirrorReceiver.ps1`

Default behavior:

- listens on TCP port `19620` by default
- receives one JSON line per trial row
- writes rows into per-session CSV files
- appends the raw JSON line to `received_rows.jsonl`
- sends an `ack` back to the HoloLens so the row is removed from the retry queue

## Run on the laptop

Open PowerShell in the repo root and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\StudyLoggerMirrorReceiver.ps1
```

If you are already inside the `Tools` folder, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\StudyLoggerMirrorReceiver.ps1
```

Or choose a custom output folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\StudyLoggerMirrorReceiver.ps1 -OutputDir C:\StudyMirrorInbox
```

## HoloLens side settings

In the Unity scene, `StudyLogger` should point to the laptop:

- `enableMirrorSend = true`
- `mirrorHost = <laptop IP>`
- `mirrorPort = 19620`

## Output files

The receiver creates:

- `StudyMirror_<participant>_<session>.csv`
- `received_rows.jsonl`

`received_rows.jsonl` is also used as the receiver-side duplicate index. Keep it with the CSV files for the session; if a row is retried with the same `row_id`, the receiver ACKs it as `duplicate` and does not append another CSV row.

Sample:

- `StudyMirror_P15_20260408_191500.csv`

## Stop the receiver

You can stop it in any of these ways:

- press `Q`
- press `Esc`
- create the file `STOP_RECEIVER` inside the output folder

Example stop file path:

- `C:\StudyMirrorInbox\STOP_RECEIVER`
- or, with the default output folder, `Tools\StudyLoggerMirrorInbox\STOP_RECEIVER`

## Notes

- The HoloLens still keeps its local CSV log.
- If the laptop is offline, the HoloLens keeps rows in `mirror_outbox.jsonl` and retries later.
- The receiver writes one CSV row per completed trial attempt.
