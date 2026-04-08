# Study Logger Mirror Receiver

This script receives backup trial rows from the HoloLens logger on a laptop.

File:

- `Tools/StudyLoggerMirrorReceiver.ps1`

What it does:

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

Example:

- `StudyMirror_P15_20260408_191500.csv`

## Notes

- The HoloLens still keeps its local CSV log.
- If the laptop is offline, the HoloLens keeps rows in `mirror_outbox.jsonl` and retries later.
- The receiver writes one CSV row per completed trial attempt.
