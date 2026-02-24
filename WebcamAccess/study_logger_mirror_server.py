#!/usr/bin/env python3
"""
study_logger_mirror_server.py

TCP receiver for StudyLogger mirror rows.

Protocol (newline-delimited JSON):
{
  "type": "trial_row",
  "protocol": "study_logger_mirror_v1",
  "row_id": "...",
  "session_timestamp": "YYYY-MM-DD_HH-mm-ss",
  "participant_id": "P001",
  "csv_header": "...",
  "csv_row": "...",
  "csv_path": "...",
  "created_unix_ms": 0
}

ACK (newline-delimited JSON):
{
  "type": "ack",
  "row_id": "...",
  "ok": true,
  "status": "ok"
}
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
from pathlib import Path
from typing import Dict, Set, Tuple


DEFAULT_PORT = 19620
DEFAULT_OUT_DIR = "study_mirror_logs"
MAX_LINE_BYTES = 2 * 1024 * 1024


def sanitize_filename(value: str, fallback: str = "TEST") -> str:
    if value is None:
        return fallback
    v = str(value).strip()
    if not v:
        return fallback
    safe_chars = []
    for ch in v:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    out = "".join(safe_chars).strip("_")
    return out or fallback


class MirrorStore:
    def __init__(self, out_dir: Path, force_fsync: bool = False, verbose: bool = False) -> None:
        self.out_dir = out_dir
        self.force_fsync = force_fsync
        self.verbose = verbose
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._rowids_cache: Dict[Path, Set[str]] = {}

    def _target_paths(self, participant_id: str, session_timestamp: str) -> Tuple[Path, Path]:
        pid = sanitize_filename(participant_id, "TEST")
        stamp = sanitize_filename(session_timestamp, "unknown")
        csv_path = self.out_dir / f"Study1_{pid}_{stamp}_mirror.csv"
        rowid_path = self.out_dir / f"Study1_{pid}_{stamp}_mirror.rowids"
        return csv_path, rowid_path

    def _load_rowids(self, rowid_path: Path) -> Set[str]:
        if rowid_path in self._rowids_cache:
            return self._rowids_cache[rowid_path]

        ids: Set[str] = set()
        if rowid_path.exists():
            try:
                with rowid_path.open("r", encoding="utf-8", newline="") as f:
                    for line in f:
                        rid = line.strip()
                        if rid:
                            ids.add(rid)
            except Exception:
                pass

        self._rowids_cache[rowid_path] = ids
        return ids

    def save_row(self, env: dict) -> Tuple[bool, str]:
        row_id = str(env.get("row_id", "")).strip()
        participant_id = str(env.get("participant_id", "")).strip()
        session_timestamp = str(env.get("session_timestamp", "")).strip()
        csv_header = str(env.get("csv_header", "")).strip()
        csv_row = str(env.get("csv_row", "")).rstrip("\r\n")

        if not row_id:
            return False, "missing_row_id"
        if not csv_row:
            return False, "missing_csv_row"

        csv_path, rowid_path = self._target_paths(participant_id, session_timestamp)

        with self._lock:
            known_ids = self._load_rowids(rowid_path)
            if row_id in known_ids:
                return True, "duplicate"

            is_new_file = not csv_path.exists()
            try:
                with csv_path.open("a", encoding="utf-8", newline="") as f:
                    if is_new_file:
                        if csv_header:
                            f.write(csv_header + "\n")
                        else:
                            f.write(
                                "session_timestamp,participant_id,task,technique,trial_index,"
                                "completion_time_s,"
                                "translation_error_cm,rotation_error_deg,scaling_error_pct,"
                                "time_to_first_within_tol_s,eligible_breaks,"
                                "micro_axis_active_duration_s,micro_axis_integral,"
                                "macro_path_length_m,"
                                "mode_switch_count\n"
                            )

                    f.write(csv_row + "\n")
                    f.flush()
                    if self.force_fsync:
                        os.fsync(f.fileno())

                with rowid_path.open("a", encoding="utf-8", newline="") as rf:
                    rf.write(row_id + "\n")
                    rf.flush()
                    if self.force_fsync:
                        os.fsync(rf.fileno())

                known_ids.add(row_id)
                if self.verbose:
                    print(f"[Mirror] saved row_id={row_id} -> {csv_path.name}")
                return True, "ok"
            except Exception as ex:
                return False, f"write_failed:{ex}"


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


class MirrorHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: "MirrorServerApp" = self.server.app  # type: ignore[attr-defined]

        try:
            line = self.rfile.readline(MAX_LINE_BYTES + 1)
            if not line:
                return
            if len(line) > MAX_LINE_BYTES:
                self._send_ack("", False, "line_too_large")
                return

            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                self._send_ack("", False, "empty_payload")
                return

            try:
                env = json.loads(text)
            except Exception:
                self._send_ack("", False, "invalid_json")
                return

            row_id = str(env.get("row_id", "")).strip()
            msg_type = str(env.get("type", "")).strip()
            protocol = str(env.get("protocol", "")).strip()

            if msg_type != "trial_row":
                self._send_ack(row_id, False, "invalid_type")
                return
            if protocol and protocol != "study_logger_mirror_v1":
                self._send_ack(row_id, False, "invalid_protocol")
                return

            ok, status = server.store.save_row(env)
            self._send_ack(row_id, ok, status)
        except Exception as ex:
            self._send_ack("", False, f"handler_error:{ex}")

    def _send_ack(self, row_id: str, ok: bool, status: str) -> None:
        ack = {
            "type": "ack",
            "row_id": row_id or "",
            "ok": bool(ok),
            "status": status,
        }
        payload = (json.dumps(ack, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except Exception:
            pass


class MirrorServerApp:
    def __init__(self, host: str, port: int, out_dir: Path, force_fsync: bool, verbose: bool) -> None:
        self.store = MirrorStore(out_dir=out_dir, force_fsync=force_fsync, verbose=verbose)
        self.server = ThreadingTCPServer((host, port), MirrorHandler)
        self.server.app = self  # type: ignore[attr-defined]
        self.host = host
        self.port = port
        self.verbose = verbose

    def run(self) -> None:
        print(f"[Mirror] listening on {self.host}:{self.port}")
        print(f"[Mirror] output dir: {self.store.out_dir.resolve()}")
        try:
            self.server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.shutdown()
            self.server.server_close()
            print("[Mirror] stopped")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="StudyLogger mirror receiver (TCP)")
    p.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Listen port (default: {DEFAULT_PORT})")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--fsync", action="store_true", help="Call fsync after each row write")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    app = MirrorServerApp(
        host=args.host,
        port=int(args.port),
        out_dir=out_dir,
        force_fsync=bool(args.fsync),
        verbose=bool(args.verbose),
    )
    app.run()


if __name__ == "__main__":
    main()
