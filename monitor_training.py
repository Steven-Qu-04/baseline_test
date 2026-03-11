from __future__ import annotations

import argparse
import json
import os
import re
import smtplib
import ssl
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Optional


PROGRESS_PATTERNS = {
    "preprocess": re.compile(r"LMDB writer progress entries=(?P<entries>\d+) invalid=(?P<invalid>\d+)"),
    "preprocess_done": re.compile(r"Preprocessing finished\. LMDB=(?P<lmdb>\S+) elapsed_sec=(?P<elapsed>[0-9.]+)"),
    "epoch": re.compile(
        r"Epoch (?P<epoch>\d+) loss (?P<loss>[0-9.]+) batch_per_gpu=(?P<batch>\d+) grad_accum_steps=(?P<grad>\d+) lr=(?P<lr>[0-9.]+)"
    ),
    "train_start": re.compile(r"Training start ddp=(?P<ddp>\S+) rank=(?P<rank>\d+) local_rank=(?P<local_rank>\d+) world_size=(?P<world_size>\d+)"),
    "platform": re.compile(r"Platform summary gpu_count=(?P<gpu>\d+) logical_cpu_count=(?P<cpu>\d+)"),
}


@dataclass
class MonitorState:
    stage: str = "unknown"
    last_entries: int = 0
    invalid_count: int = 0
    last_epoch: int = 0
    last_loss: float = 0.0
    batch_per_gpu: int = 0
    grad_accum_steps: int = 0
    learning_rate: float = 0.0
    gpu_count: int = 0
    logical_cpu_count: int = 0
    last_log_line: str = ""
    last_email_type: str = ""
    last_email_ts: float = 0.0
    process_alive: bool = True
    issue: str = ""


def read_tail(path: Path, line_count: int = 200) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    return lines[-line_count:]


def detect_stage(lines: list[str], state: MonitorState) -> MonitorState:
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        state.last_log_line = line
        if "Starting preprocess" in line or "LMDB writer progress" in line or "Preprocessing finished" in line:
            if state.stage not in {"training", "completed"}:
                state.stage = "preprocessing"
        if "Step 2/3: medium DDP stress test" in line:
            state.stage = "stress_test"
        if "Step 3/3: full-scale distributed pretraining" in line or "Training start ddp=True" in line:
            state.stage = "training"
        if "==== Done ====" in line:
            state.stage = "completed"
        if "CRITICAL:" in line or "Traceback" in line or "ncclSystemError" in line:
            state.issue = line

        match = PROGRESS_PATTERNS["preprocess"].search(line)
        if match:
            state.last_entries = int(match.group("entries"))
            state.invalid_count = int(match.group("invalid"))
            continue

        match = PROGRESS_PATTERNS["epoch"].search(line)
        if match:
            state.stage = "training"
            state.last_epoch = int(match.group("epoch"))
            state.last_loss = float(match.group("loss"))
            state.batch_per_gpu = int(match.group("batch"))
            state.grad_accum_steps = int(match.group("grad"))
            state.learning_rate = float(match.group("lr"))
            continue

        match = PROGRESS_PATTERNS["platform"].search(line)
        if match:
            state.gpu_count = int(match.group("gpu"))
            state.logical_cpu_count = int(match.group("cpu"))
            continue
    return state


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def collect_ps_snapshot(pid: int) -> str:
    try:
        completed = subprocess.run(
            ["ps", "-o", "pid,ppid,etime,%cpu,%mem,cmd", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"ps unavailable: {exc}"


def maybe_collect_gpu_snapshot() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def build_email(subject: str, body: str, sender: str, recipient: str) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)
    return message


def send_email(subject: str, body: str, recipient: str) -> None:
    smtp_host = os.environ.get("MONITOR_SMTP_HOST")
    smtp_port = int(os.environ.get("MONITOR_SMTP_PORT", "465"))
    smtp_user = os.environ.get("MONITOR_SMTP_USER")
    smtp_password = os.environ.get("MONITOR_SMTP_PASSWORD")
    smtp_sender = os.environ.get("MONITOR_SMTP_SENDER", smtp_user or "")
    use_ssl = os.environ.get("MONITOR_SMTP_SSL", "1") == "1"

    if not smtp_host or not smtp_user or not smtp_password or not smtp_sender:
        raise RuntimeError(
            "SMTP environment is incomplete. Please set MONITOR_SMTP_HOST, MONITOR_SMTP_USER, MONITOR_SMTP_PASSWORD, and optionally MONITOR_SMTP_SENDER."
        )

    message = build_email(subject, body, smtp_sender, recipient)
    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=30) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(message)
        return

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(smtp_user, smtp_password)
        server.send_message(message)


def build_hourly_report(state: MonitorState, pid: int, output_dir: Path) -> str:
    parts = [
        f"Stage: {state.stage}",
        f"Main PID: {pid}",
        f"Process alive: {state.process_alive}",
    ]
    if state.stage == "preprocessing":
        parts.extend(
            [
                f"Serialized entries: {state.last_entries}",
                f"Invalid rows: {state.invalid_count}",
            ]
        )
    if state.last_epoch > 0:
        parts.extend(
            [
                f"Last epoch: {state.last_epoch}",
                f"Last loss: {state.last_loss:.6f}",
                f"Batch/GPU: {state.batch_per_gpu}",
                f"Grad accum: {state.grad_accum_steps}",
                f"LR: {state.learning_rate}",
            ]
        )
    parts.extend(
        [
            f"GPU count: {state.gpu_count}",
            f"Logical CPU count: {state.logical_cpu_count}",
            "",
            "Process snapshot:",
            collect_ps_snapshot(pid),
            "",
            "GPU snapshot:",
            maybe_collect_gpu_snapshot(),
            "",
            "Latest log line:",
            state.last_log_line or "(none)",
            "",
            f"Output dir: {output_dir}",
        ]
    )
    if state.issue:
        parts.extend(["", "Detected issue:", state.issue])
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight sidecar monitor for final_fullscale_run.sh")
    parser.add_argument("--pid", type=int, required=True, help="PID of the main final_fullscale_run.sh process")
    parser.add_argument("--log-path", default="/hy-tmp/result/project.log")
    parser.add_argument("--output-dir", default="/hy-tmp/result")
    parser.add_argument("--recipient", default="2243387748@qq.com")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--hourly-seconds", type=int, default=3600)
    parser.add_argument("--state-path", default="/hy-tmp/result/monitor_state.json")
    parser.add_argument("--send-start-email", action="store_true")
    args = parser.parse_args()

    pid = args.pid
    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir)
    state_path = Path(args.state_path)
    state = MonitorState()

    if args.send_start_email:
        send_email(
            subject="[mol_gtn] Monitor started",
            body=f"Monitoring PID {pid}\nLog path: {log_path}\nOutput dir: {output_dir}",
            recipient=args.recipient,
        )
        state.last_email_type = "start"
        state.last_email_ts = time.time()

    next_hourly_ts = time.time() + args.hourly_seconds

    while True:
        state.process_alive = is_process_alive(pid)
        lines = read_tail(log_path)
        state = detect_stage(lines, state)
        state_path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))

        if not state.process_alive:
            subject = "[mol_gtn] Process exited"
            body = build_hourly_report(state, pid, output_dir)
            if state.issue:
                subject = "[mol_gtn] ALERT: process exited after issue"
            send_email(subject=subject, body=body, recipient=args.recipient)
            return

        now = time.time()
        if state.issue and (state.last_email_type != "alert" or now - state.last_email_ts > args.hourly_seconds):
            send_email(
                subject="[mol_gtn] ALERT: issue detected",
                body=build_hourly_report(state, pid, output_dir),
                recipient=args.recipient,
            )
            state.last_email_type = "alert"
            state.last_email_ts = now
        elif now >= next_hourly_ts:
            send_email(
                subject=f"[mol_gtn] Hourly update: {state.stage}",
                body=build_hourly_report(state, pid, output_dir),
                recipient=args.recipient,
            )
            state.last_email_type = "hourly"
            state.last_email_ts = now
            next_hourly_ts = now + args.hourly_seconds

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"monitor failed: {exc}", file=sys.stderr)
        sys.exit(1)
