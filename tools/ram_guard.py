#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Run a child process under a sampled RSS limit.

macOS does not honour RLIMIT_AS/RLIMIT_DATA, and cgroups are not available on a
normal macOS host.  This guard therefore samples the child's resident set size
(RSS) and kills it when the limit is crossed.  The child's ru_maxrss is also
reported after exit, so a short spike that slipped between samples is visible.

This is a simulation of a hard memory cap, not a kernel-level enforcement
mechanism.  It is intentionally kept outside the package.

Usage:
    python tools/ram_guard.py --limit-mb 100 -- python train_script.py
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

IS_DARWIN = sys.platform == "darwin"


def _rss_bytes(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total
    except psutil.Error:
        return 0


def _child_peak_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    # macOS reports bytes, Linux reports KiB.
    divisor = 1024 * 1024 if IS_DARWIN else 1024
    return round(usage.ru_maxrss / divisor, 2)


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        data = path.read_text(errors="replace")
    except OSError:
        return ""
    return data[-limit:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-mb", type=float, required=True)
    parser.add_argument("--interval-ms", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    limit_mb = args.limit_mb
    limit_bytes = int(limit_mb * 1024 * 1024)
    started = time.perf_counter()

    with (
        tempfile.TemporaryFile(mode="w+") as stdout,
        tempfile.TemporaryFile(mode="w+") as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        peak_sampled = 0
        killed = False
        try:
            while process.poll() is None:
                rss = _rss_bytes(process.pid)
                peak_sampled = max(peak_sampled, rss)
                if rss > limit_bytes:
                    process.kill()
                    killed = True
                    break
                time.sleep(args.interval_ms / 1000.0)
        finally:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        elapsed = time.perf_counter() - started
        stdout.seek(0)
        stderr.seek(0)
        out_text = stdout.read()[-4000:]
        err_text = stderr.read()[-4000:]

    peak_child = _child_peak_mb()
    exceeded = killed or peak_child > limit_mb
    result = {
        "limit_mb": limit_mb,
        "peak_sampled_mb": round(peak_sampled / 1024 / 1024, 2),
        "peak_child_ru_maxrss_mb": peak_child,
        "exceeded_limit": exceeded,
        "killed": killed,
        "exit_code": process.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "stdout": out_text,
        "stderr": err_text,
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 137 if exceeded else int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
