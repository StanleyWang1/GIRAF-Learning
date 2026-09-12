"""Run giraf-train under GDB, tee its log, and sample CPU temperatures."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def sample_temperatures(path: Path, stopped: threading.Event) -> None:
    sensors = []
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hwmon / "name").read_text().strip() == "coretemp":
                sensors.extend(hwmon.glob("temp*_label"))
        except OSError:
            continue
    with path.open("x") as output:
        while True:
            values = {}
            errors = {}
            for label in sensors:
                try:
                    name = label.read_text().strip()
                    reading = label.with_name(label.name.replace("_label", "_input"))
                    values[name] = int(reading.read_text()) / 1000
                except (OSError, ValueError) as error:
                    errors[str(label)] = str(error)
            output.write(
                json.dumps(
                    {"time": now(), "temperatures_c": values, "read_errors": errors}
                )
                + "\n"
            )
            output.flush()
            if stopped.wait(2):
                break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output-dir", required=True, type=Path)
    args, _ = parser.parse_known_args()
    # Match the fresh-run guard in the training command; never reuse old logs.
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"ERROR: run directory already exists: {args.output_dir}")
    debugger = Path(__file__).resolve().with_name("train_diagnostics.gdb")
    command = [
        "gdb",
        "-nx",
        "-q",
        "-batch",
        "-x",
        str(debugger),
        "--args",
        sys.executable,
        "-m",
        "giraf.learning.train_cli",
        *sys.argv[1:],
    ]
    environment = os.environ.copy()
    environment.setdefault("WANDB_CONSOLE", "off")
    status = {
        "state": "starting",
        "started_at": now(),
        "monitor_pid": os.getpid(),
        "command": command,
        "affinity": sorted(os.sched_getaffinity(0)),
        "wandb_mode": environment.get("WANDB_MODE"),
        "wandb_name": environment.get("WANDB_NAME"),
    }
    status_path = args.output_dir / "monitor_status.json"

    def save_status() -> None:
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    save_status()
    stopped = threading.Event()
    sampler = threading.Thread(
        target=sample_temperatures,
        args=(args.output_dir / "cpu_temperatures.jsonl", stopped),
        daemon=True,
    )
    sampler.start()
    try:
        with (args.output_dir / "train.log").open("x") as log:
            print(
                f"[MONITOR] {args.output_dir}; temperature sampling every 2 seconds",
                flush=True,
            )
            with subprocess.Popen(
                command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            ) as process:
                status.update(state="running", debugger_pid=process.pid)
                save_status()
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                exit_code = process.wait()
            status.update(
                state="completed" if exit_code == 0 else "failed",
                exit_code=exit_code,
                finished_at=now(),
            )
            (args.output_dir / "exit_status.txt").write_text(str(exit_code) + "\n")
            return exit_code
    except BaseException as error:
        status.update(state="monitor_error", error=str(error), finished_at=now())
        raise
    finally:
        stopped.set()
        sampler.join(timeout=5)
        save_status()


if __name__ == "__main__":
    raise SystemExit(main())
