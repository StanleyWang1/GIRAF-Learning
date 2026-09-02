"""Process-isolated access to the native pyCandle MAB driver."""

from __future__ import annotations

import math
import queue
import selectors
import subprocess
import sys
import threading
import time
from pathlib import Path


READY = "__MAB_WORKER_READY__"
ERROR = "__MAB_WORKER_ERROR__ "


class MabWorker:
    def __init__(self, startup_timeout: float = 15.0, watchdog: float = 0.5) -> None:
        self.startup_timeout = startup_timeout
        self.watchdog = watchdog
        self.process: subprocess.Popen[str] | None = None
        self.messages: queue.Queue[tuple[str, str]] = queue.Queue()
        self.reader: threading.Thread | None = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("MAB worker is already started")
        script = Path(__file__).resolve()
        self.process = subprocess.Popen(
            (sys.executable, str(script), "--child", "--watchdog", str(self.watchdog)),
            cwd=script.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.reader = threading.Thread(target=self._read_output, name="mab-output", daemon=True)
        self.reader.start()

        deadline = time.monotonic() + self.startup_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate()
                raise RuntimeError("MAB worker did not become ready")
            try:
                kind, detail = self.messages.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError(self._exit_message("during startup"))
                continue
            if kind == "ready":
                print("[INIT][MAB] Worker ready.", flush=True)
                return
            if kind == "error":
                self._terminate()
                raise RuntimeError(f"MAB worker startup failed: {detail}")

    def command(self, roll: float, pitch: float, boom: float) -> None:
        values = (float(roll), float(pitch), float(boom))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("MAB target is not finite")
        process = self.process
        if process is None:
            raise RuntimeError("MAB worker is not started")
        if process.poll() is not None:
            raise RuntimeError(self._exit_message("while controlling motors"))
        try:
            assert process.stdin is not None
            process.stdin.write(f"SET {values[0]:.9g} {values[1]:.9g} {values[2]:.9g}\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(self._exit_message("while sending a command")) from exc

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write("STOP\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            print("[STOP][MAB] Worker did not stop; terminating it.", flush=True)
            self._terminate()
        if self.reader is not None:
            self.reader.join(timeout=1.0)
        self.process = None

    def _read_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for raw_line in self.process.stdout:
            line = raw_line.rstrip()
            if line == READY:
                self.messages.put(("ready", ""))
            elif line.startswith(ERROR):
                self.messages.put(("error", line[len(ERROR) :]))
            elif line:
                print(line, flush=True)

    def _exit_message(self, context: str) -> str:
        assert self.process is not None
        code = self.process.poll()
        if code is None:
            return f"MAB worker failed {context}"
        if code < 0:
            return f"MAB worker received signal {-code} {context}"
        return f"MAB worker exited with code {code} {context}"

    def _terminate(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)


def child_main(watchdog: float) -> int:
    candle = motors = None
    result = 0
    try:
        from motor_driver import motor_connect, motor_disconnect, motor_drive

        candle, motors = motor_connect()
        print(READY, flush=True)
        selector = selectors.DefaultSelector()
        selector.register(sys.stdin, selectors.EVENT_READ)
        last_command = None

        while True:
            if selector.select(timeout=0.05):
                line = sys.stdin.readline()
                if not line or line.strip() == "STOP":
                    break
                fields = line.split()
                if len(fields) != 4 or fields[0] != "SET":
                    raise RuntimeError(f"invalid command: {line.strip()}")
                targets = tuple(float(value) for value in fields[1:])
                if not all(math.isfinite(value) for value in targets):
                    raise RuntimeError("received a non-finite target")
                motor_drive(candle, motors, *targets)
                last_command = time.monotonic()
            elif last_command is not None and time.monotonic() - last_command > watchdog:
                raise RuntimeError(f"command watchdog exceeded {watchdog:.2f} s")
    except BaseException as exc:
        print(f"{ERROR}{type(exc).__name__}: {exc}", flush=True)
        result = 1
    finally:
        if candle is not None:
            try:
                motor_disconnect(candle)
            except BaseException as exc:
                print(f"{ERROR}shutdown failed: {exc}", flush=True)
                result = 1
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--watchdog", type=float, default=0.5)
    args = parser.parse_args()
    if not args.child:
        parser.error("this script is launched by teleop.py")
    raise SystemExit(child_main(args.watchdog))
