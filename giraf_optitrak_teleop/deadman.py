"""Physical Linux evdev Space-bar hold-to-run input."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    from evdev import InputDevice, ecodes
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in the container
    InputDevice = None
    ecodes = None
    EVDEV_IMPORT_ERROR = exc
else:
    EVDEV_IMPORT_ERROR = None


@dataclass(frozen=True)
class DeadmanSnapshot:
    held: bool
    device_ok: bool
    last_event_ns: int
    error: str


class SpaceDeadmanReader:
    """Read genuine Space press/release events; keyboard repeat never changes state."""

    def __init__(self, device_path: str) -> None:
        if EVDEV_IMPORT_ERROR is not None:
            raise RuntimeError("python3-evdev is required") from EVDEV_IMPORT_ERROR
        self.device_path = device_path
        self._lock = threading.Lock()
        self._held = False
        self._device_ok = False
        self._last_event_ns = 0
        self._error = "keyboard has not been opened"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._device = None

    def start(self) -> str:
        device = InputDevice(self.device_path)
        active_keys = set(device.active_keys(verbose=False))
        with self._lock:
            self._device = device
            self._held = ecodes.KEY_SPACE in active_keys
            self._device_ok = True
            self._last_event_ns = time.monotonic_ns()
            self._error = ""
        self._thread = threading.Thread(
            target=self._run,
            name="giraf-space-deadman",
            daemon=True,
        )
        self._thread.start()
        return "%s (%s)" % (device.name, self.device_path)

    def _run(self) -> None:
        try:
            assert self._device is not None
            for event in self._device.read_loop():
                if self._stop_event.is_set():
                    break
                if event.type != ecodes.EV_KEY or event.code != ecodes.KEY_SPACE:
                    continue
                if event.value not in (0, 1):
                    continue
                with self._lock:
                    self._held = event.value == 1
                    self._last_event_ns = time.monotonic_ns()
        except Exception as exc:  # device removal and read failures are safety faults
            if not self._stop_event.is_set():
                with self._lock:
                    self._held = False
                    self._device_ok = False
                    self._error = "keyboard read failed: %s" % exc
        finally:
            with self._lock:
                self._held = False
                self._device_ok = False
                if not self._error and not self._stop_event.is_set():
                    self._error = "keyboard reader stopped"

    def snapshot(self) -> DeadmanSnapshot:
        with self._lock:
            return DeadmanSnapshot(
                self._held,
                self._device_ok,
                self._last_event_ns,
                self._error,
            )

    def stop(self) -> None:
        self._stop_event.set()
        device = None
        with self._lock:
            device = self._device
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
