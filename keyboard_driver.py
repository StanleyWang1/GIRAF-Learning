"""Keyboard input for clutch and grasp controls using Linux evdev."""

from __future__ import annotations

import threading
import time

from evdev import InputDevice, ecodes, list_devices


DEFAULT_CONTROL_HZ = 30.0
DEFAULT_PRINT_HZ = 10.0


class KeyboardState:
    """Thread-safe keyboard state shared with the application."""

    def __init__(self, devices: list[InputDevice]) -> None:
        self.devices = devices
        self.clutch = False
        self.grasp = False
        self._space_down = {device.path: False for device in devices}
        self._lock = threading.Lock()


def keyboard_connect() -> KeyboardState:
    """Open every available evdev input device."""
    devices = [InputDevice(path, readonly=True) for path in list_devices()]
    if not devices:
        raise RuntimeError("No input devices found in /dev/input.")
    return KeyboardState(devices)


def keyboard_read(state: KeyboardState) -> dict[str, bool]:
    """Process queued key events and return the updated controls."""
    space_events: list[tuple[str, bool]] = []
    b_presses = 0

    for device in state.devices:
        try:
            for event in device.read():
                if event.type != ecodes.EV_KEY:
                    continue
                if event.code == ecodes.KEY_SPACE:
                    space_events.append((device.path, event.value != 0))
                elif event.code == ecodes.KEY_B and event.value == 1:
                    b_presses += 1
        except BlockingIOError:
            continue

    with state._lock:
        for path, is_down in space_events:
            state._space_down[path] = is_down

        state.clutch = any(state._space_down.values())
        if b_presses % 2 == 1:
            state.grasp = not state.grasp

        return {"clutch": state.clutch, "grasp": state.grasp}


def keyboard_status(state: KeyboardState) -> dict[str, bool]:
    """Return the most recently processed control state."""
    with state._lock:
        return {"clutch": state.clutch, "grasp": state.grasp}


def keyboard_control(
    state: KeyboardState,
    stop_event: threading.Event,
    hz: float = DEFAULT_CONTROL_HZ,
) -> None:
    """Process keyboard state at ``hz`` until ``stop_event`` is set."""
    if hz <= 0:
        raise ValueError("hz must be greater than zero")

    period = 1.0 / hz
    while not stop_event.is_set():
        start = time.monotonic()
        keyboard_read(state)
        remaining = max(0.0, period - (time.monotonic() - start))
        stop_event.wait(remaining)


def keyboard_disconnect(state: KeyboardState) -> None:
    """Close every evdev input device."""
    for device in state.devices:
        device.close()


def main() -> None:
    state = keyboard_connect()
    stop_event = threading.Event()
    control_thread = threading.Thread(
        target=keyboard_control,
        args=(state, stop_event, DEFAULT_CONTROL_HZ),
        name="keyboard-control",
        daemon=True,
    )
    control_thread.start()

    print("Hold SPACE for clutch; press B to toggle grasp; Ctrl+C to exit.")
    try:
        while not stop_event.wait(1.0 / DEFAULT_PRINT_HZ):
            status = keyboard_status(state)
            print(
                f"\rclutch={str(status['clutch']):<5} "
                f"grasp={str(status['grasp']):<5}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        control_thread.join()
        keyboard_disconnect(state)
        print()


if __name__ == "__main__":
    main()
