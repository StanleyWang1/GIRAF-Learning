"""Keyboard input for clutch and grasp controls using Linux evdev."""

from __future__ import annotations

import threading
import time
from collections import deque

from evdev import InputDevice, ecodes, list_devices

KEYBOARD_POLL_HZ = 30.0
STATUS_PRINT_HZ = 10.0


class KeyboardState:
    """Thread-safe keyboard state shared with the application."""

    def __init__(
        self, devices: list[InputDevice], *, session_events: bool = False
    ) -> None:
        self.devices = devices
        self.clutch = False
        self.grasp = False
        self.record_toggle_count = 0
        self.record_event_timestamp_ns = 0
        self._pressed = {device.path: set(device.active_keys()) for device in devices}
        self._space_down = {
            path: ecodes.KEY_SPACE in pressed for path, pressed in self._pressed.items()
        }
        self.clutch = any(self._space_down.values())
        self.quit_requested = False
        self._events = deque() if session_events else None
        self._lock = threading.Lock()


def keyboard_connect(*, session_events: bool = False) -> KeyboardState:
    """Open every available evdev input device."""
    devices = [InputDevice(path, readonly=True) for path in list_devices()]
    if not devices:
        raise RuntimeError("No input devices found in /dev/input.")
    return KeyboardState(devices, session_events=session_events)


def keyboard_read(state: KeyboardState) -> dict[str, bool | int]:
    """Process queued key events and return the updated controls."""
    pending = []
    for device in state.devices:
        try:
            for event in device.read():
                if event.type == ecodes.EV_SYN and event.code == ecodes.SYN_DROPPED:
                    raise RuntimeError("keyboard input events were lost")
                if event.type == ecodes.EV_KEY:
                    pending.append((event.timestamp(), device.path, event))
        except BlockingIOError:
            continue

    with state._lock:
        for _, path, event in sorted(pending, key=lambda item: item[0]):
            if event.value == 2:
                continue  # autorepeat is never a new press
            pressed = state._pressed[path]
            was_down = event.code in pressed
            is_down = event.value == 1
            if is_down:
                pressed.add(event.code)
            else:
                pressed.discard(event.code)
            if was_down == is_down:
                continue
            transition = None
            if event.code == ecodes.KEY_SPACE:
                state._space_down[path] = is_down
                clutch = any(state._space_down.values())
                if clutch != state.clutch:
                    state.clutch = clutch
                    transition = ("clutch", clutch)
            elif is_down:
                if event.code == ecodes.KEY_B:
                    state.grasp = not state.grasp
                    transition = ("grasp", True)
                elif event.code == ecodes.KEY_R:
                    state.record_toggle_count += 1
                    state.record_event_timestamp_ns = time.monotonic_ns()
                elif event.code == ecodes.KEY_D:
                    transition = ("mode", True)
                elif event.code == ecodes.KEY_Q:
                    state.quit_requested = True
                    transition = ("quit", True)
            if transition is not None and state._events is not None:
                if len(state._events) >= 1024:
                    raise RuntimeError("session keyboard event queue overflow")
                state._events.append(transition)
        return _status_unlocked(state)


def _status_unlocked(state: KeyboardState) -> dict[str, bool | int]:
    return {
        "clutch": state.clutch,
        "grasp": state.grasp,
        "record_toggle_count": state.record_toggle_count,
        "record_event_timestamp_ns": state.record_event_timestamp_ns,
        "quit_requested": state.quit_requested,
    }


def keyboard_session_events(state: KeyboardState):
    """Atomically drain ordered session controls and snapshot held keys."""
    with state._lock:
        if state._events is None:
            raise RuntimeError("session events were not enabled")
        events = list(state._events)
        state._events.clear()
        return events, _status_unlocked(state)


def keyboard_status(state: KeyboardState) -> dict[str, bool | int]:
    """Return the most recently processed control state."""
    with state._lock:
        return _status_unlocked(state)


def keyboard_control(
    state: KeyboardState,
    stop_event: threading.Event,
    hz: float = KEYBOARD_POLL_HZ,
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
        args=(state, stop_event, KEYBOARD_POLL_HZ),
        name="keyboard-control",
        daemon=True,
    )
    control_thread.start()

    print(
        "Hold SPACE for clutch; press B to toggle grasp; "
        "press R to toggle recording; Ctrl+C to exit."
    )
    try:
        while not stop_event.wait(1.0 / STATUS_PRINT_HZ):
            status = keyboard_status(state)
            print(
                f"\rclutch={status['clutch']!s:<5} grasp={status['grasp']!s:<5}",
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
