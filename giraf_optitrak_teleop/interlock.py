"""Hold-to-run deadman and fault re-arm state machine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    engaged: bool
    newly_engaged: bool
    newly_disengaged: bool


class HoldToRunGate:
    """Engage only on a fresh press after a complete released state."""

    def __init__(self) -> None:
        self._previous_held = (
            True  # Startup while held can never look like a fresh press.
        )
        self._released_since_fault = False
        self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    def invalidate(self, held: bool) -> None:
        self._engaged = False
        self._released_since_fault = not held
        self._previous_held = held

    def step(self, held: bool, prerequisites_ok: bool) -> GateResult:
        was_engaged = self._engaged
        if not prerequisites_ok:
            self._engaged = False
            if not held:
                self._released_since_fault = True
        elif not held:
            self._engaged = False
            self._released_since_fault = True
        elif not self._previous_held and self._released_since_fault:
            self._engaged = True
            self._released_since_fault = False

        newly_engaged = self._engaged and not was_engaged
        newly_disengaged = was_engaged and not self._engaged
        self._previous_held = held
        return GateResult(self._engaged, newly_engaged, newly_disengaged)
