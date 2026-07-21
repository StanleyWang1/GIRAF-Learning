"""Thread-safe ROS feedback and publisher-ownership health checks."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


class JointStateHealthTracker:
    def __init__(self, expected_names: Sequence[str]) -> None:
        self.expected_names = tuple(expected_names)
        self._lock = threading.Lock()
        self._receipt_ns = 0
        self._valid_messages = 0
        self._error = "no MD80 joint state received"

    def update(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        receipt_ns: Optional[int] = None,
    ) -> bool:
        try:
            index = {name: offset for offset, name in enumerate(names)}
            values = np.array([positions[index[name]] for name in self.expected_names])
            if not np.all(np.isfinite(values)):
                raise ValueError("positions are non-finite")
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            with self._lock:
                self._valid_messages = 0
                self._error = "invalid MD80 joint state: %s" % exc
            return False
        with self._lock:
            self._receipt_ns = time.monotonic_ns() if receipt_ns is None else receipt_ns
            self._valid_messages = min(2, self._valid_messages + 1)
            self._error = ""
        return True

    def health(self, now_ns: int, max_age_ms: float) -> Tuple[bool, str, float]:
        with self._lock:
            receipt_ns = self._receipt_ns
            count = self._valid_messages
            error = self._error
        if receipt_ns <= 0:
            return False, error, math.inf
        age_ms = (now_ns - receipt_ns) / 1e6
        if error:
            return False, error, age_ms
        if count < 2:
            return False, "waiting for a second valid MD80 joint state", age_ms
        if age_ms > max_age_ms:
            return False, "MD80 joint state is stale", age_ms
        return True, "ready", age_ms


@dataclass(frozen=True)
class PublisherHealth:
    healthy: bool
    age_ms: float
    competitors: Tuple[str, ...]
    error: str


def competing_publishers(
    publishers: Iterable[Sequence[object]], topic: str, own_node: str
) -> Tuple[str, ...]:
    """Return every publisher on ``topic`` except this node, deterministically."""

    for published_topic, nodes in publishers:
        if published_topic != topic:
            continue
        if not isinstance(nodes, (list, tuple)):
            raise ValueError("ROS master returned invalid publisher nodes")
        return tuple(sorted(str(node) for node in nodes if str(node) != own_node))
    return ()


class PublisherMonitor:
    """Poll ROS master off the command thread and reject competing publishers."""

    def __init__(self, master, topic: str, own_node: str, period_sec: float = 1.0):
        self.master = master
        self.topic = topic
        self.own_node = own_node
        self.period_sec = float(period_sec)
        self._lock = threading.Lock()
        self._receipt_ns = 0
        self._competitors: Tuple[str, ...] = ()
        self._error = "publisher ownership has not been checked"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="giraf-publisher-monitor",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                publishers, _subscribers, _services = self.master.getSystemState()
                competitors = competing_publishers(
                    publishers, self.topic, self.own_node
                )
                with self._lock:
                    self._receipt_ns = time.monotonic_ns()
                    self._competitors = competitors
                    self._error = ""
            except Exception as exc:
                with self._lock:
                    self._error = "ROS master ownership query failed: %s" % exc
            self._stop_event.wait(self.period_sec)

    def health(self, now_ns: int, max_age_ms: float = 2500.0) -> PublisherHealth:
        with self._lock:
            receipt_ns = self._receipt_ns
            competitors = self._competitors
            error = self._error
        age_ms = math.inf if receipt_ns <= 0 else (now_ns - receipt_ns) / 1e6
        if error:
            return PublisherHealth(False, age_ms, competitors, error)
        if age_ms > max_age_ms:
            return PublisherHealth(
                False, age_ms, competitors, "publisher ownership check is stale"
            )
        if competitors:
            return PublisherHealth(
                False, age_ms, competitors, "competing teleop publisher detected"
            )
        return PublisherHealth(True, age_ms, competitors, "")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
