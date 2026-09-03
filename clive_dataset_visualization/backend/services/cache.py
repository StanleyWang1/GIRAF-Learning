from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Generic, Hashable, Optional, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, max_items: int = 512):
        if max_items <= 0:
            raise ValueError("max_items must be > 0")
        self._max_items = max_items
        self._data: "OrderedDict[K, V]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key not in self._data:
                return None
            value = self._data.pop(key)
            self._data[key] = value
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            self._data[key] = value
            while len(self._data) > self._max_items:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
