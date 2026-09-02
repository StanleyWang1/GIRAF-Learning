"""Fixed-shape shared-memory buffers for single-producer streams.

The sizing and rolling-window behavior follow the design used by Diffusion
Policy's SharedMemoryRingBuffer, while adding explicit absolute sequence
indices so consumers can detect overwritten samples instead of silently
reading the wrong slot.
"""

from __future__ import annotations

import numbers
import time
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.sharedctypes import RawValue

import numpy as np


class RingBufferOverrun(RuntimeError):
    """Raised when a consumer falls behind the rolling retention window."""


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """Name, per-sample shape, and dtype for one ring-buffer field."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype

    def __init__(self, name: str, shape: tuple[int, ...], dtype) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", tuple(shape))
        object.__setattr__(self, "dtype", np.dtype(dtype))


class SharedNDArray:
    """Pickleable NumPy view backed by ``multiprocessing.shared_memory``."""

    def __init__(self, block: shared_memory.SharedMemory, shape, dtype) -> None:
        self.block = block
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    @classmethod
    def create(
        cls,
        manager: SharedMemoryManager,
        shape: tuple[int, ...],
        dtype,
    ) -> SharedNDArray:
        dtype = np.dtype(dtype)
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        return cls(manager.SharedMemory(size=size), shape, dtype)

    def array(self) -> np.ndarray:
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.block.buf)

    def __getstate__(self):
        return self.block.name, self.shape, self.dtype.str

    def __setstate__(self, state) -> None:
        name, shape, dtype_str = state
        self.block = shared_memory.SharedMemory(name=name, create=False)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype_str)

    def close(self) -> None:
        self.block.close()


class SharedMemoryRingBuffer:
    """Rolling shared-memory window for one single-producer stream.

    Payloads never pass through a Python queue. The producer fills a complete
    slot and publishes it by advancing an aligned 64-bit counter. Consumers
    address samples using absolute counts and receive an explicit overrun if a
    requested sample has already been overwritten.
    """

    def __init__(
        self,
        manager: SharedMemoryManager,
        specs: list[ArraySpec],
        *,
        get_max_k: int,
        get_time_budget: float,
        put_desired_frequency: float,
        safety_margin: float = 1.5,
    ) -> None:
        if not specs:
            raise ValueError("at least one ArraySpec is required")
        if get_max_k <= 0:
            raise ValueError("get_max_k must be positive")
        if get_time_budget <= 0 or put_desired_frequency <= 0:
            raise ValueError("time budget and producer frequency must be positive")
        if safety_margin < 1.0:
            raise ValueError("safety_margin must be at least 1.0")
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError("ring-buffer field names must be unique")

        self.specs = tuple(specs)
        self.get_max_k = int(get_max_k)
        self.get_time_budget = float(get_time_budget)
        self.put_desired_frequency = float(put_desired_frequency)
        self.safety_margin = float(safety_margin)
        self.buffer_size = (
            int(np.ceil(put_desired_frequency * get_time_budget * safety_margin))
            + get_max_k
        )
        self.fields = {
            spec.name: SharedNDArray.create(
                manager,
                (self.buffer_size,) + spec.shape,
                spec.dtype,
            )
            for spec in self.specs
        }
        self._slot_write_times = SharedNDArray.create(
            manager, (self.buffer_size,), np.float64
        )
        self._slot_write_times.array().fill(-np.inf)
        # RawValue avoids a lock on the real-time producer path. There is one
        # producer per ring, and the counter is advanced only after all fields
        # in the slot have been copied.
        self._count = RawValue("Q", 0)

    @classmethod
    def create_from_examples(
        cls,
        manager: SharedMemoryManager,
        examples: Mapping[str, np.ndarray | numbers.Number],
        *,
        get_max_k: int = 32,
        get_time_budget: float = 0.01,
        put_desired_frequency: float = 60.0,
        safety_margin: float = 1.5,
    ) -> SharedMemoryRingBuffer:
        specs: list[ArraySpec] = []
        for name, value in examples.items():
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError(f"object dtype is unsupported for {name!r}")
            specs.append(ArraySpec(name, array.shape, array.dtype))
        return cls(
            manager,
            specs,
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_desired_frequency,
            safety_margin=safety_margin,
        )

    @property
    def count(self) -> int:
        return int(self._count.value)

    def put(
        self,
        data: Mapping[str, np.ndarray | numbers.Number],
        *,
        wait: bool = False,
    ) -> int:
        """Publish one sample and return its absolute zero-based count."""

        expected = {spec.name for spec in self.specs}
        if set(data) != expected:
            missing = expected.difference(data)
            extra = set(data).difference(expected)
            raise KeyError(f"ring payload mismatch; missing={missing}, extra={extra}")

        count = self.count
        slot = count % self.buffer_size
        protected_slot = (slot + self.get_max_k - 1) % self.buffer_size
        now = time.monotonic()
        age = now - float(self._slot_write_times.array()[protected_slot])
        if age < self.get_time_budget:
            if not wait:
                raise TimeoutError(
                    "producer exceeded configured shared-memory copy budget"
                )
            time.sleep(self.get_time_budget - age)

        for spec in self.specs:
            value = np.asarray(data[spec.name], dtype=spec.dtype)
            if value.shape != spec.shape:
                raise ValueError(
                    f"{spec.name} has shape {value.shape}, expected {spec.shape}"
                )
            self.fields[spec.name].array()[slot] = value
        self._slot_write_times.array()[slot] = time.monotonic()
        self._count.value = count + 1
        return count

    def get(self) -> dict[str, np.ndarray]:
        count = self.count
        if count == 0:
            raise LookupError("ring buffer is empty")
        return {
            key: value[0] for key, value in self.get_range(count - 1, count).items()
        }

    def get_last_k(self, k: int) -> dict[str, np.ndarray]:
        if k <= 0 or k > self.get_max_k:
            raise ValueError(f"k must be between 1 and {self.get_max_k}")
        count = self.count
        if k > count:
            raise LookupError(f"requested {k} samples but only {count} exist")
        return self.get_range(count - k, count)

    def get_range(self, start_count: int, end_count: int) -> dict[str, np.ndarray]:
        """Copy the half-open absolute count range ``[start_count, end_count)``."""

        if start_count < 0 or end_count < start_count:
            raise ValueError("invalid absolute ring-buffer range")
        snapshot_count = self.count
        if end_count > snapshot_count:
            raise ValueError("requested samples have not been produced yet")
        if start_count < max(0, snapshot_count - self.buffer_size):
            raise RingBufferOverrun(
                f"sample {start_count} was overwritten; oldest is "
                f"{max(0, snapshot_count - self.buffer_size)}"
            )
        n_items = end_count - start_count
        if n_items > self.get_max_k:
            raise ValueError(
                f"requested {n_items} items; configured get_max_k={self.get_max_k}"
            )

        started = time.monotonic()
        indices = np.arange(start_count, end_count, dtype=np.int64) % self.buffer_size
        result = {
            spec.name: self.fields[spec.name].array()[indices].copy()
            for spec in self.specs
        }
        elapsed = time.monotonic() - started
        if elapsed > self.get_time_budget:
            raise TimeoutError(
                f"shared-memory copy took {elapsed:.4f}s; budget is "
                f"{self.get_time_budget:.4f}s"
            )
        return result

    def close(self) -> None:
        for field in self.fields.values():
            field.close()
        self._slot_write_times.close()
