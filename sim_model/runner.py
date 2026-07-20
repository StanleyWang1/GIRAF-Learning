"""Optional interactive and headless runners around the simulation core."""

from __future__ import annotations

import time

import mujoco.viewer

from .simulation import GirafSimulation


def run_headless(simulation: GirafSimulation, steps: int) -> None:
    """Advance a simulation without wall-clock throttling."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    for _ in range(steps):
        simulation.step()


def run_viewer(
    simulation: GirafSimulation,
    *,
    duration: float | None = None,
    realtime: bool = True,
) -> None:
    """Launch MuJoCo's passive viewer and step until closed or timed out."""

    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")

    start_sim_time = simulation.data.time
    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
        while viewer.is_running():
            step_started = time.perf_counter()
            simulation.step()
            viewer.sync()

            if (
                duration is not None
                and simulation.data.time - start_sim_time >= duration
            ):
                break
            if realtime:
                remaining = simulation.step_dt - (time.perf_counter() - step_started)
                if remaining > 0:
                    time.sleep(remaining)
