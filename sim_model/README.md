# GIRAF MuJoCo simulation

`sim_model` is the simulation-only package for the GIRAF manipulator. It owns
MuJoCo model loading, state, stepping, and rendering. Controllers, teleoperation,
tasks, policies, and data collection should consume this API from their own
packages.

See [`sim.md`](sim.md) for the package boundary, a detailed comparison of the
current MJCF files, and the model-consolidation plan.

Install its runtime dependencies with:

```bash
python -m pip install -r sim_model/requirements.txt
```

## Sanity launcher

Open the arm-only scene in MuJoCo's viewer:

```bash
python -m sim_model --scene arm
```

Run any built-in scene headlessly:

```bash
python -m sim_model --scene bananas --headless --steps 1000
```

Built-in scenes are `arm`, `banana`, `bananas`, and `ycb`.

## Python API

```python
from sim_model import GirafSimulation

with GirafSimulation("banana", frame_skip=10) as sim:
    state = sim.reset()
    sim.set_actuator_target("actuator_P3", 0.5)
    state = sim.step()
    wrist_pose = sim.body_pose("wrist")
    wrist_rgb = sim.render("wrist_cam", width=640, height=480)
```

One `GirafSimulation` instance owns one independent `MjModel` and `MjData` pair.
The core never sleeps, opens windows, reads input devices, or chooses control
semantics. `frame_skip` controls how many physics steps one `step()` call runs.

Run the simulation tests with:

```bash
python -m unittest discover -s sim_model/tests -v
```

## Layout

- `scenes.py`: built-in scene registry and custom MJCF path resolution.
- `simulation.py`: simulation lifecycle, state, named accessors, and camera rendering.
- `runner.py`: optional headless and passive-viewer loops.
- `models/`: MJCF scenes and YCB assets.
- `utils/`: asset download and MJCF generation utilities.
- `tests/`: model-loading and simulation API regression tests.
