# GIRAF simulation design

## Scope

`sim_model` is the MuJoCo simulation package for GIRAF. It is responsible for:

- loading a GIRAF scene;
- owning an independent `MjModel` and `MjData` pair;
- resetting and stepping physics;
- accepting raw MuJoCo actuator targets;
- exposing simulation state, named joints, body poses, and cameras;
- rendering RGB or depth images; and
- optionally running MuJoCo's passive viewer.

Task-space control, teleoperation, learned policies, task rewards, and data
collection do not belong in this package. Those systems should use
`GirafSimulation` as a backend.

## Package structure

| Path | Responsibility |
| --- | --- |
| `simulation.py` | Simulation lifecycle, stepping, state snapshots, named accessors, and rendering |
| `scenes.py` | Registry of built-in scenes and resolution of custom MJCF paths |
| `runner.py` | Optional headless and passive-viewer loops |
| `__main__.py` | Command-line sanity launcher |
| `models/` | MJCF scenes and YCB assets |
| `utils/` | YCB download and MJCF generation utilities |
| `tests/` | Model-loading and simulation API regression tests |

The core simulation never reads an input device, opens a viewer automatically,
sleeps to enforce wall-clock timing, or implements a control policy.

## Runtime API

```python
from sim_model import GirafSimulation

with GirafSimulation("banana", frame_skip=10) as sim:
    initial_state = sim.reset()
    sim.set_actuator_target("actuator_P3", 0.5)
    next_state = sim.step()
    wrist_pose = sim.body_pose("wrist")
    wrist_rgb = sim.render("wrist_cam", width=1280, height=720)
```

The MJCF physics timestep is 1 ms. `frame_skip` is the number of physics steps
performed by one `step()` call, so the effective caller timestep is:

```text
step_dt = 0.001 seconds * frame_skip
```

## Current MJCF files

The four current files are complete standalone models. Each repeats the full
robot, actuators, defaults, lighting, and floor rather than composing a shared
robot definition with scene-specific objects.

| File | Scene key | Contents | Compiled `nq` / `nv` | Robot variant |
| --- | --- | --- | ---: | --- |
| `GIRAF.xml` | `arm` | Arm only | 8 / 8 | Base robot variant |
| `GIRAF_banana.xml` | `banana` | Arm, one banana, bin | 15 / 14 | Narrow-jaw variant |
| `GIRAF_bananas.xml` | `bananas` | Arm, ten bananas, bin | 78 / 68 | Narrow-jaw variant |
| `GIRAF_ycb.xml` | `ycb` | Arm, eleven YCB objects, bin | 85 / 74 | Base robot variant |

### Settings shared by all four files

| Setting | Value |
| --- | --- |
| Gravity | `0 0 -9.81` m/s^2 |
| Physics timestep | `0.001` s |
| Solver | Newton |
| Solver iterations | 200 |
| Solver tolerance | `1e-10` |
| No-slip iterations | 10 |
| Arm joints | `R1`, `R2`, `P3`, `R4`, `R5`, `R6` |
| Gripper joints | `left_grip_joint`, `right_grip_joint` |
| Actuator gains | Identical, including gripper `kp=200` |
| Floor, lights, and contact defaults | Identical |

The compiler configuration, global defaults, joint names, non-wrist arm geometry,
joint damping, arm actuator ranges, and actuator gains are shared. They should be
defined once. Wrist axes and anchors currently differ between variants.

## Robot variant differences

There are two internally consistent variants:

- `GIRAF.xml` and `GIRAF_ycb.xml` are structurally identical for the
  robot, visual settings, tendons, actuators, and contact section.
- `GIRAF_banana.xml` and `GIRAF_bananas.xml` are structurally identical for those
  same sections.

| Property | Large-jaw variant | Narrow-jaw variant | Consequence |
| --- | --- | --- | --- |
| Files | `GIRAF.xml`, `GIRAF_ycb.xml` | `GIRAF_banana.xml`, `GIRAF_bananas.xml` | The selected scene currently changes the robot itself |
| Wrist axes and anchors | R4: `0 -1 0` at origin; R5: `0 0 -1` and R6: `1 0 0` at `x=0.0597` | R4: `0.866025 0 0.5`; R5: `0 -1 0`; R6: `1 0 0`, all at origin | Wrist kinematics differ by scene |
| Wrist camera `fovy` | 55 degrees | 63 degrees | Policies receive different camera intrinsics by scene |
| Default offscreen size | Unspecified | 1280 x 720 | Render defaults differ, although the Python renderer can override size |
| Jaw geom `size` | `0.05 0.01 0.025` | `0.04 0.005 0.02` | Full boxes are 100 x 20 x 50 mm versus 80 x 10 x 40 mm |
| Jaw-base site offsets | `x=-0.025`, `y=+/-0.0125` | `x=-0.02`, `y=+/-0.01` | Tendon routing/display endpoints differ |
| Gripper actuator range | `0..0.05` m per jaw | `0..0.04` m per jaw | Maximum commanded aperture differs by 20 mm overall |
| Boom/wrist tendon width | `0.04` / `0.02` | `0.03` / `0.01` | Visual thickness differs; no tendon stiffness is specified |
| Jaw-to-world exclusions | None | Both gripper bodies excluded from `world` | Jaw contact with floor/world geometry differs |

MJCF box `size` values are half-extents, which is why the full dimensions in the
table are twice the XML values. Both variants retain gripper joint ranges of
`0..0.05`; only the narrow variant restricts actuator commands to `0..0.04`.
Positive gripper position moves both jaws outward, so zero is the minimum modeled
opening and increasing position opens the gripper.

The 63-degree vertical FOV is consistent with approximately 95 degrees horizontal
FOV at 16:9. Since MuJoCo stores vertical FOV, rendering at another aspect ratio
changes horizontal FOV. Camera resolution and calibration therefore need to be
chosen together.

## Scene differences

### Arm

`GIRAF.xml` is the canonical arm-only entry scene. It contains no free objects,
object meshes, or bin.

### Single banana

`GIRAF_banana.xml` adds one banana at `1.0 0.0 0.25` and a bin centered at
`1.0 0.5 0`. Its free joint is explicitly named `banana_joint`, making direct
state lookup straightforward.

### Ten bananas

`GIRAF_bananas.xml` adds ten bodies named `banana_1` through `banana_10` and the
same bin as the single-banana scene. The ten free joints are unnamed. They should
be named consistently if callers need stable joint-level access.

### YCB collection

`GIRAF_ycb.xml` adds eleven YCB bodies, eleven mesh/texture/material sets, and a bin
centered at `1.0 0.4 0`. The free joints are unnamed. It is the only scene that
loads all 134 MB of current YCB assets.

Object density, mesh scale, friction, and initial pose are embedded directly in
each scene. These values have not yet been separated into reusable object
definitions or validated against a single physical-data source.

## Consolidation decisions

Before merging the robot definitions, the following physical choices need an
authoritative answer:

1. Confirm the updated R4 axis and the 59.7 mm R4-to-R5/R6 offset from CAD.
2. Which jaw dimensions and jaw-site locations match the current GIRAF hardware?
3. Is the usable per-jaw travel 40 mm or 50 mm?
4. Should the camera represent 95-degree horizontal FOV at 1280 x 720? If so,
   `fovy=63` is the consistent current option.
5. Should jaw contact with floor/world geometry be enabled or excluded?
6. Are the tendon widths purely visual, and which display dimensions are desired?

These should be answered from CAD, measured travel, and camera calibration rather
than inferred from whichever task script was edited most recently.

## Target MJCF layout

After those decisions, the model should move toward:

```text
models/
├── robot/
│   ├── giraf.xml
│   └── actuators.xml
├── objects/
│   ├── bin.xml
│   └── banana.xml
├── scenes/
│   ├── arm.xml
│   ├── banana.xml
│   ├── bananas.xml
│   └── ycb.xml
└── ycb/
```

Each scene should include the same canonical robot and then add only its objects
and initial layout. The scene registry can retain the public keys `arm`, `banana`,
`bananas`, and `ycb`, so downstream Python code will not change.

Regression tests should then assert that all scenes have identical robot joint
types, limits, actuator gains/ranges, camera intrinsics, jaw geometry, and robot
body parameters. Separate tests should cover only scene-specific object counts,
names, and initial poses.
