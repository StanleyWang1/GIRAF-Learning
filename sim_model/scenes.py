"""Named MuJoCo scenes shipped with the simulation package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class SceneSpec:
    """Description of an MJCF scene available to the simulator."""

    name: str
    path: Path
    description: str


SceneInput = str | Path | SceneSpec


_SCENES = {
    "arm": SceneSpec(
        name="arm",
        path=MODELS_DIR / "GIRAF.xml",
        description="GIRAF arm in an empty workspace",
    ),
    "banana": SceneSpec(
        name="banana",
        path=MODELS_DIR / "GIRAF_banana.xml",
        description="GIRAF arm, one banana, and a collection bin",
    ),
    "bananas": SceneSpec(
        name="bananas",
        path=MODELS_DIR / "GIRAF_bananas.xml",
        description="GIRAF arm, ten bananas, and a collection bin",
    ),
    "ycb": SceneSpec(
        name="ycb",
        path=MODELS_DIR / "GIRAF_ycb.xml",
        description="GIRAF arm with the current YCB object collection",
    ),
}


def available_scenes() -> tuple[SceneSpec, ...]:
    """Return all built-in scenes in command-line display order."""

    return tuple(_SCENES.values())


def resolve_scene(scene: SceneInput) -> SceneSpec:
    """Resolve a built-in scene name or a custom MJCF path."""

    if isinstance(scene, SceneSpec):
        spec = scene
    elif isinstance(scene, Path):
        path = scene.expanduser().resolve()
        spec = SceneSpec(path.stem, path, "Custom MJCF scene")
    elif scene in _SCENES:
        spec = _SCENES[scene]
    else:
        path = Path(scene).expanduser().resolve()
        spec = SceneSpec(path.stem, path, "Custom MJCF scene")

    if not spec.path.is_file():
        names = ", ".join(_SCENES)
        raise FileNotFoundError(
            f"MuJoCo scene not found: {spec.path}. Built-in scenes: {names}"
        )
    return spec
