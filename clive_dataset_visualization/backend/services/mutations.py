from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Dict, List


EPISODE_DIR_RE = re.compile(r"^ep_(\d+)$")


def parse_episode_dir_indices(videos_root: Path) -> List[int]:
    if not videos_root.exists() or not videos_root.is_dir():
        return []

    indices: List[int] = []
    for child in videos_root.iterdir():
        if not child.is_dir():
            continue
        match = EPISODE_DIR_RE.match(child.name)
        if match is None:
            continue
        indices.append(int(match.group(1)))
    return sorted(indices)


def validate_sidecar_episode_layout(videos_root: Path, expected_episode_count: int) -> None:
    if expected_episode_count < 0:
        raise ValueError(f"invalid episode count: {expected_episode_count}")

    if not videos_root.exists() or not videos_root.is_dir():
        raise ValueError(f"videos root does not exist or is not a directory: {videos_root}")

    actual = parse_episode_dir_indices(videos_root)
    expected = list(range(expected_episode_count))
    if actual != expected:
        raise ValueError(
            "sidecar episode folders are inconsistent: "
            f"expected ep_0..ep_{max(0, expected_episode_count - 1)} exactly, "
            f"found indices={actual}"
        )


def delete_and_compact_sidecar_episode(videos_root: Path, episode_index: int, episode_count_before: int) -> Dict[str, object]:
    if episode_index < 0 or episode_index >= episode_count_before:
        raise ValueError(
            f"episode index {episode_index} out of range [0, {max(0, episode_count_before - 1)}]"
        )

    target_dir = videos_root / f"ep_{episode_index}"
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"expected episode directory not found: {target_dir}")

    # Rename first so compaction can proceed even when directory cleanup is delayed
    # by transient filesystem behavior (e.g., NFS stale/hidden files).
    tombstone = videos_root / f".deleting_ep_{episode_index}_{int(time.time() * 1000)}"
    target_dir.rename(tombstone)
    renamed: List[Dict[str, str]] = []

    for src_idx in range(episode_index + 1, episode_count_before):
        src = videos_root / f"ep_{src_idx}"
        dst = videos_root / f"ep_{src_idx - 1}"

        if not src.exists() or not src.is_dir():
            raise ValueError(f"expected episode directory not found during compaction: {src}")
        if dst.exists():
            raise ValueError(f"destination already exists during compaction: {dst}")

        src.rename(dst)
        renamed.append({"from": src.name, "to": dst.name})

    cleanup_pending = None
    cleanup_exc: Exception | None = None
    for _ in range(6):
        try:
            shutil.rmtree(tombstone)
            cleanup_exc = None
            break
        except Exception as exc:
            cleanup_exc = exc
            time.sleep(0.15)

    if cleanup_exc is not None:
        cleanup_pending = tombstone.name

    return {
        "deleted_dir": target_dir.name,
        "renamed": renamed,
        "cleanup_pending": cleanup_pending,
    }
