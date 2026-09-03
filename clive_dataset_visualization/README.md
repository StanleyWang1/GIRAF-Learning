# Dataset Visualization

Isolated episode debugging web app for ARX zarr datasets.

## Features

- Timeline scrubbing with frame/time jump.
- Primary camera view + multi-stream thumbnails.
- RGB/depth rendering with selectable colormap.
- Key-value graph panel with presets and rolling history window.
- 3D trajectory pane when `robot*_eef_pos` exists.
- Event markers (contact transitions, spikes, action jumps, missing modalities).
- Episode delete action with optional sidecar folder compaction.

## Supported Input Formats (v1)

1. Full extracted/pruned zarr containing in-zarr camera arrays (`camera_*_rgb`, optional `camera_*_depth`).
2. Raw zarr with sidecar media at `<zarr_stem>_videos/ep_X/camera_*_rgb.mp4` and optional `camera_*_depth.bin`.

## Explicitly Unsupported (v1)

- Legacy index-based zarr (`camera_*_indices`).

The API and UI will report an explicit unsupported-format message.

## Install

```bash
pip install -r dataset_visualization/requirements.txt
```

## Run

```bash
python -m dataset_visualization \
  --input /home/ajay/mnt/data_storage/full_screw_sorting_rgb_v11_withleader.zarr \
  --port 8080
```

Raw + sidecar example:

```bash
python -m dataset_visualization \
  --input /home/ajay/mnt/data_storage/screw_sorting_new_grippers_screws_v5.zarr \
  --videos-root /home/ajay/mnt/data_storage/screw_sorting_new_grippers_screws_v5_videos \
  --port 8080
```

Then open `http://127.0.0.1:8080`.

Without `-m`, you can also run:

```bash
python dataset_visualization/main.py --input /path/to/dataset.zarr --port 8080
```

## CLI

```text
python -m dataset_visualization --input <zarr_path> --port 8080 \
  [--videos-root <path>] [--episode <idx>] [--history-frames 180] \
  [--prefetch 60] [--depth-shape H W]
```

## API Endpoints

- `GET /api/dataset/summary`
- `GET /api/episodes`
- `GET /api/episode/{ep}/schema`
- `GET /api/episode/{ep}/timing`
- `GET /api/episode/{ep}/signals?keys=...&start=...&end=...&stride=...`
- `GET /api/episode/{ep}/frame?camera=...&idx=...&modality=rgb|depth&colormap=...`
- `GET /api/episode/{ep}/events`
- `GET /api/episode/{ep}/trajectory3d?idx=...&window=...`
- `DELETE /api/episode/{ep}?delete_videos=true`
- `GET /api/health`

## Notes

- Deleting an episode mutates the zarr dataset on disk (`drop_episode_by_index`).
- For sidecar mode, delete also removes and compacts `ep_*` folders (`ep_{k+1} -> ep_k`).
- Sidecar delete enforces strict consistency before mutation: folders must be exactly `ep_0..ep_{N-1}` with no gaps/extras.
- In strict sidecar mode, `delete_videos` must remain `true` to preserve zarr/video parity.
- If `timestamps` is absent, frame index is used as timeline.
- Playback defaults to a `30 FPS` cap with `Timestamp` mode preferred when reliable timestamps exist.
- Preview tiles run full-speed first, then automatically reduce update rate only when request backlog forms.
