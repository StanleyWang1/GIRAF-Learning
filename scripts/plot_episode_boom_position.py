"""Save commanded d3 versus elapsed episode time in all 60 rollout folders.

Run with Matplotlib available, for example using the temporary dependencies:
PYTHONPATH=/tmp/giraf-rollout-plot-deps MPLCONFIGDIR=/tmp/giraf-rollout-matplotlib \
    .venv/bin/python scripts/plot_episode_boom_position.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from rollout_boom_travel import episode_stats


def load_trace(path: Path) -> tuple[list[float], list[float]]:
    metadata = json.loads(path.read_text())
    start = metadata['started_monotonic_ns']
    times = [0.0]
    positions = [metadata['initial_joints'][2]]
    with (path.parent / metadata['data']).open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['event'] != 'control':
                continue
            if not (record['source'] == 'policy' and record['active']
                    and record['clutch']):
                raise ValueError(f'Control sample outside clutch-held policy: {path}')
            times.append((record['monotonic_ns'] - start) / 1e9)
            positions.append(record['joint_position'][2])
    times.append((metadata['ended_monotonic_ns'] - start) / 1e9)
    positions.append(metadata['final_joints'][2])
    if metadata['reason'] != 'clutch released':
        raise ValueError(f'Unexpected end reason: {path}')
    if not all(math.isfinite(p) for p in positions):
        raise ValueError(f'Nonfinite position: {path}')
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError(f'Unordered timestamps: {path}')
    if not math.isclose(times[-1], metadata['duration_seconds'], abs_tol=1e-9):
        raise ValueError(f'Duration mismatch: {path}')
    return times, positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset', nargs='?', type=Path,
        default=Path('deployment_runs/ring_over_peg_joint_angle_ROLLOUTS'),
    )
    args = parser.parse_args()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 13,
        'axes.labelsize': 14, 'axes.linewidth': 1.3,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size': 5, 'ytick.major.size': 5,
    })
    report = {
        'source': 'control.joint_position[2] (commanded meters, not feedback)',
        'time': 'Seconds since policy activation, using monotonic timestamps',
        'selection': 'All 60 valid episodes, including _F failures',
        'boundaries': 'Initial activation and final clutch-release positions included',
        'style': 'Unsmoothed trace; common y limits within each length group',
        'line_colors': 'Blue: increasing d3; orange: decreasing d3; '
                       'gray: exactly unchanged d3. No threshold or smoothing.',
        'episodes': [],
    }
    for length in (40, 60, 80):
        paths = sorted((args.dataset / f'{length}in/episodes').glob('*/episode.json'))
        if len(paths) != 20:
            raise ValueError(f'Expected 20 episodes for {length}in')
        traces = [(path, *load_trace(path), episode_stats(path)) for path in paths]
        lower = min(min(positions) for _, _, positions, _ in traces)
        upper = max(max(positions) for _, _, positions, _ in traces)
        padding = max((upper - lower) * 0.10, 0.01)
        for path, times, positions, stats in traces:
            failed = path.parent.name.endswith('_F')
            fig, ax = plt.subplots(figsize=(9, 5.8))
            fig.subplots_adjust(left=0.13, right=0.96, bottom=0.19, top=0.78)
            fig.text(0.13, 0.93, f'{length} in boom  |  {path.parent.name}',
                     fontsize=18, weight='bold', ha='left')
            status = 'Valid failure' if failed else 'Success'
            fig.text(0.13, 0.865,
                     f'{status}   ·   Duration: {times[-1]:.2f} s   ·   '
                     f"Extension: {stats['extension_travel_m']:.3f} m   ·   "
                     f"Retraction: {stats['retraction_travel_m']:.3f} m",
                     fontsize=12, color='#505866', ha='left')
            points = list(zip(times, positions))
            segments = list(zip(points, points[1:]))
            colors = [
                '#1976D2' if end[1] > start[1] else
                '#EF8A17' if end[1] < start[1] else '#88929E'
                for start, end in segments
            ]
            ax.add_collection(LineCollection(
                segments, colors=colors, linewidths=2.4, capstyle='round',
            ))
            ax.scatter([times[0], times[-1]], [positions[0], positions[-1]],
                       s=45, color=[colors[0], colors[-1]],
                       edgecolors='white', linewidths=0.8,
                       clip_on=False, zorder=4)
            ax.legend(handles=[
                Line2D([0], [0], color=color, linewidth=2.8, label=label)
                for label, color in (
                    ('Extension', '#1976D2'), ('Retraction', '#EF8A17'),
                    ('Hold', '#88929E'),
                )
            ], loc='best', fontsize=11, framealpha=0.95, edgecolor='#DDE2E8')
            ax.set_xlim(0, times[-1])
            ax.set_ylim(lower - padding, upper + padding)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.set_xlabel('Episode time (s)', labelpad=10)
            ax.set_ylabel('Commanded boom position, d3 (m)', labelpad=10)
            ax.set_axisbelow(True)
            ax.grid(color='#DDE2E8', linewidth=0.9)
            for name in ('top', 'right'):
                ax.spines[name].set_visible(False)
            for name in ('left', 'bottom'):
                ax.spines[name].set_color('#566170')
            ax.tick_params(colors='#354050')
            fig.text(0.13, 0.055,
                     'Policy activation → clutch release. '
                     f'Unsmoothed commands; shared y scale for {length} in episodes.',
                     fontsize=11, color='#505866')
            output = path.parent / 'd3_vs_episode_time.png'
            fig.savefig(output, dpi=220, facecolor='white')
            plt.close(fig)
            report['episodes'].append({
                'plot': str(output.relative_to(args.dataset)),
                'control_samples': stats['control_samples'],
                'duration_s': times[-1],
                'y_limits_m': [lower - padding, upper + padding],
            })
        print(f'{length}in: saved 20 episode plots', flush=True)
    report_path = args.dataset / 'analysis/boom_travel/episode_plot_manifest.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved all 60 plots; manifest: {report_path}', flush=True)


if __name__ == '__main__':
    main()
