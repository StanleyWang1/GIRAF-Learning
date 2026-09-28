"""Plot successful rollout extension/retraction versus time-averaged d3.

Requires Matplotlib. The x coordinate is the trapezoidal time average of
commanded d3, including the initial and final episode positions. The y
coordinates sum positive and negative commanded position changes separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rollout_boom_travel import episode_stats


def mean_boom_length(path: Path) -> float:
    metadata = json.loads(path.read_text())
    samples = [(metadata['started_monotonic_ns'], metadata['initial_joints'][2])]
    with (path.parent / metadata['data']).open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['event'] == 'control':
                samples.append((record['monotonic_ns'], record['joint_position'][2]))
    samples.append((metadata['ended_monotonic_ns'], metadata['final_joints'][2]))
    duration = samples[-1][0] - samples[0][0]
    if duration <= 0:
        raise ValueError(f'Nonpositive episode duration: {path}')
    if any(b[0] < a[0] for a, b in zip(samples, samples[1:])):
        raise ValueError(f'Unordered timestamps: {path}')
    mean = math.fsum(
        (b[0] - a[0]) / duration * (a[1] + b[1]) / 2
        for a, b in zip(samples, samples[1:])
    )
    assert min(p for _, p in samples) <= mean <= max(p for _, p in samples)
    return mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset', nargs='?', type=Path,
        default=Path('deployment_runs/ring_over_peg_joint_angle_ROLLOUTS'),
    )
    args = parser.parse_args()
    rows = []
    for length, expected in ((40, 19), (60, 17), (80, 14)):
        paths = sorted((args.dataset / f'{length}in/episodes').glob('*/episode.json'))
        selected = [p for p in paths if not p.parent.name.endswith('_F')
                    and not p.parent.name.startswith(('X_', '2m_test'))]
        if len(selected) != expected:
            raise ValueError(f'Expected {expected} successes at {length}in')
        for path in selected:
            rows.append({
                'boom_length_group_in': length,
                'mean_commanded_d3_m': mean_boom_length(path),
                **episode_stats(path),
            })

    output = args.dataset / 'analysis/boom_travel'
    output.mkdir(parents=True, exist_ok=True)
    stem = output / 'extension_retraction_vs_mean_boom_length'
    with stem.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 13,
        'axes.labelsize': 14, 'axes.linewidth': 1.3,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'legend.fontsize': 12, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(9, 6.4))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.20, top=0.81)
    fig.text(0.12, 0.93, 'Boom extension and retraction per rollout',
             fontsize=18, weight='bold', ha='left')
    fig.text(0.12, 0.875,
             'Successful episodes only  |  40 in: n=19   ·   60 in: n=17   ·   80 in: n=14',
             fontsize=12, color='#505866', ha='left')
    x = [row['mean_commanded_d3_m'] for row in rows]
    for field, label, color in (
        ('extension_travel_m', 'Total extension', '#1976D2'),
        ('retraction_travel_m', 'Total retraction', '#EF8A17'),
    ):
        ax.scatter(x, [row[field] for row in rows], label=label,
                   color=color, s=64, alpha=0.82,
                   edgecolors='white', linewidths=0.7, zorder=3)
    ax.set_xlabel('Average commanded boom length, d3 (m)', labelpad=12)
    ax.set_ylabel('Total commanded travel (m)', labelpad=10)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.07, y=0.10)
    ax.set_axisbelow(True)
    ax.grid(axis='both', color='#DDE2E8', linewidth=0.9, alpha=0.8)
    for name in ('top', 'right'):
        ax.spines[name].set_visible(False)
    for name in ('left', 'bottom'):
        ax.spines[name].set_color('#566170')
    ax.tick_params(colors='#354050')
    ax.legend(loc='upper left', frameon=True, facecolor='white',
              edgecolor='#DDE2E8', framealpha=0.97, borderpad=0.8)
    fig.text(0.12, 0.06,
             'Each episode contributes one dot per direction. '
             'Boom length is time-averaged over the episode.',
             fontsize=11, color='#505866', ha='left')
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(stem.with_suffix(suffix), dpi=300, facecolor='white')
    plt.close(fig)
    print(f'Saved PNG, PDF, SVG, and source CSV: {stem}')
    print(f'{len(rows)} successful episodes; {2 * len(rows)} scatter points.')


if __name__ == '__main__':
    main()
