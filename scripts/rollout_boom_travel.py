"""Summarize commanded d3 travel from a combined rollout dataset.

Uses the initial position, every control.joint_position[2], and final position.
Total travel is sum(abs(diff(d3))) in meters, including extension and retraction.
These are commanded positions, not encoder measurements. Standard deviations
are sample standard deviations across episodes (ddof=1).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def episode_stats(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    if not metadata['complete']:
        raise ValueError(f'Incomplete episode: {path}')
    positions = [float(metadata['initial_joints'][2])]
    previous_time = metadata['started_monotonic_ns']
    count = 0
    with (path.parent / metadata['data']).open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['event'] != 'control':
                continue
            if (
                record['episode'] != metadata['episode']
                or record['source'] != 'policy'
                or not record['active']
                or record['command_accepted'] is not True
            ):
                raise ValueError(f'Unexpected control record in {path}')
            timestamp = record['monotonic_ns']
            if not previous_time <= timestamp <= metadata['ended_monotonic_ns']:
                raise ValueError(f'Out-of-order or out-of-episode control in {path}')
            previous_time = timestamp
            positions.append(float(record['joint_position'][2]))
            count += 1
    if count == 0:
        raise ValueError(f'No control samples in {path}')
    positions.append(float(metadata['final_joints'][2]))
    if not all(math.isfinite(value) for value in positions):
        raise ValueError(f'Nonfinite d3 in {path}')
    differences = [b - a for a, b in zip(positions, positions[1:])]
    extension = math.fsum(max(delta, 0.0) for delta in differences)
    retraction = math.fsum(max(-delta, 0.0) for delta in differences)
    total = math.fsum(abs(delta) for delta in differences)
    net = positions[-1] - positions[0]
    assert math.isclose(total, extension + retraction, abs_tol=1e-12)
    assert math.isclose(net, extension - retraction, abs_tol=1e-12)
    assert total + 1e-12 >= abs(net)
    return {
        'episode': path.parent.name,
        'outcome': 'failure' if path.parent.name.endswith('_F') else 'success',
        'control_samples': count,
        'duration_s': metadata['duration_seconds'],
        'initial_d3_m': positions[0],
        'final_d3_m': positions[-1],
        'total_travel_m': total,
        'extension_travel_m': extension,
        'retraction_travel_m': retraction,
        'net_displacement_m': net,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset', type=Path, nargs='?',
        default=Path('deployment_runs/ring_over_peg_joint_angle_ROLLOUTS'),
    )
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    rows = []
    summary = {
        'metric': 'Commanded boom joint travel (meters), not measured travel',
        'source_field': 'control.joint_position[2]',
        'method': 'Sum absolute consecutive position differences per episode, '
                  'including initial and final metadata positions; no smoothing.',
        'std_convention': 'Sample standard deviation across episodes (ddof=1)',
        'selection': 'All 20 valid episodes per length, including _F failures; '
                     'exclude X_ and 2m_test prefixes.',
        'lengths': {},
    }
    for length in (40, 60, 80):
        paths = sorted((args.dataset / f'{length}in' / 'episodes').glob('*/episode.json'))
        paths = [p for p in paths if not p.parent.name.startswith(('X_', '2m_test'))]
        if len(paths) != 20:
            raise ValueError(f'Expected 20 episodes for {length}in, found {len(paths)}')
        group = [dict(boom_length_in=length, **episode_stats(path)) for path in paths]
        rows.extend(group)
        result = {'n': len(group)}
        for metric in ('total_travel_m', 'extension_travel_m',
                       'retraction_travel_m', 'net_displacement_m'):
            values = [row[metric] for row in group]
            result[metric] = {
                'mean': statistics.mean(values), 'std': statistics.stdev(values),
            }
        summary['lengths'][f'{length}in'] = result
        travel = result['total_travel_m']
        print(f"{length}in: {travel['mean']:.6f} +/- {travel['std']:.6f} m (n=20)")
    output = args.output_dir or args.dataset / 'analysis' / 'boom_travel'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_episode.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
