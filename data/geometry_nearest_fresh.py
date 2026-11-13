"""Nearest-center assignment of detections to target tables.

Example:
    python scripts/geometry_nearest.py --data data --out experiments/nearest/assignments.csv

Every detection is associated with at most one table *within its frame*.
No confidence threshold or maximum distance is applied in this first experiment.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def assign_nearest(tables: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    """Return detection rows with matching table_id, row_id and distance_px."""
    needed = {'frame_id', 'x1', 'y1', 'x2', 'y2'}
    if not needed.issubset(tables.columns) or not needed.issubset(detections.columns):
        raise ValueError(f'Both inputs must have columns {sorted(needed)}')
    if not {'table_id', 'row_id'}.issubset(tables.columns):
        raise ValueError('tables must include table_id and row_id')

    parts = []
    grouped_tables = {fid: group.sort_values(['row_id'], kind='stable')
                      for fid, group in tables.groupby('frame_id', sort=False)}
    for fid, det in detections.groupby('frame_id', sort=False):
        t = grouped_tables.get(fid)
        if t is None or t.empty:
            continue  # No target table in this frame.
        txy = t[['x1','y1','x2','y2']].to_numpy(float)
        dxy = det[['x1','y1','x2','y2']].to_numpy(float)
        tc = (txy[:, :2] + txy[:, 2:]) / 2
        dc = (dxy[:, :2] + dxy[:, 2:]) / 2
        dist_sq = ((dc[:, None, :] - tc[None, :, :]) ** 2).sum(axis=2)
        nearest = dist_sq.argmin(axis=1)
        part = det.copy()
        part['table_id'] = t['table_id'].to_numpy()[nearest]
        part['row_id'] = t['row_id'].to_numpy()[nearest]
        part['distance_px'] = np.sqrt(dist_sq[np.arange(len(det)), nearest])
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=[*detections.columns, 'table_id', 'row_id', 'distance_px'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--tables', type=Path)
    parser.add_argument('--detections', type=Path)
    parser.add_argument('--out', type=Path, default=Path('experiments/nearest/assignments.csv'))
    args = parser.parse_args()
    tables = pd.read_csv(args.tables or args.data / 'tables.csv')
    detections = pd.read_csv(args.detections or args.data / 'detections.csv')
    result = assign_nearest(tables, detections)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f'Assigned {len(result)} detections to tables; saved {args.out}')


if __name__ == '__main__':
    main()
