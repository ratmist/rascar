"""RASCAR: assign detected tabletop objects to one physical table per frame.

Modes:
  strict  - object box center inside the original table rectangle
  margin  - object center inside a table rectangle enlarged by a fraction of width/height
  overlap - fraction of object's box intersecting the table >= minimum threshold

The algorithm does not infer whether a box is physically on a tabletop: perspective
and sofa/chair placement require separate visual verification. It never uses people.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

COORDS = ['x1', 'y1', 'x2', 'y2']
ITEM_CLASSES = {'phone', 'wallet', 'headphones', 'handbag', 'backpack', 'personal_items', 'paper', 'papers'}
CUP_CLASSES = {'cup'}
DISH_CLASSES = {'empty_plate', 'full_plate', 'plate', 'bowl', 'tray'}
ALLOWED_CLASSES = ITEM_CLASSES | CUP_CLASSES | DISH_CLASSES


def load_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Cannot find {path}')
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    df = df.copy()
    for col in COORDS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    valid = df[COORDS].notna().all(axis=1) & (df.x2 > df.x1) & (df.y2 > df.y1)
    if (~valid).any():
        print(f'Warning: removed {(~valid).sum()} invalid boxes from {path.name}')
    return df.loc[valid].reset_index(drop=True)


def calculate_pair(obj, table, mode: str, margin: float, min_overlap: float):
    ox = (obj.x1 + obj.x2) / 2
    oy = (obj.y1 + obj.y2) / 2
    tx = (table.x1 + table.x2) / 2
    ty = (table.y1 + table.y2) / 2
    dx, dy = abs(ox - tx), abs(oy - ty)
    center_distance = float(np.hypot(dx, dy))

    left = max(float(obj.x1), float(table.x1))
    top = max(float(obj.y1), float(table.y1))
    right = min(float(obj.x2), float(table.x2))
    bottom = min(float(obj.y2), float(table.y2))
    intersection = max(0., right-left) * max(0., bottom-top)
    obj_area = float((obj.x2-obj.x1)*(obj.y2-obj.y1))
    fraction = intersection / obj_area

    if mode == 'strict':
        eligible = (table.x1 <= ox <= table.x2 and table.y1 <= oy <= table.y2)
        # Prioritize box share actually on the tabletop; center distance breaks ties.
        priority = (-fraction, center_distance)
    elif mode == 'margin':
        pad_x = margin * float(table.x2-table.x1)
        pad_y = margin * float(table.y2-table.y1)
        eligible = (table.x1-pad_x <= ox <= table.x2+pad_x and
                    table.y1-pad_y <= oy <= table.y2+pad_y)
        # Prefer original bbox over merely extended region.
        outside_dx = max(float(table.x1)-ox, 0., ox-float(table.x2))
        outside_dy = max(float(table.y1)-oy, 0., oy-float(table.y2))
        dist_to_box = float(np.hypot(outside_dx, outside_dy))
        priority = (dist_to_box, -fraction, center_distance)
    else:
        eligible = fraction >= min_overlap
        priority = (-fraction, center_distance)
    return eligible, priority, center_distance, fraction


def assign_objects(tables: pd.DataFrame, detections: pd.DataFrame,
                   mode: str, margin: float, min_overlap: float) -> pd.DataFrame:
    lookup = {frame: group for frame, group in tables.groupby('frame_id', sort=False)}
    result = []
    for object_idx, obj in enumerate(detections.itertuples(index=False)):
        candidates = []
        for table in lookup.get(obj.frame_id, ()).itertuples(index=False) if obj.frame_id in lookup else ():
            eligible, priority, distance, overlap = calculate_pair(obj, table, mode, margin, min_overlap)
            if eligible:
                # Deterministic tie-breakers independent of pandas iteration order.
                candidates.append((priority, str(table.row_id), table, distance, overlap))
        data = obj._asdict()
        data['object_index'] = object_idx
        data['mode'] = mode
        if candidates:
            _, _, best_table, dist, overlap = min(candidates, key=lambda x: (x[0], x[1]))
            data.update(row_id=best_table.row_id, table_id=best_table.table_id,
                        distance_px=dist, overlap_obj_frac=overlap,
                        assigned=True, reason='matched')
        else:
            data.update(row_id='', table_id='', distance_px=np.nan,
                        overlap_obj_frac=0., assigned=False,
                        reason='no_matching_table' if obj.frame_id in lookup else 'frame_without_tables')
        result.append(data)
    return pd.DataFrame(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tables', default='data/tables.csv')
    ap.add_argument('--detections', default='vision_items_candidates.csv')
    ap.add_argument('--out', default='experiments/objects')
    ap.add_argument('--mode', choices=['strict', 'margin', 'overlap', 'all'], default='all')
    ap.add_argument('--margin', type=float, default=0.07, help='Fraction of table width/height per side')
    ap.add_argument('--min-overlap', type=float, default=0.25, help='Minimum fraction of object bbox inside table')
    ap.add_argument('--classes', default='items', choices=['items', 'cups', 'dishes', 'all'], help='Relevant object classes only')
    ap.add_argument('--frame', default=None, help='Process one frame_id only')
    args = ap.parse_args()
    if not 0 <= args.margin <= 1:
        ap.error('--margin must be between 0 and 1')
    if not 0 <= args.min_overlap <= 1:
        ap.error('--min-overlap must be between 0 and 1')

    tables = load_csv(Path(args.tables), {'frame_id', 'row_id', 'table_id', *COORDS})
    objects = load_csv(Path(args.detections), {'frame_id', 'class_name', *COORDS})
    selected = {'items': ITEM_CLASSES, 'cups': CUP_CLASSES, 'dishes': DISH_CLASSES, 'all': ALLOWED_CLASSES}[args.classes]
    objects = objects[objects.class_name.isin(selected)].copy().reset_index(drop=True)
    if args.frame:
        tables = tables[tables.frame_id == args.frame]
        objects = objects[objects.frame_id == args.frame].reset_index(drop=True)
    if not len(objects):
        print('No selected detections found. Check --classes and --detections.')
        return
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = ['strict', 'margin', 'overlap'] if args.mode == 'all' else [args.mode]
    for mode in modes:
        result = assign_objects(tables, objects, mode, args.margin, args.min_overlap)
        result.to_csv(output_dir / f'{mode}_assignments.csv', index=False)
        result.loc[~result.assigned].to_csv(output_dir / f'{mode}_rejected.csv', index=False)
        print(f'{mode:8s} assigned={int(result.assigned.sum())}/{len(result)} rejected={int((~result.assigned).sum())}')
    if len(modes) == 3:
        output = objects[['frame_id', 'class_name']].copy()
        output.insert(0, 'object_index', range(len(output)))
        for mode in modes:
            r = pd.read_csv(output_dir / f'{mode}_assignments.csv')
            output[f'{mode}_row_id'] = r.row_id
        output.to_csv(output_dir / 'comparison.csv', index=False)
        print('Saved comparison.csv')
    print('Output:', output_dir.resolve())


if __name__ == '__main__':
    main()
