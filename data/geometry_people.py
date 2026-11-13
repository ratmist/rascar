"""RASCAR Geometry v2 / stage 1: assign detected people to target tables.

Run from repository root:
  python scripts/geometry_people.py --data data --out experiments/people
  python scripts/geometry_people.py --data data --out experiments/people --frame im_3babbd611502bb2e51b5

The anchor is the lower-centre of a person's bounding box (not box centre).
A candidate table is ranked by normalized distance of the anchor to its
rectangle; overlap fraction provides a secondary tie-break. Distant persons
are rejected as unmatched rather than forcing every passerby onto a table.
No test labels are read.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def overlap_ratio(person, table):
    x1 = max(person.x1, table.x1)
    y1 = max(person.y1, table.y1)
    x2 = min(person.x2, table.x2)
    y2 = min(person.y2, table.y2)
    inter = max(0., x2-x1) * max(0., y2-y1)
    area = max(1., (table.x2-table.x1)*(table.y2-table.y1))
    return inter/area


def assign_people(detections, tables, max_normalized_distance=0.65):
    """Return a row for every person detection, assigned or not."""
    people = detections[detections.class_name.eq('person')].copy()
    results = []
    for p in people.itertuples(index=False):
        px, py = (float(p.x1)+float(p.x2))/2, float(p.y2)
        candidates = []
        for t in tables.itertuples(index=False):
            # Distance to TABLE RECTANGLE; zero if anchor is within it.
            dx = max(float(t.x1)-px, 0., px-float(t.x2))
            dy = max(float(t.y1)-py, 0., py-float(t.y2))
            dist = float(np.hypot(dx, dy))
            diagonal = float(np.hypot(float(t.x2)-float(t.x1), float(t.y2)-float(t.y1)))
            norm = dist / max(diagonal, 1.)
            overlap = overlap_ratio(p, t)
            # Prefer proximity; only use overlap to disambiguate equal distances.
            candidates.append((norm, -overlap, str(t.table_id), str(t.row_id), dist, overlap))
        candidates.sort()
        best = candidates[0] if candidates else None
        accepted = best is not None and best[0] <= max_normalized_distance
        results.append({
            'frame_id': p.frame_id,
            'detection_id': p.detection_id,
            'person_confidence': float(p.confidence),
            'person_x1': float(p.x1), 'person_y1': float(p.y1),
            'person_x2': float(p.x2), 'person_y2': float(p.y2),
            'anchor_x': px, 'anchor_y': py,
            'table_id': best[2] if accepted else '',
            'row_id': best[3] if accepted else '',
            'distance_px': best[4] if best else np.nan,
            'distance_norm': best[0] if best else np.nan,
            'table_overlap_ratio': best[5] if best else np.nan,
            'status': 'assigned' if accepted else 'unmatched'
        })
    return pd.DataFrame(results, columns=[
        'frame_id','detection_id','person_confidence',
        'person_x1','person_y1','person_x2','person_y2',
        'anchor_x','anchor_y','table_id','row_id','distance_px',
        'distance_norm','table_overlap_ratio','status'])


def main():
    parser = argparse.ArgumentParser(description='RASCAR person-to-table geometry diagnostics')
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('experiments/people'))
    parser.add_argument('--frame', help='Optional frame_id for a targeted diagnostic')
    parser.add_argument('--max-distance', type=float, default=0.65,
                        help='Max distance from person bottom-center to table box, divided by table diagonal')
    parser.add_argument('--min-confidence', type=float, default=0.0)
    args = parser.parse_args()
    if args.max_distance < 0 or not 0 <= args.min_confidence <= 1:
        parser.error('Invalid max-distance or min-confidence')
    tables = pd.read_csv(args.data/'tables.csv')
    det = pd.read_csv(args.data/'detections.csv')
    if args.frame:
        tables = tables[tables.frame_id.eq(args.frame)].copy()
        det = det[det.frame_id.eq(args.frame)].copy()
        if tables.empty:
            parser.error(f'No target tables for frame: {args.frame}')
    det = det[det.class_name.eq('person') & (det.confidence >= args.min_confidence)].copy()
    # Ignore detections from unrelated context frames in this stage.
    det = det[det.frame_id.isin(tables.frame_id)]
    grouped = {frame: group for frame, group in tables.groupby('frame_id', sort=False)}
    outputs = []
    for frame, people in det.groupby('frame_id', sort=True):
        outputs.append(assign_people(people, grouped[frame], args.max_distance))
    result = pd.concat(outputs, ignore_index=True) if outputs else assign_people(det, tables.iloc[:0])
    counts = tables[['row_id','frame_id','table_id']].copy()
    valid = result[result.status.eq('assigned')]
    amount = valid.groupby('row_id').size().rename('people_geometry')
    counts = counts.join(amount, on='row_id').fillna({'people_geometry':0})
    counts['people_geometry'] = counts.people_geometry.astype(int)
    args.out.mkdir(parents=True, exist_ok=True)
    path1 = args.out/'person_assignments.csv'
    path2 = args.out/'person_table_counts.csv'
    result.to_csv(path1, index=False)
    counts.to_csv(path2, index=False)
    print(f'Frames with target tables: {tables.frame_id.nunique()}')
    print(f'Person detections: {len(result)}; assigned: {len(valid)}; unmatched: {len(result)-len(valid)}')
    print('Saved:', path1)
    print('Saved:', path2)
    if args.frame:
        print('\nAssignments for selected frame:')
        print(result[['detection_id','anchor_x','anchor_y','table_id','distance_px','status']].to_string(index=False))


if __name__ == '__main__':
    main()
