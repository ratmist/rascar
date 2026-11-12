"""Validate the participant package; --decode also reads every JPEG."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from score import COUNTS, validate_solution, validate_submission


def check(data, decode=False):
    data = Path(data).resolve()
    frames = pd.read_csv(data / 'frames.csv')
    targets = pd.read_csv(data / 'targets.csv').fillna('')
    tables = pd.read_csv(data / 'tables.csv')
    det = pd.read_csv(data / 'detections.csv')
    labels = pd.read_csv(data / 'train_labels.csv')
    sample = pd.read_csv(data / 'sample_submission.csv')
    for title, table, key in [('frames', frames, 'frame_id'), ('targets', targets, 'frame_id'),
                              ('tables', tables, 'row_id'), ('detections', det, 'detection_id'),
                              ('train_labels', labels, 'row_id')]:
        if table[key].isna().any() or table[key].duplicated().any():
            raise ValueError(f'{title}: missing/duplicate {key}')
    if not set(targets.split) <= {'train', 'test'}:
        raise ValueError('Unknown target split.')
    if set(targets.frame_id) != set(tables.frame_id):
        raise ValueError('Targets and table frames do not match.')
    if not set(targets.frame_id) <= set(frames.frame_id) or not set(det.frame_id) <= set(frames.frame_id):
        raise ValueError('A referenced frame is missing.')
    train = set(targets.loc[targets.split == 'train', 'frame_id'])
    test = set(targets.loc[targets.split == 'test', 'frame_id'])
    if train & test:
        raise ValueError('Train/test overlap.')
    expected_train = set(tables.loc[tables.frame_id.isin(train), 'row_id'])
    expected_test = set(tables.loc[tables.frame_id.isin(test), 'row_id'])
    if set(labels.row_id) != expected_train:
        raise ValueError('Training labels do not cover every train frame-table row exactly.')
    validate_solution(labels)
    if (labels[COUNTS] % 1 != 0).any().any():
        raise ValueError('Gold training counters must be integers.')
    validate_submission(sample, expected_test)
    coords = ['x1', 'y1', 'x2', 'y2']
    for title, table in [('tables', tables), ('detections', det)]:
        a = table[coords].to_numpy(float)
        if not np.isfinite(a).all() or (a[:, 0] < 0).any() or (a[:, 1] < 0).any() or (a[:, 2] > 1280).any() or (a[:, 3] > 720).any() or (a[:, 2] <= a[:, 0]).any() or (a[:, 3] <= a[:, 1]).any():
            raise ValueError(f'Invalid {title} geometry.')
    if not det.confidence.between(0, 1).all():
        raise ValueError('Invalid detector confidence.')
    merged = labels.merge(tables[['row_id', 'frame_id', 'table_id']], on='row_id',
                          validate='one_to_one', suffixes=('', '_table'))
    if not (merged.frame_id == merged.frame_id_table).all() or not (merged.table_id == merged.table_id_table).all():
        raise ValueError('Label frame/table identity mismatch.')
    lookup = frames.set_index('frame_id')
    links = 0
    for r in targets.itertuples():
        ns = [n for n in r.context_frames.split(';') if n]
        if len(ns) != len(set(ns)) or r.frame_id in ns or len(ns) > 2:
            raise ValueError('Invalid context list.')
        for n in ns:
            if n not in lookup.index:
                raise ValueError('Missing context image.')
            delta = abs((pd.Timestamp(lookup.loc[n, 'timestamp']) - pd.Timestamp(lookup.loc[r.frame_id, 'timestamp'])).total_seconds())
            if abs(delta - 30) > 2:
                raise ValueError('A context link is not 30 seconds apart.')
            links += 1
    decoded = 0
    hashes = {}
    aux = pd.read_csv(data / 'aux_frames.csv')
    boxes = pd.read_csv(data / 'aux_boxes.csv')
    if aux.frame_id.duplicated().any() or not set(boxes.frame_id) <= set(aux.frame_id):
        raise ValueError('Invalid auxiliary frame keys.')
    if not (pd.to_datetime(aux.timestamp, utc=True) < pd.Timestamp('2026-10-25', tz='UTC')).all():
        raise ValueError('Auxiliary manual boxes must precede the October competition days.')
    for r in pd.concat([frames[['frame_id', 'image_path']], aux[['frame_id', 'image_path']]]).itertuples():
        p = (data / r.image_path).resolve()
        if not p.is_relative_to(data) or not p.is_file():
            raise ValueError(f'Missing/unsafe image path: {r.image_path}')
        if decode:
            import hashlib
            from PIL import Image
            with Image.open(p) as im:
                if im.size != (1280, 720):
                    raise ValueError('Unexpected JPEG dimensions.')
                digest = hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
                if digest in hashes:
                    raise ValueError('Duplicate image pixels: ' + r.frame_id + ' / ' + hashes[digest])
                hashes[digest] = r.frame_id
            decoded += 1
    return dict(train_frames=len(train), train_rows=len(labels), test_frames=len(test),
                test_rows=len(sample), images=len(frames), context_links=links,
                auxiliary_images=len(aux), auxiliary_boxes=len(boxes), decoded=decoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--decode', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.data, args.decode), indent=2))
