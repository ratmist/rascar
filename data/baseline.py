"""CPU starters. Reads only the participant package, never organizer answers."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from score import COUNTS, validate_submission, breakdown

CLASSES = ['person', 'nude_hands', 'cup', 'tray']


def assign_detections(detections, tables):
    """Assign a detection to at most one target table. Coordinate units: pixels."""
    counts = np.zeros((len(tables), 4), dtype=float)
    if tables.empty or detections.empty:
        return counts
    d = detections[detections.class_name.isin(CLASSES)]
    if d.empty:
        return counts
    b = tables[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
    v = d[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
    inter = np.maximum(0, np.minimum(v[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(v[:, None, :2], b[None, :, :2])).prod(axis=-1)
    area = np.maximum((v[:, 2:] - v[:, :2]).prod(axis=1), 1)
    table_area = np.maximum((b[:, 2:] - b[:, :2]).prod(axis=1), 1)
    share = inter / area[:, None]
    person = d.class_name.to_numpy() == 'person'
    share[person] = inter[person] / table_area[None, :]
    owner = share.argmax(axis=1)
    keep = share.max(axis=1) >= np.where(person, .15, .5)
    for j, cls in enumerate(CLASSES):
        np.add.at(counts[:, j], owner[keep & (d.class_name.to_numpy() == cls)], 1)
    return counts


def make_features(data):
    data = Path(data)
    frames = pd.read_csv(data / 'frames.csv').set_index('frame_id')
    tables = pd.read_csv(data / 'tables.csv')
    targets = pd.read_csv(data / 'targets.csv').fillna('').set_index('frame_id')
    det = pd.read_csv(data / 'detections.csv')
    grouped = dict(tuple(det.groupby('frame_id', sort=False)))
    empty = det.iloc[:0]
    rows = []
    for frame_id, t in tables.groupby('frame_id', sort=False):
        # Deterministic tie-break for objects overlapping two tables equally.
        t = t.sort_values('table_id', kind='stable')
        current = assign_detections(grouped.get(frame_id, empty), t)
        neighbours = [n for n in targets.loc[frame_id, 'context_frames'].split(';') if n]
        context = np.stack([assign_detections(grouped.get(n, empty), t) for n in neighbours]) if neighbours else current[None, :, :]
        mean, maximum = context.mean(axis=0), context.max(axis=0)
        stamp = pd.Timestamp(frames.loc[frame_id, 'timestamp'])
        hour = (stamp.hour + stamp.minute / 60 + 3) % 24
        for j, r in enumerate(t.itertuples()):
            row = dict(row_id=r.row_id, frame_id=frame_id,
                       date_utc=stamp.strftime('%Y-%m-%d'), hour=hour,
                       x1=r.x1 / 1280, y1=r.y1 / 720, x2=r.x2 / 1280, y2=r.y2 / 720,
                       n_tables=len(t))
            for k, c in enumerate(COUNTS[:4]):
                row['det_' + c] = current[j, k]
                row['context_' + c] = mean[j, k]
                row['context_max_' + c] = maximum[j, k]
            rows.append(row)
    return pd.DataFrame(rows).set_index('row_id')


def detector_prediction(features):
    pred = features[['det_' + c for c in COUNTS[:4]]].copy()
    pred.columns = COUNTS[:4]
    pred['items'] = 0.0
    pred['occ_prob'] = (pred.people > 0).astype(float)
    return pred


def calibrated_prediction(features, labels, predict_ids):
    """Fit count scales with L1 loss; learn items and occupancy from train only."""
    labels = labels.set_index('row_id')
    ids = labels.index
    result = detector_prediction(features.loc[predict_ids])
    scales = {}
    for c in COUNTS[:4]:
        raw = features.loc[ids, 'det_' + c].to_numpy(float)
        y = labels[c].to_numpy(float)
        grid = np.unique(np.r_[np.linspace(0, 3, 61), 1.0])
        # Stable tie-break: prefer the scale closest to 1.
        scale = min(grid, key=lambda s: (float(np.abs(y - s * raw).sum()), abs(s - 1)))
        scales[c] = float(scale)
        result[c] *= scale
    columns = [c for c in features if c not in ['frame_id', 'date_utc']]
    x = features.loc[ids, columns].to_numpy(float)
    xp = features.loc[predict_ids, columns].to_numpy(float)
    model = ExtraTreesRegressor(n_estimators=120, criterion='absolute_error',
                               min_samples_leaf=3, random_state=419, n_jobs=2)
    model.fit(x, labels['items'])
    result['items'] = model.predict(xp)
    # A median-oriented starter: very small estimates default to zero.
    result.loc[result['items'] < .35, 'items'] = 0.0
    if labels.occupied.nunique() > 1:
        occupancy = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=3,
                                        class_weight='balanced', random_state=419, n_jobs=2)
        occupancy.fit(x, labels.occupied)
        result['occ_prob'] = occupancy.predict_proba(xp)[:, list(occupancy.classes_).index(1)]
    else:
        result['occ_prob'] = float(labels.occupied.iloc[0])
    result[COUNTS] = result[COUNTS].clip(0, 50)
    return result, scales


def run(data, out='submission.csv', mode='detector', validate=False, features=None):
    data = Path(data)
    features = make_features(data) if features is None else features
    labels = pd.read_csv(data / 'train_labels.csv')
    if validate:
        days = sorted(labels.date_utc.unique())
        if len(days) < 2:
            raise ValueError('Validation needs two train dates.')
        training = labels[labels.date_utc < days[-1]]
        key = labels[labels.date_utc == days[-1]]
        ids = key.row_id.tolist()
    else:
        training = labels
        key = pd.read_csv(data / 'sample_submission.csv')
        ids = key.row_id.tolist()
    scales = None
    if mode == 'detector':
        prediction = detector_prediction(features.loc[ids])
    elif mode == 'median':
        prediction = pd.DataFrame(index=pd.Index(ids, name='row_id'))
        for c in COUNTS:
            prediction[c] = training[c].median()
        prediction['occ_prob'] = training.occupied.mean()
    else:
        prediction, scales = calibrated_prediction(features, training, ids)
    prediction = prediction.reset_index()[['row_id'] + COUNTS + ['occ_prob']]
    prediction = validate_submission(prediction, ids)
    if validate:
        result = breakdown(key, prediction)
        result['validation_date'] = days[-1]
        result['training_frames'] = training.frame_id.nunique()
        result['validation_frames'] = key.frame_id.nunique()
        result['scales'] = scales
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return prediction, result
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(out, index=False)
    print(f'{mode}: {len(prediction)} rows saved to {out}')
    return prediction, {'mode': mode, 'scales': scales}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path, default=Path('submission.csv'))
    parser.add_argument('--mode', choices=['median', 'detector', 'calibrated'], default='detector')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    run(args.data, args.out, args.mode, args.validate)
