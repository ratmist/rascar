"""RASCAR: reproducible baseline audit on the fixed date validation split.

Example:
  python scripts/audit_baseline.py --data data --out experiments/baseline
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('experiments/baseline'))
    args = parser.parse_args()
    data = args.data.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # baseline.py and score.py are supplied by the organizers.
    # They may be at repository root or alongside the CSV files.
    for location in (Path.cwd(), data):
        if (location / 'baseline.py').exists():
            sys.path.insert(0, str(location))
            break
    else:
        raise FileNotFoundError('Cannot find baseline.py at project root or --data directory')

    from baseline import make_features, detector_prediction, calibrated_prediction
    from score import COUNTS, breakdown, validate_submission

    labels = pd.read_csv(data / 'train_labels.csv')
    days = sorted(labels.date_utc.unique())
    if len(days) < 2:
        raise ValueError('Need at least two labelled dates for validation')
    train = labels[labels.date_utc < days[-1]].copy()
    val = labels[labels.date_utc == days[-1]].copy()
    ids = val.row_id.tolist()
    features = make_features(data)
    predictions = {}
    details = {}

    median = pd.DataFrame(index=pd.Index(ids, name='row_id'))
    for c in COUNTS:
        median[c] = train[c].median()
    median['occ_prob'] = train.occupied.mean()
    predictions['median'] = median
    predictions['detector'] = detector_prediction(features.loc[ids])
    calibrated, scales = calibrated_prediction(features, train, ids)
    predictions['calibrated'] = calibrated

    out_errors = val[['row_id', 'frame_id', 'table_id', 'date_utc', 'occupied'] + COUNTS].copy()
    geometry = pd.read_csv(data / 'tables.csv')
    out_errors = out_errors.merge(geometry[['row_id', 'x1', 'y1', 'x2', 'y2']], on='row_id', how='left', validate='one_to_one')

    for mode, pred in predictions.items():
        submission = pred.reset_index()[['row_id'] + COUNTS + ['occ_prob']]
        submission = validate_submission(submission, ids)
        report = breakdown(val, submission)
        details[mode] = report
        submission.to_csv(output / f'{mode}_predictions.csv', index=False)
        aligned = submission.set_index('row_id').loc[out_errors.row_id]
        for c in COUNTS:
            out_errors[f'{mode}_pred_{c}'] = aligned[c].to_numpy()
            out_errors[f'{mode}_ae_{c}'] = (out_errors[c] - aligned[c].to_numpy()).abs()
        out_errors[f'{mode}_occ_prob'] = aligned['occ_prob'].to_numpy()
        out_errors[f'{mode}_total_ae'] = sum(out_errors[f'{mode}_ae_{c}'] for c in COUNTS)

    out_errors.sort_values('calibrated_total_ae', ascending=False).to_csv(output / 'validation_errors.csv', index=False)
    metrics = {
        'split': {'train_date_before': days[-1], 'validation_date': days[-1],
                  'train_rows': len(train), 'validation_rows': len(val),
                  'train_frames': int(train.frame_id.nunique()),
                  'validation_frames': int(val.frame_id.nunique())},
        'scales': scales,
        'models': details,
    }
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Validation:', metrics['split'])
    for mode in predictions:
        print(f'{mode:10s} score={details[mode]["score"]:.6f} AP={details[mode]["occupancy_ap"]:.6f}')
        for c in COUNTS:
            r = details[mode]['counters'][c]
            print(f'  {c:7s} quality={r["quality"]:.4f} abs_error={r["absolute_error"]:.3f}')
    print('Files written to:', output)


if __name__ == '__main__':
    main()
