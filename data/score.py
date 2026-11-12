"""Table-state metric. Larger is better. No files/network are read by score()."""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

COUNTS = ['people', 'hands', 'cups', 'dishes', 'items']
PREDICTIONS = COUNTS + ['occ_prob']
CAP = 50.0


class ParticipantVisibleError(Exception):
    pass


def validate_submission(submission, expected_ids, row_id_column_name='row_id', allow_extra=False):
    required = [row_id_column_name] + PREDICTIONS
    if len(submission.columns) != len(required) or set(submission.columns) != set(required):
        raise ParticipantVisibleError('Required columns: ' + ', '.join(required))
    ids = submission[row_id_column_name]
    if ids.isna().any() or ids.duplicated().any():
        raise ParticipantVisibleError('Every row_id must be nonempty and unique.')
    expected = set(expected_ids)
    missing = expected - set(ids)
    extra = set(ids) - expected
    if missing or (extra and not allow_extra):
        raise ParticipantVisibleError(f'ID mismatch: {len(missing)} missing, {len(extra)} extra.')
    clean = submission.copy()
    for c in PREDICTIONS:
        value = pd.to_numeric(clean[c], errors='coerce').to_numpy(dtype=float)
        high = 1.0 if c == 'occ_prob' else CAP
        if not np.isfinite(value).all() or (value < 0).any() or (value > high).any():
            raise ParticipantVisibleError(f'{c} must contain finite numbers in [0, {high:g}].')
        clean[c] = value
    return clean


def validate_solution(solution, row_id_column_name='row_id'):
    required = [row_id_column_name] + COUNTS
    if any(c not in solution for c in required):
        raise ValueError('Solution is missing required columns.')
    if solution[row_id_column_name].isna().any() or solution[row_id_column_name].duplicated().any():
        raise ValueError('Solution IDs must be nonempty and unique.')
    clean = solution.copy()
    for c in COUNTS:
        v = pd.to_numeric(clean[c], errors='coerce').to_numpy(float)
        if not np.isfinite(v).all() or (v < 0).any() or (v > CAP).any():
            raise ValueError(f'Invalid solution counter: {c}')
        clean[c] = v
    occupied = (clean.people > 0).astype(int)
    if 'occupied' in clean and not np.array_equal(clean.occupied.to_numpy(), occupied.to_numpy()):
        raise ValueError('occupied must equal (people > 0).')
    clean['occupied'] = occupied
    if 'Usage' in clean and not clean.Usage.isin(['Public', 'Private', 'Ignored']).all():
        raise ValueError('Unknown Usage value in solution.')
    return clean


def breakdown(solution, submission, row_id_column_name='row_id'):
    solution = validate_solution(solution, row_id_column_name)
    # A full submission is allowed when Kaggle passes one solution split only.
    submission = validate_submission(submission, solution[row_id_column_name],
                                     row_id_column_name, allow_extra=True)
    if 'Usage' in solution:
        solution = solution[solution.Usage != 'Ignored'].copy()
    if solution.empty:
        raise ValueError('There are no scored rows.')
    pred = submission.set_index(row_id_column_name).loc[solution[row_id_column_name]]
    qualities, details = [], {}
    for c in COUNTS:
        y = solution[c].to_numpy(float)
        p = pred[c].to_numpy(float)
        error = float(np.abs(y - p).sum())
        support = float(y.sum())
        # Explicit zero-support convention: quality = max(0, 1 - MAE).
        denominator = support if support > 0 else len(y)
        quality = max(0.0, 1.0 - error / denominator)
        qualities.append(quality)
        details[c] = dict(objects=support, absolute_error=error,
                          quality=quality, zero_support=support == 0)
    occupied = solution.occupied.to_numpy(int)
    probability = pred.occ_prob.to_numpy(float)
    if occupied.sum() == 0:
        ap = float(1.0 - probability.mean())
    elif occupied.sum() == len(occupied):
        ap = 1.0
    else:
        ap = float(average_precision_score(occupied, probability))
    total = float(.2 * ap + .8 * np.mean(qualities))
    return dict(score=total, rows=len(solution), occupied_tables=int(occupied.sum()),
                occupancy_ap=ap, counters=details)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    return breakdown(solution, submission, row_id_column_name)['score']


def evaluate_file(solution, submission):
    solution = validate_solution(solution)
    submission = validate_submission(submission, solution.row_id)
    result = {'AllScored': breakdown(solution, submission)}
    if 'Usage' in solution:
        for usage in ['Public', 'Private']:
            split = solution[solution.Usage == usage]
            if not split.empty:
                result[usage] = breakdown(split, submission)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score labels/solution against a full submission.')
    parser.add_argument('solution')
    parser.add_argument('submission')
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args()
    result = evaluate_file(pd.read_csv(args.solution), pd.read_csv(args.submission))
    if not args.details:
        result = {k: {'score': v['score'], 'rows': v['rows']} for k, v in result.items()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
