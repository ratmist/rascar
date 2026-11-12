"""Create an image review sheet and an operator-queue CSV from predictions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from score import validate_submission, COUNTS


def preview(data, submission, out, count=6):
    data, out = Path(data), Path(out)
    pred = pd.read_csv(submission)
    sample = pd.read_csv(data / 'sample_submission.csv')
    pred = validate_submission(pred, sample.row_id)
    tables = pd.read_csv(data / 'tables.csv')
    frames = pd.read_csv(data / 'frames.csv').set_index('frame_id')
    merged = tables.merge(pred, on='row_id', validate='one_to_one')
    # These are heuristic prompts to an operator, not labels or new scoring targets.
    merged['operator_hint'] = np.select(
        [merged.occ_prob >= .5, (merged.occ_prob < .5) & (merged['items'] >= .5),
         (merged.occ_prob < .5) & ((merged.cups + merged.dishes) >= .5)],
        ['occupied', 'check_belongings', 'check_dishes'], default='no_prompt')
    out.parent.mkdir(parents=True, exist_ok=True)
    queue = out.with_suffix('.csv')
    merged[['row_id', 'frame_id', 'table_id'] + COUNTS + ['occ_prob', 'operator_hint']].to_csv(queue, index=False)
    activity = merged.groupby('frame_id')[COUNTS].sum().sum(axis=1).sort_values(ascending=False)
    chosen = list(activity.index[:count])
    width, height = 960, 650
    canvas = Image.new('RGB', (2 * width, ((len(chosen) + 1) // 2) * height), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    for i, frame in enumerate(chosen):
        im = Image.open(data / frames.loc[frame, 'image_path']).convert('RGB').resize((width, 540))
        d = ImageDraw.Draw(im)
        rows = merged[merged.frame_id == frame].sort_values('table_id')
        for r in rows.itertuples():
            color = '#ffb000' if r.occ_prob >= .5 else '#db2777' if r.operator_hint != 'no_prompt' else '#10b981'
            b = [r.x1 * .75, r.y1 * .75, r.x2 * .75, r.y2 * .75]
            d.rectangle(b, outline=color, width=3)
            d.text((b[0], max(0, b[1] - 20)), r.table_id, fill=color, font=font,
                   stroke_width=1, stroke_fill='black')
        x, y = (i % 2) * width, (i // 2) * height
        canvas.paste(im, (x, y))
        draw.text((x + 6, y + 545), frames.loc[frame, 'timestamp'], fill='black', font=font)
        for j, r in enumerate(rows.itertuples()):
            text = r.table_id + ': ' + '/'.join(f'{getattr(r,c):.1f}' for c in COUNTS) + f'  P={r.occ_prob:.2f}'
            col, line = j % 2, j // 2
            draw.text((x + 6 + col * 475, y + 568 + line * 22), text, fill='black', font=font)
    canvas.save(out)
    print(f'Review image: {out}; operator queue: {queue}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('preview.jpg'))
    parser.add_argument('--count', type=int, default=6)
    a = parser.parse_args()
    preview(a.data, a.submission, a.out, a.count)
