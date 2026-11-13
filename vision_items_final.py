"""RASCAR: reproducible YOLO-World *candidate* detection for personal items.

Outputs one row per unique item candidate in original image coordinates.
`handbag`, `backpack` and `purse` are all grouped as `bag`.
NO train/test ground-truth is used for inference; test labels are never loaded.
Do not treat exported candidates as validated table counts.

Requires: ultralytics==8.3.220, torch==2.7.1+cu118, pandas, opencv-python.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO, __version__ as ultralytics_version

# Separate passes help laptop prompts avoid competing with bag/phone prompts.
GENERAL_PROMPTS = ["handbag", "backpack", "purse", "cell phone", "wallet", "headphones"]
LAPTOP_PROMPTS = ["laptop", "closed laptop", "notebook computer", "open laptop"]
CATEGORY = {
    "handbag": "bag", "backpack": "bag", "purse": "bag",
    "cell phone": "phone", "phone": "phone",
    "wallet": "wallet", "headphones": "headphones",
    "laptop": "laptop", "closed laptop": "laptop",
    "notebook computer": "laptop", "open laptop": "laptop",
}
CSV_FIELDS = ["frame_id", "class_name", "x1", "y1", "x2", "y2", "confidence", "source"]
DEBUG_FIELDS = CSV_FIELDS + ["item_type", "original_class", "pass_name", "crop_table_id"]


def file_hash(path: Path):
    if not path.is_file():
        return None
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def box_area(b):
    return max(0., b[2] - b[0]) * max(0., b[3] - b[1])


def intersection(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    inter = intersection(a, b)
    return inter / max(1e-9, box_area(a) + box_area(b) - inter)


def as_box(r):
    return (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))


def deduplicate(rows, threshold):
    """Class-agnostic NMS: one object, even if called bag AND backpack etc."""
    ordered = sorted(rows, key=lambda r: (-float(r["confidence"]), r["original_class"], r["x1"]))
    kept = []
    for candidate in ordered:
        if all(iou(as_box(candidate), as_box(accepted)) < threshold for accepted in kept):
            kept.append(candidate)
    return kept


def crop_rect(b, width, height, margin):
    x1, y1, x2, y2 = b
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)),
            min(width, int(x2 + dx + .999)), min(height, int(y2 + dy + .999)))


def on_table(candidate, table_boxes, overlap_min, area_max, center_margin):
    """Geometry PROXY, not proof that an object is on the tabletop."""
    b = as_box(candidate)
    area = box_area(b)
    if area < 1:
        return False
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    for t in table_boxes:
        tw, th = t[2] - t[0], t[3] - t[1]
        if tw <= 0 or th <= 0:
            continue
        inside_center = (t[0] - center_margin * tw <= cx <= t[2] + center_margin * tw and
                         t[1] - center_margin * th <= cy <= t[3] + center_margin * th)
        if (inside_center and intersection(b, t) / area >= overlap_min
                and area / (tw * th) <= area_max):
            return True
    return False


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def preview(img, rows, path):
    canvas = img.copy()
    for row in rows:
        p1 = (round(row["x1"]), round(row["y1"]))
        p2 = (round(row["x2"]), round(row["y2"]))
        cv2.rectangle(canvas, p1, p2, (0, 230, 255), 2)
        label = f'{row["item_type"]} ({row["original_class"]}) {row["confidence"]:.2f}'
        pos = (p1[0], max(18, p1[1] - 5))
        cv2.putText(canvas, label, pos, cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, pos, cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="vision_items_final")
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--mode", choices=["train", "all"], default="train",
                        help="train=39 labeled frame IDs for diagnostics, all=every frame in frames.csv")
    parser.add_argument("--frame-id", default=None, help="single frame from frames.csv")
    parser.add_argument("--limit", type=int, default=0, help="optional max number of selected frames; 0=all")
    parser.add_argument("--passes", choices=["both", "general", "laptop"], default="both")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-general", type=float, default=.08)
    parser.add_argument("--conf-laptop", type=float, default=.05)
    parser.add_argument("--margin-general", type=float, default=.20)
    parser.add_argument("--margin-laptop", type=float, default=.08)
    parser.add_argument("--nms-iou", type=float, default=.60)
    parser.add_argument("--min-overlap", type=float, default=.65,
                        help="minimum fraction of detection overlapping a table box")
    parser.add_argument("--max-area-ratio", type=float, default=.55,
                        help="reject candidate bigger than this fraction of the table rectangle")
    parser.add_argument("--center-margin", type=float, default=.03)
    parser.add_argument("--preview", type=int, default=25, help="max preview images WITH candidates")
    args = parser.parse_args()
    if args.limit < 0 or args.imgsz < 64 or not (0 < args.nms_iou <= 1):
        parser.error("Invalid limit, imgsz, or nms-iou")
    if not (0 <= args.conf_general <= 1 and 0 <= args.conf_laptop <= 1):
        parser.error("confidence thresholds must be between 0 and 1")
    if args.margin_general < 0 or args.margin_laptop < 0:
        parser.error("crop margins must be nonnegative")
    if not (0 <= args.min_overlap <= 1 and args.max_area_ratio > 0):
        parser.error("invalid geometry limits")

    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    frames = pd.read_csv(root / "frames.csv", dtype={"frame_id": str, "image_path": str})
    tables = pd.read_csv(root / "tables.csv", dtype={"frame_id": str, "table_id": str})
    if args.frame_id:
        frames = frames[frames.frame_id == args.frame_id]
        if frames.empty:
            parser.error("frame-id not found in frames.csv")
    elif args.mode == "train":
        train_ids = set(pd.read_csv(root / "train_labels.csv", usecols=["frame_id"], dtype=str)["frame_id"])
        frames = frames[frames.frame_id.isin(train_ids)]
    frames = frames.drop_duplicates("frame_id").sort_values("frame_id")
    if args.limit:
        frames = frames.head(args.limit)
    groups = {fid: group for fid, group in tables.groupby("frame_id")}

    pass_cfg = []
    if args.passes in ("both", "general"):
        pass_cfg.append(("general", GENERAL_PROMPTS, args.conf_general, args.margin_general))
    if args.passes in ("both", "laptop"):
        pass_cfg.append(("laptop", LAPTOP_PROMPTS, args.conf_laptop, args.margin_laptop))
    # Two separate models avoid regenerating text embeddings for every frame.
    models = []
    for name, prompts, confidence, margin in pass_cfg:
        detector = YOLO(args.model)
        if not hasattr(detector, "set_classes"):
            raise RuntimeError("Installed Ultralytics model does not support set_classes")
        detector.set_classes(prompts)
        models.append((name, detector, prompts, confidence, margin))

    weight = Path(str(getattr(models[0][1], "ckpt_path", "") or args.model))
    sha = file_hash(weight)
    if sha is None:
        raise RuntimeError(f"Cannot locate local weights for SHA-256: {weight}")
    raw_all, unique_all, filtered_all = [], [], []
    preview_count = 0
    for index, fr in enumerate(frames.itertuples(index=False), 1):
        fid = str(fr.frame_id)
        img_path = root / str(fr.image_path)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        h, w = image.shape[:2]
        table_entries = []
        if fid in groups:
            for row in groups[fid].itertuples(index=False):
                coords = tuple(float(getattr(row, k)) for k in ("x1", "y1", "x2", "y2"))
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    table_entries.append((str(getattr(row, "table_id")), coords))
        proposals = []
        for pass_name, model, prompts, conf, margin in models:
            for table_id, table_box in table_entries:
                x1, y1, x2, y2 = crop_rect(table_box, w, h, margin)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                result = model.predict(image[y1:y2, x1:x2], device=args.device,
                                       imgsz=args.imgsz, conf=conf, half=False,
                                       verbose=False)[0]
                if result.boxes is None:
                    continue
                for det in result.boxes:
                    cls_id = int(det.cls.item())
                    # The prompt list, rather than global COCO labels, defines the custom classes.
                    if cls_id < 0 or cls_id >= len(prompts):
                        continue
                    original = prompts[cls_id]
                    item_type = CATEGORY.get(original)
                    if not item_type:
                        continue
                    bx1, by1, bx2, by2 = [float(v) for v in det.xyxy[0].cpu().tolist()]
                    b = (max(0., min(w, bx1 + x1)), max(0., min(h, by1 + y1)),
                         max(0., min(w, bx2 + x1)), max(0., min(h, by2 + y1)))
                    if box_area(b) < 4:
                        continue
                    proposals.append({"frame_id": fid, "class_name": "items",
                                      "x1": round(b[0], 2), "y1": round(b[1], 2),
                                      "x2": round(b[2], 2), "y2": round(b[3], 2),
                                      "confidence": round(float(det.conf.item()), 6),
                                      "source": "yolov8s_worldv2_items_final",
                                      "item_type": item_type, "original_class": original,
                                      "pass_name": pass_name, "crop_table_id": table_id})
        unique = deduplicate(proposals, args.nms_iou)
        table_boxes = [coords for _, coords in table_entries]
        filtered = [r for r in unique if on_table(r, table_boxes, args.min_overlap,
                                                  args.max_area_ratio, args.center_margin)]
        raw_all.extend(proposals)
        unique_all.extend(unique)
        filtered_all.extend(filtered)
        if filtered and preview_count < args.preview:
            preview(image, filtered, out / "previews" / f"{fid}_items.jpg")
            preview_count += 1
        print(f"[{index}/{len(frames)}] {fid}: raw={len(proposals)} "
              f"unique={len(unique)} tabletop_candidates={len(filtered)}")

    write_csv(out / "vision_items_raw.csv", raw_all, DEBUG_FIELDS)
    write_csv(out / "vision_items_unique.csv", unique_all, DEBUG_FIELDS)
    write_csv(out / "vision_items_filtered_debug.csv", filtered_all, DEBUG_FIELDS)
    write_csv(out / "vision_detections.csv", filtered_all, CSV_FIELDS)
    metadata = {
        "model": args.model, "weights_path": str(weight), "weights_sha256": sha,
        "model_source": "https://github.com/ultralytics/assets/releases",
        "model_docs": "https://docs.ultralytics.com/models/yolo-world/",
        "license_note": "Check terms of original YOLO-World weights and Ultralytics AGPL-3.0 before final submission.",
        "python": __import__("sys").version.split()[0], "torch": torch.__version__,
        "ultralytics": ultralytics_version, "device": args.device, "mode": args.mode,
        "passes": [{"name": name, "prompts": prompts, "conf": conf, "margin": margin}
                   for name, _, prompts, conf, margin in models],
        "imgsz": args.imgsz, "nms_iou": args.nms_iou,
        "min_overlap": args.min_overlap, "max_area_ratio": args.max_area_ratio,
        "center_margin": args.center_margin,
        "processed_frames": len(frames), "raw_boxes": len(raw_all),
        "after_class_agnostic_nms": len(unique_all), "tabletop_candidates": len(filtered_all),
        "item_type_counts": dict(Counter(r["item_type"] for r in filtered_all)),
        "warnings": ["Candidates are NOT validated object detections or per-table counts.",
                     "Bounding-box tabletop filter is a proxy, not a segmentation mask.",
                     "No test labels or manual test corrections used.",
                     "Tune confidence/geometry on training dates before using in a submission."]
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out.resolve())
    print("Unique candidates:", len(unique_all), "filtered:", len(filtered_all))
    print("Types:", metadata["item_type_counts"])
    print("Send vision_detections.csv + metadata.json to teammate after validation")


if __name__ == "__main__":
    main()
