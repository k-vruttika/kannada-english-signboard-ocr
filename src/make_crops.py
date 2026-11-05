import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from urllib.parse import unquote

import cv2
import yaml

from akshara import n_units
from preprocess import read_image

FIELDS = ["path", "text", "script", "photo_id", "n_units", "split", "char_boxes"]
SCRIPTS = ("kannada", "english")


def photo_path(task, photos_dir):
    ref = unquote(next(v for v in task["data"].values() if isinstance(v, str)))
    name = os.path.basename(ref.split("?d=")[-1])
    path = os.path.join(photos_dir, name)
    # uploaded files get an 8-character prefix in Label Studio, e.g. "1a2b3c4d-IMG_0012.jpg"
    if not os.path.exists(path) and len(name) > 9 and name[8] == "-":
        path = os.path.join(photos_dir, name[9:])
    return path


def to_pixels(v, img_w, img_h):
    x, y = v["x"] * img_w / 100, v["y"] * img_h / 100
    w, h = v["width"] * img_w / 100, v["height"] * img_h / 100
    # Label Studio rotates clockwise about the top-left corner; keep the upright bounding box
    a = math.radians(v.get("rotation", 0))
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    xs = [x + dx * math.cos(a) - dy * math.sin(a) for dx, dy in corners]
    ys = [y + dx * math.sin(a) + dy * math.cos(a) for dx, dy in corners]
    return (max(0, round(min(xs))), max(0, round(min(ys))),
            min(img_w, round(max(xs))), min(img_h, round(max(ys))))


def regions(annotation):
    # the rectangle, its label and its transcription are separate results sharing one id
    by_id = {}
    for r in annotation["result"]:
        reg = by_id.setdefault(r["id"], {})
        v = r["value"]
        if "x" in v and "width" in v:
            reg["box"] = to_pixels(v, r["original_width"], r["original_height"])
        for key in ("rectanglelabels", "labels"):
            if key in v:
                reg["labels"] = [s.lower() for s in v[key]]
        if "text" in v:
            reg["text"] = " ".join(v["text"]).strip()
    return [r for r in by_id.values() if "box" in r]


def script_of(region):
    found = [s for s in SCRIPTS if s in region.get("labels", [])]
    return found[0] if found else None


def has_kannada(text):
    return any("ಀ" <= ch <= "೿" for ch in text)


def inside(char_box, word_box):
    cx, cy = (char_box[0] + char_box[2]) / 2, (char_box[1] + char_box[3]) / 2
    return word_box[0] <= cx <= word_box[2] and word_box[1] <= cy <= word_box[3]


def split_photos(photo_ids, train_frac, seed):
    photos = sorted(set(photo_ids))
    random.Random(seed).shuffle(photos)
    n_train = round(train_frac * len(photos))
    return {p: "train" if i < n_train else "test" for i, p in enumerate(photos)}


def crop_task(task, photos_dir, out_dir, skipped):
    path = photo_path(task, photos_dir)
    photo_id = os.path.splitext(os.path.basename(path))[0]
    img = read_image(path)
    regs = regions(task["annotations"][0])
    chars = [r["box"] for r in regs if "char" in r.get("labels", [])]
    words = sorted([r for r in regs if script_of(r)], key=lambda r: (r["box"][1], r["box"][0]))
    rows = []
    for k, w in enumerate(words):
        x0, y0, x1, y1 = w["box"]
        script, text = script_of(w), w.get("text", "")
        if not text or x1 - x0 < 2 or y1 - y0 < 2:
            skipped["no text or empty box"] += 1
            continue
        if (script == "kannada") != has_kannada(text):
            print(f"check label: {photo_id} '{text}' is marked {script}")
        name = f"{photo_id}_{k:03d}.png"
        cv2.imencode(".png", img[y0:y1, x0:x1])[1].tofile(os.path.join(out_dir, name))
        mine = sorted([c for c in chars if inside(c, w["box"])], key=lambda c: c[0])
        rel = [[c[0] - x0, c[1] - y0, c[2] - x0, c[3] - y0] for c in mine]
        if rel and len(rel) != n_units(text):
            print(f"check char boxes: {photo_id} '{text}' has {len(rel)} boxes but {n_units(text)} units")
        rows.append({"path": name, "text": text, "script": script, "photo_id": photo_id,
                     "n_units": n_units(text), "char_boxes": json.dumps(rel) if rel else ""})
    unused = [c for c in chars if not any(inside(c, w["box"]) for w in words)]
    skipped["char box outside every word"] += len(unused)
    skipped["region without script label"] += sum(1 for r in regs if not script_of(r) and "char" not in r.get("labels", []))
    return rows


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=cfg["paths"]["labelstudio"])
    ap.add_argument("--photos", default=cfg["paths"]["photos"])
    ap.add_argument("--out", default=cfg["paths"]["crops"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tasks = json.load(open(args.export, encoding="utf-8"))
    rows, skipped = [], Counter()
    for task in tasks:
        task["annotations"] = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if task["annotations"]:
            rows += crop_task(task, args.photos, args.out, skipped)

    splits = split_photos([r["photo_id"] for r in rows], cfg["split"]["train"], cfg["seed"])
    for r in rows:
        r["split"] = splits[r["photo_id"]]
    with open(os.path.join(args.out, "labels.tsv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} word crops from {len(splits)} photos written to {args.out}")
    for split in ("train", "test"):
        n_photos = sum(1 for s in splits.values() if s == split)
        per_script = Counter(r["script"] for r in rows if r["split"] == split)
        boxed = sum(1 for r in rows if r["split"] == split and r["char_boxes"])
        print(f"  {split}: {n_photos} photos, {dict(per_script)}, {boxed} with char boxes")
    for reason, n in skipped.items():
        if n:
            print(f"  skipped: {reason}: {n}")


if __name__ == "__main__":
    main()
