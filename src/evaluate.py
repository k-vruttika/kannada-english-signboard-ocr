import argparse
import csv
import json
import os
import random
import time

import joblib
import numpy as np

from akshara import is_vowel_sign, split_aksharas
from classical import recognise
from datasets import RARE, clean_label, load_charset, load_config, load_public, read_labels
from metrics import bootstrap_ci, char_accuracy, levenshtein, norm, rare_share, segmentation_ok, wer
from models import cnn_recognise, crnn_recognise, load_model
from plots import charts
from preprocess import read_image

SETTINGS = {"none": (), "clahe": ("clahe",), "sharpen": ("sharpen",), "deskew": ("deskew",)}
PUBLIC_TEST = {"kannada": "indicstr12_kannada", "english": "iiit5k"}
MODELS = [("A-SVM", "svm_{}.joblib"), ("B-RF", "rf_{}.joblib"), ("C-CNN", "cnn_{}.pth"), ("D-CRNN", "crnn_{}.pth")]
CATEGORIES = "segmentation failure | confusable characters | dropped vowel sign | split/merged akshara | stylised font"


def load_models(cfg, script, wanted):
    seg, models = cfg["segment"], {}
    for name, pattern in MODELS:
        path = os.path.join(cfg["paths"]["models"], pattern.format(script))
        if name[0] not in wanted:
            continue
        pretrained = path.replace(".pth", "_pretrained.pth")
        if name == "D-CRNN" and not os.path.exists(path) and os.path.exists(pretrained):
            # no fine-tuned CRNN yet, so the pre-trained one is evaluated under its own name
            name, path = "D (pre-trained only)", pretrained
        if not os.path.exists(path):
            print(f"{name} {script}: {path} not found, skipped")
            continue
        size = os.path.getsize(path) / 1e6
        if path.endswith(".joblib"):
            b = joblib.load(path)
            run = lambda img, o, b=b: recognise(b, img, o, seg)
            models[name] = dict(run=run, size_mb=size, train_seconds=b["train_seconds"], segmented=True)
        else:
            m, info = load_model(path)
            if info["kind"] == "cnn":
                run = lambda img, o, m=m, info=info: cnn_recognise(m, info, img, o, seg)
                seconds = info["train_seconds"]
            else:
                def run(img, o, m=m, info=info):
                    text, conf = crnn_recognise(m, info, img, o)
                    return text, None, None, conf
                seconds = info["pretrain"]["train_seconds"] + info.get("finetune", {}).get("train_seconds", 0.0)
            models[name] = dict(run=run, size_mb=size, train_seconds=seconds, segmented=info["kind"] == "cnn")
    return models


def my_test_set(cfg, script):
    if not os.path.exists(os.path.join(cfg["paths"]["crops"], "labels.tsv")):
        return None
    rows = [r for r in read_labels(cfg["paths"]["crops"]) if r["split"] == "test" and r["script"] == script]
    return dict(name="mine", script=script, paths=[os.path.join(cfg["paths"]["crops"], r["path"]) for r in rows],
                refs=[clean_label(r["text"]) for r in rows], photos=[r["photo_id"] for r in rows],
                boxes=[json.loads(r["char_boxes"]) if r["char_boxes"] else None for r in rows])


def public_test_set(cfg, script, limit):
    name = PUBLIC_TEST[script]
    p = cfg["public"][name]
    needed = p["test"]["gt"] if "test" in p else os.path.join(p["root"], "testdata.mat")
    if not os.path.exists(needed):
        print(f"{name}: {needed} not found, public {script} test set skipped")
        return None
    samples = load_public(name, "test", cfg, limit)
    # no photos here, so the bootstrap resamples single images
    return dict(name=name, script=script, paths=[s[0] for s in samples], refs=[s[1] for s in samples],
                photos=[s[0] for s in samples], boxes=[None] * len(samples))


def ci(values, photos, cfg):
    m = cfg["metrics"]
    return bootstrap_ci(values, photos, m["bootstrap_samples"], m["ci_level"], cfg["seed"])


def evaluate(model, test, setting, cfg, charset):
    preds, units, boxes, confs, ms = [], [], [], [], []
    for img in test["images"]:
        start = time.perf_counter()
        text, u, b, conf = model["run"](img, SETTINGS[setting])
        ms.append(1000 * (time.perf_counter() - start))
        preds.append(text)
        units.append(u)
        boxes.append(b)
        confs.append(conf)
    refs, photos = test["refs"], test["photos"]
    ref_units = [split_aksharas(r) for r in refs]
    edits = np.array([levenshtein(split_aksharas(norm(p)), u) for p, u in zip(preds, ref_units)])
    lens = np.array([len(u) for u in ref_units])
    exact = np.array([norm(p) == r for p, r in zip(preds, refs)], dtype=float)
    cer_of = lambda idx: 100 * edits[idx].sum() / lens[idx].sum()
    exact_of = lambda idx: 100 * exact[idx].mean()
    everything = np.arange(len(refs))
    row = dict(n_words=len(refs), cer=cer_of(everything), exact=exact_of(everything), wer=wer(preds, refs),
               rare_share=rare_share(ref_units, charset), ms_per_word=float(np.mean(ms)),
               size_mb=model["size_mb"], train_seconds=model["train_seconds"])
    row["cer_lo"], row["cer_hi"] = ci(cer_of, photos, cfg)
    row["exact_lo"], row["exact_hi"] = ci(exact_of, photos, cfg)

    pairs = []
    if model["segmented"]:
        ok = np.array([len(b) == n for b, n in zip(boxes, lens)], dtype=float)
        seg_of = lambda idx: 100 * ok[idx].mean()
        row["seg_success"] = seg_of(everything)
        row["seg_success_lo"], row["seg_success_hi"] = ci(seg_of, photos, cfg)
        boxed = [k for k, t in enumerate(test["boxes"]) if t is not None]
        row["n_boxed"] = len(boxed)
        # deskew rotates the crop, so its segments are no longer in the coordinates of the true boxes
        if boxed and setting != "deskew":
            thr = cfg["metrics"]["iou_threshold"]
            row["seg_success_iou"] = 100 * np.mean([segmentation_ok(boxes[k], lens[k], test["boxes"][k], thr) for k in boxed])
        true_units = [[u if u in charset else RARE for u in ru] for ru in ref_units]
        good = [k for k in everything if ok[k]]
        if good:
            row["char_acc"] = char_accuracy([units[k] for k in good], [true_units[k] for k in good])
            pairs = [(t, p) for k in good for t, p in zip(true_units[k], units[k])]
    words = [dict(path=test["paths"][k], ref=refs[k], pred=preds[k], n_units=int(lens[k]),
                  n_segments=len(boxes[k]) if boxes[k] is not None else "", confidence=round(confs[k], 3))
             for k in everything if not exact[k]]
    return row, words, pairs


def suggest(ref, pred, n_segments, segmented):
    """A first guess to speed up sorting by hand; the category column is left for the final decision."""
    r, p = split_aksharas(ref), split_aksharas(norm(pred))
    if segmented and n_segments != len(r):
        return "segmentation failure"
    strip = lambda t: "".join(c for c in t if not is_vowel_sign(c))
    signs = lambda t: sum(is_vowel_sign(c) for c in t)
    if strip(ref) == strip(norm(pred)) and signs(norm(pred)) < signs(ref):
        return "dropped vowel sign"
    if len(p) != len(r):
        return "split/merged akshara"
    return "confusable characters"


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts", nargs="+", default=["kannada", "english"])
    ap.add_argument("--models", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--limit", type=int, help="random subset of each public test set")
    ap.add_argument("--errors-per-run", type=int, default=25)
    args = ap.parse_args()
    out = cfg["paths"]["results"]
    os.makedirs(out, exist_ok=True)

    rows, errors, confusions = [], [], {}
    rng = random.Random(cfg["seed"])
    for script in args.scripts:
        charset = set(load_charset(cfg["charsets"][script]))
        models = load_models(cfg, script, args.models)
        tests = [t for t in (my_test_set(cfg, script), public_test_set(cfg, script, args.limit)) if t and t["refs"]]
        for test in tests:
            test["images"] = [read_image(p) for p in test["paths"]]
            for name, model in models.items():
                for setting in SETTINGS:
                    row, words, pairs = evaluate(model, test, setting, cfg, charset)
                    row.update(model=name, script=script, test_set=test["name"], preprocessing=setting)
                    rows.append(row)
                    print(f"{name} {script} {test['name']} {setting}: CER {row['cer']:.1f}%, exact {row['exact']:.1f}%"
                          + (f", segmentation {row['seg_success']:.1f}%" if "seg_success" in row else ""))
                    if setting != "none":
                        continue
                    if test["name"] == "mine":
                        confusions[(name, script)] = pairs
                    for w in rng.sample(words, min(args.errors_per_run, len(words))):
                        w.update(model=name, script=script, test_set=test["name"], preprocessing=setting,
                                 suggested_category=suggest(w["ref"], w["pred"], w["n_segments"], model["segmented"]),
                                 category="")
                        errors.append(w)

    fields = ["model", "script", "test_set", "preprocessing", "n_words", "rare_share",
              "seg_success", "seg_success_lo", "seg_success_hi", "n_boxed", "seg_success_iou", "char_acc",
              "cer", "cer_lo", "cer_hi", "wer", "exact", "exact_lo", "exact_hi", "ms_per_word", "size_mb", "train_seconds"]
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 2)
    write_csv(os.path.join(out, "results.csv"), rows, fields)
    errors.sort(key=lambda e: (e["script"], e["suggested_category"], e["model"]))
    write_csv(os.path.join(out, "errors.csv"), errors,
              ["model", "script", "test_set", "preprocessing", "path", "ref", "pred", "n_units", "n_segments",
               "confidence", "suggested_category", "category"])
    # model names as they appear in the rows, so "D (pre-trained only)" gets its own bar
    charts(rows, confusions, out, list(SETTINGS), list(dict.fromkeys(r["model"] for r in rows)))
    print(f"{len(rows)} result rows, {len(errors)} sample errors -> {out}")
    print(f"fill the category column of errors.csv with one of: {CATEGORIES}")


if __name__ == "__main__":
    main()
