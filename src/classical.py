import argparse
import csv
import hashlib
import os
import random
import time
from collections import defaultdict

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from datasets import RARE, clean_label, load_charset, load_config, load_public, per_class_for  # noqa: F401  (per_class_for is used by train_cnn.ipynb)
from preprocess import dark_on_light, fit_square, read_image, to_gray
from segment import segment_word

HOG = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))


def char_vector(gray, size=32):
    return hog(fit_square(gray, size), **HOG)


def rendered_samples(root, script):
    path = os.path.join(root, script, "labels.tsv")
    with open(path, encoding="utf-8", newline="") as f:
        return [(os.path.join(root, script, r["path"]), clean_label(r["text"])) for r in csv.DictReader(f, delimiter="\t")]


def per_class(samples, n, seed):
    groups = defaultdict(list)
    for s in samples:
        groups[s[1]].append(s)
    rng = random.Random(seed)
    picked = []
    for label in sorted(groups):
        # sorted by file name, not full path, so Colab and the laptop pick the same files
        items = sorted(groups[label], key=lambda s: os.path.basename(s[0]))
        rng.shuffle(items)
        picked += items[:n] if n else items
    return picked


def char_split(cfg, script, n_per_class):
    """The characters A, B and C train on, and the held-out tenth used for validation."""
    samples = rendered_samples(cfg["paths"]["rendered"], script)
    extra = load_public(f"chars74k_{script}", "train", cfg)
    print(f"{len(samples)} rendered + {len(extra)} Chars74K training characters")
    charset = set(load_charset(cfg["charsets"][script]))
    samples = [(p, t if t in charset else RARE) for p, t in samples + extra]
    samples = per_class(samples, n_per_class, cfg["seed"])
    random.Random(cfg["seed"]).shuffle(samples)
    n_val = len(samples) // 10  # held out from the training characters, never from the test photos
    print(f"{len(samples) - n_val} train / {n_val} validation characters, fingerprint {fingerprint(samples)}")
    return samples[n_val:], samples[:n_val]


def fingerprint(samples):
    names = "\n".join(f"{os.path.basename(p)}\t{t}" for p, t in samples)
    return hashlib.md5(names.encode("utf-8")).hexdigest()[:10]


def make_model(kind, cfg):
    c = cfg["classical"]
    if kind == "svm":
        # probability=True would cost a 5-fold refit, so confidence comes from the decision scores
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c["svm_c"], gamma="scale", random_state=cfg["seed"]))
    return RandomForestClassifier(n_estimators=c["rf_trees"], min_samples_leaf=c["rf_min_samples_leaf"],
                                  n_jobs=-1, random_state=cfg["seed"])


def memory_mb(model):
    # rough: the arrays that dominate each model's size in RAM
    if hasattr(model, "estimators_"):
        states = [t.tree_.__getstate__() for t in model.estimators_]
        return sum(s["nodes"].nbytes + s["values"].nbytes for s in states) / 1e6
    svc = model[-1]
    return (svc.support_vectors_.nbytes + svc.dual_coef_.nbytes) / 1e6


def predict_chars(bundle, X, batch=64):
    """Labels and confidences, a few rows at a time: an SVM with ~860 classes compares ~370,000 class pairs,
    so scoring thousands of characters at once would need several GB."""
    model = bundle["model"]
    labels, confs = [], []
    for i in range(0, len(X), batch):
        part = X[i:i + batch]
        if bundle["kind"] == "rf":
            proba = model.predict_proba(part)
        else:
            scores = model.decision_function(part)
            if scores.ndim == 1:
                scores = np.stack([-scores, scores], axis=1)
            scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            proba = scores / scores.sum(axis=1, keepdims=True)
        labels.append(model.classes_[proba.argmax(axis=1)])
        confs.append(proba.max(axis=1))
    return np.concatenate(labels), np.concatenate(confs)


def recognise(bundle, img, options=(), seg_params=None):
    """Word image -> (text, unit labels, boxes, confidence) using the shared segmenter."""
    gray, boxes = segment_word(img, options, seg_params)
    if not boxes:
        return "", [], [], 0.0
    X = np.stack([char_vector(gray[y0:y1, x0:x1], bundle["size"]) for x0, y0, x1, y1 in boxes])
    labels, conf = predict_chars(bundle, X)
    text = "".join("?" if u == RARE else u for u in labels)
    return text, list(labels), boxes, float(conf.mean())


def features(samples, size):
    X = np.stack([char_vector(dark_on_light(to_gray(read_image(p))), size) for p, _ in samples])
    return X, np.array([t for _, t in samples])


def train(kind, script, train_set, val_set, cfg):
    print(f"computing HOG for {len(train_set) + len(val_set)} characters")
    X, y = features(train_set, cfg["char_size"])
    X_val, y_val = features(val_set, cfg["char_size"])
    model = make_model(kind, cfg)
    start = time.perf_counter()
    model.fit(X, y)
    seconds = time.perf_counter() - start
    bundle = {"model": model, "kind": kind, "script": script, "size": cfg["char_size"], "hog": HOG,
              "train_seconds": seconds, "n_train": len(train_set)}
    val_acc = 100 * float(np.mean(predict_chars(bundle, X_val)[0] == y_val)) if len(val_set) else float("nan")
    print(f"{kind} {script}: {len(set(y))} classes, trained in {seconds:.1f} s, "
          f"validation accuracy {val_acc:.1f}% on {len(val_set)} characters")
    return bundle


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, choices=["kannada", "english"])
    ap.add_argument("--model", default="both", choices=["svm", "rf", "both"])
    ap.add_argument("--per-class", type=int, default=None,
                    help="at most this many characters per class (0 = all); default: char_per_class in "
                         "config.yaml. Keep it equal to the CNN notebook")
    args = ap.parse_args()
    if args.per_class is None:
        args.per_class = per_class_for(cfg, args.script)

    train_set, val_set = char_split(cfg, args.script, args.per_class)
    print(f"{sum(1 for _, t in train_set if t == RARE)} training characters are {RARE}")
    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    for kind in (["svm", "rf"] if args.model == "both" else [args.model]):
        bundle = train(kind, args.script, train_set, val_set, cfg)
        path = os.path.join(cfg["paths"]["models"], f"{kind}_{args.script}.joblib")
        joblib.dump(bundle, path, compress=3)
        print(f"saved {path}: {os.path.getsize(path) / 1e6:.1f} MB on disk, about {memory_mb(bundle['model']):.0f} MB in memory")


if __name__ == "__main__":
    main()
