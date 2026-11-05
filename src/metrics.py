import os
import time

import jiwer
import numpy as np

from akshara import split_aksharas

# boxes are (x0, y0, x1, y1) in pixels; metrics are returned as percentages.
# Text is lowercased before comparing, so English is case-insensitive; Kannada has no case.


def norm(text):
    return text.strip().lower()


def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def iou(a, b):
    inter = box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def segmentation_ok(pred_boxes, n_units, true_boxes=None, iou_thr=0.5):
    if len(pred_boxes) != n_units:
        return False
    if true_boxes is None:
        return True
    # both lists are read left to right, so the k-th segment is compared with the k-th true box
    pred = sorted(pred_boxes, key=lambda b: b[0])
    true = sorted(true_boxes, key=lambda b: b[0])
    return len(true) == n_units and all(iou(p, t) >= iou_thr for p, t in zip(pred, true))


def rate(flags):
    return 100.0 * float(np.mean(flags))


def char_accuracy(pred_units, true_units):
    """Lists of per-word unit labels. Only pass words that were segmented correctly."""
    correct = total = 0
    for pred, true in zip(pred_units, true_units):
        assert len(pred) == len(true), "char accuracy is only defined on correctly segmented words"
        correct += sum(norm(p) == norm(t) for p, t in zip(pred, true))
        total += len(true)
    return 100.0 * correct / total


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(preds, refs):
    # aksharas for Kannada, letters for English (split_aksharas handles both)
    edits = sum(levenshtein(split_aksharas(norm(p)), split_aksharas(norm(r))) for p, r in zip(preds, refs))
    total = sum(len(split_aksharas(norm(r))) for r in refs)
    return 100.0 * edits / total


def wer(preds, refs):
    return 100.0 * jiwer.wer([norm(r) for r in refs], [norm(p) for p in preds])


def exact_match(preds, refs):
    return rate([norm(p) == norm(r) for p, r in zip(preds, refs)])


def rare_share(true_units, classes):
    """Share of test units that fall outside the classes, i.e. would be <rare> for models A-C."""
    known = set(classes)
    units = [u for word in true_units for u in word]
    return rate([norm(u) not in known for u in units])


def bootstrap_ci(metric, photo_ids, n_boot=1000, level=0.95, seed=42):
    """metric(idx) computes the score on the words at positions idx. Photos, not words, are resampled."""
    _, inverse = np.unique(np.asarray(photo_ids), return_inverse=True)
    order = np.argsort(inverse, kind="stable")  # word positions grouped by photo
    counts = np.bincount(inverse)
    starts = np.cumsum(counts) - counts
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(counts), len(counts))
        # all words of each picked photo, without a Python loop over photos
        lens = counts[picks]
        offsets = np.repeat(starts[picks] - (np.cumsum(lens) - lens), lens)
        scores.append(metric(order[offsets + np.arange(lens.sum())]))
    tail = (1 - level) / 2 * 100
    return float(np.percentile(scores, tail)), float(np.percentile(scores, 100 - tail))


def model_size_mb(path):
    return os.path.getsize(path) / 1e6


def ms_per_word(predict, images, warmup=3):
    for img in images[:warmup]:
        predict(img)
    start = time.perf_counter()
    for img in images:
        predict(img)
    return 1000 * (time.perf_counter() - start) / len(images)


if __name__ == "__main__":
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 0, 30, 10)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9

    true = [(0, 0, 10, 20), (12, 0, 22, 20)]
    assert segmentation_ok(true[::-1], 2, true)
    assert not segmentation_ok([(0, 0, 22, 20)], 2)
    assert not segmentation_ok([(0, 0, 4, 20), (5, 0, 22, 20)], 2, true)
    assert segmentation_ok([(0, 0, 4, 20), (5, 0, 22, 20)], 2)

    assert char_accuracy([["ಕ", "ನ", "ಡ"], ["a", "b"]], [["ಕ", "ನ್ನ", "ಡ"], ["a", "b"]]) == 80.0
    assert levenshtein("kitten", "sitting") == 3
    assert abs(cer(["ಕನಡ"], ["ಕನ್ನಡ"]) - 100 / 3) < 1e-9
    assert cer(["hotel"], ["Hotel"]) == 0.0
    assert exact_match(["SHOP"], ["shop"]) == 100.0
    assert rare_share([["ಕ", "ನ್ನ", "ಡ"], ["s"]], ["ಕ", "ಡ", "s", "<rare>"]) == 25.0
    assert cer([""], ["Shop"]) == 100.0
    assert wer(["Hotel", "Bakry"], ["Hotel", "Bakery"]) == 50.0
    assert exact_match(["ಅಂಗಡಿ", "Shop"], ["ಅಂಗಡಿ", "Shoe"]) == 50.0

    correct = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    photos = ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"]
    lo, hi = bootstrap_ci(lambda idx: rate(correct[idx]), photos, n_boot=1000)
    assert 0 <= lo <= rate(correct) <= hi <= 100
    assert bootstrap_ci(lambda idx: 7.0, photos, n_boot=50) == (7.0, 7.0)

    assert ms_per_word(lambda img: sum(range(1000)), list(range(20))) > 0
    print("metrics tests passed, bootstrap CI on toy data:", (round(lo, 1), round(hi, 1)))
