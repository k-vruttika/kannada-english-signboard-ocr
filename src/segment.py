import cv2
import numpy as np

from preprocess import binarize, dark_on_light, preprocess

# One plain segmenter for both scripts: Otsu, connected components, merge by x-overlap,
# split wide blobs at vertical-projection minima. Boxes are (x0, y0, x1, y1), left to right.


def x_overlap(a, b):
    inter = min(a[2], b[2]) - max(a[0], b[0])
    return inter / max(1, min(a[2] - a[0], b[2] - b[0]))


def union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def components(bw, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    parts = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area >= min_area:
            parts.append({"box": (int(x), int(y), int(x + w), int(y + h)), "ids": [k]})
    return labels, parts


def merge_overlapping(parts, min_overlap):
    parts = sorted(parts, key=lambda p: p["box"][0])
    changed = True
    while changed:
        changed = False
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                if x_overlap(parts[i]["box"], parts[j]["box"]) >= min_overlap:
                    parts[i] = {"box": union(parts[i]["box"], parts[j]["box"]), "ids": parts[i]["ids"] + parts[j]["ids"]}
                    del parts[j]
                    changed = True
                    break
            if changed:
                break
    return parts


def tight(mask, x0, x1):
    rows = np.flatnonzero(mask[:, x0:x1].any(axis=1))
    cols = np.flatnonzero(mask[:, x0:x1].any(axis=0))
    if len(rows) == 0:
        return None
    return (x0 + int(cols[0]), int(rows[0]), x0 + int(cols[-1]) + 1, int(rows[-1]) + 1)


def split_wide(mask, box, max_width, margin):
    x0, y0, x1, y1 = box
    w = x1 - x0
    # lo >= 1 so the cut never falls on the left edge (that would hand back the same box forever)
    lo, hi = max(1, int(margin * w)), int((1 - margin) * w)
    if w <= max_width or hi <= lo:
        return [box]
    cols = mask[y0:y1, x0:x1].sum(axis=0)
    cut = x0 + lo + int(np.argmin(cols[lo:hi]))
    pieces = []
    for a, b in ((x0, cut), (cut, x1)):
        piece = tight(mask, a, b)
        if piece == tuple(box):  # cut did not shrink anything, stop here
            pieces.append(piece)
        elif piece is not None:
            pieces += split_wide(mask, piece, max_width, margin)
    return pieces


def segment(gray, noise_frac=0.002, min_overlap=0.5, wide_ratio=1.5, split_margin=0.2):
    bw = binarize(gray)
    labels, parts = components(bw, noise_frac * gray.shape[0] ** 2)
    if not parts:
        return []
    parts = merge_overlapping(parts, min_overlap)
    text_h = np.median([p["box"][3] - p["box"][1] for p in parts])
    boxes = []
    for p in parts:
        mask = np.isin(labels, p["ids"])
        boxes += split_wide(mask, p["box"], wide_ratio * text_h, split_margin)
    return sorted(boxes, key=lambda b: b[0])


def segment_word(img, options=(), seg_params=None):
    """Preprocess a word crop and segment it; models A-C classify gray[y0:y1, x0:x1] for each box."""
    gray = dark_on_light(preprocess(img, options))
    return gray, segment(gray, **(seg_params or {}))


def draw_boxes(gray, boxes):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 1)
    return img


if __name__ == "__main__":
    from preprocess import synthetic_word

    assert len(segment(synthetic_word("SHOP"))) == 4
    assert len(segment(synthetic_word("SHOP", dark_text=False))) == 4
    assert len(segment(synthetic_word("Hill"))) == 4, "the dot of i should merge with its stem"
    noisy = synthetic_word("SHOP")
    noisy[5, 5] = noisy[50, -3] = 0
    assert len(segment(noisy)) == 4, "single-pixel noise should be dropped"

    # thicker strokes make neighbouring letters touch, which the splitter has to undo
    touching = 255 - cv2.dilate(255 - synthetic_word("MEDICALS"), np.ones((1, 5), np.uint8))
    n_blobs = cv2.connectedComponents(binarize(touching))[0] - 1
    boxes = segment(touching)
    assert n_blobs < 8 and len(boxes) == 8, (n_blobs, len(boxes))
    print(f"touching MEDICALS: {n_blobs} blobs before splitting, {len(boxes)} segments after")
    print("segment tests passed")
