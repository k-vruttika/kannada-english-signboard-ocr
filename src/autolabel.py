"""First-pass word labels for the shop-sign photos, to be checked by hand in review_labels.py.

Kannada words come from Google Cloud Vision (GCV) and English words from PaddleOCR, as the internship
guide advised. These labels are only a starting point: a word counts as ground truth once it has been
accepted or corrected in the review tool, and only those words reach data/crops/labels.tsv.

Run it with the labelling environment (.venv-label, Python 3.13), because PaddlePaddle has no wheels for
newer Pythons:

    .venv-label/bin/python src/autolabel.py

GCV needs an API key in the environment variable GCV_API_KEY (or a service-account JSON file in
GOOGLE_APPLICATION_CREDENTIALS). Raw OCR output is cached per photo in data/ocr_cache/, so a rerun
never pays for the same photo twice.
"""
import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

import cv2
import yaml

from akshara import split_aksharas
from make_crops import has_kannada, split_photos
from preprocess import read_image

AUTO_FIELDS = ["path", "text", "script", "photo_id", "split", "box", "source", "conf", "char_boxes"]
GCV_URL = "https://vision.googleapis.com/v1/images:annotate"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ---------------------------------------------------------------- Google Cloud Vision (Kannada)

def gcv_auth():
    key = os.environ.get("GCV_API_KEY", "").strip()
    if key:
        return {"params": f"?key={key}", "headers": {}}
    creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_file and os.path.exists(creds_file):
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            creds_file, scopes=["https://www.googleapis.com/auth/cloud-vision"])
        creds.refresh(Request())
        return {"params": "", "headers": {"Authorization": f"Bearer {creds.token}"}}
    return None


def gcv_annotate(img, auth):
    # the photo is sent as decoded by OpenCV (EXIF rotation applied), so GCV's coordinates match ours
    content = base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()).decode()
    body = json.dumps({"requests": [{
        "image": {"content": content},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        "imageContext": {"languageHints": ["kn", "en"]}}]}).encode()
    for attempt in range(5):
        req = urllib.request.Request(GCV_URL + auth["params"], data=body, method="POST",
                                     headers={"Content-Type": "application/json", **auth["headers"]})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.load(resp)["responses"][0]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(f"GCV request failed ({e.code}): {e.read().decode(errors='replace')[:500]}")
        if "error" in result:
            raise SystemExit(f"GCV error: {result['error']}")
        return result
    raise SystemExit("GCV kept failing; try again later.")


def poly_box(poly):
    xs = [v.get("x", 0) for v in poly["vertices"]]
    ys = [v.get("y", 0) for v in poly["vertices"]]
    return [min(xs), min(ys), max(xs), max(ys)]


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def is_kannada_char(ch):
    return "ಀ" <= ch <= "೿"


def akshara_boxes(symbols):
    """GCV symbols (text, box) of one word -> one box per akshara, or None if they don't line up."""
    text = "".join(t for t, _ in symbols)
    units = split_aksharas(text)
    boxes, i = [], 0
    for unit in units:
        acc, members = "", []
        while i < len(symbols) and len(acc) < len(unit):
            acc += symbols[i][0]
            members.append(symbols[i][1])
            i += 1
        if unicodedata.normalize("NFC", acc) != unit:
            return None
        boxes.append(union(members))
    return boxes if i == len(symbols) else None


def gcv_kannada_words(result):
    words = []
    for page in result.get("fullTextAnnotation", {}).get("pages", []):
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    # keep the Kannada symbols only: stray punctuation or Latin letters glued to a word drop out
                    symbols = [(unicodedata.normalize("NFC", s["text"]), poly_box(s["boundingBox"]))
                               for s in word.get("symbols", [])]
                    symbols = [(t, b) for t, b in symbols if t and all(is_kannada_char(c) for c in t)]
                    text = unicodedata.normalize("NFC", "".join(t for t, _ in symbols))
                    if not text or not split_aksharas(text):
                        continue
                    words.append({"text": text, "box": union([b for _, b in symbols]),
                                  "conf": round(word.get("confidence", 0.0), 3),
                                  "char_boxes": akshara_boxes(symbols), "source": "gcv"})
    return words


# ---------------------------------------------------------------- PaddleOCR (English)

_paddle = None


def paddle_lines(img):
    global _paddle
    if _paddle is None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR
        _paddle = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
                            use_textline_orientation=False)
    r = _paddle.predict(img, return_word_box=True)[0]
    lines = []
    for text, score, words, regions in zip(r["rec_texts"], r["rec_scores"], r["text_word"], r["text_word_region"]):
        lines.append({"text": text, "score": round(float(score), 3),
                      "words": [{"text": w, "box": union([[int(x), int(y), int(x), int(y)] for x, y in poly])}
                                for w, poly in zip(words, regions)]})
    return lines


def paddle_english_words(lines, min_score):
    words = []
    for line in lines:
        if line["score"] < min_score:
            continue
        for w in line["words"]:
            # the English charset is a-z, 0-9 and &; punctuation tokens and edges are dropped
            text = re.sub(r"^[^A-Za-z0-9&]+|[^A-Za-z0-9&]+$", "", w["text"])
            if not text or not text.isascii():
                continue
            words.append({"text": text, "box": w["box"], "conf": line["score"], "char_boxes": None,
                          "source": "paddle"})
    return words


# ---------------------------------------------------------------- merging and cropping

def overlap_share(a, b):
    """Share of box a covered by box b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return max(0, w) * max(0, h) / area if area > 0 else 0.0


def padded(box, img_w, img_h, pad):
    x0, y0, x1, y1 = box
    p = round(pad * (y1 - y0))
    return [max(0, x0 - p), max(0, y0 - p), min(img_w, x1 + p), min(img_h, y1 + p)]


def cached(path, compute):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    data = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    return data


def label_photo(path, args, auth):
    photo_id = os.path.splitext(os.path.basename(path))[0]
    img = read_image(path)
    h, w = img.shape[:2]
    kannada, english = [], []
    gcv_file = os.path.join(args.cache, "gcv", photo_id + ".json")
    if auth or os.path.exists(gcv_file):
        kannada = gcv_kannada_words(cached(gcv_file, lambda: gcv_annotate(img, auth)))
    if not args.no_paddle:
        lines = cached(os.path.join(args.cache, "paddle", photo_id + ".json"), lambda: paddle_lines(img))
        english = paddle_english_words(lines, args.min_score)
        # Paddle's English model reads Kannada letters as junk; drop whatever sits on a Kannada word
        english = [e for e in english if not any(overlap_share(e["box"], k["box"]) > 0.3 for k in kannada)]
    words = []
    for word, script in [(k, "kannada") for k in kannada] + [(e, "english") for e in english]:
        x0, y0, x1, y1 = word["box"]
        if y1 - y0 < args.min_height or x1 - x0 < 4:
            continue
        if script == "kannada" and not has_kannada(word["text"]):
            continue
        crop_box = padded(word["box"], w, h, args.pad)
        # OCR boxes can reach past the photo edge; skip any that are empty once clipped to it
        if crop_box[2] - crop_box[0] < 4 or crop_box[3] - crop_box[1] < args.min_height:
            continue
        rel = None
        if word["char_boxes"]:
            rel = [[b[0] - crop_box[0], b[1] - crop_box[1], b[2] - crop_box[0], b[3] - crop_box[1]]
                   for b in sorted(word["char_boxes"], key=lambda b: b[0])]
        words.append({"text": word["text"], "script": script, "photo_id": photo_id, "box": crop_box,
                      "source": word["source"], "conf": word["conf"], "char_boxes": rel})
    words.sort(key=lambda r: (r["box"][1], r["box"][0]))
    for k, r in enumerate(words):
        r["path"] = f"{photo_id}_{k:03d}.png"
        x0, y0, x1, y1 = r["box"]
        cv2.imencode(".png", img[y0:y1, x0:x1])[1].tofile(os.path.join(args.out, r["path"]))
    return words


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--photos", default=cfg["paths"]["photos"])
    ap.add_argument("--out", default=cfg["paths"]["crops"])
    ap.add_argument("--cache", default="data/ocr_cache")
    ap.add_argument("--limit", type=int, help="only the first N photos (for a quick test)")
    ap.add_argument("--min-score", type=float, default=0.6, help="lowest PaddleOCR line score kept")
    ap.add_argument("--min-height", type=int, default=12, help="smallest word height in pixels")
    ap.add_argument("--pad", type=float, default=0.12, help="margin around each word, as a share of its height")
    ap.add_argument("--no-paddle", action="store_true", help="skip English (PaddleOCR)")
    ap.add_argument("--force", action="store_true", help="overwrite even if a review is already in progress")
    args = ap.parse_args()

    review_file = os.path.join(args.out, "review.json")
    if os.path.exists(review_file) and not args.force:
        sys.exit(f"{review_file} exists, so a review is in progress and new crops would not match it.\n"
                 f"Rerun with --force to start over (your review decisions would be lost).")
    auth = gcv_auth()
    if auth is None:
        print("No GCV credentials (GCV_API_KEY or GOOGLE_APPLICATION_CREDENTIALS): Kannada words come only "
              "from photos already in the GCV cache.")

    photos = sorted(f for f in os.listdir(args.photos) if f.lower().endswith(IMAGE_EXT))[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.out):
        if f.endswith(".png"):
            os.remove(os.path.join(args.out, f))

    rows, start = [], time.time()
    for i, name in enumerate(photos, 1):
        try:
            rows += label_photo(os.path.join(args.photos, name), args, auth)
        except ValueError as e:
            print(f"  skipped {name}: {e}")
        if i % 10 == 0 or i == len(photos):
            print(f"{i}/{len(photos)} photos, {len(rows)} words, {time.time() - start:.0f} s", flush=True)

    # the split is fixed here, by photo, before anyone looks at the labels
    splits = split_photos([os.path.splitext(n)[0] for n in photos], cfg["split"]["train"], cfg["seed"])
    with open(os.path.join(args.out, "auto_labels.tsv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, AUTO_FIELDS, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "split": splits[r["photo_id"]], "box": json.dumps(r["box"]),
                             "char_boxes": json.dumps(r["char_boxes"]) if r["char_boxes"] else ""})
    per_script = {s: sum(r["script"] == s for r in rows) for s in ("kannada", "english")}
    print(f"{len(rows)} candidate words {per_script} from {len(photos)} photos in {args.out}/auto_labels.tsv")
    print("Next: check them with  python app/review_labels.py")


if __name__ == "__main__":
    main()
