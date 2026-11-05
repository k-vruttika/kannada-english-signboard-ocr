import argparse
import csv
import glob
import os
import random
import shutil
import sys

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from datasets import load_charset, load_config

SIZES = (28, 36, 44)


def check_raqm():
    if not features.check("raqm"):
        sys.exit("Pillow has no RAQM layout support, so Kannada vowel signs and ottaksharas would be placed wrongly.\n"
                 "In Colab run:  !apt-get -qq install -y libraqm-dev  and restart the session.")


def font_files(folder):
    return sorted(f for ext in ("ttf", "otf", "ttc") for f in glob.glob(os.path.join(folder, f"*.{ext}")))


def glyphs(path):
    return set(TTFont(path, fontNumber=0, lazy=True).getBestCmap())


def variants(cls, script):
    # English classes are lowercase but signs are mostly capitals, so both cases are drawn
    return sorted({cls, cls.upper()}) if script == "english" else [cls]


def render(text, font, rng, noise_rng):
    size = font.size
    img = Image.new("L", (size * (3 + len(text)), size * 4), 255)
    ImageDraw.Draw(img).text((size, size), text, font=font, fill=0)
    img = img.rotate(rng.uniform(-5, 5), resample=Image.BILINEAR, fillcolor=255)
    sx, sy = rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)
    img = img.resize((round(img.width * sx), round(img.height * sy)), Image.BILINEAR)
    box = img.point(lambda v: 255 if v < 128 else 0).getbbox()
    if box is None:
        return None
    img = img.crop((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.0)))
    arr = np.asarray(img, np.float32) + noise_rng.normal(0, rng.uniform(0, 10), (img.height, img.width))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, choices=["kannada", "english"])
    ap.add_argument("--out", default=cfg["paths"]["rendered"])
    ap.add_argument("--fonts", help="default: fonts/<script>")
    ap.add_argument("--per-class", type=int, default=cfg["render"]["per_class"])
    ap.add_argument("--zip", help="also write the output folder to this .zip")
    args = ap.parse_args()

    if args.script == "kannada":
        check_raqm()
    layout = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
    fonts = font_files(args.fonts or os.path.join(cfg["paths"]["fonts"], args.script))
    if not fonts:
        sys.exit(f"no .ttf/.otf/.ttc fonts in {args.fonts or os.path.join(cfg['paths']['fonts'], args.script)}")
    cmaps = {f: glyphs(f) for f in fonts}
    loaded = {(f, s): ImageFont.truetype(f, s, layout_engine=layout) for f in fonts for s in SIZES}
    rng, noise_rng = random.Random(cfg["seed"]), np.random.default_rng(cfg["seed"])

    out = os.path.join(args.out, args.script)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    rows, missing, used = [], [], set()
    for k, cls in enumerate(load_charset(cfg["charsets"][args.script])):
        texts = variants(cls, args.script)
        usable = [f for f in fonts if all(ord(ch) in cmaps[f] for t in texts for ch in t)]
        if not usable:
            missing.append(cls)
            continue
        for n in range(args.per_class):
            font_path = rng.choice(usable)
            img = render(rng.choice(texts), loaded[(font_path, rng.choice(SIZES))], rng, noise_rng)
            if img is None:
                continue
            name = f"images/{k:04d}_{n:03d}.png"  # ASCII names keep zips and Windows paths simple
            img.save(os.path.join(out, name))
            rows.append({"path": name, "text": cls, "font": os.path.basename(font_path)})
            used.add(font_path)

    with open(os.path.join(out, "labels.tsv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, ["path", "text", "font"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} images of {len({r['text'] for r in rows})} classes from {len(used)} fonts -> {out}")
    if missing:
        print(f"no font has the glyphs for {len(missing)} classes: {' '.join(missing[:30])}")
    if args.zip:
        shutil.make_archive(args.zip.removesuffix(".zip"), "zip", args.out, args.script)
        print("zipped to", args.zip)


if __name__ == "__main__":
    main()
