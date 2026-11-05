import argparse
import os
import shutil
import zipfile

from datasets import load_config, load_gt, subset


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="copy a random subset of an 'image<TAB>text' dataset into a new folder")
    ap.add_argument("--images", required=True, help="image folder; with --zip, the folder inside the zip")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--vocab", help="vocabulary.txt to copy along, for build_charset.py")
    ap.add_argument("--zip", help="read the chosen images straight from this zip instead of an unpacked folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200000)
    args = ap.parse_args()

    samples = subset(load_gt(args.images, args.gt), args.n, cfg["seed"])
    # only the chosen images are extracted, so the other ~1.8 million never touch the disk
    archive = zipfile.ZipFile(args.zip) if args.zip else None
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    lines = []
    for i, (path, text) in enumerate(samples):
        name = f"{i:07d}{os.path.splitext(path)[1]}"
        target = os.path.join(args.out, "images", name)
        if archive:
            with archive.open(path.replace(os.sep, "/")) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(path, target)
        lines.append(f"{name}\t{text}")
        if (i + 1) % 20000 == 0:
            print(f"{i + 1} of {len(samples)} images copied")
    with open(os.path.join(args.out, "train_gt.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if args.vocab:
        shutil.copyfile(args.vocab, os.path.join(args.out, "vocabulary.txt"))
    print(f"copied {len(lines)} words ({len({t for _, t in samples})} unique) to {args.out}")


if __name__ == "__main__":
    main()
