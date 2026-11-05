import argparse
import csv
import os
import random
import string
import unicodedata

import scipy.io
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from preprocess import dark_on_light, fit_square, preprocess, read_image, resize, resize_to_height

RARE = "<rare>"


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_lines(path):
    with open(path, encoding="utf-8-sig") as f:
        return [line.rstrip("\r\n") for line in f if line.strip()]


def load_charset(path):
    return [line.strip() for line in read_lines(path)]


def class_list(charset):
    # anything outside the charset is <rare>
    return list(charset) + [RARE]


def clean_label(text):
    # English is case-insensitive everywhere in this project; lower() leaves Kannada unchanged
    return unicodedata.normalize("NFC", text).strip().lower()


def codepoints(text):
    text = unicodedata.normalize("NFC", text)
    return [ch for ch in text if not ch.isspace() and ch not in "‌‍"]


def alphabet(charset):
    return sorted({ch for c in charset for ch in codepoints(c)})


def encodable(samples, alpha):
    keep = set(alpha)
    good = [s for s in samples if codepoints(s[1]) and set(codepoints(s[1])) <= keep]
    if len(good) < len(samples):
        print(f"dropped {len(samples) - len(good)} of {len(samples)} words with characters outside the charset")
    return good


def subset(samples, limit, seed=42):
    if limit and len(samples) > limit:
        samples = random.Random(seed).sample(samples, limit)
    return samples


def split_list(samples, split, train_frac, seed=42):
    samples = sorted(samples)
    random.Random(seed).shuffle(samples)
    n = round(train_frac * len(samples))
    return samples[:n] if split == "train" else samples[n:]


def read_labels(crops_dir):
    with open(os.path.join(crops_dir, "labels.tsv"), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def crop_samples(crops_dir, split, script):
    return [(os.path.join(crops_dir, r["path"]), clean_label(r["text"]))
            for r in read_labels(crops_dir) if r["split"] == split and r["script"] == script]


def crop_train_val(crops_dir, script, val_frac, seed=42):
    # validation photos come out of the train split only; the test split stays untouched
    rows = [r for r in read_labels(crops_dir) if r["split"] == "train" and r["script"] == script]
    photos = sorted({r["photo_id"] for r in rows})
    random.Random(seed).shuffle(photos)
    val_photos = set(photos[:max(1, round(val_frac * len(photos)))])
    pick = [(os.path.join(crops_dir, r["path"]), clean_label(r["text"]), r["photo_id"] in val_photos) for r in rows]
    return [(p, t) for p, t, v in pick if not v], [(p, t) for p, t, v in pick if v]


def load_gt(images_dir, gt_path):
    # IndicSTR12 / IIIT synthetic: "image_name<TAB>text" per line
    samples = []
    for line in read_lines(gt_path):
        name, text = line.split("\t", 1) if "\t" in line else line.split(None, 1)
        samples.append((os.path.join(images_dir, name.strip()), clean_label(text)))
    return samples


def load_mjsynth(root, annotation):
    # lines look like "./2697/6/466_MONIKER_49537.jpg 49537"; the word is in the file name
    samples = []
    for line in read_lines(os.path.join(root, annotation)):
        rel = line.split()[0]
        samples.append((os.path.normpath(os.path.join(root, rel)), clean_label(os.path.basename(rel).split("_")[1])))
    return samples


def load_iiit5k(root, split):
    records = scipy.io.loadmat(os.path.join(root, f"{split}data.mat"))[f"{split}data"][0]
    return [(os.path.join(root, str(r["ImgName"][0])), clean_label(str(r["GroundTruth"][0]))) for r in records]


def english_map():
    symbols = string.digits + string.ascii_uppercase + string.ascii_lowercase
    return {f"Sample{i + 1:03d}": ch for i, ch in enumerate(symbols)}


def read_map(path):
    return dict(line.split("\t", 1) for line in read_lines(path))


def load_class_folders(root, mapping):
    samples, unmapped = [], []
    for folder in sorted(os.listdir(root)):
        if folder not in mapping:
            unmapped.append(folder)
            continue
        for name in sorted(os.listdir(os.path.join(root, folder))):
            samples.append((os.path.join(root, folder, name), clean_label(mapping[folder])))
    if unmapped:
        print(f"no class text for {len(unmapped)} folders, e.g. {unmapped[:3]}")
    return samples


def load_public(name, split, cfg, limit=None):
    pub, seed = cfg["public"], cfg["seed"]
    if name in ("kannada_synth", "indicstr12_kannada"):
        samples = load_gt(pub[name][split]["images"], pub[name][split]["gt"])
    elif name == "mjsynth":
        samples = load_mjsynth(pub[name]["root"], pub[name][split])
    elif name == "iiit5k":
        samples = load_iiit5k(pub[name]["root"], split)
    elif name in ("chars74k_english", "chars74k_kannada"):
        # optional extra character data; rendered characters are the main training set for A-C
        map_path = pub["chars74k_kannada_map"]
        if not os.path.isdir(pub[name]):
            print(f"{name}: no folder {pub[name]}, skipping it")
            return []
        if name == "chars74k_kannada" and not os.path.exists(map_path):
            print(f"{name}: no class map {map_path}, skipping it")
            return []
        # Chars74K has no official split, so a fixed shuffle is cut with the same ratio as our photos
        mapping = english_map() if name == "chars74k_english" else read_map(map_path)
        samples = split_list(load_class_folders(pub[name], mapping), split, cfg["split"]["train"], seed)
    else:
        raise ValueError(f"unknown dataset {name}")
    return subset(samples, limit, seed)


class CharDataset(Dataset):
    def __init__(self, samples, classes, size=32, options=()):
        self.samples, self.size, self.options = samples, size, options
        self.index = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        gray = dark_on_light(preprocess(read_image(path), self.options))
        x = torch.from_numpy(fit_square(gray, self.size)).float().div(255).unsqueeze(0)
        return x, self.index.get(label, self.index[RARE])


class WordDataset(Dataset):
    def __init__(self, samples, alpha, height=32, max_width=512, options=()):
        self.samples, self.height, self.max_width, self.options = samples, height, max_width, options
        self.index = {ch: i + 1 for i, ch in enumerate(alpha)}  # 0 is the CTC blank

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, text = self.samples[i]
        gray = resize_to_height(dark_on_light(preprocess(read_image(path), self.options)), self.height)
        if gray.shape[1] > self.max_width:
            gray = resize(gray, self.max_width, self.height)
        x = torch.from_numpy(gray).float().div(255).unsqueeze(0)
        y = torch.tensor([self.index[ch] for ch in codepoints(text)], dtype=torch.long)
        return x, y


def collate_words(batch):
    width = max(x.shape[2] for x, _ in batch)
    images = torch.ones(len(batch), 1, batch[0][0].shape[1], width)  # pad with white
    for k, (x, _) in enumerate(batch):
        images[k, :, :, :x.shape[2]] = x
    targets = torch.cat([y for _, y in batch])
    target_lengths = torch.tensor([len(y) for _, y in batch])
    widths = torch.tensor([x.shape[2] for x, _ in batch])
    return images, targets, target_lengths, widths


def main():
    ap = argparse.ArgumentParser(description="load a dataset and print what was found")
    ap.add_argument("name", help="crops, kannada_synth, indicstr12_kannada, mjsynth, iiit5k, chars74k_english, chars74k_kannada")
    ap.add_argument("--split", default="train")
    ap.add_argument("--script", default="kannada", help="only used for crops")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.name == "crops":
        samples = subset(crop_samples(cfg["paths"]["crops"], args.split, args.script), args.limit)
        script = args.script
    else:
        samples = load_public(args.name, args.split, cfg, args.limit)
        script = "english" if args.name in ("mjsynth", "iiit5k", "chars74k_english") else "kannada"
    print(f"{len(samples)} samples, e.g. {samples[:3]}")
    if not samples:
        return
    charset = load_charset(cfg["charsets"][script])

    if args.name.startswith("chars74k"):
        classes = class_list(charset)
        rare = sum(1 for _, label in samples if label not in classes)
        print(f"{rare} samples fall outside the charset and would be {RARE}")
        x, y = next(iter(DataLoader(CharDataset(samples, classes, cfg["char_size"]), batch_size=8)))
        print("char batch", tuple(x.shape), "labels", [classes[k] for k in y])
    else:
        samples = encodable(samples, alphabet(charset))
        data = WordDataset(samples, alphabet(charset), cfg["img_height"])
        images, targets, lengths, widths = next(iter(DataLoader(data, batch_size=8, collate_fn=collate_words)))
        print("word batch", tuple(images.shape), "target lengths", lengths.tolist(), "widths", widths.tolist())


if __name__ == "__main__":
    main()
