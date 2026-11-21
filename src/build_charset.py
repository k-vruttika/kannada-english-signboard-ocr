import argparse
import os
from collections import Counter

from akshara import split_aksharas
from datasets import codepoints, load_config, per_class_for, read_labels, read_lines


def is_kannada(unit):
    return "ಀ" <= unit[0] <= "೿"


def count_aksharas(words):
    counts = Counter()
    for word in words:
        counts.update(u for u in split_aksharas(word) if is_kannada(u))
    return counts


def choose(counts, coverage):
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(counts.values())
    kept, covered = [], 0
    for unit, n in ranked:
        if covered / total >= coverage:
            break
        kept.append(unit)
        covered += n
    return kept, ranked[len(kept):], covered / total


def rf_memory_gb(n_classes, cfg):
    # rough: a tree has about n_train / min_samples_leaf nodes, and every node stores one float64 per class
    # plus a 64-byte node record; this agrees with the ~2 GB measured for the earlier 972-class charset
    per_class = per_class_for(cfg, "kannada") or cfg["render"]["per_class"]
    n_train = 0.9 * n_classes * per_class  # classical.py holds out a tenth for validation
    nodes = n_train / cfg["classical"]["rf_min_samples_leaf"]
    return cfg["classical"]["rf_trees"] * nodes * (8 * n_classes + 64) / 1e9


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=cfg["public"]["kannada_synth"]["vocabulary"])
    ap.add_argument("--crops", default=cfg["paths"]["crops"])
    ap.add_argument("--coverage", type=float, default=cfg["charset_coverage"])
    ap.add_argument("--out", default=cfg["charsets"]["kannada"])
    ap.add_argument("--alphabet-out", default=cfg["crnn_alphabets"]["kannada"])
    args = ap.parse_args()

    # one entry per vocabulary word, so this counts word types, not how often a word is printed
    vocab = [line.split()[0] for line in read_lines(args.vocab)]
    words = list(vocab)
    if os.path.exists(os.path.join(args.crops, "labels.tsv")):
        # train split only, so the test photos play no part in choosing the classes
        mine = [r["text"] for r in read_labels(args.crops) if r["split"] == "train" and r["script"] == "kannada"]
        words += mine
        print(f"{len(vocab)} vocabulary words + {len(mine)} train-split Kannada labels")
    else:
        print(f"{len(vocab)} vocabulary words, no labels.tsv found in {args.crops}")

    counts = count_aksharas(words)
    kept, left_out, reached = choose(counts, args.coverage)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")

    # the CRNN reads code points, not aksharas, so its alphabet is every Kannada code point seen,
    # independent of the coverage cut above
    alphabet = sorted({ch for word in words for ch in codepoints(word) if is_kannada(ch)})
    with open(args.alphabet_out, "w", encoding="utf-8") as f:
        f.write("\n".join(alphabet) + "\n")

    print(f"{len(counts)} distinct aksharas, kept {len(kept)} covering {100 * reached:.2f}% of occurrences -> {args.out}")
    print(f"{len(alphabet)} Kannada code points -> {args.alphabet_out} (CRNN alphabet)")
    c = cfg["classical"]
    print(f"\nclasses kept at other coverages, with a rough Kannada RF memory estimate "
          f"({per_class_for(cfg, 'kannada')} per class, {c['rf_trees']} trees, min_samples_leaf {c['rf_min_samples_leaf']}):")
    for cov in (0.95, 0.97, 0.98, 0.99):
        n = len(choose(counts, cov)[0])
        print(f"  {cov:.2f}: {n:5d} classes, RF about {rf_memory_gb(n, cfg):.1f} GB")
    print(f"\nleft out {len(left_out)} aksharas ({sum(n for _, n in left_out)} occurrences); most frequent of them:")
    for unit, n in left_out[:40]:
        print(f"  {unit}\t{n}")


if __name__ == "__main__":
    main()
