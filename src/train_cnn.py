"""Model C on the laptop CPU: the same steps as notebooks/train_cnn.ipynb, without the Colab parts.

    python src/train_cnn.py --script english

The English CNN trains in a few minutes on a CPU. The Kannada one has ~860 classes and is faster on the
Colab GPU, but works here too. It prints the same fingerprint as classical.py, so A, B and C can be
checked to have seen the same characters.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from classical import char_split, fingerprint
from datasets import CharDataset, class_list, load_charset, load_config, per_class_for
from models import CharCNN, cnn_step, fit, save_model, set_seed


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Train model C (CNN) on the CPU.")
    ap.add_argument("--script", required=True, choices=["kannada", "english"])
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    set_seed(cfg["seed"])
    train_set, val_set = char_split(cfg, args.script, per_class_for(cfg, args.script))
    classes = class_list(load_charset(cfg["charsets"][args.script]))
    t = cfg["train"]
    # loading in the main process: worker processes would re-import this script on Windows and Python 3.14
    train_loader = DataLoader(CharDataset(train_set, classes, cfg["char_size"]), batch_size=t["batch_size"], shuffle=True)
    val_loader = DataLoader(CharDataset(val_set, classes, cfg["char_size"]), batch_size=t["batch_size"])

    model = CharCNN(len(classes))
    step = lambda m, batch, decode: cnn_step(m, batch, "cpu", classes, decode)
    result = fit(model, step, train_loader, val_loader, t["lr"], t["max_epochs"], t["patience"])

    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    path = os.path.join(cfg["paths"]["models"], f"cnn_{args.script}.pth")
    info = dict(kind="cnn", script=args.script, classes=classes, size=cfg["char_size"], seed=cfg["seed"],
                char_per_class=per_class_for(cfg, args.script), n_train=len(train_set),
                fingerprint=fingerprint(val_set + train_set), **result)
    save_model(model, info, path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"saved {path}: {os.path.getsize(path) / 1e6:.1f} MB, {n_params:,} parameters")


if __name__ == "__main__":
    main()
