"""Model D on the laptop CPU: the same steps as notebooks/train_crnn.ipynb, without the Colab parts.

    python src/mjsynth_subset.py --url <mjsynth.tar.gz link> --n 60000
    python src/train_crnn.py --script english --max-hours 3

Pre-training writes models/crnn_<script>_pretrained.pth after every new best epoch, so an interrupted
run keeps its best weights. Once data/crops/labels.tsv holds reviewed words, running the script again
reuses that model and fine-tunes it into models/crnn_<script>.pth.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets import (WordDataset, alphabet, collate_words, crop_train_val, encodable, load_charset, load_config,
                      load_public, read_labels, split_list)
from models import CRNN, crnn_step, fit, load_model, save_model, set_seed

PUBLIC = {"english": "mjsynth", "kannada": "kannada_synth"}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Train model D (CRNN) on the CPU.")
    ap.add_argument("--script", required=True, choices=sorted(PUBLIC))
    ap.add_argument("--words", type=int, default=60000, help="public words used for pre-training")
    ap.add_argument("--max-hours", type=float, default=3, help="stop each stage after the epoch that passes this")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--restart", action="store_true", help="pre-train again even if a pre-trained model exists")
    ap.add_argument("--resume", action="store_true", help="continue pre-training from the saved pre-trained model")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    set_seed(cfg["seed"])

    t = cfg["train"]
    alpha = alphabet(load_charset(cfg["crnn_alphabets"][args.script]))
    models_dir = cfg["paths"]["models"]
    os.makedirs(models_dir, exist_ok=True)
    pretrained = os.path.join(models_dir, f"crnn_{args.script}_pretrained.pth")
    limit = args.max_hours * 3600

    def loader(samples, shuffle):
        data = WordDataset(samples, alpha, cfg["img_height"], t["crnn_max_width"])
        return DataLoader(data, batch_size=t["batch_size"], shuffle=shuffle, num_workers=args.workers,
                          collate_fn=collate_words, persistent_workers=args.workers > 0)

    step = lambda m, batch, decode: crnn_step(m, batch, "cpu", alpha, decode)

    if os.path.exists(pretrained) and not args.restart and not args.resume:
        model, info = load_model(pretrained)
        assert info["alphabet"] == alpha, "the alphabet changed since pre-training; rerun with --restart"
        print(f"reusing {pretrained}")
    else:
        words = encodable(load_public(PUBLIC[args.script], "train", cfg, args.words), alpha)
        pre_train = split_list(words, "train", 1 - t["pretrain_val"], cfg["seed"])
        pre_val = split_list(words, "test", 1 - t["pretrain_val"], cfg["seed"])
        print(f"{len(alpha)} symbols; pre-training on {len(pre_train)} words, validating on {len(pre_val)}")
        earlier = {}
        if args.resume:
            model, info = load_model(pretrained)
            assert info["alphabet"] == alpha, "the alphabet changed since pre-training; rerun with --restart"
            earlier = info.get("pretrain", {})
            print(f"continuing after {earlier.get('epochs', 0)} earlier epochs")
        else:
            model = CRNN(len(alpha))
            info = dict(kind="crnn", script=args.script, alphabet=alpha, hidden=256, height=cfg["img_height"],
                        max_width=t["crnn_max_width"], seed=cfg["seed"], pretrain_words=len(pre_train))

        def checkpoint(epoch, val_loss):
            done = {"epochs": earlier.get("epochs", 0) + epoch, "best_val_loss": val_loss,
                    "train_seconds": earlier.get("train_seconds", 0)}
            save_model(model, {**info, "pretrain": done}, pretrained)

        info["pretrain"] = fit(model, step, loader(pre_train, True), loader(pre_val, False), t["lr"], t["max_epochs"],
                               t["patience"], on_best=checkpoint, max_seconds=limit)
        info["pretrain"]["epochs"] += earlier.get("epochs", 0)
        info["pretrain"]["train_seconds"] += earlier.get("train_seconds", 0)
        save_model(model, info, pretrained)
        print(f"saved {pretrained}")

    labels = os.path.join(cfg["paths"]["crops"], "labels.tsv")
    reviewed = os.path.exists(labels) and any(r["script"] == args.script for r in read_labels(cfg["paths"]["crops"]))
    if not reviewed:
        print(f"no reviewed {args.script} words in {labels} yet, so fine-tuning is skipped")
        return
    mine_train, mine_val = crop_train_val(cfg["paths"]["crops"], args.script, t["finetune_val"], cfg["seed"])
    mine_train, mine_val = encodable(mine_train, alpha), encodable(mine_val, alpha)
    print(f"fine-tuning on {len(mine_train)} reviewed words, validating on {len(mine_val)}")
    info["finetune"] = fit(model, step, loader(mine_train, True), loader(mine_val, False), t["finetune_lr"],
                           t["max_epochs"], t["patience"], max_seconds=limit)
    info["finetune_words"] = len(mine_train)
    path = os.path.join(models_dir, f"crnn_{args.script}.pth")
    save_model(model, info, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
