import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import RARE
from metrics import cer, exact_match
from preprocess import dark_on_light, fit_square, preprocess, resize, resize_to_height
from segment import segment_word


def conv_bn(cin, cout, kernel=3, padding=1):
    return [nn.Conv2d(cin, cout, kernel, padding=padding), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CharCNN(nn.Module):
    """Model C: three conv blocks and a linear layer on 32x32 character crops."""

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(*conv_bn(1, 32), nn.MaxPool2d(2),
                                      *conv_bn(32, 64), nn.MaxPool2d(2),
                                      *conv_bn(64, 128), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 4 * 4, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class CRNN(nn.Module):
    """Model D: conv stack -> 2-layer BiLSTM -> linear, trained with CTC. Input height 32."""

    def __init__(self, n_symbols, hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(*conv_bn(1, 64), nn.MaxPool2d(2),                        # 16 x W/2
                                 *conv_bn(64, 128), nn.MaxPool2d(2),                      # 8 x W/4
                                 *conv_bn(128, 256), *conv_bn(256, 256), nn.MaxPool2d((2, 1)),  # 4 x W/4
                                 *conv_bn(256, 512), *conv_bn(512, 512), nn.MaxPool2d((2, 1)),  # 2 x W/4
                                 *conv_bn(512, 512, kernel=(2, 1), padding=0))            # 1 x W/4
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(2 * hidden, n_symbols + 1)  # +1 for the CTC blank at index 0

    def forward(self, x):
        features = self.cnn(x).squeeze(2).permute(2, 0, 1)  # (T, batch, 512)
        out, _ = self.rnn(features)
        return self.fc(out).log_softmax(2)

    @staticmethod
    def output_lengths(widths):
        return widths // 4


def greedy_decode(log_probs, lengths, alphabet):
    best_p, best = log_probs.exp().max(2)  # (T, batch)
    results = []
    for b in range(best.shape[1]):
        chars, confs, prev = [], [], 0
        for t in range(int(lengths[b])):
            k = int(best[t, b])
            if k != 0 and k != prev:
                chars.append(alphabet[k - 1])
                confs.append(float(best_p[t, b]))
            prev = k
        results.append(("".join(chars), float(np.mean(confs)) if confs else 0.0))
    return results


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cnn_step(model, batch, device, classes, decode):
    x, y = batch[0].to(device), batch[1].to(device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    if not decode:
        return loss, [], []
    return loss, [classes[i] for i in logits.argmax(1).tolist()], [classes[i] for i in y.tolist()]


def crnn_step(model, batch, device, alphabet, decode):
    images, targets, target_lengths, widths = batch
    log_probs = model(images.to(device))
    input_lengths = model.output_lengths(widths).clamp(max=log_probs.shape[0])
    # zero_infinity: a word with more code points than output frames gives no gradient instead of inf
    loss = F.ctc_loss(log_probs, targets.to(device), input_lengths, target_lengths, blank=0, zero_infinity=True)
    if not decode:
        return loss, [], []
    preds = [t for t, _ in greedy_decode(log_probs.detach().cpu(), input_lengths, alphabet)]
    refs = ["".join(alphabet[k - 1] for k in seq.tolist()) for seq in targets.split(target_lengths.tolist())]
    return loss, preds, refs


def run_epoch(model, step, loader, optimizer=None):
    total, n, preds, refs = 0.0, 0, [], []
    for batch in loader:
        loss, p, r = step(model, batch, optimizer is None)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * len(batch[0])
        n += len(batch[0])
        preds += p
        refs += r
    return total / n, preds, refs


def fit(model, step, train_loader, val_loader, lr, max_epochs, patience):
    """Adam with early stopping on validation loss; the best weights are restored at the end."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, bad, epoch = float("inf"), None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss, _, _ = run_epoch(model, step, train_loader, optimizer)
        model.eval()
        with torch.no_grad():
            val_loss, preds, refs = run_epoch(model, step, val_loader)
        print(f"epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
              f"val exact {exact_match(preds, refs):.1f}%, val CER {cer(preds, refs):.1f}%")
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop: no improvement for {patience} epochs")
                break
    model.load_state_dict(best_state)
    seconds = time.perf_counter() - start
    print(f"training time {seconds / 60:.1f} min, best val loss {best_loss:.4f}")
    return {"train_seconds": seconds, "epochs": epoch, "best_val_loss": best_loss}


def save_model(model, info, pth_path):
    torch.save(model.state_dict(), pth_path)
    with open(pth_path.replace(".pth", ".json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)


def load_model(pth_path):
    with open(pth_path.replace(".pth", ".json"), encoding="utf-8") as f:
        info = json.load(f)
    if info["kind"] == "cnn":
        model = CharCNN(len(info["classes"]))
    else:
        model = CRNN(len(info["alphabet"]), info["hidden"])
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    return model.eval(), info


def to_tensor(gray):
    return torch.from_numpy(gray).float().div(255).unsqueeze(0)


def cnn_recognise(model, info, img, options=(), seg_params=None):
    """Same output as classical.recognise: (text, unit labels, boxes, confidence)."""
    gray, boxes = segment_word(img, options, seg_params)
    if not boxes:
        return "", [], [], 0.0
    x = torch.stack([to_tensor(fit_square(gray[y0:y1, x0:x1], info["size"])) for x0, y0, x1, y1 in boxes])
    with torch.no_grad():
        conf, idx = model(x).softmax(1).max(1)
    units = [info["classes"][i] for i in idx.tolist()]
    return "".join("?" if u == RARE else u for u in units), units, boxes, float(conf.mean())


def crnn_recognise(model, info, img, options=()):
    gray = resize_to_height(dark_on_light(preprocess(img, options)), info["height"])
    if gray.shape[1] > info["max_width"]:
        gray = resize(gray, info["max_width"], info["height"])
    with torch.no_grad():
        log_probs = model(to_tensor(gray).unsqueeze(0))
    return greedy_decode(log_probs, [log_probs.shape[0]], info["alphabet"])[0]


if __name__ == "__main__":
    x = torch.rand(2, 1, 32, 32)
    assert CharCNN(10)(x).shape == (2, 10)
    crnn = CRNN(20)
    out = crnn(torch.rand(3, 1, 32, 100))
    assert out.shape == (25, 3, 21), out.shape
    assert int(CRNN.output_lengths(torch.tensor([100]))) == out.shape[0]
    # blank=0; "a a" separated by blank must decode to "aa", repeats without blank collapse
    lp = torch.full((6, 1, 3), -10.0)
    for t, k in enumerate([1, 1, 0, 1, 2, 2]):
        lp[t, 0, k] = 0.0
    assert greedy_decode(lp, [6], ["a", "b"])[0][0] == "aab"
    n_params = sum(p.numel() for p in crnn.parameters())
    print(f"models tests passed; CNN(500 classes) {sum(p.numel() for p in CharCNN(500).parameters()):,} params, "
          f"CRNN {n_params:,} params")
