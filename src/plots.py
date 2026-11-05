import csv
import os
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

COLOURS = {"kannada": "#2a78d6", "english": "#eb6834"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e6e5e0"
TOP_CLASSES = {"kannada": 30, "english": 20}


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK)
    ax.set_ylabel(ylabel, color=INK)


def grouped_bars(rows, group_key, groups, title, metric, path):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    x, width = np.arange(len(groups)), 0.38
    scripts = [s for s in COLOURS if any(r["script"] == s for r in rows)]
    for k, script in enumerate(scripts):
        found = {r[group_key]: r for r in rows if r["script"] == script}
        vals = np.array([found[g][metric] if g in found else np.nan for g in groups], dtype=float)
        low = np.array([found[g][metric + "_lo"] if g in found else np.nan for g in groups], dtype=float)
        high = np.array([found[g][metric + "_hi"] if g in found else np.nan for g in groups], dtype=float)
        bars = ax.bar(x + (k - (len(scripts) - 1) / 2) * (width + 0.02), vals, width, color=COLOURS[script],
                      label=script, yerr=[vals - low, high - vals], error_kw=dict(ecolor=INK, elinewidth=1, capsize=3))
        # values are printed so that a 0% bar is not mistaken for a missing one
        ax.bar_label(bars, labels=["" if np.isnan(v) else f"{v:.0f}" for v in vals], padding=2, fontsize=8, color=INK)
    ax.set_xticks(x, groups)
    style(ax, title, f"{metric.replace('_', ' ')} (%), 95% CI")
    top = np.nanmax([r.get(metric + "_hi", np.nan) for r in rows] + [100])  # CER can pass 100% with insertions
    ax.set_ylim(0, top * 1.08)
    ax.legend(frameon=False, ncol=len(scripts), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def confusion(pairs, script, title, path):
    counts = Counter(t for t, _ in pairs)
    labels = sorted(counts, key=lambda c: (-counts[c], c))[:TOP_CLASSES[script]]
    index = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels) + 1))
    for t, p in pairs:
        if t in index:
            m[index[t], index.get(p, len(labels))] += 1
    m = m / m.sum(axis=1, keepdims=True)
    # matplotlib cannot shape Kannada, so Kannada classes are numbered; the CSV next to the chart has the text
    ticks = labels if script == "english" else [str(i) for i in range(len(labels))]
    with open(path.replace(".png", ".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "true class", "count"] + labels + ["other"])
        for i, c in enumerate(labels):
            w.writerow([i, c, counts[c]] + [round(v, 3) for v in m[i]])
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
    size = 6.5 if len(labels) <= 20 else 8.5
    fig, ax = plt.subplots(figsize=(size + 1, size), facecolor=SURFACE)
    image = ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels) + 1), ticks + ["other"], fontsize=8, rotation=90)
    ax.set_yticks(range(len(labels)), ticks, fontsize=8)
    ax.set_xlabel("predicted", color=INK)
    ax.set_ylabel("true", color=INK)
    ax.set_title(title, loc="left", color=INK)
    fig.colorbar(image, ax=ax, label="share of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def top_confusions(confusions, out, n=20):
    """results/top_confusions_<script>.csv: the n most frequent wrong (true, predicted) pairs per model."""
    for script in sorted({s for _, s in confusions}):
        rows = []
        for (model, s), pairs in sorted(confusions.items()):
            if s == script:
                wrong = Counter((t, p) for t, p in pairs if t != p)
                rows += [[model, t, p, c] for (t, p), c in wrong.most_common(n)]
        # utf-8-sig so that Excel shows the Kannada text instead of mojibake
        with open(os.path.join(out, f"top_confusions_{script}.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "true", "predicted", "count"])
            w.writerows(rows)


def charts(rows, confusions, out, settings, model_names):
    # bar charts use my test set; the public sets only when mine has not been evaluated
    which = "my" if any(r["test_set"] == "mine" for r in rows) else "public"
    rows = [r for r in rows if (r["test_set"] == "mine") == (which == "my")]
    seg = [r for r in rows if "seg_success" in r]
    if seg:
        # A-C share the segmenter, so the first segmentation model of each script stands for all three
        first = {}
        for r in seg:
            first.setdefault(r["script"], r["model"])
        grouped_bars([r for r in seg if r["model"] == first[r["script"]]], "preprocessing", settings,
                     f"Segmentation success by script ({which} test set)", "seg_success",
                     os.path.join(out, "seg_success_by_script.png"))
    plain = [r for r in rows if r["preprocessing"] == "none"]
    if plain:
        grouped_bars(plain, "model", model_names, f"CER by model and script ({which} test set, no preprocessing)",
                     "cer", os.path.join(out, "cer_by_model_script.png"))
    for (model, script), pairs in confusions.items():
        if pairs:
            confusion(pairs, script, f"{model} {script}: confusions on correctly segmented words",
                      os.path.join(out, f"confusion_{model}_{script}.png"))
    top_confusions(confusions, out)
