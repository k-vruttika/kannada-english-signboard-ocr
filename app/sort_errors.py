"""Fill the category column of results/errors.csv, one error at a time.

Rows from models A-C whose segment count differs from the number of units are segmentation failures
by definition, so they are filled in on start. The rest are shown one by one with the word image, the
true text and the model's output.

Keys: 1-5 pick a category, Enter takes the suggestion, Backspace / PgUp goes back, PgDn skips.
Every choice is saved to errors.csv straight away, so the window can be closed at any time.
"""
import csv
import os
import sys
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_labels as rl  # noqa: E402  (also sets the working directory to the project root)
from preprocess import read_image  # noqa: E402

ERRORS = os.path.join(rl.cfg["paths"]["results"], "errors.csv")
CATEGORIES = ["segmentation failure", "confusable characters", "dropped vowel sign",
              "split/merged akshara", "stylised font"]
VIEW = (640, 200)


def load():
    with open(ERRORS, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def save(fields, rows):
    tmp = ERRORS + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, ERRORS)


def certain_segmentation_failure(r):
    return r["n_segments"] != "" and int(r["n_segments"]) != int(r["n_units"])


def main():
    fields, rows = load()
    auto = 0
    for r in rows:
        if certain_segmentation_failure(r):
            r["category"] = r["category"] or "segmentation failure"
            auto += 1
    save(fields, rows)
    todo = [i for i, r in enumerate(rows) if not certain_segmentation_failure(r)]
    font_path = rl.find_kannada_font()
    rl.KANNADA_FONT = font_path

    root = tk.Tk()
    root.title("Sort errors")
    root.geometry("760x640")
    big = ("TkDefaultFont", 14)
    state = {"pos": next((k for k, i in enumerate(todo) if not rows[i]["category"]), 0)}
    keep = {}  # PhotoImage references, or Tk drops the images

    info = ttk.Label(root, font=big, padding=(10, 8))
    info.pack(anchor="w")
    word = ttk.Label(root, padding=10)
    word.pack()
    grid = ttk.Frame(root, padding=10)
    grid.pack(fill="x")
    labels = {}
    for row, name in enumerate(("Correct", "Model read")):
        ttk.Label(grid, text=name + ":", font=big, width=11).grid(row=row, column=0, sticky="w")
        labels[name] = (ttk.Label(grid, font=big), ttk.Label(grid))
        labels[name][0].grid(row=row, column=1, sticky="w")
        labels[name][1].grid(row=row, column=2, sticky="w", padx=10)
    note = ttk.Label(root, padding=(10, 0), foreground="#555555")
    note.pack(anchor="w")
    buttons = ttk.Frame(root, padding=10)
    buttons.pack(fill="x")
    progress = ttk.Label(root, padding=10)
    progress.pack(anchor="w")
    ttk.Label(root, padding=(10, 0), foreground="#555555",
              text="1-5 choose   Enter take the suggestion   Backspace / PgUp back   PgDn skip").pack(anchor="w")

    def show_text(name, text):
        plain, drawn = labels[name]
        plain.config(text=text or "(nothing)")
        img = rl.render_text(text) if rl.has_kannada(text or "") else None
        if img is not None:
            keep[name] = ImageTk.PhotoImage(img)
            drawn.config(image=keep[name])
        else:
            drawn.config(image="")

    def show():
        done = sum(1 for i in todo if rows[i]["category"])
        progress.config(text=f"{done} of {len(todo)} checked by hand  "
                             f"({auto} segmentation failures filled in automatically)")
        if state["pos"] >= len(todo):
            info.config(text="All done. You can close this window.")
            return
        r = rows[todo[state["pos"]]]
        info.config(text=f"{state['pos'] + 1}/{len(todo)}   model {r['model']}   {r['script']}   test set: {r['test_set']}")
        try:
            img = Image.fromarray(read_image(r["path"])[:, :, ::-1])
            scale = min(VIEW[0] / img.width, VIEW[1] / img.height, 4)
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
            keep["word"] = ImageTk.PhotoImage(img)
            word.config(image=keep["word"], text="")
        except Exception as e:  # a missing public test image should not stop the sorting
            word.config(image="", text=f"cannot open {r['path']}: {e}")
        show_text("Correct", r["ref"])
        show_text("Model read", r["pred"])
        segs = f"{r['n_segments']} pieces for {r['n_units']} units" if r["n_segments"] else f"{r['n_units']} units"
        note.config(text=f"{segs}   suggestion: {r['suggested_category']}"
                         + (f"   current: {r['category']}" if r["category"] else ""))
        for k, b in enumerate(button_list):
            b.state(["pressed"] if CATEGORIES[k] == (r["category"] or r["suggested_category"]) else ["!pressed"])

    def choose(category):
        if state["pos"] >= len(todo):
            return
        rows[todo[state["pos"]]]["category"] = category
        save(fields, rows)
        state["pos"] += 1
        show()

    def move(step):
        state["pos"] = max(0, min(len(todo), state["pos"] + step))
        show()

    button_list = []
    for k, c in enumerate(CATEGORIES):
        b = ttk.Button(buttons, text=f"{k + 1}  {c}", command=lambda c=c: choose(c))
        b.pack(fill="x", pady=2)
        button_list.append(b)
        root.bind(str(k + 1), lambda e, c=c: choose(c))
        root.bind(f"<KP_{k + 1}>", lambda e, c=c: choose(c))
    for seq in ("<Return>", "<KP_Enter>"):
        root.bind(seq, lambda e: state["pos"] < len(todo) and choose(rows[todo[state["pos"]]]["suggested_category"]))
    for seq in ("<BackSpace>", "<Prior>"):
        root.bind(seq, lambda e: move(-1))
    root.bind("<Next>", lambda e: move(1))

    show()
    root.mainloop()


if __name__ == "__main__":
    main()
