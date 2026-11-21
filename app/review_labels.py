"""Check the automatic labels from src/autolabel.py, one word at a time.

    python app/review_labels.py

Enter accepts the word as shown (after any correction), Ctrl+D deletes it. Only accepted words go to
data/crops/labels.tsv, which is what training and evaluation read, so every label there has been seen
and approved by a person. Decisions are saved after every word in data/crops/review.json; close the
window at any time and carry on later.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tkinter as tk
from tkinter import font, messagebox, ttk

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from akshara import n_units  # noqa: E402
from datasets import load_config  # noqa: E402
from make_crops import FIELDS, has_kannada  # noqa: E402
from preprocess import read_image  # noqa: E402

PHOTO_VIEW = (560, 430)
CROP_VIEW = (640, 230)
GOAL_TEST = 100      # reviewed test-split words per script: the results are measured on these
GOAL_BOXES = 50      # test-split words per script with checked char boxes, for segmentation against ground truth
ENGLISH_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789&")
HELP = ("Enter accept   Ctrl+D delete   PgUp/PgDn previous/next   Ctrl+U next unchecked   Ctrl+G grow crop   "
        "Ctrl+R reset crop   Ctrl+B char boxes checked   Ctrl+X clear char boxes   "
        "Drag on the crop to draw a char box, right-click one to remove it")

cfg = load_config()
CROPS = cfg["paths"]["crops"]
AUTO_FILE = os.path.join(CROPS, "auto_labels.tsv")
REVIEW_FILE = os.path.join(CROPS, "review.json")
LABELS_FILE = os.path.join(CROPS, "labels.tsv")


class State:
    rows = []          # auto_labels.tsv, in photo order
    review = {}        # path -> decision
    queue = []         # indices of rows in review order, spread over the photos
    order = []         # the part of queue that passes the filter
    pos = 0            # position in order
    boxes = []         # char boxes of the current word, relative to its crop
    box = None         # current crop box in the photo
    scale = 1.0        # crop view zoom
    drag = None
    photo_cache = {}
    photo_files = {}


S = State()


# ---------------------------------------------------------------- files

def write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def write_labels():
    out = []
    for r in S.rows:
        d = S.review.get(r["path"])
        if not d or d["status"] != "accepted":
            continue
        boxes = d["char_boxes"] if d.get("boxes_ok") and d.get("char_boxes") else None
        out.append({"path": r["path"], "text": d["text"], "script": d["script"], "photo_id": r["photo_id"],
                    "n_units": n_units(d["text"]), "split": r["split"],
                    "char_boxes": json.dumps(boxes) if boxes else ""})
    tmp = LABELS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(out)
    os.replace(tmp, LABELS_FILE)


def review_queue(rows, seed):
    """Test-split words first, because the results are measured on them; within each split one Kannada
    and one English word from every photo, then the next of each, so that a few hundred reviewed words
    cover many different signs instead of the first few photos."""
    photos = sorted({r["photo_id"] for r in rows})
    # a different seed from the train/test split, which shuffles the same sorted list
    random.Random(seed + 1).shuffle(photos)
    photo_pos = {p: i for i, p in enumerate(photos)}
    seen, keys = {}, []
    for i, r in enumerate(rows):
        group = (r["photo_id"], r["script"])
        seen[group] = seen.get(group, 0) + 1
        keys.append((r["split"] != "test", seen[group], photo_pos[r["photo_id"]], r["script"], i))
    return [k[-1] for k in sorted(keys)]


def load():
    if not os.path.exists(AUTO_FILE):
        messagebox.showerror("Review labels", f"{AUTO_FILE} not found.\nRun src/autolabel.py first (see README).")
        sys.exit(1)
    with open(AUTO_FILE, encoding="utf-8", newline="") as f:
        S.rows = list(csv.DictReader(f, delimiter="\t"))
    for r in S.rows:
        r["box"] = json.loads(r["box"])
        r["char_boxes"] = json.loads(r["char_boxes"]) if r["char_boxes"] else []
    if os.path.exists(REVIEW_FILE):
        S.review = json.load(open(REVIEW_FILE, encoding="utf-8"))
    photos = cfg["paths"]["photos"]
    S.photo_files = {os.path.splitext(f)[0]: os.path.join(photos, f) for f in os.listdir(photos)}
    S.queue = review_queue(S.rows, cfg["seed"])
    write_labels()


# ---------------------------------------------------------------- Kannada rendering

def find_kannada_font():
    candidates = [os.path.join("fonts", "kannada", f) for f in sorted(os.listdir(os.path.join("fonts", "kannada")))
                  ] if os.path.isdir(os.path.join("fonts", "kannada")) else []
    candidates += [r"C:\Windows\Fonts\Nirmala.ttc", r"C:\Windows\Fonts\tunga.ttf",
                   "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf"]
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans Kannada"], capture_output=True, text=True, timeout=5)
        candidates.append(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return next((c for c in candidates if c and os.path.exists(c) and c.lower().endswith((".ttf", ".otf", ".ttc"))), None)


KANNADA_FONT = None


def render_text(text):
    """Tk cannot shape Kannada conjuncts on every system, so the label is also drawn with Pillow + raqm."""
    if not KANNADA_FONT or not text:
        return None
    try:
        f = ImageFont.truetype(KANNADA_FONT, 34, layout_engine=ImageFont.Layout.RAQM)
    except (OSError, KeyError, ImportError):
        f = ImageFont.truetype(KANNADA_FONT, 34)
    x0, y0, x1, y1 = f.getbbox(text)
    img = Image.new("RGB", (max(1, x1 - x0) + 20, max(1, y1 - y0) + 16), "white")
    ImageDraw.Draw(img).text((10 - x0, 8 - y0), text, font=f, fill="black")
    return img


# ---------------------------------------------------------------- current word

def current():
    return S.rows[S.order[S.pos]] if S.order else None


def decision(r):
    return S.review.get(r["path"], {})


def photo(photo_id):
    if photo_id not in S.photo_cache:
        S.photo_cache = {photo_id: read_image(S.photo_files[photo_id])}  # keep one photo in memory
    return S.photo_cache[photo_id]


def crop_image():
    r = current()
    x0, y0, x1, y1 = S.box
    if S.box == r["box"]:
        img = read_image(os.path.join(CROPS, r["path"]))
    else:
        img = photo(r["photo_id"])[y0:y1, x0:x1]
    return img


def apply_filter(keep_path=None):
    status_f, script_f = ui.filter.get(), ui.script_filter.get()
    S.order = []
    for i in S.queue:
        r = S.rows[i]
        d = decision(r)
        st = d.get("status", "unchecked")
        script = d.get("script", r["script"])
        if status_f != "all" and st != status_f:
            continue
        if script_f != "both" and script != script_f:
            continue
        S.order.append(i)
    S.pos = 0
    if keep_path:
        idx = [S.rows[i]["path"] for i in S.order]
        if keep_path in idx:
            S.pos = idx.index(keep_path)
    show()


def show():
    r = current()
    update_progress()
    if r is None:
        ui.info.configure(text="No words match this filter.")
        ui.photo_canvas.delete("all")
        ui.crop_canvas.delete("all")
        ui.text.set("")
        return
    d = decision(r)
    ui.text.set(d.get("text", r["text"]))
    S.box = list(d.get("box", r["box"]))
    S.boxes = [list(b) for b in d.get("char_boxes", r["char_boxes"]) or []]
    ui.boxes_ok.set(bool(d.get("boxes_ok", False)))
    status = d.get("status", "unchecked")
    same_photo = [x for x in S.rows if x["photo_id"] == r["photo_id"]]
    ui.info.configure(text=f"{r['photo_id']}   word {same_photo.index(r) + 1} of {len(same_photo)}   "
                           f"{r['source'].upper()} {float(r['conf']):.2f}   split {r['split']}   "
                           f"[{S.pos + 1}/{len(S.order)}]   status: {status}")
    ui.status_tag.configure(text=status.upper(), foreground={"accepted": "#1b7a1b", "deleted": "#b00000"}.get(status, "#666666"))
    draw_photo()
    draw_crop()
    on_text_change()
    ui.entry.focus_set()
    ui.entry.icursor("end")


def draw_photo():
    r = current()
    img = photo(r["photo_id"])
    h, w = img.shape[:2]
    s = min(PHOTO_VIEW[0] / w, PHOTO_VIEW[1] / h)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ui.photo_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    c = ui.photo_canvas
    c.delete("all")
    c.create_image(0, 0, anchor="nw", image=ui.photo_img)
    x0, y0, x1, y1 = S.box
    c.create_rectangle(x0 * s - 2, y0 * s - 2, x1 * s + 2, y1 * s + 2, outline="#ff1a1a", width=3)


def draw_crop():
    img = crop_image()
    h, w = img.shape[:2]
    S.scale = min(CROP_VIEW[0] / w, CROP_VIEW[1] / h, 8.0)
    big = cv2.resize(img, (max(1, round(w * S.scale)), max(1, round(h * S.scale))), interpolation=cv2.INTER_CUBIC)
    ui.crop_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
    c = ui.crop_canvas
    c.delete("all")
    c.create_image(0, 0, anchor="nw", image=ui.crop_img)
    colour = "#00a000" if ui.boxes_ok.get() else "#ff8c00"
    for b in S.boxes:
        c.create_rectangle(*[v * S.scale for v in b], outline=colour, width=2)
    update_box_note()


def update_box_note():
    units = n_units(ui.text.get().strip()) if ui.text.get().strip() else 0
    ui.box_note.configure(text=f"{len(S.boxes)} char boxes, {units} units in the word"
                               + ("" if not S.boxes or len(S.boxes) == units else "  (counts differ)"))


def update_progress():
    done = [(r, S.review[r["path"]]) for r in S.rows if r["path"] in S.review]
    acc = [(r, d) for r, d in done if d["status"] == "accepted"]

    def count(script, split, boxed=False):
        return sum(d["script"] == script and r["split"] == split and (not boxed or (d.get("boxes_ok") and bool(d.get("char_boxes"))))
                   for r, d in acc)

    ui.progress.configure(text=f"Checked {len(done)} of {len(S.rows)}.   Test words: Kannada {count('kannada', 'test')}/{GOAL_TEST}, "
                               f"English {count('english', 'test')}/{GOAL_TEST}.   Test char boxes: Kannada "
                               f"{count('kannada', 'test', True)}/{GOAL_BOXES}, English {count('english', 'test', True)}/{GOAL_BOXES}."
                               f"   Train words: Kannada {count('kannada', 'train')}, English {count('english', 'train')}.")


def on_text_change(*_):
    text = ui.text.get().strip()
    img = render_text(text) if has_kannada(text) else None
    if img is not None:
        ui.render_img = ImageTk.PhotoImage(img)
        ui.render.configure(image=ui.render_img, text="")
    else:
        ui.render.configure(image="", text="")
    script = "kannada" if has_kannada(text) else "english"
    warn = ""
    if script == "english" and set(text.lower()) - ENGLISH_CHARS:
        warn = f"   outside the English charset: {''.join(sorted(set(text.lower()) - ENGLISH_CHARS))}"
    ui.script_note.configure(text=f"Script: {script}{warn}")
    update_box_note()


# ---------------------------------------------------------------- actions

def save(status):
    r = current()
    text = " ".join(ui.text.get().split())
    if status == "accepted":
        if not text:
            ui.message.configure(text="Type the word, or press Ctrl+D to delete it.")
            return False
        if ui.boxes_ok.get() and S.boxes and len(S.boxes) != n_units(text):
            ui.message.configure(text=f"{len(S.boxes)} char boxes but {n_units(text)} units: fix the boxes, "
                                      f"or untick 'char boxes checked' (Ctrl+B).")
            return False
    S.review[r["path"]] = {"status": status, "text": text, "script": "kannada" if has_kannada(text) else "english",
                           "box": S.box, "char_boxes": sorted(S.boxes, key=lambda b: b[0]),
                           "boxes_ok": bool(ui.boxes_ok.get() and S.boxes)}
    if S.box != r["box"]:
        # the crop was grown: rewrite the crop file so training and evaluation see the same pixels
        x0, y0, x1, y1 = S.box
        cv2.imencode(".png", photo(r["photo_id"])[y0:y1, x0:x1])[1].tofile(os.path.join(CROPS, r["path"]))
        r["box"] = list(S.box)
        save_auto_rows()
    write_json_atomic(REVIEW_FILE, S.review)
    write_labels()
    ui.message.configure(text=f"{status.capitalize()}: {text or r['path']}")
    return True


def save_auto_rows():
    tmp = AUTO_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(S.rows[0]), delimiter="\t")
        writer.writeheader()
        for r in S.rows:
            writer.writerow({**r, "box": json.dumps(r["box"]),
                             "char_boxes": json.dumps(r["char_boxes"]) if r["char_boxes"] else ""})
    os.replace(tmp, AUTO_FILE)


def advance():
    if not S.order:
        return
    path = current()["path"]
    if ui.filter.get() != "all":
        # the word may have left the filter (e.g. 'unchecked'); rebuild and stay at the same place
        old = S.pos
        apply_filter()
        S.pos = min(old, max(0, len(S.order) - 1))
        if S.order and S.rows[S.order[S.pos]]["path"] == path:
            S.pos = min(S.pos + 1, len(S.order) - 1)
        show()
    elif S.pos < len(S.order) - 1:
        S.pos += 1
        show()
    else:
        show()


def accept(*_):
    if current() and save("accepted"):
        advance()
    return "break"


def delete(*_):
    if current() and save("deleted"):
        advance()
    return "break"


def move(step):
    if S.order:
        S.pos = max(0, min(len(S.order) - 1, S.pos + step))
        show()
    return "break"


def next_unchecked(*_):
    for k in list(range(S.pos + 1, len(S.order))) + list(range(0, S.pos + 1)):
        if S.rows[S.order[k]]["path"] not in S.review:
            S.pos = k
            show()
            return "break"
    ui.message.configure(text="Every word in this filter has been checked.")
    return "break"


def grow(*_):
    r = current()
    if not r:
        return "break"
    img = photo(r["photo_id"])
    h, w = img.shape[:2]
    x0, y0, x1, y1 = S.box
    p = max(2, round(0.1 * (y1 - y0)))
    new = [max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)]
    dx, dy = x0 - new[0], y0 - new[1]
    S.boxes = [[b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy] for b in S.boxes]
    S.box = new
    draw_photo()
    draw_crop()
    return "break"


def reset_crop(*_):
    r = current()
    if r:
        S.box = list(r["box"])
        S.boxes = [list(b) for b in r["char_boxes"]]
        draw_photo()
        draw_crop()
    return "break"


def toggle_boxes(*_):
    ui.boxes_ok.set(not ui.boxes_ok.get())
    draw_crop()
    return "break"


def clear_boxes(*_):
    S.boxes = []
    ui.boxes_ok.set(False)
    draw_crop()
    return "break"


def crop_coords(event):
    return event.x / S.scale, event.y / S.scale


def on_press(event):
    S.drag = crop_coords(event)


def on_drag(event):
    if S.drag:
        ui.crop_canvas.delete("drag")
        x, y = S.drag
        ui.crop_canvas.create_rectangle(x * S.scale, y * S.scale, event.x, event.y, outline="#0055ff", width=2, tags="drag")


def on_release(event):
    if not S.drag or not current():
        return
    (x0, y0), (x1, y1) = S.drag, crop_coords(event)
    S.drag = None
    cw, ch = S.box[2] - S.box[0], S.box[3] - S.box[1]
    b = [max(0, round(min(x0, x1))), max(0, round(min(y0, y1))), min(cw, round(max(x0, x1))), min(ch, round(max(y0, y1)))]
    if (b[2] - b[0]) * S.scale >= 4 and (b[3] - b[1]) * S.scale >= 4:
        S.boxes.append(b)
        S.boxes.sort(key=lambda v: v[0])
    draw_crop()


def on_right_click(event):
    x, y = crop_coords(event)
    hits = [b for b in S.boxes if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
    if hits:
        S.boxes.remove(min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
        draw_crop()


# ---------------------------------------------------------------- window

ui = type("UI", (), {})()


def build(root):
    root.title("Review labels")
    root.geometry("1280x800")
    families = set(font.families())
    text_family = next((f for f in ("Noto Sans Kannada", "Nirmala UI", "Tunga") if f in families), None)
    big = (text_family, 22) if text_family else ("TkDefaultFont", 22)

    top = ttk.Frame(root, padding=(10, 8))
    top.pack(fill="x")
    ttk.Label(top, text="Show").pack(side="left")
    ui.filter = tk.StringVar(value="unchecked")
    f = ttk.Combobox(top, textvariable=ui.filter, values=["unchecked", "accepted", "deleted", "all"], width=11, state="readonly")
    f.pack(side="left", padx=(4, 12))
    ui.script_filter = tk.StringVar(value="both")
    g = ttk.Combobox(top, textvariable=ui.script_filter, values=["both", "kannada", "english"], width=9, state="readonly")
    g.pack(side="left")
    for box in (f, g):
        box.bind("<<ComboboxSelected>>", lambda e: apply_filter(current()["path"] if current() else None))
    ui.progress = ttk.Label(top, text="")
    ui.progress.pack(side="left", padx=20)

    body = ttk.Frame(root, padding=(10, 0))
    body.pack(fill="both", expand=True)
    ui.photo_canvas = tk.Canvas(body, width=PHOTO_VIEW[0], height=PHOTO_VIEW[1], background="#d9d9d9", highlightthickness=0)
    ui.photo_canvas.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 14))

    right = ttk.Frame(body)
    right.grid(row=0, column=1, sticky="nw")
    ui.info = ttk.Label(right, text="")
    ui.info.pack(anchor="w")
    ui.status_tag = ttk.Label(right, text="", font=("TkDefaultFont", 11, "bold"))
    ui.status_tag.pack(anchor="w", pady=(2, 4))
    ui.crop_canvas = tk.Canvas(right, width=CROP_VIEW[0], height=CROP_VIEW[1], background="#f0f0f0",
                               highlightthickness=1, highlightbackground="#bdbdbd", cursor="crosshair")
    ui.crop_canvas.pack(anchor="w")
    ui.crop_canvas.bind("<ButtonPress-1>", on_press)
    ui.crop_canvas.bind("<B1-Motion>", on_drag)
    ui.crop_canvas.bind("<ButtonRelease-1>", on_release)
    ui.crop_canvas.bind("<Button-3>", on_right_click)

    row = ttk.Frame(right)
    row.pack(anchor="w", fill="x", pady=(6, 0))
    ui.boxes_ok = tk.BooleanVar(value=False)
    ttk.Checkbutton(row, text="Char boxes checked (Ctrl+B)", variable=ui.boxes_ok, command=draw_crop).pack(side="left")
    ui.box_note = ttk.Label(row, text="", foreground="#555555")
    ui.box_note.pack(side="left", padx=12)

    ttk.Label(right, text="Word").pack(anchor="w", pady=(12, 0))
    ui.text = tk.StringVar()
    ui.text.trace_add("write", on_text_change)
    ui.entry = tk.Entry(right, textvariable=ui.text, font=big, width=30)
    ui.entry.pack(anchor="w", pady=(2, 4))
    ui.render = tk.Label(right, background="#ffffff")
    ui.render.pack(anchor="w")
    ui.script_note = ttk.Label(right, text="", foreground="#555555")
    ui.script_note.pack(anchor="w", pady=(2, 8))

    buttons = ttk.Frame(right)
    buttons.pack(anchor="w", pady=(4, 0))
    for text, cmd in (("◀ Previous", lambda: move(-1)), ("Accept  ⏎", accept), ("Delete", delete),
                      ("Next ▶", lambda: move(1)), ("Next unchecked", next_unchecked), ("Grow crop", grow),
                      ("Reset crop", reset_crop), ("Clear char boxes", clear_boxes)):
        ttk.Button(buttons, text=text, command=cmd).pack(side="left", padx=(0, 6))
    ui.message = ttk.Label(right, text="", foreground="#8a4b00", wraplength=640)
    ui.message.pack(anchor="w", pady=(8, 0))

    ttk.Separator(root).pack(fill="x")
    ttk.Label(root, text=HELP, foreground="#555555", padding=(10, 4), wraplength=1250).pack(fill="x")

    for seq, fn in (("<Return>", accept), ("<KP_Enter>", accept), ("<Control-d>", delete), ("<Prior>", lambda e: move(-1)),
                    ("<Next>", lambda e: move(1)), ("<Control-u>", next_unchecked), ("<Control-g>", grow),
                    ("<Control-r>", reset_crop), ("<Control-b>", toggle_boxes), ("<Control-x>", clear_boxes)):
        # on the entry itself too, so that its own Ctrl bindings (Ctrl+D deletes a letter) never run
        ui.entry.bind(seq, fn)
        root.bind_all(seq, fn)


def main():
    global KANNADA_FONT, CROPS, AUTO_FILE, REVIEW_FILE, LABELS_FILE
    ap = argparse.ArgumentParser(description="Check the automatic labels from src/autolabel.py.")
    ap.add_argument("--crops", default=CROPS, help="folder with auto_labels.tsv (default: paths.crops in config.yaml)")
    CROPS = ap.parse_args().crops
    AUTO_FILE, REVIEW_FILE, LABELS_FILE = (os.path.join(CROPS, f) for f in ("auto_labels.tsv", "review.json", "labels.tsv"))
    root = tk.Tk()
    build(root)
    load()
    KANNADA_FONT = find_kannada_font()
    apply_filter()
    if not S.order:
        ui.filter.set("all")
        apply_filter()
    root.mainloop()


if __name__ == "__main__":
    main()
