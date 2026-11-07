import csv
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)  # the paths in config.yaml are relative to the project folder

from datasets import load_config  # noqa: E402
from preprocess import read_image  # noqa: E402
from segment import draw_boxes, segment_word  # noqa: E402

MODELS = [("A", "A  HOG + SVM", "svm_{}.joblib"), ("B", "B  HOG + Random Forest", "rf_{}.joblib"),
          ("C", "C  CNN", "cnn_{}.pth"), ("D", "D  CRNN", "crnn_{}.pth")]
LABELS = {letter: label for letter, label, _ in MODELS} | {"All": "All available models"}
RF_NOTE = "Model B for Kannada needs about 2 GB of RAM and takes a while to load the first time."
PREVIEW = (300, 110)

app = SimpleNamespace(cfg=load_config(), image=None, path="", scale=1.0, selection=None, drag_start=None,
                      suggestions=[], loaded={}, records=[], available=[], photo=None, preview_photo=None)


def status(text):
    app.status.configure(text=text)
    app.root.update_idletasks()


def model_path(letter, script):
    pattern = next(p for l, _, p in MODELS if l == letter)
    path = os.path.join(app.cfg["paths"]["models"], pattern.format(script))
    pretrained = path.replace(".pth", "_pretrained.pth")
    # until D is fine-tuned, the pre-trained CRNN stands in for it
    if letter == "D" and not os.path.exists(path) and os.path.exists(pretrained):
        return pretrained
    return path


def label(letter):
    if letter == "D" and model_path("D", app.script.get()).endswith("_pretrained.pth"):
        return "D  CRNN (pre-trained only)"
    return LABELS[letter]


def get_runner(letter, script):
    # models are loaded on first use only; torch and the large forests are not touched before that
    key = (letter, script)
    if key in app.loaded:
        return app.loaded[key]
    status(f"Loading model {letter} ({script}) ...")
    seg, path = app.cfg["segment"], model_path(letter, script)
    if path.endswith(".joblib"):
        import joblib
        from classical import recognise
        bundle = joblib.load(path)
        run = lambda img, options: recognise(bundle, img, options, seg)
    else:
        from models import cnn_recognise, crnn_recognise, load_model
        model, info = load_model(path)
        if info["kind"] == "cnn":
            run = lambda img, options: cnn_recognise(model, info, img, options, seg)
        else:
            def run(img, options):
                text, conf = crnn_recognise(model, info, img, options)
                return text, None, None, conf
    app.loaded[key] = run
    return run


def refresh_models(*_):
    script = app.script.get()
    app.available = [l for l, _, _ in MODELS if os.path.exists(model_path(l, script))]
    for i, (letter, _, _) in enumerate(MODELS):
        app.model_menu.entryconfigure(i, label=label(letter), state="normal" if letter in app.available else "disabled")
    app.model_menu.entryconfigure(len(MODELS), state="normal" if app.available else "disabled")
    if app.model.get() != "All" and app.model.get() not in app.available:
        app.model.set(app.available[0] if app.available else "All")
    on_model()
    missing = [l for l, _, _ in MODELS if l not in app.available]
    app.run_button.state(["!disabled"] if app.available else ["disabled"])
    if missing:
        status(f"{script.capitalize()}: no model file for {', '.join(missing)} in {app.cfg['paths']['models']}/, "
               f"so {'it is' if len(missing) == 1 else 'they are'} disabled.")
    else:
        status(f"{script.capitalize()}: all four models found.")


def on_model():
    app.model_button.configure(text=label(app.model.get()))
    uses_rf = app.model.get() == "B" or (app.model.get() == "All" and "B" in app.available)
    app.note.configure(text=RF_NOTE if uses_rf and app.script.get() == "kannada" else "")


def open_image(event=None):
    path = filedialog.askopenfilename(title="Open image", initialdir=app.cfg["paths"]["photos"],
                                      filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")])
    if not path:
        return
    try:
        image = read_image(path)
    except ValueError:
        messagebox.showerror("Open image", f"Could not read {os.path.basename(path)} as an image.")
        return
    app.image, app.path, app.selection, app.suggestions = image, path, None, []
    zoom_to_fit()
    status(f"{os.path.basename(path)}: {image.shape[1]} x {image.shape[0]} px. Drag a rectangle around one word.")


def show_image():
    canvas = app.canvas
    canvas.delete("all")
    if app.image is None:
        return
    h, w = app.image.shape[:2]
    size = (max(1, round(w * app.scale)), max(1, round(h * app.scale)))
    rgb = cv2.cvtColor(cv2.resize(app.image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    app.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
    canvas.create_image(0, 0, anchor="nw", image=app.photo)
    canvas.configure(scrollregion=(0, 0, size[0], size[1]))
    for box in app.suggestions:
        canvas.create_rectangle(*[v * app.scale for v in box], outline="#606060", dash=(4, 3))
    if app.selection:
        canvas.create_rectangle(*[v * app.scale for v in app.selection], outline="#c00000", width=2)
    app.zoom_label.configure(text=f"{100 * app.scale:.0f}%")


def zoom_to_fit():
    if app.image is None:
        return
    h, w = app.image.shape[:2]
    app.scale = min(max(app.canvas.winfo_width(), 50) / w, max(app.canvas.winfo_height(), 50) / h)
    show_image()


def on_wheel(event):
    if event.state & 0x0004:  # Ctrl + wheel zooms
        if app.image is not None:
            app.scale = min(8.0, max(0.05, app.scale * (1.25 if event.delta > 0 else 0.8)))
            show_image()
    elif event.state & 0x0001:  # Shift + wheel scrolls sideways
        app.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
    else:
        app.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")


def to_image(event):
    return app.canvas.canvasx(event.x) / app.scale, app.canvas.canvasy(event.y) / app.scale


def on_press(event):
    app.drag_start = to_image(event)


def on_drag(event):
    if app.image is None or app.drag_start is None:
        return
    (x0, y0), (x1, y1) = app.drag_start, to_image(event)
    app.canvas.delete("drag")
    app.canvas.create_rectangle(x0 * app.scale, y0 * app.scale, x1 * app.scale, y1 * app.scale,
                                outline="#c00000", width=2, tags="drag")


def on_release(event):
    if app.image is None or app.drag_start is None:
        return
    (x0, y0), (x1, y1) = app.drag_start, to_image(event)
    h, w = app.image.shape[:2]
    box = (max(0, round(min(x0, x1))), max(0, round(min(y0, y1))), min(w, round(max(x0, x1))), min(h, round(max(y0, y1))))
    if (box[2] - box[0]) * app.scale < 5 or (box[3] - box[1]) * app.scale < 5:
        # a click instead of a drag picks the suggested box under the pointer, if any
        inside = [b for b in app.suggestions if b[0] <= x1 <= b[2] and b[1] <= y1 <= b[3]]
        if not inside:
            show_image()
            return
        box = inside[0]
    app.selection = box
    show_image()
    status(f"Selected {box[2] - box[0]} x {box[3] - box[1]} px. Press Run.")


def suggest_boxes():
    if app.image is None:
        status("Open an image first.")
        return
    gray = cv2.cvtColor(app.image, cv2.COLOR_BGR2GRAY)
    # a light blur makes MSER find stable regions on crisp printed text as well as on photos
    _, regions = cv2.MSER_create(delta=5, min_area=30, max_area=20000, max_variation=0.5).detectRegions(
        cv2.GaussianBlur(gray, (3, 3), 0))
    h, w = gray.shape
    regions = [r for r in regions if 0.01 * h < r[3] < 0.4 * h and r[2] < 0.5 * w]
    if not len(regions):
        app.suggestions = []
        show_image()
        status("MSER found no text-like regions; drag a rectangle by hand.")
        return
    # character regions become words by smearing them sideways by about half a character height
    mask = np.zeros_like(gray)
    for x, y, bw, bh in regions:
        mask[y:y + bh, x:x + bw] = 255
    gap = max(3, int(0.5 * np.median([r[3] for r in regions])))
    mask = cv2.dilate(mask, np.ones((1, gap), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    app.suggestions = [(int(x), int(y), int(x + bw), int(y + bh)) for x, y, bw, bh, _ in stats[1:] if bw > 8 and bh > 8]
    show_image()
    status(f"{len(app.suggestions)} suggested boxes (dashed). Click one to select it, or drag your own.")


def run():
    if app.image is None or app.selection is None:
        status("Open an image and drag a rectangle around one word first.")
        return
    x0, y0, x1, y1 = app.selection
    crop = app.image[y0:y1, x0:x1]
    options = tuple(name for name, var in app.options.items() if var.get())
    script = app.script.get()
    letters = app.available if app.model.get() == "All" else [app.model.get()]
    app.root.configure(cursor="watch")
    for letter in letters:
        runner = get_runner(letter, script)
        status(f"Running model {letter} ...")
        start = time.perf_counter()
        text, _, boxes, conf = runner(crop, options)
        ms = 1000 * (time.perf_counter() - start)
        item = app.table.insert("", "end", values=(label(letter), text, f"{conf:.2f}", f"{ms:.0f}"))
        app.table.see(item)
        app.records.append({"image": app.path, "box": f"{x0} {y0} {x1} {y1}", "script": script,
                            "preprocessing": "+".join(options) or "none", "model": label(letter),
                            "text": text, "confidence": round(conf, 3), "time_ms": round(ms, 1)})
    show_preview(crop, options, any(l in "ABC" for l in letters))
    app.root.configure(cursor="")
    status(f"Ran {', '.join(letters)} on the selected word ({script}, preprocessing: {'+'.join(options) or 'none'}).")


def show_preview(crop, options, with_segments):
    if with_segments:
        gray, boxes = segment_word(crop, options, app.cfg["segment"])
        shown = draw_boxes(gray, boxes)
        app.preview_note.configure(text=f"{len(boxes)} segments from the shared segmenter (models A-C)")
    else:
        shown = crop
        app.preview_note.configure(text="Model D reads the whole word, so there are no segments")
    h, w = shown.shape[:2]
    scale = min(PREVIEW[0] / w, PREVIEW[1] / h)
    shown = cv2.resize(shown, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_NEAREST)
    app.preview_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))
    app.preview.configure(image=app.preview_photo)


def export_csv():
    if not app.records:
        messagebox.showinfo("Export results", "There are no results to export yet.")
        return
    path = filedialog.asksaveasfilename(title="Export results to CSV", defaultextension=".csv",
                                        initialdir=app.cfg["paths"]["results"], filetypes=[("CSV", "*.csv")])
    if not path:
        return
    # utf-8-sig so that Excel shows the Kannada text correctly
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, list(app.records[0]))
        writer.writeheader()
        writer.writerows(app.records)
    status(f"Exported {len(app.records)} results to {path}")


def build_menu(root):
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=False)
    file_menu.add_command(label="Open image...", accelerator="Ctrl+O", command=open_image)
    file_menu.add_command(label="Export results to CSV...", command=export_csv)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.destroy)
    tools_menu = tk.Menu(menubar, tearoff=False)
    tools_menu.add_command(label="Suggest boxes (MSER)", command=suggest_boxes)
    menubar.add_cascade(label="File", menu=file_menu)
    menubar.add_cascade(label="Tools", menu=tools_menu)
    root.configure(menu=menubar)
    root.bind("<Control-o>", open_image)


def build_canvas(parent):
    frame = ttk.Frame(parent)
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    app.canvas = tk.Canvas(frame, background="#d9d9d9", highlightthickness=0, cursor="crosshair")
    xbar = ttk.Scrollbar(frame, orient="horizontal", command=app.canvas.xview)
    ybar = ttk.Scrollbar(frame, orient="vertical", command=app.canvas.yview)
    app.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
    app.canvas.grid(row=0, column=0, sticky="nsew")
    ybar.grid(row=0, column=1, sticky="ns")
    xbar.grid(row=1, column=0, sticky="ew")
    bar = ttk.Frame(frame)
    bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
    ttk.Button(bar, text="Zoom to fit", command=zoom_to_fit).pack(side="left")
    app.zoom_label = ttk.Label(bar, text="", width=6)
    app.zoom_label.pack(side="left", padx=8)
    ttk.Label(bar, text="Drag to select a word. Ctrl + wheel zooms.", foreground="#555555").pack(side="left")
    for event, handler in (("<ButtonPress-1>", on_press), ("<B1-Motion>", on_drag), ("<ButtonRelease-1>", on_release),
                           ("<MouseWheel>", on_wheel)):
        app.canvas.bind(event, handler)
    return frame


def build_panel(parent):
    panel = ttk.Frame(parent, width=340)
    ttk.Label(panel, text="Model").pack(anchor="w")
    app.model = tk.StringVar(value="A")
    # a Menubutton rather than a Combobox, because single menu entries can be disabled;
    # the layout only moves the label from the centre to the left
    ttk.Style().layout("Left.TMenubutton", [
        ("Menubutton.dropdown", {"side": "right", "sticky": "ns"}),
        ("Menubutton.button", {"sticky": "nswe", "children": [
            ("Menubutton.padding", {"sticky": "we", "children": [("Menubutton.label", {"sticky": "w"})]})]})])
    app.model_button = ttk.Menubutton(panel, text=LABELS["A"], width=34, style="Left.TMenubutton")
    app.model_menu = tk.Menu(app.model_button, tearoff=False)
    for letter in [l for l, _, _ in MODELS] + ["All"]:
        app.model_menu.add_radiobutton(label=LABELS[letter], value=letter, variable=app.model, command=on_model)
    app.model_button.configure(menu=app.model_menu)
    app.model_button.pack(anchor="w", fill="x", pady=(2, 10))

    ttk.Label(panel, text="Script").pack(anchor="w")
    app.script = tk.StringVar(value="kannada")
    row = ttk.Frame(panel)
    row.pack(anchor="w", pady=(2, 10))
    for value in ("kannada", "english"):
        ttk.Radiobutton(row, text=value.capitalize(), value=value, variable=app.script,
                        command=refresh_models).pack(side="left", padx=(0, 12))

    ttk.Label(panel, text="Preprocessing").pack(anchor="w")
    app.options = {name: tk.BooleanVar(value=False) for name in ("clahe", "sharpen", "deskew")}
    row = ttk.Frame(panel)
    row.pack(anchor="w", pady=(2, 6))
    for name, text in (("clahe", "CLAHE"), ("sharpen", "Sharpen"), ("deskew", "Deskew")):
        ttk.Checkbutton(row, text=text, variable=app.options[name]).pack(side="left", padx=(0, 12))

    app.note = ttk.Label(panel, text="", foreground="#8a4b00", wraplength=320, justify="left")
    app.note.pack(anchor="w", fill="x", pady=(0, 6))
    app.run_button = ttk.Button(panel, text="Run", command=run)
    app.run_button.pack(anchor="w", pady=(0, 12))

    ttk.Label(panel, text="Selected word").pack(anchor="w")
    box = tk.Frame(panel, width=PREVIEW[0] + 10, height=PREVIEW[1] + 10, background="#ffffff",
                   highlightthickness=1, highlightbackground="#bdbdbd")
    box.pack(anchor="w", pady=(2, 2))
    box.pack_propagate(False)
    app.preview = tk.Label(box, background="#ffffff")
    app.preview.pack(expand=True)
    app.preview_note = ttk.Label(panel, text="", foreground="#555555", wraplength=320)
    app.preview_note.pack(anchor="w", pady=(0, 10))

    ttk.Label(panel, text="Results").pack(anchor="w")
    table_frame = ttk.Frame(panel)
    table_frame.pack(fill="both", expand=True, pady=(2, 0))
    columns = (("model", "Model", 190), ("text", "Text", 160), ("conf", "Confidence", 75), ("ms", "Time (ms)", 70))
    # the default row height cuts off Kannada vowel signs, and large screen fonts make it worse
    ttk.Style().configure("Treeview", rowheight=round(1.8 * font.nametofont("TkDefaultFont").metrics("linespace")))
    app.table = ttk.Treeview(table_frame, columns=[c for c, _, _ in columns], show="headings", height=8)
    for key, heading, width in columns:
        app.table.heading(key, text=heading)
        app.table.column(key, width=width, anchor="w" if key in ("model", "text") else "e")
    scroll = ttk.Scrollbar(table_frame, orient="vertical", command=app.table.yview)
    app.table.configure(yscrollcommand=scroll.set)
    app.table.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    return panel


def main():
    root = tk.Tk()
    app.root = root
    root.title("Shop-sign OCR (Kannada / English)")
    root.geometry("1200x760")
    root.minsize(900, 560)
    build_menu(root)
    main_frame = ttk.Frame(root, padding=8)
    main_frame.pack(fill="both", expand=True)
    main_frame.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    build_canvas(main_frame).grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    build_panel(main_frame).grid(row=0, column=1, sticky="ns")
    ttk.Separator(root).pack(fill="x")
    app.status = ttk.Label(root, text="", padding=(8, 3))
    app.status.pack(fill="x")
    refresh_models()
    root.mainloop()


if __name__ == "__main__":
    main()
