# OCR for Kannada–English shop signs

MCA internship project: *Optical Character Recognition – Quantification with Various ML
Approaches*. It recognises cropped word images from shop signs in Kannada and English with four
models, all trained from scratch:

| | Model | Needs segmentation |
|---|---|---|
| A | HOG + SVM | yes |
| B | HOG + Random Forest | yes |
| C | small CNN | yes |
| D | CRNN (CNN + BiLSTM + CTC) | no |

A–C share one ordinary segmenter (Otsu + connected components + projection profile), used unchanged
for both scripts. The main question is how often that segmenter fails on Kannada, where one akshara is
often several blobs and neighbouring letters touch. The main metric is therefore the segmentation
success rate. No pretrained OCR models or weights are used anywhere.

This README contains no results. `results/` is filled by `src/evaluate.py` from your own runs.

## Setup

Tested with Python 3.13 on Windows. Run every command from the project folder.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows the PyPI build of PyTorch is CPU-only, which is all the laptop needs. GPU training happens in
Colab.

The scripts print Kannada text. That works in an interactive terminal, but fails with
`UnicodeEncodeError` when output is piped or redirected to a file. Turn on Python's UTF-8 mode once per
terminal to avoid it:

```
set PYTHONUTF8=1          (cmd)
$env:PYTHONUTF8 = "1"     (PowerShell)
```

Each core module checks itself:

```
python src/akshara.py
python src/preprocess.py
python src/metrics.py
python src/segment.py
python src/models.py
```

All settings live in `config.yaml`: paths, image height, charsets, seeds, split, segmenter, training
and Random Forest settings.

English is case-insensitive throughout. Labels are lowercased when they are loaded, and every metric
compares lowercased text, so `SHOP` and `shop` count as the same word.

## Full run order

Everything still needed for real results, in order. Steps marked **Colab** run in a notebook with a
T4 GPU; everything else runs on the laptop.

**1. Get the data**

- [ ] Download IIIT5K and put `test/` and `testdata.mat` in `data/public/IIIT5K/`.
- [ ] Download IndicSTR12 `test.zip` and unzip it inside `data/public/indicstr12_kannada/`, which gives
      `test/images/` and `test/test_gt.txt`.
- [ ] Download Chars74K `EnglishImg.tgz` into `data/public/chars74k/` (optional).
- [ ] Optional: Chars74K Kannada, with your own `kannada_classes.tsv`.
- [ ] Find the download links for MJSynth and the IIIT synthetic Kannada set. These two are never
      downloaded to the laptop or to Drive in full.
- [ ] Check `python src/datasets.py iiit5k --split test` and
      `python src/datasets.py indicstr12_kannada --split test`.

**2. Annotate and crop your photos**

- [ ] Put the photos in `data/photos/` and annotate them in Label Studio (see [Your photos](#your-photos-and-label-studio)).
- [ ] Draw `char` boxes for about 50 words per script.
- [ ] Export as JSON to `data/labelstudio.json`.
- [ ] `python src/make_crops.py`, and fix any labels it reports. Rerun until it is clean.

**3. Prepare the Kannada vocabulary and build the charset**

- [ ] Upload the project to Drive as `MyDrive/ocr-project` (see [What to upload](#what-to-upload-to-drive)).
- [ ] **Colab**: open `notebooks/train_crnn.ipynb` with `SCRIPT = 'kannada'` and run the cells down to
      the one that says *training reads the public words*. `KANNADA_URL` is already filled in. Stop
      there.
- [ ] Download `data/public/kannada_synth/vocabulary.txt` from Drive into the same place on the laptop.
- [ ] `python src/build_charset.py` and look at the aksharas it leaves out.
- [ ] Upload the new `charsets/kannada.txt` to Drive. Every model below depends on it, so redo
      steps 4–7 whenever the charset changes.

**4. Re-render the training characters**

- [ ] **Colab**: run `notebooks/render_chars.ipynb`. It should print `raqm: True`.
- [ ] Download `data/rendered_kannada.zip` and `data/rendered_english.zip`. Delete the old
      `data/rendered/` and unzip both into it.

**5. Retrain A and B**

- [ ] `python src/classical.py --script english`
- [ ] `python src/classical.py --script kannada`
- [ ] Note the two fingerprints these print.

**6. Train C**

- [ ] **Colab**: run `notebooks/train_cnn.ipynb` once with `SCRIPT = 'english'` and once with
      `'kannada'`.
- [ ] Check that each printed fingerprint matches the one from step 5 for that script. If one
      differs, A, B and C did not see the same characters. Check `char_per_class`, the rendered zips,
      and that Chars74K is on Drive whenever it is on the laptop.
- [ ] Download `models/cnn_<script>.pth` and `.json` into `models/`.

**7. Train D**

- [ ] **Colab**: run `notebooks/train_crnn.ipynb` with `SCRIPT = 'kannada'`, now to the end. The
      subset zip from step 3 is reused.
- [ ] Before it trains, check how many sampled words have fewer frames than code points.
- [ ] Each run first pre-trains on the public words, then fine-tunes on your crops.
- [ ] If `models/crnn_<script>_pretrained.pth` already exists, for example because you trained D
      before annotating, the notebook fine-tunes that model instead of pre-training again
      (`REUSE_PRETRAINED`). It stops only if the set of code points in the charset has changed; new
      aksharas built from known code points are fine.
- [ ] Run it again with `SCRIPT = 'english'`, with `MJSYNTH_URL` filled in. Check that the subset has
      tens of thousands of unique words.
- [ ] Download `models/crnn_<script>.pth` and `.json` into `models/`. The `_pretrained` copies are
      optional.

**8. Evaluate**

- [ ] `python src/evaluate.py`
- [ ] Look at `results/results.csv` and the charts.
- [ ] Fill the `category` column of `results/errors.csv` by hand.

**9. Run the app**

- [ ] `python app/app.py`, and try a few test-split photos with each model.

### What to upload to Drive

Drive needs `config.yaml`, `requirements.txt`, `src/`, `notebooks/`, `charsets/` and `data/crops/`, plus
`data/public/chars74k/` if you use it. Leave out `.venv/`, `data/photos/`, `data/rendered/`, `models/`
and the other public sets. The notebooks write their zips, models and `vocabulary.txt` back to Drive.

## Data

All paths below are the defaults in `config.yaml`. If your folders are named differently, change the
config instead of moving files.

### Your photos and Label Studio

Put the shop-sign photos in `data/photos/`, with the same file names you uploaded to Label Studio.

Labelling config (OCR template):

```xml
<View>
  <Image name="image" value="$ocr"/>
  <Labels name="label" toName="image">
    <Label value="kannada"/>
    <Label value="english"/>
    <Label value="char"/>
  </Labels>
  <Rectangle name="bbox" toName="image"/>
  <TextArea name="transcription" toName="image" editable="true" perRegion="true"/>
</View>
```

- Draw one rectangle per word, labelled `kannada` or `english`, and type the word in the transcription
  box.
- For the boxed subset, also draw one rectangle per letter or akshara inside the word, labelled `char`.
  These need no transcription. The number of `char` boxes should equal the number of units in the word,
  and `make_crops.py` prints the words where it doesn't.
- Export with **Export → JSON**, not JSON-MIN.

`python src/make_crops.py` writes `data/crops/*.png` and `data/crops/labels.tsv` with the columns
`path, text, script, photo_id, n_units, split, char_boxes`. The split is 70/30 by photo with the config
seed, and the test split is never used for training or tuning. `char_boxes` is empty, or a JSON list of
`[x0, y0, x1, y1]` boxes relative to the crop, from left to right.

### Public datasets

Sizes are approximate. Where I don't know a size, check the download page.

| Dataset | Used for | Where it goes | Approx. download |
|---|---|---|---|
| IIIT-Synthetic-IndicSTR-Kannada | CRNN pre-training, charset vocabulary | Colab only; just `vocabulary.txt` on the laptop, in `data/public/kannada_synth/` | not known |
| IndicSTR12, Kannada part | Kannada public test set | unzip `test.zip` inside `data/public/indicstr12_kannada/`, giving `test/images/` and `test/test_gt.txt` | ~2 MB |
| MJSynth (90kDICT32px) | English CRNN pre-training | Colab only, streamed | full archive ~10 GB; the subset is a few hundred MB |
| IIIT5K | English public test set | `data/public/IIIT5K/` with `test/` and `testdata.mat` | ~100 MB |
| Chars74K English | extra characters for A–C (optional) | `data/public/chars74k/English/Img/GoodImg/Bmp/Sample001 ...` | ~130 MB (`EnglishImg.tgz` only) |
| Chars74K Kannada | extra characters for A–C (optional) | `data/public/chars74k/Kannada/Img/GoodImg/Bmp/...` plus `kannada_classes.tsv` | not known |

The Kannada gt files are read as `image_name<TAB>text`, one line per image. If a Chars74K folder or the
Kannada class map is missing, that set is skipped with a message. `kannada_classes.tsv` has one line per
folder, such as `Sample001<TAB>ಅ`. Chars74K has no official split, so a fixed shuffle is cut 70/30 with
the config seed.

`python src/datasets.py <name> [--split test] [--limit N]` loads any set and prints a few samples and
the shape of one batch. The names are `crops`, `kannada_synth`, `indicstr12_kannada`, `mjsynth`,
`iiit5k`, `chars74k_english` and `chars74k_kannada`.

### The two large sets stay on Colab

Drive has too little space for the full Kannada synthetic set or MJSynth. `notebooks/train_crnn.ipynb`
downloads and unpacks them only on the Colab disk (`/content`). It then zips a random subset of `SUBSET`
words, 200,000 by default, to Drive as `data/public/<set>_subset.zip`. Later sessions just unzip that
file, and training always reads from `/content`.

- **MJSynth.** `src/mjsynth_subset.py` streams the `.tar.gz` and stops after `SUBSET` readable images.
  It writes an annotation file in the original format, and prints how many unique words it kept. If
  that number is very low, the archive is ordered by word. In that case, delete the zip and add
  `--every 45` to the command in that cell. The script then streams the whole archive once and keeps
  every 45th image. The link is on https://www.robots.ox.ac.uk/~vgg/data/text/.
- **Kannada synthetic set.** The link and paths are already set in the notebook. It downloads the
  4.6 GB zip with `wget -c`, which shows progress and can resume within the same Colab session. From
  the zip it unpacks only `train/train_gt.txt` and `train/vocabulary.txt`. `src/subset_words.py --zip`
  then picks `SUBSET` random words and extracts only their images. The 2 million images are never
  unpacked. The notebook then puts `vocabulary.txt` on Drive for `build_charset.py`. These steps need no
  GPU, so run them on a CPU runtime first.

### The Kannada charset

`python src/build_charset.py` counts aksharas in `vocabulary.txt` and in the **train** split of
`labels.tsv`. Each vocabulary word counts once. It then keeps the most frequent aksharas that together
cover `charset_coverage` (0.99) of all occurrences, writes them to `charsets/kannada.txt`, and prints
what it left out. For A–C, anything outside the charset is the class `<rare>`. The CRNN's alphabet is
separate: the script also writes every Kannada code point it found to `charsets/kannada_alphabet.txt`,
whatever the coverage. It also prints how many classes each of 0.95, 0.97, 0.98 and 0.99 coverage would
keep, with a rough Kannada Random Forest memory estimate for each. The English charset is `a–z`, `0–9`
and `&`, and the English CRNN uses that same file.

## Training

### A and B: HOG + SVM and HOG + Random Forest (laptop)

Kannada rendering needs Pillow's RAQM layout engine, which Pillow on Windows usually lacks, so the
characters are rendered in Colab with `notebooks/render_chars.ipynb`. It uses free fonts (Noto
Sans/Serif Kannada, Lohit Kannada, Noto, DejaVu, Liberation). It applies light rotation, blur, noise
and scaling, and draws English in both cases.

`python src/classical.py --script <english|kannada>` then trains both models on HOG features of 32×32
crops. It uses at most `char_per_class` characters per class, and holds out a tenth of them for the
validation accuracy it prints. It saves `models/svm_<script>.joblib` and `models/rf_<script>.joblib`,
and prints their size on disk and approximate memory use. The forest uses 50 trees with at least 5
samples per leaf, because scikit-learn stores one value per class in every node. Even so, the Kannada
forest needs about 2 GB of RAM.

### C and D: CNN and CRNN (Colab, T4 GPU)

Set `SCRIPT` in the second cell of each notebook. Both notebooks import from `src/`, copy what they
read to `/content`, fix the seed, use Adam with early stopping on validation loss, and print the
training time. Download the `.pth` and `.json` files into `models/`. The app and the evaluation run
them on the CPU.

- **`train_cnn.ipynb`** trains model C on the same characters as A and B, checked by the fingerprint.
- **`train_crnn.ipynb`** trains model D in two stages. It pre-trains on the public subset, holding out 5%
  for validation. It then fine-tunes on the train split of your crops, holding out 15% of those photos
  for validation. The network shrinks the width by 4, so a word gets `width / 4` output frames. CTC
  can't learn from words with more code points than frames, so the notebook counts them first.

## Evaluation

```
python src/evaluate.py
python src/evaluate.py --scripts kannada --models A B --limit 500
```

This runs every model found in `models/` on your test split, and on IndicSTR12 (Kannada) and IIIT5K
(English). It runs each one without preprocessing, then with CLAHE, sharpening and deskew, one at a
time. Missing models or test sets are skipped with a message. `--limit` evaluates a random subset of
each public set. The outputs go to `results/`:

- **`results.csv`** has one row per model × script × test set × preprocessing:
  - `seg_success` (A–C) is the share of words whose segment count equals `n_units`.
  - `seg_success_iou` uses only the boxed words, and also needs IoU ≥ 0.5 per segment. It is empty for
    deskew, because deskew rotates the crop away from the box coordinates.
  - `char_acc` (A–C) is counted only on correctly segmented words.
  - `cer` is computed over aksharas for Kannada and letters for English, and can exceed 100% when a
    model outputs extra units. `wer` and `exact` are the other text metrics.
  - `_lo` and `_hi` give 95% bootstrap intervals, from 1000 resamples of photos. The public sets
    resample single images instead.
  - `rare_share` is the share of test units outside the charset.
  - `ms_per_word` (CPU), `size_mb` and `train_seconds` cover efficiency.
- **`errors.csv`** has up to 25 random failures per model, script and test set. `suggested_category` is
  an automatic first guess. Write the final category by hand: segmentation failure, confusable
  characters, dropped vowel sign, split/merged akshara or stylised font.
- **`seg_success_by_script.png`** and **`cer_by_model_script.png`** use your test split.
- **`confusion_<model>_<script>.png`** covers A–C on correctly segmented words: the 30 most frequent
  classes for Kannada and 20 for English. Kannada classes are numbered, because Matplotlib can't shape
  Kannada; the matching `.csv` maps each number to its akshara.
- **`top_confusions_<script>.csv`** lists the 20 most frequent wrong (true, predicted) pairs for each of
  A–C.

## The app

```
python app/app.py
```

- **Open a photo** with *File → Open image*. Ctrl + wheel zooms, and *Zoom to fit* resets the view.
- **Select a word** by dragging a rectangle around it. *Tools → Suggest boxes (MSER)* offers dashed
  boxes, and clicking one selects it. It is only a helper and often misses words.
- **Run** after choosing a model (A, B, C, D or All), the script and any preprocessing. Each model adds
  a row with its text, confidence and recognition time.
- **The preview** shows the segments that A–C classify.
- **Export** the session's results with *File → Export results to CSV*.

Models load from `models/` on first use. A missing model is greyed out, and the status bar names it.
Selecting the Kannada Random Forest shows a note that it needs about 2 GB of RAM. Confidence means
something different for each model, so compare it within one model only.

## Project layout

```
config.yaml            settings: paths, image height, charsets, seeds, segmenter, training
charsets/              english.txt, kannada.txt (built by build_charset.py)
data/                  photos/, labelstudio.json, crops/, public/, rendered/
src/                   akshara, preprocess, make_crops, datasets, build_charset, render_chars,
                       mjsynth_subset, subset_words, segment, classical, models, metrics, evaluate, plots
notebooks/             render_chars, train_cnn, train_crnn (Colab)
app/app.py             Tkinter app
models/                trained .joblib / .pth + .json files
results/               written by evaluate.py
```
