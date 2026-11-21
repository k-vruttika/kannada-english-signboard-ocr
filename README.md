# OCR for Kannada–English Shop Signs

MCA internship project: *Optical Character Recognition – Quantification with Various ML Approaches*.

The project reads cropped word images from Kannada and English shop signs and compares four models.
All four are trained from scratch; no pretrained OCR models or weights are used.

| | Model | Needs segmentation |
|---|---|---|
| A | HOG + SVM | yes |
| B | HOG + Random Forest | yes |
| C | Small CNN | yes |
| D | CRNN (CNN + BiLSTM + CTC) | no |

Models A–C use one shared segmenter (Otsu thresholding, connected components and projection profiles)
for both scripts. The main question is how often this segmenter fails on Kannada, where one akshara is
often made of several pieces and neighbouring letters often touch. So the main metric is the
segmentation success rate.

## Status

All four models are trained for both scripts and evaluated. The results are in the
[Evaluation](#evaluation) section.

- 333 shop-sign photos, split 70/30 by photo.
- 652 checked word labels: 244 test words (114 Kannada, 130 English) and 408 training words.
  198 of the test words also have checked character boxes (105 Kannada, 93 English).
- The labels were first filled in automatically (Google Cloud Vision for Kannada, PaddleOCR for English)
  and then every word was checked and corrected by hand. All results are measured against these
  corrected labels, not against another OCR system's output.
- Models A–C are trained on the same characters for each script (Kannada: rendered letters, 30 per
  class; English: rendered letters and Chars74K, 250 per class).
- Model D is pre-trained on 190,000 synthetic words per script (IIIT Synthetic Kannada, MJSynth) and
  then fine-tuned on the training words.
- Test sets: the photo test split, IndicSTR12 (Kannada) and IIIT5K (English).

Still left: sorting the sample errors in `results/errors.csv` into categories.

## Setup

Tested with Python 3.13/3.14 on Windows and Ubuntu. Run all commands from the project folder.

```
python -m venv .venv
.venv\Scripts\activate            (Windows)
source .venv/bin/activate         (Linux)
pip install -r requirements.txt
```

PyTorch from PyPI is CPU-only on Windows, which is fine for running the models. GPU training was done
on Google Colab.

The scripts print Kannada text. If the output is redirected to a file on Windows, Python can raise
`UnicodeEncodeError`. Turning on UTF-8 mode fixes this:

```
set PYTHONUTF8=1          (cmd)
$env:PYTHONUTF8 = "1"     (PowerShell)
```

The core modules have small self-checks:

```
python src/akshara.py
python src/preprocess.py
python src/metrics.py
python src/segment.py
python src/models.py
```

All settings (paths, image height, charsets, seed, split, segmenter, training and Random Forest
parameters) are in `config.yaml`.

English is case-insensitive: labels are lowercased when loaded, so `SHOP` and `shop` count as the same
word.

## Trained Models

The trained model files are not stored in the repository. They are on the
[Releases page](https://github.com/k-vruttika/kannada-english-signboard-ocr/releases) and go in the
`models/` folder. The datasets are not included either.

## Steps

The full pipeline, in order. Steps marked **Colab** were run in a notebook on a T4 GPU; the rest run
locally.

1. **Get the data.** IIIT synthetic Kannada (Colab, only `vocabulary.txt` is needed locally), IIIT5K,
   IndicSTR12 Kannada test set, Chars74K English and an MJSynth subset (streamed on Colab). Locations
   are listed under [Public datasets](#public-datasets).
2. **Label and crop the photos.** `src/autolabel.py` finds, reads and crops the words, and
   `app/review_labels.py` is used to accept, correct or delete each one and to check the character
   boxes. See [Labelling](#labelling).
3. **Build the Kannada charset.** `python src/build_charset.py` (859 classes). Models A–D depend on
   this file, so steps 4–7 have to be repeated if it changes.
4. **Render the training characters** (Colab): `notebooks/render_chars.ipynb`, then extract the two
   zip files into `data/rendered/`.
5. **Train A and B:** `python src/classical.py --script english` and `--script kannada`. Each run
   prints a fingerprint of the characters used.
6. **Train C:** `python src/train_cnn.py --script <english|kannada>` or `notebooks/train_cnn.ipynb`.
   The printed fingerprint must match the one from step 5, so A, B and C use the same characters.
7. **Train D:** `notebooks/train_crnn.ipynb` (Colab) or `python src/train_crnn.py --script <script>`.
   It pre-trains on the public words and then fine-tunes on the training split.
8. **Evaluate:** `python src/evaluate.py`.

For Colab, Drive needs `config.yaml`, `requirements.txt`, `src/`, `notebooks/`, `charsets/` and
`data/crops/` (and `data/public/chars74k/` for English). The notebooks save their models back to Drive.

## Data

All paths are the defaults in `config.yaml`. If folders are named differently, change the config
instead of moving the files.

### Labelling

Typing 300+ words by hand would take too long, so an existing OCR system gives a first guess for each
word and every guess is checked by hand. Kannada is read with **Google Cloud Vision**, which is the
best available baseline for Kannada, and English with **PaddleOCR**. These are only used to fill in
draft labels; none of the models A–D use them. A word goes into `data/crops/labels.tsv` only after it
is accepted or corrected in the review tool.

PaddlePaddle has no wheels for Python 3.14, so labelling uses a separate Python 3.13 environment:

```
pip install uv
uv python install 3.13
uv venv --python 3.13 .venv-label
uv pip install --python .venv-label -r requirements-label.txt
```

(On Windows use `.venv-label\Scripts\python` instead of `.venv-label/bin/python`.)

The Cloud Vision API key is created in Google Cloud Console (*APIs & Services → Credentials*, with the
Cloud Vision API enabled; the first 1000 images a month are free). It is passed through an environment
variable and never saved in the repository:

```
export GCV_API_KEY=...         (Linux)
set GCV_API_KEY=...            (cmd)
$env:GCV_API_KEY = "..."       (PowerShell)
```

```
.venv-label/bin/python src/autolabel.py
.venv-label/bin/python src/autolabel.py --limit 5      (quick test on five photos)
```

`autolabel.py` keeps the Kannada words from Cloud Vision and the English words from PaddleOCR. PaddleOCR
words that overlap a Kannada word are dropped, since PaddleOCR reads Kannada as random letters. Each
word is cropped with a small margin into `data/crops/`, and the candidates are written to
`data/crops/auto_labels.tsv`. For Kannada, Cloud Vision's letter boxes are grouped into aksharas as
draft character boxes. The OCR output is cached in `data/ocr_cache/`, so running it again does not call
the APIs again. The train/test split by photo is fixed at this point using the config seed. Once the
review has started, the script will not run again without `--force`.

```
python app/review_labels.py          (or Tools → Review automatic labels in the app)
```

The review window shows the photo with the word marked, the enlarged crop and the OCR text in an edit
box. Kannada text is also drawn with Pillow below the box, because Tk does not shape Kannada conjuncts
correctly on every system.

| Key | Action |
|---|---|
| Enter | accept the word and go to the next one |
| Ctrl+D | delete the crop (not a word, cut off, unreadable) |
| PgUp / PgDn | previous / next word |
| Ctrl+U | next unchecked word |
| Ctrl+G / Ctrl+R | grow the crop if a letter is cut off / reset it |
| drag / right-click on the crop | draw / remove a character box |
| Ctrl+B | mark the character boxes as checked |
| Ctrl+X | clear the character boxes |

Decisions are saved immediately to `data/crops/review.json`, and `labels.tsv` is rewritten with only the
accepted words. Character boxes are kept only for words marked as checked, and only when the number of
boxes equals the number of units in the word. Test-split words are shown first.

`src/make_crops.py` can also read a Label Studio JSON export (`data/labelstudio.json`) if the photos are
labelled fully by hand instead. Both ways produce `data/crops/*.png` and `data/crops/labels.tsv` with the
columns `path, text, script, photo_id, n_units, split, char_boxes`. The test split is never used for
training or tuning.

### Public datasets

| Dataset | Used for | Location |
|---|---|---|
| IIIT-Synthetic-IndicSTR-Kannada | CRNN pre-training, charset vocabulary | Colab; locally only `data/public/kannada_synth/vocabulary.txt` |
| IndicSTR12, Kannada | Kannada test set | `data/public/indicstr12_kannada/test/images/` and `test/test_gt.txt` |
| MJSynth (90kDICT32px) | English CRNN pre-training | Colab, streamed |
| IIIT5K | English test set | `data/public/IIIT5K/` with `test/` and `testdata.mat` |
| Chars74K English | extra real letters for A–C | `data/public/chars74k/English/Img/GoodImg/Bmp/` |
| Chars74K Kannada (optional) | extra letters for A–C | `data/public/chars74k/Kannada/...` and `kannada_classes.tsv` |

The Kannada ground-truth files have one `image_name<TAB>text` line per image. Missing datasets are
skipped with a message. Chars74K has no official split, so it is shuffled and split 70/30 with the
config seed.

`python src/datasets.py <name> [--split test] [--limit N]` prints a few samples from any dataset
(`crops`, `kannada_synth`, `indicstr12_kannada`, `mjsynth`, `iiit5k`, `chars74k_english`,
`chars74k_kannada`).

Google Drive does not have space for the full Kannada synthetic set or MJSynth, so
`notebooks/train_crnn.ipynb` downloads them only to the Colab disk and saves a random subset of
`SUBSET` words to Drive. For Kannada, `src/subset_words.py --zip` extracts only the chosen images from the
4.6 GB archive. For MJSynth, `src/mjsynth_subset.py` streams the `.tar.gz`, stops after `SUBSET` images
and can resume a broken download. The MJSynth link is on https://www.robots.ox.ac.uk/~vgg/data/text/.

### Kannada charset

`python src/build_charset.py` counts aksharas in `vocabulary.txt` and in the training split of
`labels.tsv`. It keeps the most frequent aksharas that together cover 99% of all occurrences and writes
them to `charsets/kannada.txt`. For models A–C, any akshara outside this list becomes the class
`<rare>`. The CRNN uses every Kannada code point instead (`charsets/kannada_alphabet.txt`). The English
charset is `a–z`, `0–9` and `&`.

## Training

**A and B.** Kannada characters are rendered on Colab with `notebooks/render_chars.ipynb`, because
Kannada rendering needs Pillow with RAQM. Free fonts are used (Noto Sans/Serif Kannada, Lohit Kannada,
DejaVu, Liberation) with small rotation, blur, noise and scaling. `python src/classical.py --script
<english|kannada>` trains both models on HOG features of 32×32 characters, with at most
`char_per_class` samples per class and 10% held out for validation. The forest has 50 trees and at
least 5 samples per leaf; even then the Kannada forest needs about 1.6 GB of RAM.

**C.** Trained on exactly the same characters as A and B (checked with the fingerprint), using Adam and
early stopping.

**D.** Trained in two stages: pre-training on the public synthetic words (5% held out), then
fine-tuning on the training split of my crops (15% of those photos held out). The network reduces the
width by 4, so a word gives `width / 4` output frames; CTC cannot learn words longer than that, so these
are counted before training. English pre-training was done on Colab (190,000 MJSynth words, 14 epochs,
about 1 hour 50 minutes). On the laptop CPU one epoch of 60,000 words takes about 45 minutes, which is
too slow.

## Evaluation

The final evaluation was run on 21 November 2026. The outputs are in `results/`: `results.csv` (all
metrics with 95% bootstrap confidence intervals), `errors.csv` (400 sample errors), charts and confusion
matrices.

### Results (no preprocessing)

CER is counted over aksharas for Kannada and letters for English. It can go above 100% when a model
outputs more units than the word has. The ranges in brackets are 95% confidence intervals.

| Model | Kannada, own photos (114 words) | Kannada, IndicSTR12 (270) | English, own photos (130) | English, IIIT5K (3,000) |
|---|---|---|---|---|
| A: HOG + SVM | CER 128% [119–138], exact 0.0% | CER 118% [113–123], exact 0.0% | CER 85% [79–91], exact 8.5% | CER 57% [55–60], exact 19.5% |
| B: HOG + Random Forest | CER 135% [125–146], exact 0.0% | CER 121% [116–126], exact 0.4% | CER 84% [79–90], exact 6.9% | CER 61% [60–64], exact 15.5% |
| C: CNN | CER 140% [130–150], exact 0.0% | CER 129% [124–134], exact 0.4% | CER 85% [79–91], exact 6.9% | CER 56% [54–58], exact 21.2% |
| D: CRNN | CER 37% [30–43], exact 36.8% | CER 33% [29–37], exact 41.5% | CER 51% [42–59], exact 25.4% | CER 10% [9–11], exact 71.5% |

### Segmentation (A–C)

| | Kannada, own | Kannada, IndicSTR12 | English, own | English, IIIT5K |
|---|---|---|---|---|
| Correct number of segments | 23.7% | 26.7% | 30.8% | 48.0% |
| Correct number and IoU ≥ 0.5 for every box | 7.6% (105 words) | – | 16.1% (93 words) | – |
| Character accuracy on correctly segmented words (A / B / C) | 4.6 / 4.6 / 1.5% | 13.7 / 9.7 / 3.5% | 54.3 / 47.0 / 55.0% | 70.6 / 65.1 / 73.3% |

### Speed and size (CPU)

| Model | Time per word | Size (Kannada) | Size (English) |
|---|---|---|---|
| A: SVM | 2.7 s Kannada, 10–12 ms English | 246 MB | 10.6 MB |
| B: Random Forest | 15 ms | 27.9 MB (about 1.6 GB in RAM) | 2.2 MB |
| C: CNN | 2–4 ms | 7.4 MB | 0.7 MB |
| D: CRNN | 9–14 ms | 32.9 MB | 32.9 MB |

### Observations

1. The CRNN (D) is the most accurate model on all four test sets. Its CER is two to six times lower
   than A–C and the confidence intervals do not overlap.
2. Segmentation is the main weakness of A–C. Only about a quarter of Kannada words and a third of
   English sign words are cut into the right number of pieces, and even fewer match the true boxes.
   Kannada is harder because vowel signs and conjuncts are attached to or float near the consonant.
3. Even when a Kannada word is segmented correctly, A–C get only 2–14% of the characters right, since
   they were trained only on rendered font characters. For English, where Chars74K photo letters were
   also used, they get 47–73% right.
4. Preprocessing does not help. None of the settings improves on no preprocessing, and CLAHE makes every
   model worse (for example D on IIIT5K drops from 71.5% to 59.7% exact).
5. My own sign photos are harder than IIIT5K for English (D: 25% vs 72% exact) because of stylised
   fonts, low resolution and only 159 fine-tuning words.
6. Tuning the segmenter settings on the training split only raised the correct-segment rate from 31% to
   35% (English) and from 24% to 32% (Kannada), so the gap to the CRNN is not just a tuning problem.
7. The Kannada SVM is too slow for interactive use (2.7 s per word, 246 MB), because one-vs-one
   classification over 859 classes compares about 370,000 class pairs.

### Running the evaluation

```
python src/evaluate.py
python src/evaluate.py --scripts kannada --models A B --limit 500
```

`src/evaluate.py` runs every model in `models/` on the photo test split, IndicSTR12 (Kannada) and IIIT5K
(English), once without preprocessing and once each with CLAHE, sharpening and deskew. It reports:

- segmentation success for A–C (correct segment count, and the stricter IoU ≥ 0.5 version on the boxed
  words);
- character accuracy for A–C on correctly segmented words;
- CER, WER and exact match for all models;
- 95% bootstrap confidence intervals, resampled by photo;
- the share of `<rare>` units in each test set;
- time per word, model size and training time.

It also saves a sample of errors for manual sorting (segmentation failure, confusable characters,
dropped vowel sign, split/merged akshara, stylised font), comparison charts and confusion matrices for
A–C.

## Application

```
python app/app.py
```

- *File → Open image* opens a photo. Ctrl + mouse wheel zooms and *Zoom to fit* resets the view.
- Drag a rectangle around a word to select it. *Tools → Suggest boxes (MSER)* shows dashed candidate
  boxes that can be clicked, but it often misses words.
- Choose the model (A, B, C, D or All), the script and any preprocessing, then click *Run*. Each model
  adds a row with its text, confidence and time.
- The preview shows the segments that A–C classified.
- *File → Export results to CSV* saves the session's results.
- *Tools → Review automatic labels* opens the label review tool.

Models are loaded from `models/` when first used; missing ones are greyed out. The Kannada Random
Forest needs about 1.6 GB of RAM. Confidence values are calculated differently for each model, so they
should only be compared within the same model.

## Project Structure

```
config.yaml            settings: paths, image height, charsets, seed, segmenter, training
charsets/              english.txt, kannada.txt, kannada_alphabet.txt
data/                  photos/, ocr_cache/, crops/, public/, rendered/
src/                   akshara, preprocess, autolabel, make_crops, datasets, build_charset, render_chars,
                       mjsynth_subset, subset_words, segment, classical, train_cnn, train_crnn, models,
                       metrics, evaluate, plots
notebooks/             render_chars, train_cnn, train_crnn (Colab)
app/app.py             desktop application (Tkinter)
app/review_labels.py   label review tool (Tkinter)
requirements-label.txt packages for the labelling environment (.venv-label, Python 3.13)
models/                trained .joblib / .pth + .json files
results/               evaluation outputs
```
