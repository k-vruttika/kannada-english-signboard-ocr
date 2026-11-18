# OCR for Kannada–English Shop Signs

MCA internship project: *Optical Character Recognition – Quantification with Various ML
Approaches*. The project recognises cropped word images from Kannada and English shop signs using
four models, all trained from scratch:

| | Model | Requires segmentation |
|---|---|---|
| A | HOG + SVM | yes |
| B | HOG + Random Forest | yes |
| C | Small CNN | yes |
| D | CRNN (CNN + BiLSTM + CTC) | no |

Models A–C share a single conventional segmenter (Otsu thresholding, connected components and
projection profiles), applied unchanged to both scripts. The central question of the project is how
often this segmenter fails on Kannada, where a single akshara frequently consists of several
components and adjacent characters often touch. The primary metric is therefore the segmentation
success rate. No pretrained OCR models or weights are used.

## Project Status

The project is in progress. No results have been produced yet.

**Completed**

- Core modules: akshara splitting, preprocessing, metrics, segmentation, dataset loaders, charset
  builder, character rendering, model definitions, Colab training notebooks, the evaluation pipeline
  and the desktop application.
- 360 shop-sign photos collected (`data/photos/`).
- Kannada vocabulary extracted from the IIIT synthetic Kannada set, and the Kannada charset built
  from it.
- Training characters rendered for both scripts.
- Models A and B trained for English and Kannada.
- Model C trained for English and Kannada.
- Model D (Kannada) pre-trained on a subset of the synthetic Kannada words.

**To be done**

- Annotation of the photos: automatic labelling (GCV + PaddleOCR), then checking every word by hand in `app/review_labels.py`.
- Word crops and the train/test split, generated from the annotations.
- Rebuilding the Kannada charset with the train-split labels, and retraining A–C if it changes.
- Downloading the public test sets (IndicSTR12 Kannada, IIIT5K).
- Fine-tuning model D (Kannada) on the annotated crops.
- Training model D (English) on an MJSynth subset.
- Running the evaluation and analysing the results.

## Setup

The project has been tested with Python 3.13 on Windows. All commands are run from the project root.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, the PyPI build of PyTorch is CPU-only, which is sufficient for local use. GPU training is
done in Google Colab.

The scripts print Kannada text. This works in an interactive terminal, but raises
`UnicodeEncodeError` when output is piped or redirected to a file. Enabling Python's UTF-8 mode in the
terminal avoids this:

```
set PYTHONUTF8=1          (cmd)
$env:PYTHONUTF8 = "1"     (PowerShell)
```

Each core module includes a self-check:

```
python src/akshara.py
python src/preprocess.py
python src/metrics.py
python src/segment.py
python src/models.py
```

All settings are defined in `config.yaml`: paths, image height, charsets, seeds, data split, segmenter,
training and Random Forest parameters.

English recognition is case-insensitive. Labels are lowercased on load and all metrics compare
lowercased text, so `SHOP` and `shop` are treated as the same word.

## Trained Models

Trained model files are not stored in the repository. They are published as release assets on the
[GitHub Releases page](https://github.com/k-vruttika/kannada-english-signboard-ocr/releases) and should
be placed in the `models/` directory. The datasets are distributed separately.

## Workflow

The pipeline is listed below in order. Steps marked **Colab** run in a notebook on a T4 GPU; all other
steps run locally. Each stage is marked **Done**, **Partly done** or **To be done**.

**1. Obtain the data** (Partly done)

1. Download the IIIT synthetic Kannada set on Colab and extract its vocabulary. *(Done)*
2. Download IIIT5K and place `test/` and `testdata.mat` in `data/public/IIIT5K/`. *(To be done)*
3. Download IndicSTR12 `test.zip` and extract it inside `data/public/indicstr12_kannada/`, which
   produces `test/images/` and `test/test_gt.txt`. *(To be done)*
4. Optionally, add Chars74K English and Kannada (see [Public datasets](#public-datasets)).
5. Locate the MJSynth download link. This dataset is never downloaded in full, either locally or to
   Google Drive. *(To be done)*

**2. Label and crop the photos** (To be done)

- [ ] Put the photos in `data/photos/`.
- [ ] Set up the labelling environment once, and set your GCV key (see [Automatic labelling](#automatic-labelling-gcv--paddleocr)).
- [ ] `.venv-label/bin/python src/autolabel.py` finds, reads and crops every word.
- [ ] `python app/review_labels.py`: accept, correct or delete each word until you have enough.
- [ ] While reviewing, draw or check `char` boxes for about 50 words per script.

Label Studio still works instead (see [Your photos and Label Studio](#your-photos-and-label-studio)),
but it is all manual.

**3. Build the Kannada charset** (Done, to be repeated after annotation)

1. **Colab**: run `notebooks/train_crnn.ipynb` with `SCRIPT = 'kannada'` up to the cell marked
   *training reads the public words*.
2. Download `data/public/kannada_synth/vocabulary.txt` from Drive to the same path locally.
3. Run `python src/build_charset.py` and review the aksharas it excludes.
4. Upload the new `charsets/kannada.txt` to Drive. All models depend on this file, so steps 4–7 must
   be repeated whenever the charset changes.

**4. Render the training characters** (Done)

1. **Colab**: run `notebooks/render_chars.ipynb`. It should print `raqm: True`.
2. Download `data/rendered_kannada.zip` and `data/rendered_english.zip` and extract both into
   `data/rendered/`.

**5. Train models A and B** (Done)

1. `python src/classical.py --script english`
2. `python src/classical.py --script kannada`
3. Record the two fingerprints printed by these commands.

**6. Train model C** (Done)

1. **Colab**: run `notebooks/train_cnn.ipynb` once with `SCRIPT = 'english'` and once with
   `SCRIPT = 'kannada'`.
2. Confirm that each printed fingerprint matches the one from step 5 for the same script. A mismatch
   means A, B and C were not trained on the same characters.
3. Download `models/cnn_<script>.pth` and the matching `.json` into `models/`.

**7. Train model D** (Partly done)

1. **Colab**: run `notebooks/train_crnn.ipynb` with `SCRIPT = 'kannada'`. Without annotated crops, the
   notebook only pre-trains and saves `models/crnn_kannada_pretrained.pth`. *(Done)*
2. Once crops exist, run it again. It reuses the pre-trained model (`REUSE_PRETRAINED`) and fine-tunes
   it on the train split, saving `models/crnn_kannada.pth`. *(To be done)*
3. Repeat with `SCRIPT = 'english'` and `MJSYNTH_URL` set, and confirm that the subset contains tens
   of thousands of unique words. *(To be done)*

Until fine-tuning is complete, the application and the evaluation fall back to the pre-trained
Kannada model and label it as pre-trained only.

**8. Evaluate** (To be done)

See [Evaluation](#evaluation).

### Files required on Google Drive

Drive requires `config.yaml`, `requirements.txt`, `src/`, `notebooks/`, `charsets/` and `data/crops/`,
plus `data/public/chars74k/` if Chars74K is used. `.venv/`, `data/photos/`, `data/rendered/`, `models/`
and the other public datasets are not needed. The notebooks write their archives, models and
`vocabulary.txt` back to Drive.

## Data

All paths below are the defaults from `config.yaml`. If the directories are named differently, update
the configuration rather than moving the files.

### Automatic labelling (GCV + PaddleOCR)

Typing 300+ words by hand takes too long, so an existing OCR system writes a first guess for every word
and a person checks each one. As the internship guide advised, the automatic labels are only an
initialisation: a word reaches `data/crops/labels.tsv` only after it has been accepted or corrected in
the review tool. Otherwise the models would be compared with another OCR system's output instead of
with ground truth. Kannada comes from **Google Cloud Vision** (the strongest baseline for Kannada) and
English from **PaddleOCR**. Neither is used by models A–D; they only produce draft labels.

**Setup (once).** PaddlePaddle has no wheels for Python 3.14, so the labelling script has its own
Python 3.13 environment, `.venv-label`:

```
pip install uv
uv python install 3.13
uv venv --python 3.13 .venv-label
uv pip install --python .venv-label -r requirements-label.txt
```

On Windows use `.venv-label\Scripts\python` wherever this section says `.venv-label/bin/python`.

**GCV key.** In Google Cloud Console, create a project, enable the *Cloud Vision API* (billing must be
on; the first 1000 images a month are free), and create an API key under *APIs & Services →
Credentials*. Put it in the environment, never in a file in the repository:

```
export GCV_API_KEY=...         (Linux)
set GCV_API_KEY=...            (cmd)
$env:GCV_API_KEY = "..."       (PowerShell)
```

A service-account JSON file in `GOOGLE_APPLICATION_CREDENTIALS` works as well.

**Run.**

```
.venv-label/bin/python src/autolabel.py
.venv-label/bin/python src/autolabel.py --limit 5     a quick test on five photos
```

For each photo it keeps the Kannada words found by GCV and the English words found by PaddleOCR.
PaddleOCR words that lie on a Kannada word are dropped, because the English model reads Kannada as junk.
Every word is cropped with a small margin into `data/crops/`, and the candidates are listed in
`data/crops/auto_labels.tsv`. For Kannada words, GCV's per-letter boxes are grouped into aksharas and
offered as draft `char` boxes. The raw OCR output is cached in `data/ocr_cache/`, so a rerun costs
nothing and sends nothing. The train/test split by photo is fixed here, with the config seed, before
anyone looks at the labels. Once a review has started, `autolabel.py` refuses to run again unless you
pass `--force`, because new crops would not match the decisions already made.

**Review.**

```
python app/review_labels.py          (or Tools → Review automatic labels in the app)
```

The window shows the photo with the word marked, the crop enlarged, and the OCR text in an edit box.
Below the box, Kannada text is also drawn with Pillow, because Tk cannot shape Kannada conjuncts on
every system. That drawing shows how the label will actually look.

| Key | Action |
|---|---|
| Enter | accept the word as typed, then go to the next one |
| Ctrl+D | delete the crop (not a word, cut off, unreadable, wrong language) |
| PgUp / PgDn | previous / next word |
| Ctrl+U | next unchecked word |
| Ctrl+G / Ctrl+R | grow the crop when a letter is cut off / reset it |
| drag, right-click on the crop | draw / remove a `char` box |
| Ctrl+B | mark the `char` boxes as checked |
| Ctrl+X | clear the `char` boxes |

Every decision is saved straight away to `data/crops/review.json`, and `labels.tsv` is rewritten with
the accepted words only. You can close the window and carry on later. The script follows from the text
(any Kannada letter means Kannada), and English letters outside `a–z 0–9 &` get a warning. `char` boxes
reach `labels.tsv` only for words marked as checked, and only when their count equals the number of
units in the word. In the report, say that the labels were initialised with GCV and PaddleOCR and that
every word was then checked and corrected by hand.

OCR systems miss some words. That does not matter here, because the aim is a few hundred correct
words, not every word on every sign.

### Your photos and Label Studio

This is the fully manual alternative to the automatic labelling above. The photos stay in
`data/photos/`, with the same file names as uploaded to Label Studio. The annotations must provide:

- one bounding box per word, with its script (`kannada` or `english`) and transcription;
- for a subset of about 50 words per script, one box per letter or akshara, used to check
  segmentation against the ground truth.

`src/make_crops.py` reads the Label Studio JSON export (`data/labelstudio.json`). Both routes produce
`data/crops/*.png` and `data/crops/labels.tsv` with the columns
`path, text, script, photo_id, n_units, split, char_boxes`. The data is split 70/30 by photo using the
configured seed, and the test split is never used for training or tuning.

### Public datasets

| Dataset | Purpose | Location | Status |
|---|---|---|---|
| IIIT-Synthetic-IndicSTR-Kannada | CRNN pre-training, charset vocabulary | Colab only; locally, only `vocabulary.txt` in `data/public/kannada_synth/` | Done |
| IndicSTR12, Kannada subset | Kannada public test set | extract `test.zip` inside `data/public/indicstr12_kannada/`, producing `test/images/` and `test/test_gt.txt` | To be done |
| MJSynth (90kDICT32px) | English CRNN pre-training | Colab only, streamed | To be done |
| IIIT5K | English public test set | `data/public/IIIT5K/` with `test/` and `testdata.mat` | To be done |
| Chars74K English | additional characters for A–C (optional) | `data/public/chars74k/English/Img/GoodImg/Bmp/Sample001 ...` | Optional |
| Chars74K Kannada | additional characters for A–C (optional) | `data/public/chars74k/Kannada/Img/GoodImg/Bmp/...` plus `kannada_classes.tsv` | Optional |

The Kannada ground-truth files are read as `image_name<TAB>text`, one line per image. If a Chars74K
directory or the Kannada class map is missing, that dataset is skipped with a message.
`kannada_classes.tsv` contains one line per folder, for example `Sample001<TAB>ಅ`. Chars74K has no
official split, so a fixed shuffle is divided 70/30 using the configured seed.

`python src/datasets.py <name> [--split test] [--limit N]` loads any dataset and prints sample entries
and the shape of one batch. Available names: `crops`, `kannada_synth`, `indicstr12_kannada`,
`mjsynth`, `iiit5k`, `chars74k_english` and `chars74k_kannada`.

### Handling the large datasets on Colab

Google Drive does not have enough space for the full Kannada synthetic set or MJSynth.
`notebooks/train_crnn.ipynb` downloads and extracts them only on the Colab disk (`/content`), then
archives a random subset of `SUBSET` words (200,000 by default) to Drive as
`data/public/<set>_subset.zip`. Later sessions extract this archive directly, and training always reads
from `/content`.

- **Kannada synthetic set.** The link and paths are preconfigured in the notebook. The 4.6 GB archive
  is downloaded with `wget -c`, which displays progress and can resume within the same Colab session.
  Only `train/train_gt.txt` and `train/vocabulary.txt` are extracted from it. `src/subset_words.py --zip`
  then selects `SUBSET` random words and extracts only their images, so the 2 million images are never
  fully unpacked.
- **MJSynth.** `src/mjsynth_subset.py` streams the `.tar.gz` and stops after `SUBSET` readable images,
  then reports the number of unique words retained. A very low count indicates that the archive is
  ordered by word; in that case, delete the archive and add `--every 45` to the command in that cell so
  that every 45th image is kept. The download link is available at
  https://www.robots.ox.ac.uk/~vgg/data/text/.

### Kannada charset

`python src/build_charset.py` counts aksharas in `vocabulary.txt` and in the **train** split of
`labels.tsv` (once it exists), counting each vocabulary word once. It keeps the most frequent aksharas
that together cover `charset_coverage` (0.99) of all occurrences, writes them to
`charsets/kannada.txt`, and reports those excluded. For models A–C, any akshara outside the charset is
mapped to the class `<rare>`. The CRNN uses a separate alphabet: the script also writes every Kannada
code point found to `charsets/kannada_alphabet.txt`, regardless of coverage. The English charset
consists of `a–z`, `0–9` and `&`, and the English CRNN uses the same file.

The current charset was built from the vocabulary alone, since no annotated labels exist yet.

## Training

### Models A and B: HOG + SVM and HOG + Random Forest (local)

Kannada rendering requires Pillow's RAQM layout engine, which is usually unavailable in Pillow on
Windows. The training characters are therefore rendered in Colab with `notebooks/render_chars.ipynb`,
using free fonts (Noto Sans/Serif Kannada, Lohit Kannada, Noto, DejaVu, Liberation). Light rotation,
blur, noise and scaling are applied, and English characters are rendered in both cases.

`python src/classical.py --script <english|kannada>` trains both models on HOG features of 32×32
crops. It uses at most `char_per_class` samples per class and holds out 10% for the reported
validation accuracy. It saves `models/svm_<script>.joblib` and `models/rf_<script>.joblib` and reports
their disk size and approximate memory usage. The forest uses 50 trees with a minimum of 5 samples per
leaf, since scikit-learn stores one value per class in every node. Even with these limits, the Kannada
forest requires approximately 2 GB of RAM.

### Models C and D: CNN and CRNN (Colab, T4 GPU)

`SCRIPT` is set in the second cell of each notebook. Both notebooks import from `src/`, copy their
inputs to `/content`, fix the random seed, use Adam with early stopping on validation loss, and report
the training time. The resulting `.pth` and `.json` files are downloaded into `models/`; the
application and evaluation run them on the CPU.

- **`train_cnn.ipynb`** trains model C on the same characters as A and B, verified by the fingerprint.
- **`train_crnn.ipynb`** trains model D in two stages. It first pre-trains on the public subset,
  holding out 5% for validation, then fine-tunes on the train split of the annotated crops, holding out
  15% of those photos for validation. The network reduces the width by a factor of 4, so each word
  yields `width / 4` output frames. Since CTC cannot learn from words with more code points than frames,
  the notebook counts such words beforehand.

## Evaluation

*Status: to be done.* The evaluation pipeline is implemented but has not been run, as it requires the
annotated test split and the public test sets. The `results/` directory is currently empty.

```
python src/evaluate.py
python src/evaluate.py --scripts kannada --models A B --limit 500
```

`src/evaluate.py` runs every model found in `models/` on the photo test split, on IndicSTR12 (Kannada)
and on IIIT5K (English). Each model is evaluated without preprocessing, then with CLAHE, sharpening and
deskew applied individually. Missing models or test sets are skipped with a message.

Planned metrics:

- **Segmentation success** (A–C): proportion of words whose segment count matches the number of units,
  plus a stricter variant requiring IoU ≥ 0.5 per segment on the boxed subset.
- **Character accuracy** (A–C), measured on correctly segmented words.
- **CER, WER and exact match** for all models. CER is computed over aksharas for Kannada and letters
  for English.
- **95% bootstrap confidence intervals** for each metric.
- **Share of `<rare>` units** in each test set.
- **Efficiency:** recognition time per word on the CPU, model size and training time.

The pipeline will also produce an error sample for manual categorisation (segmentation failure,
confusable characters, dropped vowel sign, split/merged akshara, stylised font), per-script comparison
charts and confusion matrices for A–C.

## Application

```
python app/app.py
```

The desktop application is implemented and runs with the models currently available.

- **Open a photo** via *File → Open image*. Ctrl + mouse wheel zooms, and *Zoom to fit* resets the
  view.
- **Select a word** by dragging a rectangle around it. *Tools → Suggest boxes (MSER)* displays dashed
  candidate boxes that can be clicked to select. This is an aid only and frequently misses words.
- **Run** recognition after choosing a model (A, B, C, D or All), the script and any preprocessing.
  Each model adds a row showing its output text, confidence and recognition time.
- **The preview** shows the segments classified by models A–C.
- **Export** the session's results via *File → Export results to CSV*.
- **Check labels** via *Tools → Review automatic labels*, which opens the review tool described under
  [Automatic labelling](#automatic-labelling-gcv--paddleocr).

Models are loaded from `models/` on first use. Unavailable models are greyed out and named in the
status bar. Model D for English is not yet available, and model D for Kannada currently uses the
pre-trained model. Selecting the Kannada Random Forest displays a notice that it requires approximately
2 GB of RAM. Confidence values are computed differently for each model and should only be compared
within the same model.

## Project Structure

```
config.yaml            settings: paths, image height, charsets, seeds, segmenter, training
charsets/              english.txt, kannada.txt, kannada_alphabet.txt
data/                  photos/, ocr_cache/, crops/, public/, rendered/ (labelstudio.json if used)
src/                   akshara, preprocess, autolabel, make_crops, datasets, build_charset, render_chars,
                       mjsynth_subset, subset_words, segment, classical, models, metrics, evaluate, plots
notebooks/             render_chars, train_cnn, train_crnn (Colab)
app/app.py             Tkinter application
app/review_labels.py   Tkinter tool to check the automatic labels
requirements-label.txt the labelling environment (.venv-label, Python 3.13)
models/                trained .joblib / .pth + .json files
results/               evaluation outputs (generated by evaluate.py; currently empty)
```

