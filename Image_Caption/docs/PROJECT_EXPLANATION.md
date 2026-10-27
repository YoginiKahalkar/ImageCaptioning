---
title: "VisionCaption AI — Complete Project Explanation & Architecture"
---

# 1. What the project is

**VisionCaption AI** is a Generative-AI web application. The user uploads an image and the app writes a natural-language description of it (for example, *"A dog running through a grassy field."*). This task is called **image captioning**, a vision-language task that combines computer vision (understanding pixels) and natural-language generation (writing text).

Key properties:

- **Model:** Salesforce **BLIP** (`Salesforce/blip-image-captioning-base`), a pretrained transformer from Hugging Face.
- **Runs locally:** no API key, no cloud calls at inference time, images never leave the computer.
- **Stack:** Python, Flask (backend), PyTorch + Hugging Face Transformers (AI), Pillow (images), HTML/CSS/JavaScript (UI), NLTK + rouge-score (evaluation), pytest (tests).

# 2. Technology stack — and why each piece is used

| Technology | Purpose |
|---|---|
| Python 3 | Language of the whole backend |
| Flask | Lightweight web server and REST API |
| PyTorch | Tensor / neural-network engine that executes the model |
| Transformers (Hugging Face) | Downloads and wraps BLIP (`BlipProcessor`, `BlipForConditionalGeneration`) |
| Pillow (PIL) | Opens, validates and converts images |
| NumPy | Numerical dependency of the ML stack |
| NLTK, rouge-score | Computing BLEU and ROUGE metrics |
| pytest | Automated tests |
| HTML + CSS + vanilla JS | The front end — no framework, no build step |
| threading | Loads the model in the background without freezing the server |

# 3. Folder structure

```
VisionCaption/
├── app.py                 Flask server + API routes
├── model.py               BLIP loading, inference, caption clean-up, mock mode
├── image_utils.py         Upload validation + RGB conversion
├── config.py              All constants and defaults
├── requirements.txt       Dependencies
├── static/index.html      Entire UI (HTML + CSS + JS)
├── evaluation/
│   ├── evaluate.py        BLEU-1, BLEU-4, ROUGE-L pipeline
│   └── sample_references.json   Reference captions (placeholders)
├── tests/                 pytest suite (API, validation, cleaning)
├── docs/                  architecture.md, testing.md, this file
├── assets/  outputs/      Evaluation images / result files
└── .vscode/launch.json    One-click run (normal and mock mode)
```

# 4. High-level architecture

```
BROWSER (static/index.html)
   │  POST /api/caption (image + mode + sliders)     GET /api/status, /api/config
   ▼
FLASK (app.py)  ── validates request, clamps settings, returns JSON
   │                         │
   ▼                         ▼
image_utils.py           model.py ── BLIP (PyTorch + Transformers) ── weights cached from HF Hub
(Pillow checks)
         config.py supplies every constant
```

It is a classic **client → API → ML-inference** architecture with a clean separation of concerns: presentation (HTML), transport/validation (Flask), data-handling (image_utils), intelligence (model), settings (config).

# 5. The AI model — BLIP explained

**BLIP = Bootstrapped Language-Image Pre-training** (Li et al., ICML 2022, Salesforce).

1. **Vision encoder — ViT (Vision Transformer).** The image is resized (to 384×384), normalised, and cut into small 16×16 patches. Each patch becomes a vector; self-attention layers learn relationships between patches and output a rich "image embedding".
2. **Text decoder — BERT-style transformer with cross-attention.** It generates the caption one token at a time. At each step, *cross-attention* lets the text decoder look at the image embedding, so every word is grounded in the picture.
3. **Pre-training idea ("bootstrapping").** BLIP was trained on large web image–text data. A *captioner* writes synthetic captions and a *filter* removes noisy ones, so the model learns from cleaner data — this is the "bootstrapped" part. It was trained with three objectives: image-text contrastive, image-text matching, and language modelling (caption generation). We use the caption-generation head.
4. **Model used here:** *base* size, ~990 MB, fine-tuned for captioning on COCO. It is used **as-is for inference — no training in this project**.

## Generation settings (decoding)

| Setting | Value | Meaning |
|---|---|---|
| `num_beams` | 4 (1–8 in UI) | **Beam search**: keeps the 4 most promising partial sentences at each step instead of greedily picking one word. Better quality, slower. |
| `max_new_tokens` | 50 standard / 150 detailed | Hard cap on caption length |
| `min_length` | 5 | Avoids one-word answers |
| `repetition_penalty` | 1.5 | Discourages loops like "a dog dog dog" |
| `length_penalty` | 1.0 | Neutral preference for length in beam scoring |

Because beam search is deterministic (no random sampling), the same image + settings give the same caption.

**Detailed mode** is a *conditional prompt*: the text "a detailed description of" is fed to the decoder as the start of the sentence, which nudges BLIP toward longer output. The prompt is later stripped from the result. (Honest limitation: BLIP-base is not instruction-tuned, so this is only a hint.)

# 6. File-by-file explanation

## 6.1 `config.py` — configuration

Holds every tunable value so the rest of the code has no "magic numbers":

- Model metadata: `MODEL_ID`, framework, architecture, license, paper.
- Upload rules: `ALLOWED_EXTENSIONS = jpg, jpeg, png, webp`; `MAX_UPLOAD_SIZE_MB = 10` (and bytes); image side limits `MIN_IMAGE_DIMENSION = 10`, `MAX_IMAGE_DIMENSION = 10000`.
- Modes: `"Standard Caption"` and `"Detailed Description"`.
- `GenerationConfig` dataclass: the decoding defaults above plus slider ranges `max_tokens_range=(10,200)`, `beam_range=(1,8)` and the `detailed_prompt`.
- Logging format constants and a few unused UI strings left from an earlier Streamlit version.

## 6.2 `image_utils.py` — safe image handling

Never trust uploads. `validate_uploaded_file(filename, bytes)` runs five checks and returns `(image, None)` or `(None, "friendly error")`:

1. **Extension** allow-list (case-insensitive).
2. **File size** ≤ 10 MB.
3. **Content check**: `Image.open` → `verify()` (structure check) → re-open (verify leaves the stream unusable) → `load()` (full decode). This catches fake or corrupted files, e.g. a text file renamed to `.jpg`. Errors are converted to readable messages.
4. **Dimensions**: between 10 px and 10 000 px per side (blocks tiny/huge images and decompression-bomb style inputs).
5. **RGB conversion** (`convert_to_rgb`): the model needs 3-channel images. RGBA images are composited onto a **white** background (naive conversion would turn transparency black); palette/greyscale modes use `.convert("RGB")`.

Helpers: `validate_extension`, `validate_file_size`, `get_file_size_mb`, `validate_dimensions`, `get_image_info` (width, height, mode, format returned to the UI).

## 6.3 `model.py` — the AI engine

- **`CaptionResult` dataclass**: caption, inference time, model id, device, mode, parameters, success flag, error message. The function never raises to the caller; failures come back as `success=False`.
- **Loading state**: a lock-protected dictionary `{status, message, device}` with states `idle → loading → ready | error`, read by `/api/status`.
- **`load_model()`**: lazy, once-only (double-checked locking with `_load_guard`). Detects CUDA GPU, otherwise CPU; loads `BlipProcessor` and `BlipForConditionalGeneration`, moves the model to the device and calls `model.eval()` (turns off dropout). Stores `(processor, model, device)` in the global `_bundle`.
- **`start_background_load()`**: starts a daemon thread so Flask is responsive immediately and the UI can show "Loading model weights…". In mock mode it just marks the status ready.
- **`generate_caption()`**:
  1. Chooses tokens/prompt/beams from mode and arguments.
  2. `processor(image, text=prompt, return_tensors="pt")` → resizes, normalises, tokenises, builds PyTorch tensors.
  3. Moves tensors to the device.
  4. `model.generate(...)` inside `torch.no_grad()` (no gradient bookkeeping → faster, less memory), timed with `time.perf_counter()`.
  5. `processor.decode(..., skip_special_tokens=True)` → text.
  6. `_clean_caption()`.
- **`_clean_caption()`**: collapses whitespace, strips the conditioning prompt, capitalises the first letter, adds a full stop if missing, returns `""` if nothing remains (which triggers a friendly "empty caption" error).
- **Mock mode** (`CAPTION_MOCK=1`): returns a fixed sample caption after 1.2 s so UI and API can be developed/tested without the 990 MB download.

## 6.4 `app.py` — Flask server

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `static/index.html` |
| `/api/config` | GET | Sends limits, modes, slider ranges and defaults so the UI never hard-codes them |
| `/api/status` | GET | Model loading status (polled by the UI) |
| `/api/caption` | POST | The main endpoint |

`/api/caption` logic: get `image` from the multipart form → 400 if missing → validate (422 with message if invalid) → read `mode` (fall back to Standard if unknown) → read and **clamp** `max_new_tokens` and `num_beams` into allowed ranges (400 if not integers) → `generate_caption` → 500 on model failure, otherwise JSON: `caption, seconds, device, mode, model, params, image`.
Other details: `MAX_CONTENT_LENGTH` makes Flask reject oversized bodies early (custom 413 handler); `start_background_load()` is called at startup; the server binds to `127.0.0.1` (local only), port 5000 (override with `PORT`).

HTTP status codes used: 200 OK, 400 bad request, 413 too large, 422 unprocessable (invalid image), 500 server/model error.

## 6.5 `static/index.html` — the user interface (redesigned)

A single file with three parts.

**HTML structure:** navigation bar (brand, live model-status chip, theme toggle) → hero (badges, headline, four stats) → two-column workbench (left: image drop-zone; right: style selector, fine-tune sliders, Describe button, error box, result card) → "How it works" four-step cards → "This session" gallery → footer.

**CSS highlights (new look):**
- Design tokens as CSS variables for light and dark themes (`:root` and `[data-theme=dark]`).
- Animated **aurora** blurred gradient blobs and a faded grid background.
- **Glassmorphism** cards (`backdrop-filter: blur`), gradient brand colour, serif display font (Fraunces) with Plus Jakarta Sans for the UI.
- Animated upload icon, dashed drop-zone that highlights on drag, corner crop-marks, and a **scanning-beam animation** while the model works.
- Gradient result card with decorative quote mark; typing caret; keyword chips with pop-in animation.
- Fully responsive grid (two columns → one on small screens); `prefers-reduced-motion` support; visible focus rings.

**JavaScript logic (vanilla, no libraries):**
- *Theme*: follows system preference, toggle saved in `localStorage` (wrapped in try/catch).
- *Status polling*: `pollStatus()` calls `/api/status` every 1.5 s while loading and updates the chip colour/text.
- *Config*: `init()` fetches `/api/config` and sets slider limits/defaults; switching to Detailed changes the token default.
- *Choosing images*: click, keyboard (Enter/Space), drag-and-drop and **paste (Ctrl+V)**; client-side type/size checks give instant feedback before upload; preview shows name, pixel size and MB.
- *Generate*: `run()` posts `FormData` with `fetch`, toggles busy state, handles errors (including "server not reachable"). Ctrl+Enter is a shortcut.
- *Result display* (`show()`): types the caption word-by-word, extracts **keyword tags** (client-side stop-word removal), shows word count / time / device / beams.
- *Actions*: Copy (Clipboard API with fallback), Download `.txt` (Blob), Read aloud (Web Speech API), Describe again.
- *Session strip*: keeps the last 12 images + captions in memory; clicking restores one.
- *Captions-today counter*: stored per day in `localStorage`.

## 6.6 `evaluation/evaluate.py` — measuring quality

Offline pipeline: loads reference captions (several human captions per image) → generates a caption for each image → scores:

- **BLEU-1**: unigram (single word) overlap with references.
- **BLEU-4**: overlap of 1–4-word sequences (geometric mean) — stricter, rewards fluent phrasing. Smoothing (method1) prevents zero scores on short captions.
- **ROUGE-L**: F-measure of the *longest common subsequence*, averaged across references (with stemming).

Outputs `outputs/evaluation_results.csv` (per image) and `evaluation_summary.json` (averages, median/min/max inference time, model load time, settings). Run: `python evaluation/evaluate.py --images-dir assets --references evaluation/sample_references.json --output-dir outputs`. The sample references are **placeholders** — use real Flickr8k/COCO captions before quoting scores.

## 6.7 `tests/` — automated tests

- `test_image_utils.py`: extension, size, corrupt data, dimension limits, RGBA/palette conversion, full pipeline — all with in-memory synthetic images.
- `test_caption_validation.py`: `_clean_caption` (whitespace, capitalisation, prompt stripping, empty input, duplicate full stops).
- `test_api.py`: runs the Flask test client in mock mode — success, wrong file type (422), missing file (400).
- `conftest.py` adds the project root to `sys.path`. Run: `pytest tests -v`.

# 7. End-to-end flow

1. `python app.py` → Flask starts; background thread begins loading BLIP (first run downloads ~990 MB, cached afterwards).
2. Browser opens `/`; JS fetches `/api/config` and polls `/api/status` → chip turns green "Model ready · CPU/CUDA".
3. User drops an image; JS validates type/size and shows a preview.
4. User picks style and presses **Describe image**; JS POSTs the multipart form.
5. Flask validates (extension → size → decode → dimensions → RGB), clamps settings.
6. BLIP: ViT encodes image → decoder generates tokens with beam search → decoded and cleaned.
7. JSON returns; the UI animates the caption, shows keywords and stats, stores it in the session strip.
8. User copies, downloads, listens, or regenerates.

# 8. Security & robustness

- Extension allow-list **and** real content decoding (not trusting file names).
- Size limits at Flask level and in validation; dimension limits.
- Server bound to localhost; no uploaded file is written to disk.
- All text inserted into the page uses `textContent` (prevents XSS from captions or file names).
- Errors are JSON with clear messages; no tracebacks reach the user.
- Thread-safe, once-only model loading.

# 9. Limitations (be honest in your viva)

- BLIP-base can misidentify objects, miss details, and cannot read text or count reliably.
- "Detailed" mode is only a prompt hint.
- CPU inference takes roughly 2–10 s; GPU is much faster.
- English only; outputs reflect biases in the training data.
- Evaluation numbers are meaningless until real reference captions are supplied.

# 10. What was changed in this updated version

- **UI completely redesigned** (`static/index.html`): aurora/glass look, gradient brand, new fonts, larger hero with stats, manual light/dark toggle, redesigned drop-zone with scan animation, gradient result card, keyword tags, stats tiles, "How it works" section, polished session gallery, mobile-friendly.
- **Same API contract** — `app.py`, `model.py`, `image_utils.py` untouched, so all tests still pass.
- `docs/architecture.md` rewritten (it still described an old Streamlit version); README updated; this explanation added.

# 11. Possible future improvements

BLIP-large or BLIP-2 for better quality; multiple captions per image; Hindi/other-language translation of captions; batch upload; history saved to a database; Docker image; deployment on Hugging Face Spaces; fine-tuning on a custom dataset.

# 12. Likely viva questions and short answers

- **What model do you use and why?** BLIP-base: strong captioning quality, small enough for laptops, open licence, runs offline.
- **What is beam search?** Keeps the top-k partial sentences at each step; picks the best-scoring full sentence — better than greedy.
- **Why `model.eval()` and `torch.no_grad()`?** Inference only: disables dropout and gradient tracking, saving time and memory.
- **Why load the model in a thread?** So the server and UI respond immediately and show loading progress.
- **How is quality measured?** BLEU-1/4 and ROUGE-L against human captions.
- **Why RGBA → white background?** Avoids black artefacts from transparent pixels.
- **How do you handle bad uploads?** Five-step validation with specific error messages and HTTP 400/413/422.
