# VisionCaption

Upload an image and get a natural-language caption from the Salesforce BLIP model.
Runs locally (no API key). Flask backend + a custom single-page UI.

## Run in VS Code

1. Open this folder in VS Code (File > Open Folder).
2. Open a terminal (Ctrl+`) and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS / Linux
   ```
3. Install PyTorch (CPU build is smallest), then the rest:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
4. Start the app:
   ```
   python app.py
   ```
5. Open http://localhost:5000. The first start downloads the model (about 990 MB); the
   status pill in the top-right shows progress. Later starts use the cache.

**Try the UI without the model:** set `CAPTION_MOCK=1` (PowerShell: `$env:CAPTION_MOCK=1`)
before `python app.py`, or pick "VisionCaption (mock, no model)" in the Run and Debug panel.

## Features
- Drag and drop, click to browse, or paste an image (Ctrl+V)
- Short caption or detailed description, plus length and beam-count controls
- Live model status, scanning animation while it works, caption typed out on arrival
- Copy, download as .txt, read aloud, describe again (Ctrl+Enter)
- Session strip: click any earlier image to bring its caption back
- Redesigned glass-style UI: aurora background, light/dark toggle (remembered), animated scan effect
- Result card with keyword tags, word count, time, device and beam stats; "How it works" section
- Upload checks: type, size (10 MB), dimensions; light/dark theme follows your system

## Structure
```
app.py              Flask server and API (/api/caption, /api/status, /api/config)
model.py            BLIP loading (background thread) and inference
image_utils.py      Upload validation and RGB conversion
config.py           Limits, modes and generation defaults
static/index.html   The whole UI (HTML, CSS, JS)
tests/              pytest suite   ->  pytest tests -v
evaluation/         BLEU-1, BLEU-4, ROUGE-L pipeline (see docs/)
docs/               Architecture and testing notes
```

## Notes
- BLIP Base can make mistakes; "Detailed" mode is only a prompt hint, so output may stay short.
- CPU inference takes roughly 2-10 s per image.
- The UI loads two Google Fonts; offline it falls back to system fonts.
- Evaluation: put images in `assets/`, edit `evaluation/sample_references.json` with real
  reference captions, then `python evaluation/evaluate.py --images-dir assets --references evaluation/sample_references.json --output-dir outputs`.

Model: Salesforce/blip-image-captioning-base (BSD-3-Clause). Full write-up: `docs/PROJECT_EXPLANATION.md`. Original structure and
validation/evaluation modules adapted from a friend's VisionCaption AI project.
