# System Architecture

VisionCaption is a local, three-tier web application: a browser UI, a Flask API
server, and a BLIP vision-language model running in PyTorch. No database, no
cloud API, no API key.

```
┌──────────────────────────────────────────────────────────────┐
│  BROWSER  static/index.html  (HTML + CSS + vanilla JS)       │
│  Upload · Preview · Style/Sliders · Result · Session strip   │
└───────────────┬──────────────────────────────┬───────────────┘
   POST /api/caption (multipart)      GET /api/status · /api/config
                │                              │
┌───────────────▼──────────────────────────────▼───────────────┐
│  FLASK SERVER  app.py                                        │
│  routing · input clamping · JSON errors · 10 MB request cap  │
└───────────┬──────────────────────────────────┬───────────────┘
            │                                  │
   ┌────────▼─────────┐              ┌─────────▼──────────┐
   │  image_utils.py  │              │      model.py      │
   │ ext·size·decode· │              │ background load ·  │
   │ dims · RGB       │              │ generate_caption   │
   └────────┬─────────┘              └─────────┬──────────┘
            │                                  │
        Pillow (PIL)               Transformers + PyTorch
                                              │
                                   BLIP weights (HF Hub cache)

   config.py  ← every constant used by all modules above
```

## Module responsibilities

| File | Role |
|------|------|
| `config.py` | Single source of truth: model id, upload limits, dimension limits, caption modes, `GenerationConfig` defaults. |
| `image_utils.py` | 5-step upload validation pipeline; returns `(image, error)`. |
| `model.py` | Thread-safe one-time model load, live status, inference, caption clean-up, mock mode. |
| `app.py` | Flask routes: `/`, `/api/config`, `/api/status`, `/api/caption`; 413 handler. |
| `static/index.html` | Entire front end. No framework, no build step. |
| `evaluation/evaluate.py` | BLEU-1, BLEU-4, ROUGE-L scoring against reference captions. |
| `tests/` | pytest suite (API, validation, caption cleaning). |

## Request flow (what happens when you press "Describe image")

1. JS builds `FormData` (image, mode, max_new_tokens, num_beams) and POSTs to `/api/caption`.
2. Flask rejects missing files (400) and runs `validate_uploaded_file` (422 on failure):
   extension → size → `Image.open`+`verify`+`load` → dimensions → RGB conversion.
3. Mode is checked against the allowed list; token and beam values are clamped to the
   ranges in `config.py`.
4. `generate_caption` gets the model bundle `(processor, model, device)`. The processor
   resizes/normalises the image into a tensor (detailed mode adds the text prompt
   "a detailed description of").
5. `model.generate()` runs beam search under `torch.no_grad()`.
6. The processor decodes token ids to text, `_clean_caption` normalises it.
7. A `CaptionResult` dataclass comes back; Flask returns JSON; the UI types out the
   caption, derives keyword tags and stats, and adds the image to the session strip.

## Model loading and status

`start_background_load()` starts a daemon thread at boot so the server answers
immediately. A lock-guarded `_state` dict (`idle → loading → ready | error`) is read by
`/api/status`; the UI polls it every 1.5 s and drives the coloured status chip.

## Caching and state

| What | Where | Lifetime |
|------|-------|----------|
| Model weights | module global `_bundle` | server process |
| Downloaded weights | Hugging Face cache on disk | permanent |
| Uploaded image | request memory only | one request |
| Session history, theme, daily counter | browser memory / `localStorage` | tab / browser |

## Design decisions

1. **BLIP Base** (~990 MB) rather than Large — fits ordinary laptops.
2. **Local inference** — privacy; no keys, no cost.
3. **Beam search, no sampling** — same image + settings gives the same caption.
4. **Plain HTML/JS front end** — zero build tooling, easy to read and demo.
5. **Structured results, no raised exceptions to the UI** — failures become JSON errors.
6. **Mock mode (`CAPTION_MOCK=1`)** — UI and API tests run without downloading the model.
7. **No database** — stateless server, simple deployment.
