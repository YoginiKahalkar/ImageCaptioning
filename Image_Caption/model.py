"""
VisionCaption — Model module (Salesforce BLIP, runs locally, no API key).

Same inference logic as the reference project, but framework-free: the model
loads once in a background thread so the web UI can show live loading status.

Set CAPTION_MOCK=1 to skip the model and return fake captions (UI testing).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from PIL import Image

from config import MODEL_ID, MODE_STANDARD, MODE_DETAILED, GENERATION_DEFAULTS

logger = logging.getLogger(__name__)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

MOCK = os.environ.get("CAPTION_MOCK") == "1"


@dataclass
class CaptionResult:
    caption: str
    inference_time_seconds: float
    model_id: str
    device: str
    mode: str
    generation_params: dict = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


# ── Loading state (read by /api/status) ──────────────────────────────────
_lock = threading.Lock()
_state = {"status": "idle", "message": "Model not loaded yet.", "device": "-"}
_bundle: Optional[Tuple] = None


def get_status() -> dict:
    with _lock:
        return dict(_state)


def _set(status: str, message: str, device: Optional[str] = None) -> None:
    with _lock:
        _state.update(status=status, message=message)
        if device:
            _state["device"] = device


def load_model() -> Tuple:
    """Load (once) and return (processor, model, device). Thread-safe."""
    global _bundle
    if _bundle is not None:
        return _bundle
    with _load_guard:
        if _bundle is not None:
            return _bundle
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _set("loading", "Fetching processor…", device)
            processor = BlipProcessor.from_pretrained(MODEL_ID)
            _set("loading", "Loading model weights (about 990 MB on first run)…")
            model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
            model.eval()
            _bundle = (processor, model, device)
            _set("ready", "Model ready.", device)
            return _bundle
        except Exception as exc:  # network, disk, memory…
            logger.error("Model loading failed: %s", exc, exc_info=True)
            _set("error", f"Could not load the model: {exc}")
            raise RuntimeError(f"Could not load the model: {exc}")


_load_guard = threading.Lock()


def start_background_load() -> None:
    """Kick off model loading without blocking the web server."""
    if MOCK:
        _set("ready", "Mock mode (no real model).", "mock")
        return

    def _run():
        try:
            load_model()
        except RuntimeError:
            pass

    threading.Thread(target=_run, daemon=True).start()


# ── Inference ────────────────────────────────────────────────────────────
def generate_caption(
    image: Image.Image,
    mode: str = MODE_STANDARD,
    max_new_tokens: Optional[int] = None,
    num_beams: Optional[int] = None,
) -> CaptionResult:
    detailed = mode == MODE_DETAILED
    tokens = max_new_tokens or (
        GENERATION_DEFAULTS.max_new_tokens_detailed if detailed else GENERATION_DEFAULTS.max_new_tokens
    )
    prompt = GENERATION_DEFAULTS.detailed_prompt if detailed else None
    beams = num_beams or GENERATION_DEFAULTS.num_beams
    params = {"max_new_tokens": tokens, "num_beams": beams, "mode": mode}

    def fail(msg: str, device: str = "unknown", t: float = 0.0) -> CaptionResult:
        return CaptionResult("", t, MODEL_ID, device, mode, params, False, msg)

    if MOCK:
        time.sleep(1.2)
        text = "A person standing on a hillside at sunset, looking out over a valley."
        if detailed:
            text += " Warm light falls across the grass while mountains fade into haze behind."
        return CaptionResult(text, 1.2, MODEL_ID, "mock", mode, params)

    try:
        import torch
        processor, model, device = load_model()
    except RuntimeError as exc:
        return fail(str(exc))

    try:
        inputs = processor(image, text=prompt, return_tensors="pt") if prompt else processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        start = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=tokens,
                num_beams=beams,
                min_length=GENERATION_DEFAULTS.min_new_tokens,
                repetition_penalty=GENERATION_DEFAULTS.repetition_penalty,
                length_penalty=GENERATION_DEFAULTS.length_penalty,
            )
        elapsed = time.perf_counter() - start
        caption = _clean_caption(processor.decode(ids[0], skip_special_tokens=True), prompt)
        if not caption:
            return fail("The model returned an empty caption. Try another image.", device, elapsed)
        return CaptionResult(caption, elapsed, MODEL_ID, device, mode, params)
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        return fail(f"Inference error: {exc}")


def _clean_caption(raw: str, prompt: Optional[str] = None) -> str:
    """Normalise whitespace, strip the conditioning prompt, capitalise, add full stop."""
    text = " ".join(raw.split())
    if prompt and text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip().lstrip(",: ").strip()
    if text:
        text = text[0].upper() + text[1:]
        if not text.endswith("."):
            text += "."
    return text
