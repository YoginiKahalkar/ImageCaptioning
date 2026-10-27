"""
VisionCaption — Flask server.   Run:  python app.py   →  http://localhost:5000
"""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request, send_from_directory

from config import (
    ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB, MAX_UPLOAD_SIZE_BYTES, MODEL_ID,
    DESCRIPTION_MODES, MODE_STANDARD, GENERATION_DEFAULTS as G,
)
from image_utils import validate_uploaded_file, get_image_info
from model import generate_caption, get_status, start_background_load

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_BYTES + 1024 * 1024


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/config")
def api_config():
    return jsonify(
        model=MODEL_ID, modes=DESCRIPTION_MODES, extensions=ALLOWED_EXTENSIONS,
        max_mb=MAX_UPLOAD_SIZE_MB,
        tokens={"min": G.max_tokens_range[0], "max": G.max_tokens_range[1],
                "standard": G.max_new_tokens, "detailed": G.max_new_tokens_detailed},
        beams={"min": G.beam_range[0], "max": G.beam_range[1], "default": G.num_beams},
    )


@app.get("/api/status")
def api_status():
    return jsonify(get_status())


@app.post("/api/caption")
def api_caption():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="No image received. Choose or drop an image first."), 400

    img, err = validate_uploaded_file(f.filename, f.read())
    if err:
        return jsonify(error=err), 422

    mode = request.form.get("mode", MODE_STANDARD)
    if mode not in DESCRIPTION_MODES:
        mode = MODE_STANDARD
    try:
        tokens = max(G.max_tokens_range[0], min(G.max_tokens_range[1], int(request.form.get("max_new_tokens", 0) or 0))) or None
        beams = max(G.beam_range[0], min(G.beam_range[1], int(request.form.get("num_beams", 0) or 0))) or None
    except ValueError:
        return jsonify(error="Settings must be whole numbers."), 400

    r = generate_caption(img, mode, tokens, beams)
    if not r.success:
        return jsonify(error=r.error_message), 500
    return jsonify(
        caption=r.caption, seconds=round(r.inference_time_seconds, 2), device=r.device,
        mode=r.mode, model=r.model_id, params=r.generation_params, image=get_image_info(img),
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"File is too large. Maximum size is {MAX_UPLOAD_SIZE_MB} MB."), 413


if __name__ == "__main__":
    # Avoid loading the model twice when the Flask reloader is active.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        start_background_load()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
