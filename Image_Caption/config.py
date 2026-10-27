"""
VisionCaption AI — Application Configuration

Central configuration for model identifiers, file-upload constraints,
generation parameters, and UI constants.  Every tunable value lives here
so that the rest of the codebase stays free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ── Model configuration ─────────────────────────────────────────────────
MODEL_ID: str = "Salesforce/blip-image-captioning-base"
MODEL_FRAMEWORK: str = "PyTorch + Hugging Face Transformers"
MODEL_ARCHITECTURE: str = "BLIP (Bootstrapped Language-Image Pre-training)"
MODEL_LICENSE: str = "BSD-3-Clause"
MODEL_PAPER: str = "Li et al., BLIP: Bootstrapped Language-Image Pre-training for Unified Vision-Language Understanding and Generation, ICML 2022"

# ── File-upload constraints ──────────────────────────────────────────────
ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]
MAX_UPLOAD_SIZE_MB: int = 10
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # 10 485 760

# Image dimension guard-rails (pixels per side)
MIN_IMAGE_DIMENSION: int = 10
MAX_IMAGE_DIMENSION: int = 10_000

# ── Description modes ────────────────────────────────────────────────────
MODE_STANDARD: str = "Standard Caption"
MODE_DETAILED: str = "Detailed Description"
DESCRIPTION_MODES: List[str] = [MODE_STANDARD, MODE_DETAILED]

# ── Generation defaults ──────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """Default and boundary values for caption generation."""
    max_new_tokens: int = 50
    max_new_tokens_detailed: int = 150
    num_beams: int = 4
    min_new_tokens: int = 5
    repetition_penalty: float = 1.5
    length_penalty: float = 1.0

    # Boundaries exposed in the sidebar
    max_tokens_range: tuple = (10, 200)
    beam_range: tuple = (1, 8)

    # Conditional prompt used for "detailed" mode
    detailed_prompt: str = "a detailed description of"


GENERATION_DEFAULTS = GenerationConfig()

# ── UI constants ─────────────────────────────────────────────────────────
APP_TITLE: str = "VisionCaption AI"
APP_SUBTITLE: str = "Transform images into meaningful descriptions with AI."
APP_ICON: str = "🔬"
PAGE_LAYOUT: str = "wide"

# ── Evaluation ───────────────────────────────────────────────────────────
EVALUATION_OUTPUT_DIR: str = "outputs"

# ── Logging ──────────────────────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
