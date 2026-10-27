"""
VisionCaption AI — Image Utilities

Validates uploads, opens images safely, and converts them for model input.
Every public function returns either a clean result or raises a descriptive
ValueError / IOError so the caller can surface a friendly message.
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError

from config import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    MIN_IMAGE_DIMENSION,
    MAX_IMAGE_DIMENSION,
)

logger = logging.getLogger(__name__)


# ── Public helpers ───────────────────────────────────────────────────────

def validate_extension(filename: str) -> bool:
    """Return True when *filename* has an allowed extension (case-insensitive)."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_file_size(file_bytes: bytes) -> bool:
    """Return True when *file_bytes* is within the configured upload limit."""
    return len(file_bytes) <= MAX_UPLOAD_SIZE_BYTES


def get_file_size_mb(file_bytes: bytes) -> float:
    """Return the size of *file_bytes* in megabytes, rounded to two decimals."""
    return round(len(file_bytes) / (1024 * 1024), 2)


def open_image_safely(file_bytes: bytes) -> Image.Image:
    """Open raw bytes as a PIL Image, raising IOError on failure.

    Parameters
    ----------
    file_bytes : bytes
        Raw uploaded file content.

    Returns
    -------
    Image.Image
        A verified PIL Image (not yet converted to RGB).

    Raises
    ------
    IOError
        If the bytes cannot be interpreted as a valid image.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # lightweight structural check
        # Re-open because verify() can leave the stream in a bad state
        img = Image.open(io.BytesIO(file_bytes))
        img.load()    # force full decode
        return img
    except UnidentifiedImageError:
        raise IOError("The uploaded file is not a recognisable image.")
    except (SyntaxError, OSError) as exc:
        raise IOError(f"The image file appears to be corrupted: {exc}")
    except Exception as exc:
        logger.error("Unexpected error opening image: %s", exc, exc_info=True)
        raise IOError("Unable to process the uploaded file.")


def validate_dimensions(img: Image.Image) -> Tuple[bool, str]:
    """Check that image dimensions fall within configured limits.

    Returns
    -------
    (is_valid, message) — *message* is empty when valid.
    """
    width, height = img.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return False, (
            f"Image is too small ({width}×{height} px). "
            f"Minimum dimension is {MIN_IMAGE_DIMENSION} px."
        )
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        return False, (
            f"Image is too large ({width}×{height} px). "
            f"Maximum dimension is {MAX_IMAGE_DIMENSION} px."
        )
    return True, ""


def convert_to_rgb(img: Image.Image) -> Image.Image:
    """Convert any image mode (RGBA, P, L, etc.) to RGB safely.

    For RGBA images a white background is composited to avoid black
    artefacts from naive mode conversion.
    """
    if img.mode == "RGB":
        return img

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # alpha channel
        return background

    return img.convert("RGB")


def get_image_info(img: Image.Image) -> dict:
    """Return a small metadata dict for display in the UI."""
    return {
        "width": img.size[0],
        "height": img.size[1],
        "mode": img.mode,
        "format": getattr(img, "format", "N/A") or "N/A",
    }


# ── Full validation pipeline ────────────────────────────────────────────

def validate_uploaded_file(
    filename: str,
    file_bytes: bytes,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Run every validation step on an uploaded file.

    Returns
    -------
    (image, error_message)
        *image* is a valid RGB PIL Image when all checks pass;
        *error_message* is a human-readable string when any check fails.
    """
    # 1. Extension
    if not validate_extension(filename):
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        return None, f"Unsupported file type. Allowed formats: {allowed}."

    # 2. Size
    if not validate_file_size(file_bytes):
        size_mb = get_file_size_mb(file_bytes)
        return None, (
            f"File is too large ({size_mb} MB). "
            f"Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB."
        )

    # 3. Content
    try:
        img = open_image_safely(file_bytes)
    except IOError as exc:
        return None, str(exc)

    # 4. Dimensions
    ok, msg = validate_dimensions(img)
    if not ok:
        return None, msg

    # 5. RGB conversion
    img = convert_to_rgb(img)

    return img, None
