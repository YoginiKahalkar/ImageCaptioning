"""
Tests for image_utils — validation, opening, conversion.

All tests are self-contained: they create small synthetic images in memory
and do not require any external files or model downloads.
"""

from __future__ import annotations

import io
import struct

import pytest
from PIL import Image

from image_utils import (
    validate_extension,
    validate_file_size,
    get_file_size_mb,
    open_image_safely,
    validate_dimensions,
    convert_to_rgb,
    validate_uploaded_file,
    get_image_info,
)
from config import MAX_UPLOAD_SIZE_BYTES


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_image_bytes(
    width: int = 100,
    height: int = 100,
    mode: str = "RGB",
    fmt: str = "JPEG",
) -> bytes:
    """Create a minimal valid image in memory and return its bytes."""
    img = Image.new(mode, (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    if fmt == "JPEG" and mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_rgba_image_bytes(width: int = 50, height: int = 50) -> bytes:
    """Create a PNG image with an alpha channel."""
    img = Image.new("RGBA", (width, height), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Extension validation ────────────────────────────────────────────────

class TestValidateExtension:
    def test_valid_jpg(self):
        assert validate_extension("photo.jpg") is True

    def test_valid_jpeg(self):
        assert validate_extension("photo.JPEG") is True

    def test_valid_png(self):
        assert validate_extension("image.png") is True

    def test_valid_webp(self):
        assert validate_extension("pic.webp") is True

    def test_invalid_gif(self):
        assert validate_extension("anim.gif") is False

    def test_invalid_bmp(self):
        assert validate_extension("map.bmp") is False

    def test_no_extension(self):
        assert validate_extension("noext") is False

    def test_empty_string(self):
        assert validate_extension("") is False

    def test_double_extension(self):
        assert validate_extension("photo.raw.png") is True

    def test_hidden_file(self):
        assert validate_extension(".png") is True


# ── File-size validation ────────────────────────────────────────────────

class TestValidateFileSize:
    def test_small_file(self):
        assert validate_file_size(b"x" * 100) is True

    def test_at_limit(self):
        assert validate_file_size(b"x" * MAX_UPLOAD_SIZE_BYTES) is True

    def test_over_limit(self):
        assert validate_file_size(b"x" * (MAX_UPLOAD_SIZE_BYTES + 1)) is False


class TestGetFileSizeMB:
    def test_one_mb(self):
        assert get_file_size_mb(b"x" * (1024 * 1024)) == 1.0

    def test_zero(self):
        assert get_file_size_mb(b"") == 0.0


# ── Opening images ──────────────────────────────────────────────────────

class TestOpenImageSafely:
    def test_valid_jpeg(self):
        data = _make_image_bytes(fmt="JPEG")
        img = open_image_safely(data)
        assert img.size == (100, 100)

    def test_valid_png(self):
        data = _make_image_bytes(fmt="PNG")
        img = open_image_safely(data)
        assert img.size == (100, 100)

    def test_valid_webp(self):
        data = _make_image_bytes(fmt="WEBP")
        img = open_image_safely(data)
        assert img.size == (100, 100)

    def test_corrupted_bytes(self):
        with pytest.raises(IOError):
            open_image_safely(b"this is not an image at all")

    def test_empty_bytes(self):
        with pytest.raises(IOError):
            open_image_safely(b"")

    def test_truncated_jpeg(self):
        data = _make_image_bytes(fmt="JPEG")
        with pytest.raises(IOError):
            open_image_safely(data[:50])


# ── Dimension validation ────────────────────────────────────────────────

class TestValidateDimensions:
    def test_normal(self):
        img = Image.new("RGB", (640, 480))
        ok, msg = validate_dimensions(img)
        assert ok is True
        assert msg == ""

    def test_too_small(self):
        img = Image.new("RGB", (5, 5))
        ok, msg = validate_dimensions(img)
        assert ok is False
        assert "too small" in msg.lower()

    def test_too_large(self):
        # We do not allocate pixels — just craft the header
        img = Image.new("RGB", (1, 1))
        img._size = (20_000, 20_000)  # monkey-patch for test only
        ok, msg = validate_dimensions(img)
        assert ok is False
        assert "too large" in msg.lower()


# ── RGB conversion ───────────────────────────────────────────────────────

class TestConvertToRGB:
    def test_rgb_passthrough(self):
        img = Image.new("RGB", (10, 10))
        assert convert_to_rgb(img).mode == "RGB"

    def test_rgba_to_rgb(self):
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        result = convert_to_rgb(img)
        assert result.mode == "RGB"
        # The semi-transparent red on white should not be pure red
        px = result.getpixel((0, 0))
        assert px[0] > 100  # still reddish
        assert px[1] > 50   # blended with white

    def test_grayscale_to_rgb(self):
        img = Image.new("L", (10, 10), 128)
        result = convert_to_rgb(img)
        assert result.mode == "RGB"

    def test_palette_to_rgb(self):
        img = Image.new("P", (10, 10))
        result = convert_to_rgb(img)
        assert result.mode == "RGB"


# ── Full pipeline ────────────────────────────────────────────────────────

class TestValidateUploadedFile:
    def test_valid_jpeg(self):
        data = _make_image_bytes(fmt="JPEG")
        img, err = validate_uploaded_file("photo.jpg", data)
        assert err is None
        assert img is not None
        assert img.mode == "RGB"

    def test_valid_png(self):
        data = _make_image_bytes(fmt="PNG")
        img, err = validate_uploaded_file("photo.png", data)
        assert err is None

    def test_bad_extension(self):
        data = _make_image_bytes(fmt="JPEG")
        img, err = validate_uploaded_file("photo.gif", data)
        assert img is None
        assert "Unsupported" in err

    def test_oversized(self):
        data = b"x" * (MAX_UPLOAD_SIZE_BYTES + 1)
        img, err = validate_uploaded_file("big.jpg", data)
        assert img is None
        assert "too large" in err.lower()

    def test_corrupted(self):
        img, err = validate_uploaded_file("bad.jpg", b"not an image")
        assert img is None
        assert err is not None

    def test_rgba_conversion(self):
        data = _make_rgba_image_bytes()
        img, err = validate_uploaded_file("alpha.png", data)
        assert err is None
        assert img.mode == "RGB"


# ── Image info ───────────────────────────────────────────────────────────

class TestGetImageInfo:
    def test_basic(self):
        img = Image.new("RGB", (320, 240))
        info = get_image_info(img)
        assert info["width"] == 320
        assert info["height"] == 240
        assert info["mode"] == "RGB"
