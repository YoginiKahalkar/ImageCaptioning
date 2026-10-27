# Testing Strategy

## Overview

VisionCaption AI uses **pytest** for automated testing.  The test suite is
designed so that unit tests run quickly without downloading the ~990 MB
BLIP model.  A separate smoke-test procedure verifies end-to-end inference
with the real model.

## Test Organisation

```
tests/
├── __init__.py
├── test_image_utils.py         # Image validation & conversion
├── test_caption_validation.py  # Caption string cleaning
└── test_app_helpers.py         # Download builder, mocked inference
```

## Running Tests

### Quick unit tests (no model download)

```bash
# From the project root
pytest tests/ -v
```

### With coverage report

```bash
pip install pytest-cov
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Smoke test with the real model

This downloads the model on first run (~990 MB) and performs a single
inference.  Run it manually once before deployment:

```bash
python -c "
from PIL import Image
from model import generate_caption
img = Image.new('RGB', (224, 224), (128, 64, 32))
result = generate_caption(img)
assert result.success, result.error_message
print('Smoke test passed:', result.caption)
print('Inference time:', f'{result.inference_time_seconds:.2f}s')
"
```

## Test Categories

### Image Utilities (`test_image_utils.py`)

| Test case                    | What it verifies                                |
|------------------------------|-------------------------------------------------|
| Valid JPG/PNG/WEBP extension | Allowlist accepts known formats                 |
| Invalid extension (GIF, BMP) | Rejects unsupported formats                     |
| Empty / missing extension    | Edge case handling                              |
| File within size limit       | Boundary condition at MAX_UPLOAD_SIZE_BYTES      |
| File over size limit         | Rejects oversized uploads                       |
| Valid image bytes            | PIL opens and loads without error                |
| Corrupted / truncated bytes  | Raises IOError with a descriptive message        |
| Dimension too small / large  | Guard-rails prevent extreme dimensions           |
| RGBA → RGB conversion       | Alpha composited on white background             |
| Grayscale / palette → RGB   | Mode conversion works for all common modes       |
| Full validation pipeline     | End-to-end validate_uploaded_file()              |

### Caption Validation (`test_caption_validation.py`)

| Test case                   | What it verifies                                 |
|-----------------------------|--------------------------------------------------|
| Basic cleaning              | Whitespace normalised, capitalised, period added |
| Prompt stripping            | Conditional prompt prefix removed                |
| Prompt with colon           | Handles `prompt: caption` format                 |
| Empty / whitespace-only     | Returns empty string                             |
| Trailing period             | Not duplicated                                   |

### App Helpers (`test_app_helpers.py`)

| Test case                   | What it verifies                                 |
|-----------------------------|--------------------------------------------------|
| Download text content       | Contains caption, model ID, device               |
| Mocked standard inference   | generate_caption returns CaptionResult           |
| Mocked detailed mode        | Prompt prefix stripped in detailed mode           |
| Empty caption detection     | success=False when model returns nothing          |
| Model load failure          | RuntimeError caught, error_message populated      |
| Inference exception         | Unexpected errors handled gracefully              |

## Known Limitations

- Streamlit `session_state` cannot be fully tested without Streamlit's
  AppTest framework; the unit tests cover the underlying logic only.
- The smoke test requires internet access for the initial model download.
- GPU-specific tests are skipped when CUDA is not available.
