"""
VisionCaption AI — Evaluation Pipeline

Runs the BLIP model against a set of images that have human-written
reference captions, computes BLEU-1, BLEU-4, and ROUGE-L, and writes
per-image and aggregate results to CSV and JSON files.

Usage
-----
    python evaluation/evaluate.py                          # defaults
    python evaluation/evaluate.py --images-dir path/to/images \
                                   --references evaluation/sample_references.json \
                                   --output-dir outputs

Prerequisites
-------------
1. Download a small subset of Flickr8k or MS COCO images and place them
   in a directory.
2. Create or update ``sample_references.json`` with the correct
   filename → list-of-captions mapping.
3. Install evaluation deps:  pip install nltk rouge-score
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from PIL import Image

from config import MODEL_ID, GENERATION_DEFAULTS, MODE_STANDARD
from model import generate_caption, load_model


def ensure_nltk_data():
    """Download required NLTK tokenizer data if missing."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def compute_bleu(reference_captions: List[str], hypothesis: str) -> Dict[str, float]:
    """Compute BLEU-1 and BLEU-4 for a single hypothesis against
    multiple reference captions.

    Parameters
    ----------
    reference_captions : list[str]
        Human-written reference captions (at least one).
    hypothesis : str
        Model-generated caption.

    Returns
    -------
    dict with keys ``bleu_1`` and ``bleu_4``.
    """
    refs_tok = [nltk.word_tokenize(r.lower()) for r in reference_captions]
    hyp_tok = nltk.word_tokenize(hypothesis.lower())
    smoother = SmoothingFunction().method1

    bleu_1 = sentence_bleu(refs_tok, hyp_tok, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu_4 = sentence_bleu(refs_tok, hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    return {"bleu_1": round(bleu_1, 4), "bleu_4": round(bleu_4, 4)}


def compute_rouge_l(reference_captions: List[str], hypothesis: str) -> float:
    """Compute ROUGE-L F1 averaged over all references."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, hypothesis)["rougeL"].fmeasure for ref in reference_captions]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def run_evaluation(
    images_dir: str,
    references_path: str,
    output_dir: str,
) -> None:
    """Execute the full evaluation pipeline."""

    ensure_nltk_data()

    # Load references
    with open(references_path, "r", encoding="utf-8") as f:
        references: Dict[str, List[str]] = json.load(f)

    if not references:
        print("ERROR: No reference captions found in", references_path)
        sys.exit(1)

    # Ensure output dir
    os.makedirs(output_dir, exist_ok=True)

    # Pre-load model (measured separately)
    print(f"Loading model: {MODEL_ID} …")
    t0 = time.perf_counter()
    load_model()
    model_load_time = time.perf_counter() - t0
    print(f"Model loaded in {model_load_time:.2f}s")

    # Evaluate each image
    results = []
    inference_times = []
    failed = 0

    for filename, ref_captions in references.items():
        img_path = os.path.join(images_dir, filename)
        if not os.path.isfile(img_path):
            print(f"  SKIP {filename} — file not found")
            failed += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"  SKIP {filename} — cannot open: {exc}")
            failed += 1
            continue

        result = generate_caption(img, mode=MODE_STANDARD)
        if not result.success:
            print(f"  FAIL {filename} — {result.error_message}")
            failed += 1
            continue

        bleu = compute_bleu(ref_captions, result.caption)
        rouge_l = compute_rouge_l(ref_captions, result.caption)

        row = {
            "filename": filename,
            "generated_caption": result.caption,
            "reference_captions": " | ".join(ref_captions),
            "bleu_1": bleu["bleu_1"],
            "bleu_4": bleu["bleu_4"],
            "rouge_l": rouge_l,
            "inference_time_s": round(result.inference_time_seconds, 4),
        }
        results.append(row)
        inference_times.append(result.inference_time_seconds)
        print(f"  OK   {filename}  BLEU-1={bleu['bleu_1']:.3f}  ROUGE-L={rouge_l:.3f}  {result.inference_time_seconds:.2f}s")

    # ── Write per-image CSV ──────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nPer-image results → {csv_path}")

    # ── Aggregate summary ────────────────────────────────────────────────
    if results:
        n = len(results)
        avg_bleu1 = sum(r["bleu_1"] for r in results) / n
        avg_bleu4 = sum(r["bleu_4"] for r in results) / n
        avg_rouge = sum(r["rouge_l"] for r in results) / n
        avg_time = sum(inference_times) / n
        med_time = sorted(inference_times)[n // 2]

        summary = {
            "model_id": MODEL_ID,
            "device": results[0].get("device", "see log"),
            "images_evaluated": n,
            "images_failed": failed,
            "avg_bleu_1": round(avg_bleu1, 4),
            "avg_bleu_4": round(avg_bleu4, 4),
            "avg_rouge_l": round(avg_rouge, 4),
            "avg_inference_time_s": round(avg_time, 4),
            "median_inference_time_s": round(med_time, 4),
            "min_inference_time_s": round(min(inference_times), 4),
            "max_inference_time_s": round(max(inference_times), 4),
            "model_load_time_s": round(model_load_time, 4),
            "generation_settings": {
                "max_new_tokens": GENERATION_DEFAULTS.max_new_tokens,
                "num_beams": GENERATION_DEFAULTS.num_beams,
            },
        }

        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary          → {summary_path}")

        print("\n── Aggregate Results ──")
        print(f"  Images evaluated : {n}")
        print(f"  Images failed    : {failed}")
        print(f"  Avg BLEU-1       : {avg_bleu1:.4f}")
        print(f"  Avg BLEU-4       : {avg_bleu4:.4f}")
        print(f"  Avg ROUGE-L      : {avg_rouge:.4f}")
        print(f"  Avg inference    : {avg_time:.3f}s")
        print(f"  Median inference : {med_time:.3f}s")
    else:
        print("\nNo images were successfully evaluated.")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VisionCaption AI against reference captions."
    )
    parser.add_argument(
        "--images-dir",
        default="assets",
        help="Directory containing evaluation images.",
    )
    parser.add_argument(
        "--references",
        default=str(Path(__file__).parent / "sample_references.json"),
        help="JSON file mapping filenames to reference-caption lists.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for evaluation output files.",
    )
    args = parser.parse_args()
    run_evaluation(args.images_dir, args.references, args.output_dir)


if __name__ == "__main__":
    main()
