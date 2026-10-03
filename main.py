#!/usr/bin/env python3
"""
main.py – Traffic Scene Risk-Scoring Experiment Runner
═══════════════════════════════════════════════════════
Batch pipeline that:
  1. Loads images from  data/raw_images/
  2. Runs YOLO perception extraction on each frame
  3. Calls the VLM under Condition A (image-only) and Condition B (image + YOLO)
  4. Saves all results to  output/experiment_results.csv
  5. Prints comparative evaluation metrics against ground truth (if available)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── Project imports ──────────────────────────────────────────────────────
from src.perception.yolo_extractor import YOLOPerceptionExtractor
from src.reasoning.vlm_client import VLMClient, VLMResponse
from src.evaluation.metrics import (
    compute_correlation,
    compute_classification_metrics,
    format_correlation,
    format_classification,
)
from src.utils.file_loader import list_images, image_id_from_path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.csv"
OUTPUT_DIR = ROOT / "output"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_images"
RESULTS_CSV = OUTPUT_DIR / "experiment_results.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Traffic Scene Risk-Scoring Experiment Runner",
    )
    p.add_argument(
        "--provider", default="gemini", choices=["gemini", "openai"],
        help="VLM provider to use (default: gemini).",
    )
    p.add_argument(
        "--model", default=None,
        help="VLM model name override (e.g. 'gpt-4o', 'gemini-3.8-flash').",
    )
    p.add_argument(
        "--yolo-model", default="yolo26n.pt",
        help="YOLO model weights file (default: yolo26n.pt).",
    )
    p.add_argument(
        "--yolo-conf", type=float, default=0.35,
        help="YOLO confidence threshold (default: 0.35).",
    )
    p.add_argument(
        "--threshold", type=int, default=70,
        help="Risk score binarisation threshold for classification metrics (default: 70).",
    )
    p.add_argument(
        "--images-dir", type=str, default=str(RAW_IMAGES_DIR),
        help=f"Path to input images directory (default: {RAW_IMAGES_DIR}).",
    )
    p.add_argument(
        "--skip-vlm", action="store_true",
        help="Run YOLO only (skip VLM calls). Useful for testing perception.",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    image_paths = list_images(images_dir)
    if not image_paths:
        print(f"No images found in {images_dir}")
        sys.exit(0)

    print(f"Found {len(image_paths)} image(s) in {images_dir}\n")

    # ── Load ground truth (if available) ─────────────────────────────
    gt_map: dict[str, float] = {}
    if GROUND_TRUTH_PATH.is_file():
        gt_df = pd.read_csv(GROUND_TRUTH_PATH)
        if {"image_id", "human_score"}.issubset(gt_df.columns):
            gt_map = dict(zip(gt_df["image_id"].astype(str), gt_df["human_score"]))
            print(f"Loaded {len(gt_map)} ground-truth entries.\n")

    # ── Initialise components ────────────────────────────────────────
    print("Initialising YOLO perception extractor …")
    yolo = YOLOPerceptionExtractor(
        model_name=args.yolo_model,
        confidence_threshold=args.yolo_conf,
    )

    vlm: VLMClient | None = None
    if not args.skip_vlm:
        print(f"Initialising VLM client ({args.provider}) …")
        vlm = VLMClient(provider=args.provider, model=args.model)

    # ── Process each image ───────────────────────────────────────────
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    records: list[dict] = []

    for idx, img_path in enumerate(image_paths, start=1):
        image_id = image_id_from_path(img_path)
        print(f"\n[{idx}/{len(image_paths)}] Processing: {img_path.name}")

        # 1. YOLO perception
        perception = yolo.extract(img_path)
        yolo_summary = perception.to_text_summary()
        print(f"  YOLO: {len(perception.detections)} road user(s) detected")

        # Save annotated image
        ann_path = ANNOTATED_DIR / f"{image_id}_annotated.jpg"
        yolo.save_annotated_image(img_path, perception, ann_path)

        # 2. VLM scoring
        score_a, meaning_a = -1, ""
        score_b, meaning_b = -1, ""

        if vlm is not None:
            # Condition A: image-only
            print("  VLM Condition A (image-only) …")
            try:
                resp_a: VLMResponse = vlm.score_image(img_path, yolo_summary=None)
                score_a = resp_a.confidence_score
                meaning_a = resp_a.meaning
                print(f"    → Score: {score_a}  |  {meaning_a[:80]}")
            except Exception as e:
                print(f"    ✗ Condition A failed: {e}")

            # Condition B: image + YOLO
            print("  VLM Condition B (image + YOLO) …")
            try:
                resp_b: VLMResponse = vlm.score_image(img_path, yolo_summary=yolo_summary)
                score_b = resp_b.confidence_score
                meaning_b = resp_b.meaning
                print(f"    → Score: {score_b}  |  {meaning_b[:80]}")
            except Exception as e:
                print(f"    ✗ Condition B failed: {e}")

        # 3. Collect record
        gt_score = gt_map.get(image_id, None)
        records.append({
            "image_id": image_id,
            "yolo_detections": len(perception.detections),
            "yolo_summary": yolo_summary,
            "condition_a_score": score_a,
            "condition_a_meaning": meaning_a,
            "condition_b_score": score_b,
            "condition_b_meaning": meaning_b,
            "ground_truth_score": gt_score,
        })

    # ── Save results CSV ─────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n{'═' * 60}")
    print(f"Results saved to: {RESULTS_CSV}")
    print(f"Annotated images saved to: {ANNOTATED_DIR}")
    print(f"{'═' * 60}")

    # ── Evaluation (if ground truth available) ───────────────────────
    if not gt_map or args.skip_vlm:
        print("\nSkipping evaluation (no ground truth or VLM was skipped).")
        return

    # Filter to rows with valid scores and ground truth
    eval_df = results_df.dropna(subset=["ground_truth_score"])
    eval_df = eval_df[
        (eval_df["condition_a_score"] >= 0) & (eval_df["condition_b_score"] >= 0)
    ]

    if len(eval_df) < 3:
        print(f"\nToo few valid samples ({len(eval_df)}) for evaluation metrics.")
        return

    gt_scores = eval_df["ground_truth_score"].values
    a_scores = eval_df["condition_a_score"].values
    b_scores = eval_df["condition_b_score"].values

    print(f"\n{'─' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'─' * 60}\n")

    # Correlation
    corr_a = compute_correlation(a_scores, gt_scores)
    corr_b = compute_correlation(b_scores, gt_scores)
    print(format_correlation(corr_a, label="Condition A: Image-Only"))
    print()
    print(format_correlation(corr_b, label="Condition B: Image + YOLO"))

    # Classification
    print(f"\n{'─' * 60}\n")
    cls_a = compute_classification_metrics(a_scores, gt_scores, threshold=args.threshold)
    cls_b = compute_classification_metrics(b_scores, gt_scores, threshold=args.threshold)
    print(format_classification(cls_a, label="Condition A: Image-Only"))
    print()
    print(format_classification(cls_b, label="Condition B: Image + YOLO"))

    print(f"\n{'═' * 60}")
    print("Experiment complete.")


if __name__ == "__main__":
    main()
