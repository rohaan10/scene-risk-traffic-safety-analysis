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
from src.reasoning.vlm_client import VLMClient, VLMResponse, FACTORS
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
        "--provider", default="ollama", choices=["ollama", "gemini", "openai"],
        help="VLM provider to use (default: ollama).",
    )
    p.add_argument(
        "--model", default=None,
        help="VLM model name override (e.g. 'moondream', 'gemini-3.8-flash', 'gpt-4o').",
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
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit processing to first N images (useful for quick smoke tests).",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    all_image_paths = list_images(images_dir)
    if not all_image_paths:
        print(f"No images found in {images_dir}")
        sys.exit(0)

    total_dataset_count = len(all_image_paths)
    image_paths = all_image_paths
    if args.limit and args.limit > 0:
        image_paths = all_image_paths[:args.limit]
        print(f"Limiting execution to first {len(image_paths)} of {total_dataset_count} image(s).\n")

    print(f"Processing {len(image_paths)} image(s) from {images_dir}\n")

    # ── Load ground truth (if available) ─────────────────────────────
    gt_map: dict[str, float] = {}
    if GROUND_TRUTH_PATH.is_file():
        try:
            gt_df = pd.read_csv(GROUND_TRUTH_PATH)
            if "image_id" in gt_df.columns:
                # Option 1: 5 factor sub-scores – auto-calculate exact aggregate human score in Python
                if set(FACTORS).issubset(gt_df.columns) and any(gt_df[f].notna().any() for f in FACTORS):
                    from src.reasoning.vlm_client import aggregate_factor_scores
                    for _, row in gt_df.iterrows():
                        f_data = {}
                        for f in FACTORS:
                            val = row.get(f)
                            if pd.notna(val):
                                try:
                                    f_data[f] = {"score": float(val)}
                                except ValueError:
                                    pass
                        if len(f_data) == len(FACTORS):
                            gt_map[str(row["image_id"])] = aggregate_factor_scores(f_data)

                # Option 2: Direct numeric human_score column (if factors were not provided)
                if not gt_map and "human_score" in gt_df.columns:
                    numeric_hs = pd.to_numeric(gt_df["human_score"], errors="coerce")
                    valid_mask = numeric_hs.notna()
                    if valid_mask.any():
                        gt_map = dict(zip(gt_df.loc[valid_mask, "image_id"].astype(str), numeric_hs[valid_mask]))

                print(f"Loaded {len(gt_map)} ground-truth entries from {GROUND_TRUTH_PATH.name}.\n")
        except Exception as e:
            print(f"⚠️ Could not load ground truth from {GROUND_TRUTH_PATH}: {e}\n")

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
        resp_a: VLMResponse | None = None
        resp_b: VLMResponse | None = None
        score_a, meaning_a, raw_a = None, "", ""
        score_b, meaning_b, raw_b = None, "", ""
        duration_a, duration_b = 0.0, 0.0

        if vlm is not None:
            # Condition A: image-only
            print("  VLM Condition A (image-only) …")
            try:
                t0_a = time.perf_counter()
                resp_a = vlm.score_image(img_path, yolo_summary=None)
                duration_a = round(time.perf_counter() - t0_a, 2)
                score_a = resp_a.confidence_score
                meaning_a = resp_a.meaning
                raw_a = resp_a.raw_text
                disp_a = f"{score_a}/100" if score_a is not None else "NaN"
                print(f"    → Attention Score: {disp_a} [{resp_a.advisory_state}] ({duration_a}s)")
                for f in FACTORS:
                    sub = resp_a.factors.get(f, {})
                    if sub:
                        print(f"      • {f:22s}: {sub.get('score', 0):2d}/10 | {sub.get('reason', '')}")
            except Exception as e:
                print(f"    ✗ Condition A failed: {e}")

            # Condition B: image + YOLO
            print("  VLM Condition B (image + YOLO) …")
            try:
                t0_b = time.perf_counter()
                resp_b = vlm.score_image(img_path, yolo_summary=yolo_summary)
                duration_b = round(time.perf_counter() - t0_b, 2)
                score_b = resp_b.confidence_score
                meaning_b = resp_b.meaning
                raw_b = resp_b.raw_text
                disp_b = f"{score_b}/100" if score_b is not None else "NaN"
                print(f"    → Attention Score: {disp_b} [{resp_b.advisory_state}] ({duration_b}s)")
                for f in FACTORS:
                    sub = resp_b.factors.get(f, {})
                    if sub:
                        print(f"      • {f:22s}: {sub.get('score', 0):2d}/10 | {sub.get('reason', '')}")
            except Exception as e:
                print(f"    ✗ Condition B failed: {e}")

            img_duration = round(duration_a + duration_b, 2)
            print(f"  ⏱ Total image inference time: {img_duration}s")

        # 3. Collect record with sub-scores and advisory states
        gt_score = gt_map.get(image_id, None)
        record = {
            "image_id": image_id,
            "yolo_detections": len(perception.detections),
            "yolo_summary": yolo_summary,
            "condition_a_score": score_a,
            "condition_a_meaning": meaning_a,
            "condition_a_state": resp_a.advisory_state if resp_a else "Unknown",
            "condition_a_time_sec": duration_a,
            "condition_a_raw": raw_a,
            "condition_b_score": score_b,
            "condition_b_meaning": meaning_b,
            "condition_b_state": resp_b.advisory_state if resp_b else "Unknown",
            "condition_b_time_sec": duration_b,
            "condition_b_raw": raw_b,
            "total_time_sec": round(duration_a + duration_b, 2),
            "ground_truth_score": gt_score,
        }

        # Save individual factor sub-scores and reasons as separate columns
        for f in FACTORS:
            f_a = resp_a.factors.get(f, {}) if resp_a else {}
            record[f"cond_a_{f}_score"] = f_a.get("score", None)
            record[f"cond_a_{f}_reason"] = f_a.get("reason", "")

            f_b = resp_b.factors.get(f, {}) if resp_b else {}
            record[f"cond_b_{f}_score"] = f_b.get("score", None)
            record[f"cond_b_{f}_reason"] = f_b.get("reason", "")

        records.append(record)

    # ── Save results CSV ─────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n{'═' * 60}")
    print(f"Results saved to: {RESULTS_CSV}")
    print(f"Annotated images saved to: {ANNOTATED_DIR}")
    print(f"{'═' * 60}")

    # ── Performance summary ──────────────────────────────────────────
    if vlm is not None and records:
        times_a = [r["condition_a_time_sec"] for r in records if r.get("condition_a_time_sec", 0) > 0]
        times_b = [r["condition_b_time_sec"] for r in records if r.get("condition_b_time_sec", 0) > 0]
        if times_a and times_b:
            if len(times_a) > 1:
                # Discard 1st image cold-start (model load into VRAM) to report steady-state average
                warmup_time = round(times_a[0] + times_b[0], 2)
                bench_a = times_a[1:]
                bench_b = times_b[1:]
                warmup_msg = f" (excludes 1st image cold-start: {warmup_time}s)"
            else:
                bench_a = times_a
                bench_b = times_b
                warmup_msg = ""

            avg_a = sum(bench_a) / len(bench_a)
            avg_b = sum(bench_b) / len(bench_b)
            avg_total = avg_a + avg_b
            est_total_min = (avg_total * total_dataset_count) / 60
            print(f"\n⏱ Performance Benchmark ({args.model}):")
            print(f"  • Steady Condition A avg: {avg_a:.2f}s | Condition B avg: {avg_b:.2f}s{warmup_msg}")
            print(f"  • Steady average per image (both conditions): {avg_total:.2f}s")
            print(f"  • Estimated runtime for full dataset ({total_dataset_count} images): {est_total_min:.1f} minutes")

    # ── Evaluation (if ground truth available) ───────────────────────
    if not gt_map or args.skip_vlm:
        print("\nSkipping evaluation (no ground truth or VLM was skipped).")
        return

    # Filter to rows with valid scores and ground truth
    eval_df = results_df.dropna(
        subset=["ground_truth_score", "condition_a_score", "condition_b_score"]
    )
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
