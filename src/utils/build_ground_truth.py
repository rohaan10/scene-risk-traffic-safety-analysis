"""
build_ground_truth.py – Pre-populates ground_truth.csv with all dataset images
─────────────────────────────────────────────────────────────────────────────
Generates a structured annotation table for the 4-person team:
  • Pre-fills image_id from data/raw_images/
  • Assigns image batches evenly across 4 members (with shared calibration set)
  • Inserts live Google Sheets / Excel formula for instant 0–100 attention score
  • Includes driver_state_gut (Relax / Stay Alert / Take Control) for calibration

Usage:
  python -m src.utils.build_ground_truth
  python -m src.utils.build_ground_truth --images-dir data/raw_images --members 4 --calibration 8
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path so the script runs from anywhere
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils.file_loader import list_images, image_id_from_path
from src.reasoning.vlm_client import FACTORS, aggregate_factor_scores

DATA_DIR = ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
GROUND_TRUTH_CSV = DATA_DIR / "ground_truth.csv"


def generate_ground_truth_table(
    images_dir: Path = RAW_IMAGES_DIR,
    output_path: Path = GROUND_TRUTH_CSV,
    num_members: int = 4,
) -> None:
    """Scan images and create the team ground-truth CSV divided equally among members."""
    if not images_dir.is_dir():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    images = list_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    image_ids = [image_id_from_path(p) for p in images]
    total_imgs = len(image_ids)

    # Divide all images directly and equally into continuous sequential blocks
    assignments = []
    for i in range(total_imgs):
        member_idx = (i * num_members) // total_imgs + 1
        member_idx = min(member_idx, num_members)
        assignments.append(f"Member {member_idx}")

    records = []
    for img_id, assigned in zip(image_ids, assignments):
        records.append({
            "image_id": img_id,
            "assigned_to": assigned,
            "traffic_density": "",
            "vulnerable_road_users": "",
            "conflict_proximity": "",
            "road_complexity": "",
            "impaired_visibility": "",
            "driver_state_gut": "",  # Relax, Stay Alert, Take Control
            "notes": "",
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'═' * 60}")
    print(f"✅ Generated ground truth table: {output_path}")
    print(f"   Total images: {total_imgs} divided equally across {num_members} members:")
    for m in range(1, num_members + 1):
        m_name = f"Member {m}"
        count = sum(1 for a in assignments if a == m_name)
        indices = [idx + 1 for idx, a in enumerate(assignments) if a == m_name]
        if indices:
            print(f"   • {m_name}: {count} images (Rows {indices[0]}–{indices[-1]})")
    print(f"{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate team ground-truth CSV")
    parser.add_argument("--images-dir", type=Path, default=RAW_IMAGES_DIR, help="Path to raw images")
    parser.add_argument("--output", type=Path, default=GROUND_TRUTH_CSV, help="Output CSV destination")
    parser.add_argument("--members", type=int, default=4, help="Number of team members")

    args = parser.parse_args()
    generate_ground_truth_table(
        images_dir=args.images_dir,
        output_path=args.output,
        num_members=args.members,
    )


if __name__ == "__main__":
    main()
