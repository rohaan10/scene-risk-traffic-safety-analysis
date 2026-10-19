#!/usr/bin/env python3
"""
analysis.py – Post-experiment analysis and visualisation
══════════════════════════════════════════════════════════
Reads output/experiment_results.csv (and optionally output/experiment_reasons.csv)
and produces:

  1. Console summary  – score distributions, per-condition means, GT coverage
  2. Scatter plots    – GT vs Cond A / Cond B (with correlation line)
  3. Factor heatmap   – per-factor mean sub-scores, Cond A vs Cond B side-by-side
  4. Advisory-state   – confusion matrix GT label vs model label (bar chart)
  5. Score delta      – Cond B minus Cond A per image (does YOLO context help?)

All charts are saved to output/figures/ and displayed inline if running in a
Jupyter/Colab environment.

Usage:
    python analysis.py
    python analysis.py --results output/experiment_results.csv --threshold 70
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
RESULTS    = ROOT / "output" / "experiment_results.csv"
FIGURES    = ROOT / "output" / "figures"

FACTORS = [
    "traffic_density",
    "vulnerable_road_users",
    "conflict_proximity",
    "road_complexity",
    "impaired_visibility",
]

ADVISORY_ORDER = ["Relax", "Stay Alert", "Take Control", "Unknown"]
CMAP = "RdYlGn_r"   # green=low risk, red=high risk


# ── Helpers ──────────────────────────────────────────────────────────────────
def score_to_advisory(score: float, threshold: int = 70) -> str:
    if score is None or np.isnan(score):
        return "Unknown"
    if score >= threshold:
        return "Take Control"
    if score >= 40:
        return "Stay Alert"
    return "Relax"


def correlation_line(ax, x, y) -> None:
    """Draw OLS trend line + Pearson r annotation."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return
    x_, y_ = x[mask], y[mask]
    m, b, r, p, _ = stats.linregress(x_, y_)
    xs = np.linspace(x_.min(), x_.max(), 100)
    ax.plot(xs, m * xs + b, color="crimson", lw=1.5, ls="--", label=f"r = {r:.2f}, p = {p:.3f}")
    ax.legend(fontsize=9)


def load_gut_scores(gt_arg: str | None) -> pd.DataFrame | None:
    """Load the instinctive (gut) 0-100 score per image from ground_truth.csv."""
    candidates = [Path(gt_arg)] if gt_arg else [
        ROOT / "ground_truth.csv", ROOT / "output" / "ground_truth.csv",
        ROOT / "data" / "ground_truth.csv",
    ]
    for path in candidates:
        if path.is_file():
            gt = pd.read_csv(path)
            if "image_id" in gt.columns and "driver_state_gut" in gt.columns:
                return gt[["image_id", "driver_state_gut"]]
            print(f"  ⚠ {path} lacks image_id/driver_state_gut columns — skipping gut analysis")
            return None
    print("  ⚠ ground_truth.csv not found — skipping gut-score analysis (use --ground-truth)")
    return None


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse experiment results")
    p.add_argument("--results",   default=str(RESULTS), help="Path to experiment_results.csv")
    p.add_argument("--threshold", type=int, default=70, help="Risk score binarisation threshold")
    p.add_argument("--ground-truth", default=None,
                   help="Path to ground_truth.csv (needs image_id + driver_state_gut). "
                        "Default: searches next to the script and in output/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"Error: results file not found: {results_path}")
        sys.exit(1)

    df = pd.read_csv(results_path)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ── 1. Console summary ───────────────────────────────────────────────────
    n_total = len(df)
    n_gt    = df["ground_truth_score"].notna().sum()
    print(f"\n{'═'*60}")
    print(f"  EXPERIMENT ANALYSIS SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total images processed : {n_total}")
    print(f"  Images with GT score   : {n_gt}")
    print(f"  Cond A score  mean±std : {df['condition_a_score'].mean():.1f} ± {df['condition_a_score'].std():.1f}")
    print(f"  Cond B score  mean±std : {df['condition_b_score'].mean():.1f} ± {df['condition_b_score'].std():.1f}")
    if n_gt > 0:
        print(f"  GT score      mean±std : {df['ground_truth_score'].mean():.1f} ± {df['ground_truth_score'].std():.1f}")
    print(f"{'─'*60}")

    # Advisory state distribution
    for col, label in [("condition_a_state", "Cond A"), ("condition_b_state", "Cond B")]:
        if col in df.columns:
            counts = df[col].value_counts()
            print(f"\n  {label} advisory distribution:")
            for state in ADVISORY_ORDER:
                c = counts.get(state, 0)
                bar = "█" * int(c / max(counts.max(), 1) * 20)
                print(f"    {state:15s}  {c:3d}  {bar}")

    # Pearson & Spearman vs GT
    eval_df = df.dropna(subset=["ground_truth_score", "condition_a_score", "condition_b_score"])
    eval_df = eval_df[(eval_df["condition_a_score"] >= 0) & (eval_df["condition_b_score"] >= 0)]
    if len(eval_df) >= 3:
        gt = eval_df["ground_truth_score"].values
        for cond, label in [("condition_a_score", "Cond A"), ("condition_b_score", "Cond B")]:
            sc = eval_df[cond].values
            r, p_r = stats.pearsonr(sc, gt)
            rho, p_rho = stats.spearmanr(sc, gt)
            print(f"\n  Correlation vs GT ({label}, n={len(eval_df)}):")
            print(f"    Pearson  r  = {r:+.3f}  (p={p_r:.3f})")
            print(f"    Spearman ρ  = {rho:+.3f}  (p={p_rho:.3f})")
    else:
        print(f"\n  ⚠ Only {len(eval_df)} image(s) have GT scores — need ≥ 3 to compute correlation.")
    print(f"\n{'═'*60}\n")

    # ── 2. Score distribution – Cond A vs B vs GT ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = range(0, 105, 5)
    ax.hist(df["condition_a_score"].dropna(), bins=bins, alpha=0.6, label="Cond A (image-only)", color="#4C9BE8")
    ax.hist(df["condition_b_score"].dropna(), bins=bins, alpha=0.6, label="Cond B (image + YOLO)", color="#E8834C")
    if n_gt > 0:
        ax.hist(df["ground_truth_score"].dropna(), bins=bins, alpha=0.7,
                label="Ground truth", color="#5CBF72", edgecolor="white")
    ax.axvline(args.threshold, color="red", ls="--", lw=1.2, label=f"Threshold ({args.threshold})")
    ax.set_xlabel("Risk score (0–100)", fontsize=11)
    ax.set_ylabel("Number of images", fontsize=11)
    ax.set_title("Score Distribution: Conditions A & B vs Ground Truth", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = FIGURES / "01_score_distributions.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out.name}")
    plt.close(fig)

    # ── 3. Scatter: GT vs Cond A & Cond B ────────────────────────────────────
    if len(eval_df) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
        for ax, cond, label, color in [
            (axes[0], "condition_a_score", "Cond A: Image-only", "#4C9BE8"),
            (axes[1], "condition_b_score", "Cond B: Image + YOLO", "#E8834C"),
        ]:
            x = eval_df["ground_truth_score"].values.astype(float)
            y = eval_df[cond].values.astype(float)
            ax.scatter(x, y, alpha=0.7, s=50, color=color, edgecolors="white", linewidth=0.5)
            correlation_line(ax, x, y)
            ax.plot([0, 100], [0, 100], "k--", lw=0.8, alpha=0.4, label="Perfect agreement")
            ax.set_xlim(0, 100); ax.set_ylim(0, 100)
            ax.set_xlabel("Ground truth score", fontsize=11)
            ax.set_ylabel("Model score", fontsize=11)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
        fig.suptitle("Model Score vs Ground Truth", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = FIGURES / "02_scatter_gt_vs_model.png"
        fig.savefig(out, dpi=150)
        print(f"  Saved: {out.name}")
        plt.close(fig)

    # ── 4. Factor heatmap: mean sub-scores, Cond A vs Cond B ─────────────────
    factor_cols_a = [f"cond_a_{f}_score" for f in FACTORS]
    factor_cols_b = [f"cond_b_{f}_score" for f in FACTORS]
    # Only include columns that actually exist (old CSVs may not have them)
    fa_present = [c for c in factor_cols_a if c in df.columns]
    fb_present = [c for c in factor_cols_b if c in df.columns]

    if fa_present and fb_present:
        means_a = [df[c].mean() for c in fa_present]
        means_b = [df[c].mean() for c in fb_present]
        data = np.array([means_a, means_b])         # shape (2, n_factors)
        labels = [f.replace("_", "\n") for f in FACTORS[:len(fa_present)]]

        fig, ax = plt.subplots(figsize=(9, 2.8))
        im = ax.imshow(data, cmap=CMAP, vmin=0, vmax=10, aspect="auto")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Cond A\n(image-only)", "Cond B\n(+YOLO)"], fontsize=10)
        for i in range(2):
            for j in range(len(fa_present)):
                ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white" if data[i, j] > 5 else "black")
        plt.colorbar(im, ax=ax, label="Mean sub-score (0–10)", shrink=0.9)
        ax.set_title("Mean Factor Sub-Scores by Condition", fontsize=12, fontweight="bold")
        fig.tight_layout()
        out = FIGURES / "03_factor_heatmap.png"
        fig.savefig(out, dpi=150)
        print(f"  Saved: {out.name}")
        plt.close(fig)

    # ── 5. Score delta: Cond B – Cond A per image ────────────────────────────
    df["score_delta"] = df["condition_b_score"] - df["condition_a_score"]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#E8834C" if d > 0 else "#4C9BE8" for d in df["score_delta"]]
    ax.bar(range(len(df)), df["score_delta"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Image index", fontsize=11)
    ax.set_ylabel("Score delta (Cond B − Cond A)", fontsize=11)
    ax.set_title("Effect of Adding YOLO Context (Cond B − Cond A)\nOrange = YOLO raised score, Blue = YOLO lowered score",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "04_score_delta.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out.name}")
    plt.close(fig)

    # ── 6. Advisory-state bar chart (Cond A vs Cond B) ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    states = [s for s in ADVISORY_ORDER if s != "Unknown"]
    x = np.arange(len(states))
    w = 0.35
    for col, label, color, offset in [
        ("condition_a_state", "Cond A", "#4C9BE8", -w/2),
        ("condition_b_state", "Cond B", "#E8834C",  w/2),
    ]:
        if col in df.columns:
            counts = df[col].value_counts()
            vals = [counts.get(s, 0) for s in states]
            ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(states, fontsize=11)
    ax.set_ylabel("Number of images", fontsize=11)
    ax.set_title("Advisory State Distribution: Cond A vs Cond B", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = FIGURES / "05_advisory_states.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out.name}")
    plt.close(fig)

    # ── 7. Instinctive (gut) score vs Cond A / Cond B ────────────────────────
    gut_df = load_gut_scores(args.ground_truth)
    if gut_df is not None:
        m = df.merge(gut_df, on="image_id", how="left")
        m = m.dropna(subset=["driver_state_gut", "condition_a_score", "condition_b_score"])
        m = m[(m["condition_a_score"] >= 0) & (m["condition_b_score"] >= 0)]
        print(f"\n{'═'*60}\n  GUT-INSTINCT SCORE ANALYSIS (n={len(m)})\n{'═'*60}")
        if len(m) >= 3:
            gut = m["driver_state_gut"].values.astype(float)
            panels = [("condition_a_score", "Cond A: Image-only", "#4C9BE8"),
                      ("condition_b_score", "Cond B: Image + YOLO", "#E8834C")]
            has_formula = "ground_truth_score" in m.columns and m["ground_truth_score"].notna().sum() >= 3
            if has_formula:
                panels.append(("ground_truth_score", "Formula GT (factor-based)", "#5CBF72"))
            for col, label, _ in panels:
                sc = m[col].values.astype(float)
                ok = ~np.isnan(sc)
                r, p_r = stats.pearsonr(sc[ok], gut[ok])
                rho, p_rho = stats.spearmanr(sc[ok], gut[ok])
                mae = np.mean(np.abs(sc[ok] - gut[ok]))
                bias = np.mean(sc[ok] - gut[ok])
                print(f"\n  {label} vs gut:")
                print(f"    Pearson r = {r:+.3f} (p={p_r:.3f})   Spearman ρ = {rho:+.3f} (p={p_rho:.3f})")
                print(f"    MAE = {mae:.1f}   mean bias (model − gut) = {bias:+.1f}")

            fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5))
            for ax, (col, label, color) in zip(np.atleast_1d(axes), panels):
                x = gut
                y = m[col].values.astype(float)
                ax.scatter(x, y, alpha=0.7, s=50, color=color, edgecolors="white", linewidth=0.5)
                correlation_line(ax, x, y)
                ax.plot([0, 100], [0, 100], "k--", lw=0.8, alpha=0.4, label="Perfect agreement")
                ax.set_xlim(0, 100); ax.set_ylim(0, 100)
                ax.set_xlabel("Gut-instinct score", fontsize=11)
                ax.set_ylabel("Score", fontsize=11)
                ax.set_title(label, fontsize=11, fontweight="bold")
                ax.legend(fontsize=8)
            fig.suptitle("Scores vs Gut-Instinct Ground Truth", fontsize=13, fontweight="bold")
            fig.tight_layout()
            out = FIGURES / "06_scatter_gut_vs_model.png"
            fig.savefig(out, dpi=150)
            print(f"\n  Saved: {out.name}")
            plt.close(fig)
        else:
            print(f"  ⚠ Only {len(m)} image(s) matched — need ≥ 3.")
        print(f"\n{'═'*60}")

    print(f"\n  All figures saved to: {FIGURES}\n")


if __name__ == "__main__":
    main()