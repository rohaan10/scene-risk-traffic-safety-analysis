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
  6. Gut scatter      – Cond A / Cond B vs gut-instinct ground truth
  7. Paired error     – per-scene absolute error with bootstrap CIs vs factor GT
  8. Paired error     – per-scene absolute error with bootstrap CIs vs gut instinct
  9. Factor shifts    – per-factor paired mean score change B − A

All charts are saved to output/figures/ and displayed inline if running in a
Jupyter/Colab environment.

Usage:
    python analysis.py
    python analysis.py --results output/experiment_results.csv --threshold 70
    python analysis.py --ground-truth ground_truth.csv
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
ROOT    = Path(__file__).resolve().parent
RESULTS = ROOT / "output" / "experiment_results.csv"
FIGURES = ROOT / "output" / "figures"

FACTORS = [
    "traffic_density",
    "vulnerable_road_users",
    "conflict_proximity",
    "road_complexity",
    "impaired_visibility",
]

FACTOR_LABELS = {
    "traffic_density":       "Traffic density",
    "vulnerable_road_users": "Vulnerable road users",
    "conflict_proximity":    "Conflict proximity",
    "road_complexity":       "Road complexity",
    "impaired_visibility":   "Impaired visibility",
}

ADVISORY_ORDER = ["Relax", "Stay Alert", "Take Control", "Unknown"]
CMAP = "RdYlGn_r"   # green=low risk, red=high risk

# Colour palette shared by the improved plots
B_BETTER  = "#23856D"
A_BETTER  = "#C45A56"
NEAR_ZERO = "#858585"


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


def bootstrap_mean_ci(
    values: np.ndarray, *, n_boot: int = 10_000, seed: int = 42
) -> tuple[float, float, float, np.ndarray]:
    """Return observed mean, percentile 95% CI lower/upper, and bootstrap means."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan, np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[indices].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi), boot_means


def find_ground_truth_file(explicit_path: str | None, results_path: Path) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        if path.is_file():
            return path
        print(f"Warning: ground-truth file not found: {path}")
        return None
    candidates = [
        ROOT / "ground_truth.csv",
        ROOT / "output" / "ground_truth.csv",
        ROOT / "data" / "ground_truth.csv",
        results_path.parent / "ground_truth.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


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


# ── Improved-plot helpers (figures 07-09) ────────────────────────────────────
def paired_error_figure(
    df: pd.DataFrame,
    target_col: str,
    target_label: str,
    output_path: Path,
    seed: int = 42,
) -> None:
    """Compare absolute prediction errors per scene, not just correlations."""
    cols = ["image_id", target_col, "condition_a_score", "condition_b_score"]
    data = df[cols].copy()
    for col in cols[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=cols[1:])
    data = data[(data["condition_a_score"] >= 0) & (data["condition_b_score"] >= 0)]
    data = data[data[target_col] >= 0]
    if len(data) < 3:
        print(f"Skipping {target_label}: only {len(data)} paired scenes available.")
        return

    target = data[target_col].to_numpy(dtype=float)
    pred_a = data["condition_a_score"].to_numpy(dtype=float)
    pred_b = data["condition_b_score"].to_numpy(dtype=float)
    err_a  = np.abs(pred_a - target)
    err_b  = np.abs(pred_b - target)
    # Paired per-scene difference: negative values mean Condition B has lower error.
    delta  = err_b - err_a

    mae_a,      a_lo,     a_hi,     _ = bootstrap_mean_ci(err_a,  seed=seed)
    mae_b,      b_lo,     b_hi,     _ = bootstrap_mean_ci(err_b,  seed=seed + 1)
    delta_mean, delta_lo, delta_hi, _ = bootstrap_mean_ci(delta,  seed=seed + 2)
    wins_b = int(np.sum(delta < -1e-9))
    wins_a = int(np.sum(delta >  1e-9))
    ties   = len(delta) - wins_a - wins_b

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [0.9, 1.5]})
    fig.suptitle(
        f"Does YOLO improve score accuracy?\nReference: {target_label} (n={len(data)} scenes)",
        fontsize=14, fontweight="bold",
    )

    # Panel 1: MAE with bootstrap intervals.
    ax = axes[0]
    means  = [mae_a, mae_b]
    lower  = [mae_a - a_lo, mae_b - b_lo]
    upper  = [a_hi - mae_a, b_hi - mae_b]
    y      = np.array([1, 0])
    ax.errorbar(means, y, xerr=np.array([lower, upper]), fmt="o", markersize=8,
                capsize=5, linewidth=2, color="#34495E")
    ax.set_yticks(y, ["A: image only", "B: image + YOLO"])
    ax.set_xlabel("Mean absolute error (lower is better)")
    ax.set_title("Average error ± 95% bootstrap CI", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.7, 1.7)
    ax.text(
        0.02, 0.02,
        f"Paired ΔMAE (B − A) = {delta_mean:+.2f}\n95% bootstrap CI [{delta_lo:+.2f}, {delta_hi:+.2f}]",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    # Panel 2: every scene's change in absolute error.
    ax      = axes[1]
    order   = np.argsort(delta)
    sorted_delta = delta[order]
    ranks   = np.arange(len(delta))
    colors  = [B_BETTER if x < -1e-9 else A_BETTER if x > 1e-9 else NEAR_ZERO for x in sorted_delta]
    ax.scatter(sorted_delta, ranks, c=colors, s=34, alpha=0.9, edgecolors="white", linewidths=0.35)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="No change")
    ax.axvline(delta_mean, color="#34495E", linewidth=1.5, label=f"Mean change = {delta_mean:+.2f}")
    ax.axvspan(delta_lo, delta_hi, color="#34495E", alpha=0.12, label="95% CI for mean change")
    ax.set_xlabel("Per-scene change in absolute error: |B − reference| − |A − reference|")
    ax.set_ylabel("Scenes (sorted by change in error)")
    ax.set_title(
        f"B lower error: {wins_b}/{len(delta)} · A lower error: {wins_a}/{len(delta)} · Ties: {ties}",
        fontsize=11,
    )
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{target_label} (n={len(data)}):")
    print(f"  Condition A MAE: {mae_a:.2f} (95% bootstrap CI {a_lo:.2f}–{a_hi:.2f})")
    print(f"  Condition B MAE: {mae_b:.2f} (95% bootstrap CI {b_lo:.2f}–{b_hi:.2f})")
    print(f"  Paired ΔMAE B−A: {delta_mean:+.2f} (95% bootstrap CI {delta_lo:+.2f} to {delta_hi:+.2f})")
    print(f"  Per-scene lower absolute error: B={wins_b}, A={wins_a}, ties={ties}")
    print(f"  Saved: {output_path}")


def factor_shift_figure(df: pd.DataFrame, output_path: Path, seed: int = 42) -> None:
    """Show the paired mean change in each factor score, B minus A."""
    rows = []
    for i, factor in enumerate(FACTORS):
        col_a = f"cond_a_{factor}_score"
        col_b = f"cond_b_{factor}_score"
        if col_a not in df.columns or col_b not in df.columns:
            continue
        a    = pd.to_numeric(df[col_a], errors="coerce")
        b    = pd.to_numeric(df[col_b], errors="coerce")
        pair = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(pair) < 3:
            continue
        pair = pair[(pair["a"] >= 0) & (pair["a"] <= 10) & (pair["b"] >= 0) & (pair["b"] <= 10)]
        if len(pair) < 3:
            continue
        delta = pair["b"].to_numpy() - pair["a"].to_numpy()
        mean, lo, hi, _ = bootstrap_mean_ci(delta, seed=seed + i)
        rows.append({
            "factor":     factor,
            "label":      FACTOR_LABELS[factor],
            "n":          len(pair),
            "mean_delta": mean,
            "lo":         lo,
            "hi":         hi,
            "mean_a":     float(pair["a"].mean()),
            "mean_b":     float(pair["b"].mean()),
        })

    if not rows:
        print("Skipping factor-shift plot: no paired factor columns found.")
        return

    # Keep the rubric's original order, with the first factor at the top.
    rows.sort(key=lambda r: FACTORS.index(r["factor"]))
    labels   = [r["label"] for r in rows]
    means    = np.array([r["mean_delta"] for r in rows])
    low_err  = np.array([r["mean_delta"] - r["lo"] for r in rows])
    high_err = np.array([r["hi"] - r["mean_delta"] for r in rows])
    y        = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.errorbar(means, y, xerr=np.array([low_err, high_err]), fmt="o", markersize=7,
                capsize=5, linewidth=2, color="#34495E")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired score change (Condition B − Condition A; 0–10 scale)")
    ax.set_title("How YOLO changes the five factor scores", fontsize=13, fontweight="bold")
    ax.text(
        0.01, -0.19,
        "Negative = YOLO lowered the score; positive = YOLO raised it. Bars show 95% paired-bootstrap CIs.\n"
        "This measures score shifts, not correctness; correctness requires human reference scores for each factor.",
        transform=ax.transAxes, fontsize=9, va="top",
    )
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nMean paired factor-score changes (B − A):")
    for r in rows:
        print(
            f"  {r['label']}: {r['mean_delta']:+.2f} (95% CI {r['lo']:+.2f} to {r['hi']:+.2f}); "
            f"means A={r['mean_a']:.2f}, B={r['mean_b']:.2f}, n={r['n']}"
        )
    print(f"Saved: {output_path}")


# ── Argument parsing ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse experiment results and produce all comparison figures."
    )
    p.add_argument("--results",      default=str(RESULTS), help="Path to experiment_results.csv")
    p.add_argument("--threshold",    type=int, default=70,  help="Risk score binarisation threshold")
    p.add_argument("--ground-truth", default=None,
                   help="Path to ground_truth.csv (needs image_id + driver_state_gut). "
                        "Default: searches next to the script and in output/")
    p.add_argument("--figures",      default=str(FIGURES), help="Directory for output plots")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    figures_path = Path(args.figures)

    if not results_path.is_file():
        print(f"Error: results file not found: {results_path}")
        sys.exit(1)

    df = pd.read_csv(results_path)
    figures_path.mkdir(parents=True, exist_ok=True)

    # Merge gut scores into df if available (used by figures 06-08)
    gt_path = find_ground_truth_file(args.ground_truth, results_path)
    if gt_path is not None:
        gt = pd.read_csv(gt_path)
        if "image_id" in gt.columns and "driver_state_gut" in gt.columns:
            gut = gt[["image_id", "driver_state_gut"]].drop_duplicates("image_id")
            if "driver_state_gut" in df.columns:
                df = df.drop(columns=["driver_state_gut"])
            df = df.merge(gut, on="image_id", how="left")
            print(f"Loaded gut-instinct scores from: {gt_path}")
        else:
            print(f"Warning: {gt_path} has no image_id + driver_state_gut columns; skipping gut score.")

    # ── 1. Console summary ───────────────────────────────────────────────────
    n_total = len(df)
    n_gt    = df["ground_truth_score"].notna().sum() if "ground_truth_score" in df.columns else 0
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
                c   = counts.get(state, 0)
                bar = "█" * int(c / max(counts.max(), 1) * 20)
                print(f"    {state:15s}  {c:3d}  {bar}")

    # Pearson & Spearman vs GT
    if "ground_truth_score" in df.columns:
        eval_df = df.dropna(subset=["ground_truth_score", "condition_a_score", "condition_b_score"])
        eval_df = eval_df[(eval_df["condition_a_score"] >= 0) & (eval_df["condition_b_score"] >= 0)]
        if len(eval_df) >= 3:
            gt_vals = eval_df["ground_truth_score"].values
            for cond, label in [("condition_a_score", "Cond A"), ("condition_b_score", "Cond B")]:
                sc = eval_df[cond].values
                r,   p_r   = stats.pearsonr(sc, gt_vals)
                rho, p_rho = stats.spearmanr(sc, gt_vals)
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
    out = figures_path / "01_score_distributions.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out.name}")
    plt.close(fig)

    # ── 3. Scatter: GT vs Cond A & Cond B ────────────────────────────────────
    if "ground_truth_score" in df.columns:
        eval_df = df.dropna(subset=["ground_truth_score", "condition_a_score", "condition_b_score"])
        eval_df = eval_df[(eval_df["condition_a_score"] >= 0) & (eval_df["condition_b_score"] >= 0)]
        if len(eval_df) >= 3:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
            for ax, cond, label, color in [
                (axes[0], "condition_a_score", "Cond A: Image-only",   "#4C9BE8"),
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
            out = figures_path / "02_scatter_gt_vs_model.png"
            fig.savefig(out, dpi=150)
            print(f"  Saved: {out.name}")
            plt.close(fig)

    # ── 4. Factor heatmap: mean sub-scores, Cond A vs Cond B ─────────────────
    factor_cols_a = [f"cond_a_{f}_score" for f in FACTORS]
    factor_cols_b = [f"cond_b_{f}_score" for f in FACTORS]
    fa_present    = [c for c in factor_cols_a if c in df.columns]
    fb_present    = [c for c in factor_cols_b if c in df.columns]

    if fa_present and fb_present:
        means_a = [df[c].mean() for c in fa_present]
        means_b = [df[c].mean() for c in fb_present]
        data    = np.array([means_a, means_b])         # shape (2, n_factors)
        labels  = [f.replace("_", "\n") for f in FACTORS[:len(fa_present)]]

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
        out = figures_path / "03_factor_heatmap.png"
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
    ax.set_title(
        "Effect of Adding YOLO Context (Cond B − Cond A)\n"
        "Orange = YOLO raised score, Blue = YOLO lowered score",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = figures_path / "04_score_delta.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out.name}")
    plt.close(fig)

    # ── 6. Advisory-state bar chart (Cond A vs Cond B) ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    states = [s for s in ADVISORY_ORDER if s != "Unknown"]
    x = np.arange(len(states))
    w = 0.35
    for col, label, color, offset in [
        ("condition_a_state", "Cond A", "#4C9BE8", -w / 2),
        ("condition_b_state", "Cond B", "#E8834C",  w / 2),
    ]:
        if col in df.columns:
            counts = df[col].value_counts()
            vals   = [counts.get(s, 0) for s in states]
            ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(states, fontsize=11)
    ax.set_ylabel("Number of images", fontsize=11)
    ax.set_title("Advisory State Distribution: Cond A vs Cond B", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = figures_path / "05_advisory_states.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out.name}")
    plt.close(fig)

    # ── 7. Instinctive (gut) score vs Cond A / Cond B scatter ────────────────
    if "driver_state_gut" in df.columns:
        m = df.dropna(subset=["driver_state_gut", "condition_a_score", "condition_b_score"])
        m = m[(m["condition_a_score"] >= 0) & (m["condition_b_score"] >= 0)]
        print(f"\n{'═'*60}\n  GUT-INSTINCT SCORE ANALYSIS (n={len(m)})\n{'═'*60}")
        if len(m) >= 3:
            gut    = m["driver_state_gut"].values.astype(float)
            panels = [
                ("condition_a_score", "Cond A: Image-only",   "#4C9BE8"),
                ("condition_b_score", "Cond B: Image + YOLO", "#E8834C"),
            ]
            has_formula = "ground_truth_score" in m.columns and m["ground_truth_score"].notna().sum() >= 3
            if has_formula:
                panels.append(("ground_truth_score", "Formula GT (factor-based)", "#5CBF72"))
            for col, label, _ in panels:
                sc   = m[col].values.astype(float)
                ok   = ~np.isnan(sc)
                r,   p_r   = stats.pearsonr(sc[ok], gut[ok])
                rho, p_rho = stats.spearmanr(sc[ok], gut[ok])
                mae  = np.mean(np.abs(sc[ok] - gut[ok]))
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
            out = figures_path / "06_scatter_gut_vs_model.png"
            fig.savefig(out, dpi=150)
            print(f"\n  Saved: {out.name}")
            plt.close(fig)
        else:
            print(f"  ⚠ Only {len(m)} image(s) matched — need ≥ 3.")
        print(f"\n{'═'*60}")
    else:
        print("  ⚠ ground_truth.csv not found — skipping gut-score analysis (use --ground-truth)")

    # ── 8. Paired error figure vs factor-composite ground truth ──────────────
    if "ground_truth_score" in df.columns:
        paired_error_figure(
            df, "ground_truth_score", "Human factor-composite ground-truth score",
            figures_path / "07_paired_error_vs_factor_ground_truth.png", seed=42,
        )
    else:
        print("No ground_truth_score column; skipping factor-composite reference plot.")

    # ── 9. Paired error figure vs gut-instinct score ─────────────────────────
    if "driver_state_gut" in df.columns:
        paired_error_figure(
            df, "driver_state_gut", "Human gut-instinct score",
            figures_path / "08_paired_error_vs_gut_instinct.png", seed=142,
        )
    else:
        print("No driver_state_gut column found; skipping gut-instinct reference plot.")

    # ── 10. Factor score shifts (B − A) with bootstrap CIs ───────────────────
    factor_shift_figure(df, figures_path / "09_factor_score_shifts.png")

    print(f"\n  All figures saved to: {figures_path}\n")


if __name__ == "__main__":
    main()
