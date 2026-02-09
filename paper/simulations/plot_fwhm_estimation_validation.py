"""Plot FWHM estimation validation (GM only)

Reads a CSV with columns including: fwhm_target, method, tissue, fwhm_estimate
Filters to tissue == 'gm' and plots fwhm_estimate vs fwhm_target for each method
(defaults to the Box path used by the user). Saves a PNG to the same folder as
this script unless --out is provided.

Usage example:
    python plot_fwhm_estimation_validation.py --csv /path/to/fwhm_estimation_validation.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CSV = (
    Path.home()
    / "Library/CloudStorage/Box-Box/Dissertation/figures/ch2/simulations/fwhm_estimation_validation.csv"
)


def load_and_filter(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    expected = {"fwhm_target", "method", "tissue", "fwhm_estimate"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")

    # Filter to gray matter only
    df_gm = df[df["tissue"].str.lower() == "gm"].copy()
    if df_gm.empty:
        raise ValueError("No rows with tissue == 'gm' found in CSV")

    # Limit to target FWHM values <= 12 mm as requested
    df_gm = df_gm[df_gm["fwhm_target"] <= 12.0]
    if df_gm.empty:
        raise ValueError("No rows with tissue == 'gm' and fwhm_target <= 12 found in CSV")

    return df_gm


def plot_gm(df_gm: pd.DataFrame, out_path: Path | None = None, show: bool = False) -> Path:
    # Prepare data: pivot so each method is a column
    pivot = df_gm.pivot_table(
        index="fwhm_target", columns="method", values="fwhm_estimate", aggfunc="mean"
    )

    if pivot.empty:
        raise ValueError("Pivot produced no data to plot")

    # Sort by target
    pivot = pivot.sort_index()

    # Set plotting style robustly: prefer seaborn if installed, else fall back
    try:
        import seaborn as sns  # type: ignore

        sns.set_style("whitegrid")
    except Exception:
        if "seaborn-whitegrid" in plt.style.available:
            plt.style.use("seaborn-whitegrid")
        elif "ggplot" in plt.style.available:
            plt.style.use("ggplot")
        else:
            plt.style.use("classic")

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"baseline": "C0", "constrained": "C1"}

    for method in pivot.columns:
        y = pivot[method].values
        x = pivot.index.values
        # display 'baseline' as 'unconstrained' in the legend
        display_name = "unconstrained" if method == "baseline" else method
        ax.plot(x, y, marker="o", label=display_name, color=colors.get(method, None),
                alpha=0.8)

    # Identity line to show perfect estimation across the requested x-range 0-13
    ax.plot([0, 13], [0, 13], linestyle="--", color="gray", label="ideal (y=x)")

    # Set axis limits as requested: xlim 0..13, ylim minimum 0 (keep upper auto)
    ax.set_xlim(0, 13)
    # Determine current y upper limit then set lower bound to 0 while preserving upper
    _, ymax = ax.get_ylim()
    if ymax <= 0:
        ymax = 13
    ax.set_ylim(0, ymax)

    ax.set_xlabel("Applied FWHM")
    ax.set_ylabel("Estimated FWHM (GM)")
    ax.set_title("FWHM estimation validation (GM only)")
    ax.legend()

    # Tight layout and save
    fig.tight_layout()

    if out_path is None:
        out_dir = Path(__file__).resolve().parent
        out_path = out_dir / "fwhm_estimation_validation_gm.pdf"
    else:
        out_path = Path(out_path)

    fig.savefig(out_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot FWHM estimation validation (GM only)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to CSV file")
    parser.add_argument("--out", type=Path, default=None, help="Output figure path (png)")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    args = parser.parse_args(argv)

    try:
        df_gm = load_and_filter(args.csv)
    except Exception as exc:
        print(f"Error loading CSV: {exc}", file=sys.stderr)
        return 2

    try:
        out_path = plot_gm(df_gm, args.out, show=args.show)
    except Exception as exc:
        print(f"Error plotting data: {exc}", file=sys.stderr)
        return 3

    print(f"Saved GM FWHM estimation figure to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
