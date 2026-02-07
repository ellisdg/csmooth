import argparse
import os
import subprocess
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from .paths import load_config, script_path, DEFAULT_CONFIG_PATH, PACKAGE_ROOT


def run_cmd(cmd, dry_run=False):
    print("Running:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ensure_designs(config_path: str, dry_run: bool):
    cmd = [sys.executable, str(script_path("generate_null_designs")), "--config", config_path]
    run_cmd(cmd, dry_run=dry_run)


def run_feat_batch(config_path: str, method: str, fwhm: int, dry_run: bool):
    cmd = [
        sys.executable,
        str(script_path("run_feat_null_firstlevel")),
        "--config",
        config_path,
        "--method",
        method,
        "--fwhm",
        str(fwhm),
    ]
    if dry_run:
        cmd.append("--dry-run")
    run_cmd(cmd, dry_run=dry_run)


def analyze_batch(config_path: str, method: str, fwhm: int):
    cmd = [
        sys.executable,
        str(script_path("analyze_feat_null_results")),
        "--config",
        config_path,
        "--method",
        method,
        "--fwhm",
        str(fwhm),
    ]
    run_cmd(cmd, dry_run=False)


def aggregate_results(output_root: str, methods: list[str], fwhms: list[int], out_csv: str, out_png: str, cluster_ps: list[float]):
    frames = []
    for method in methods:
        for fwhm in fwhms:
            summary_path = os.path.join(output_root, "feat", method, f"fwhm-{fwhm}", "feat_null_summary.csv")
            if not os.path.exists(summary_path):
                print(f"Missing summary: {summary_path}")
                continue
            frames.append(pd.read_csv(summary_path))
    if not frames:
        raise SystemExit("No summaries found; run analysis first")
    df = pd.concat(frames, ignore_index=True)
    agg = (
        df.groupby(["method", "fwhm", "cluster_forming_p"])
        ["detected"]
        .mean()
        .reset_index()
        .rename(columns={"detected": "fpr"})
    )
    agg.to_csv(out_csv, index=False)
    print(f"Saved summary {out_csv}")

    plt.figure(figsize=(6, 4))
    for p_val in cluster_ps:
        sub = agg[agg["cluster_forming_p"] == p_val]
        for method in methods:
            plt.plot(
                sub[sub["method"] == method]["fwhm"],
                sub[sub["method"] == method]["fpr"],
                marker="o",
                label=f"{method} p={p_val}" if method == methods[0] else None,
            )
    plt.xlabel("FWHM (mm)")
    plt.ylabel("False positive rate")
    plt.ylim(0, 1)
    handles, labels = plt.gca().get_legend_handles_labels()
    # unique labels while preserving order
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    if uniq:
        plt.legend([h for h, _ in uniq], [l for _, l in uniq], loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot {out_png}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run full individual FPR pipeline (designs -> FEAT -> summary -> plot)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to fpr_config.yaml")
    parser.add_argument("--methods", nargs="*", default=["csmooth", "gaussian"], help="Methods to process")
    parser.add_argument("--fwhm", nargs="*", type=int, help="FWHM list (defaults to config values)")
    parser.add_argument("--skip-designs", action="store_true", help="Skip design generation step")
    parser.add_argument("--skip-feat", action="store_true", help="Skip FEAT runs (assumes outputs exist)")
    parser.add_argument("--skip-analyze", action="store_true", help="Skip zstat summarization (assumes summaries exist)")
    parser.add_argument("--feat-dry-run", action="store_true", help="Print FEAT commands without running")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    fwhms = args.fwhm or config["smoothing"].get("fwhm_values", [])
    output_root = config["paths"]["output_root"] or str(PACKAGE_ROOT / "output")
    cluster_ps = config["group"].get("cluster_forming_ps", [0.01, 0.001])

    if not args.skip_designs:
        ensure_designs(args.config, dry_run=False)

    for method in args.methods:
        for fwhm in fwhms:
            if not args.skip_feat:
                run_feat_batch(args.config, method, fwhm, dry_run=args.feat_dry_run)
            if not args.skip_analyze:
                analyze_batch(args.config, method, fwhm)

    out_csv = os.path.join(output_root, "feat", "fpr_summary.csv")
    out_png = os.path.join(output_root, "feat", "fpr_plot.png")
    aggregate_results(output_root, args.methods, fwhms, out_csv, out_png, cluster_ps)


if __name__ == "__main__":
    main()
