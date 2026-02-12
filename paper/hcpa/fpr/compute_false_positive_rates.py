from __future__ import annotations

"""Compute false positive rates from FEAT null first-level outputs.

This script scans FEAT outputs for csmooth and gaussian smoothing at each FWHM,
counts supra-threshold voxels (clusterwise and voxelwise), and writes a CSV for
analysis in R or Python.
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from nilearn import plotting


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "fpr_config.yaml"

MASK_BY_METHOD = {
    "csmooth": "/data2/david.ellis/public/HCPA/code/fpr/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz",
    "gaussian": "/data2/david.ellis/public/HCPA/code/fpr/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask_resampled.nii.gz",
}


def resolve_rel_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def load_config(config_path: str | Path | None = None) -> dict:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_dir = cfg_path.parent
    paths = cfg.get("paths", {})
    resolved_paths = {}
    for key, val in paths.items():
        if isinstance(val, str) and key != "subjects_glob":
            resolved_paths[key] = resolve_rel_path(val, base_dir)
        else:
            resolved_paths[key] = val
    cfg["paths"] = resolved_paths
    cfg["_config_path"] = str(cfg_path)
    cfg["_base_dir"] = str(base_dir)
    return cfg


def remap_path(path: str, replace_prefix: Optional[list[tuple[str, str]]]) -> str:
    """Optionally swap a leading prefix (for local mount compatibility)."""
    if not replace_prefix:
        return path
    try:
        p = Path(path)
        for old, new in replace_prefix:
            try:
                if Path(old) in p.parents or p == Path(old):
                    return str(Path(new) / p.relative_to(old))
            except Exception:
                continue
    except Exception:
        pass
    return path


@dataclass
class FeatRun:
    method: str
    fwhm: str
    subject: str
    dir: str
    run: str
    feat_dir: Path
    zstat_path: Path
    thresh_path: Path


def parse_feat_dir(feat_dir: Path) -> FeatRun:
    """Extract metadata from a FEAT directory path."""
    # Expected layout: output_root/feat/<method>/fwhm-<fwhm>/fsl/<subject>/dir-<dir>_run-<run>.feat
    method = feat_dir.parents[3].name
    fwhm_part = feat_dir.parents[2].name
    m_fwhm = re.search(r"fwhm-(?P<fwhm>[0-9]+(?:\.[0-9]+)?)", fwhm_part)
    if not m_fwhm:
        raise ValueError(f"Could not parse FWHM from {feat_dir}")
    subject = feat_dir.parent.name
    m = re.match(r"dir-(?P<dir>[^_]+)_run-(?P<run>[^.]+)\.feat", feat_dir.name)
    if not m:
        raise ValueError(f"Could not parse dir/run from {feat_dir}")
    zstat_path = feat_dir / "stats" / "zstat1.nii.gz"
    thresh_path = feat_dir / "thresh_zstat1.nii.gz"
    return FeatRun(
        method=method,
        fwhm=m_fwhm.group("fwhm"),
        subject=subject,
        dir=m.group("dir"),
        run=m.group("run"),
        feat_dir=feat_dir,
        zstat_path=zstat_path,
        thresh_path=thresh_path,
    )


def discover_feat_runs(output_root: str, methods: Iterable[str], fwhms: Iterable[str]) -> List[FeatRun]:
    runs: List[FeatRun] = []
    for method in methods:
        for fwhm in fwhms:
            base = Path(output_root) / "feat" / method / f"fwhm-{fwhm}" / "fsl"
            pattern = base / "sub-*" / "dir-*_run-*.feat"
            for feat_path in glob.glob(str(pattern)):
                try:
                    runs.append(parse_feat_dir(Path(feat_path)))
                except Exception as exc:  # keep scanning even if one path is malformed
                    print(f"Skipping {feat_path}: {exc}")
    return sorted(runs, key=lambda r: (r.method, float(r.fwhm), r.subject, r.dir, r.run))


def load_mask(mask_path: str) -> tuple[np.ndarray, int]:
    img = nib.load(mask_path)
    data = img.get_fdata()
    mask = data > 0.5
    return mask, int(np.count_nonzero(mask))


def load_mask_with_img(mask_path: str) -> tuple[nib.Nifti1Image, np.ndarray, int]:
    img = nib.load(mask_path)
    data = img.get_fdata()
    mask = data > 0
    return img, mask, int(np.count_nonzero(mask))


def count_voxels(nifti_path: Path, mask: np.ndarray, threshold: float) -> Optional[int]:
    if not nifti_path.exists():
        return None
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    if data.shape != mask.shape:
        raise ValueError(f"Shape mismatch for {nifti_path}: data {data.shape} vs mask {mask.shape}")
    return int(np.count_nonzero((data > threshold) & mask))


def evaluate_run(run: FeatRun, mask: np.ndarray, total_voxels: int, voxel_z_thresh: float) -> dict:
    cluster_count = count_voxels(run.thresh_path, mask, threshold=0.0)
    voxel_count = count_voxels(run.zstat_path, mask, threshold=voxel_z_thresh)
    cluster_pct = (cluster_count / total_voxels * 100) if cluster_count is not None else np.nan
    voxel_pct = (voxel_count / total_voxels * 100) if voxel_count is not None else np.nan
    return {
        "method": run.method,
        "fwhm": run.fwhm,
        "subject": run.subject,
        "dir": run.dir,
        "run": run.run,
        "feat_dir": str(run.feat_dir),
        "cluster_voxels": cluster_count,
        "voxel_voxels": voxel_count,
        "total_voxels": total_voxels,
        "cluster_fp_pct": cluster_pct,
        "voxel_fp_pct": voxel_pct,
    }


def accumulate_fpr_maps(
    runs: list[FeatRun], mask_path: str, voxel_z_thresh: float
) -> tuple[np.ndarray, int, np.ndarray, int, np.ndarray, nib.Nifti1Image]:
    mask_img, mask_bool, _ = load_mask_with_img(mask_path)
    cluster_counts = np.zeros(mask_bool.shape, dtype=np.uint32)
    voxel_counts = np.zeros(mask_bool.shape, dtype=np.uint32)
    cluster_runs = 0
    voxel_runs = 0
    for run in tqdm(runs, desc="FPR maps", leave=False):
        if run.thresh_path.exists():
            data = nib.load(str(run.thresh_path)).get_fdata()
            if data.shape != mask_bool.shape:
                raise ValueError(f"Shape mismatch for {run.thresh_path}: {data.shape} vs mask {mask_bool.shape}")
            cluster_counts += ((data > 0) & mask_bool)
            cluster_runs += 1
        if run.zstat_path.exists():
            data = nib.load(str(run.zstat_path)).get_fdata()
            if data.shape != mask_bool.shape:
                raise ValueError(f"Shape mismatch for {run.zstat_path}: {data.shape} vs mask {mask_bool.shape}")
            voxel_counts += ((data > voxel_z_thresh) & mask_bool)
            voxel_runs += 1
    return cluster_counts, cluster_runs, voxel_counts, voxel_runs, mask_bool, mask_img


def write_fpr_volume(counts: np.ndarray, n_runs: int, mask_img: nib.Nifti1Image, output_path: Path) -> None:
    if n_runs == 0:
        data = np.zeros_like(counts, dtype=np.float32)
    else:
        data = counts.astype(np.float32) / float(n_runs) * 100.0
    img = nib.Nifti1Image(data, mask_img.affine )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))


def save_stat_map_png(data: np.ndarray, mask_img: nib.Nifti1Image, output_path: Path, title: str) -> None:
    """Save a quick-view stat map PNG using nilearn."""
    display = plotting.plot_stat_map(
        nib.Nifti1Image(data, mask_img.affine),
        bg_img=mask_img,
        display_mode="ortho",
        threshold=0,
        colorbar=True,
        vmax=100,
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.savefig(str(output_path))
    display.close()


def save_boxplots(entries: list[tuple[str, np.ndarray]], output_path: Path, title: str) -> None:
    """Write boxplots for FPR percentage values per map."""
    if not entries:
        return
    labels = [lbl for lbl, _ in entries]
    data = [vals for _, vals in entries]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 6))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("FPR (%)")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# Globals used by worker processes
_WORKER_MASK: Optional[np.ndarray] = None
_WORKER_TOTAL_VOXELS: Optional[int] = None
_WORKER_VOXEL_Z: Optional[float] = None


def _init_worker(mask: np.ndarray, total_voxels: int, voxel_z_thresh: float) -> None:
    global _WORKER_MASK, _WORKER_TOTAL_VOXELS, _WORKER_VOXEL_Z
    _WORKER_MASK = mask
    _WORKER_TOTAL_VOXELS = total_voxels
    _WORKER_VOXEL_Z = voxel_z_thresh


def _evaluate_run_worker(run: FeatRun) -> dict:
    if _WORKER_MASK is None or _WORKER_TOTAL_VOXELS is None or _WORKER_VOXEL_Z is None:
        raise RuntimeError("Worker not initialized with mask/voxel info")
    return evaluate_run(run, _WORKER_MASK, _WORKER_TOTAL_VOXELS, voxel_z_thresh=_WORKER_VOXEL_Z)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute false positive rates from FEAT null outputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to fpr_config.yaml")
    parser.add_argument(
        "--method",
        nargs="*",
        choices=["csmooth", "gaussian"],
        default=["csmooth", "gaussian"],
        help="Smoothing methods to include (default: both)",
    )
    parser.add_argument("--fwhm", nargs="*", help="Optional list of FWHM values to include (e.g., 6 12)")
    parser.add_argument("--output-csv", help="Output CSV path (default: <output_root>/feat/fpr_false_positive_rates.csv)")
    parser.add_argument("--voxel-z", type=float, default=3.1, help="Z threshold for voxelwise map (default: 3.1)")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Raise if any expected zstat/thresh files are missing; otherwise rows get NaN",
    )
    parser.add_argument(
        "--replace-prefix",
        nargs=2,
        action="append",
        metavar=("FROM", "TO"),
        help=(
            "Replace leading path prefix FROM with TO for all data paths (repeatable). "
            "Example: --replace-prefix /data/david.ellis /media/conda2 --replace-prefix /data2/david.ellis /media/conda2"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1, i.e., serial processing).",
    )
    parser.add_argument(
        "--skip-fpr-maps",
        action="store_true",
        help="Skip writing voxelwise FPR volumes (one per method and threshold type).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    replace_prefix = [tuple(p) for p in args.replace_prefix] if args.replace_prefix else None

    output_root = config["paths"].get("output_root") or str(PACKAGE_ROOT / "output")
    output_root = remap_path(output_root, replace_prefix)
    fwhm_values = args.fwhm or [str(v) for v in config.get("smoothing", {}).get("fwhm_values", [])]
    if not fwhm_values:
        raise SystemExit("No FWHM values provided or found in config")

    methods = args.method
    output_csv = args.output_csv or os.path.join(output_root, "feat", "fpr_false_positive_rates.csv")
    counts_txt = os.path.join(os.path.dirname(output_csv), "fpr_counts.txt")

    method_info: dict[str, dict] = {}
    for method in methods:
        mask_path = MASK_BY_METHOD.get(method)
        if not mask_path:
            raise SystemExit(f"No mask configured for method '{method}'")
        mask_path = remap_path(mask_path, replace_prefix)
        if not os.path.exists(mask_path):
            raise SystemExit(f"Mask not found: {mask_path}")
        mask, total_voxels = load_mask(mask_path)
        runs = discover_feat_runs(output_root, [method], fwhm_values)
        if not runs:
            print(f"No FEAT runs found for method {method}")
        method_info[method] = {
            "mask": mask,
            "total_voxels": total_voxels,
            "mask_path": mask_path,
            "runs": runs,
        }

    if not method_info:
        raise SystemExit("No methods configured")

    # Intersect subject/dir/run/fwhm across methods so we only analyze matched outputs.
    def run_key(r: FeatRun) -> tuple[str, str, str, str]:
        return (r.subject, r.dir, r.run, r.fwhm)

    key_sets = []
    for info in method_info.values():
        key_sets.append({run_key(r) for r in info["runs"]})
    matched_keys = set.intersection(*key_sets) if key_sets else set()
    if not matched_keys:
        raise SystemExit("No matched runs across requested methods")

    matched_subjects = {k[0] for k in matched_keys}
    matched_counts: dict[str, int] = {}

    rows = []
    boxplot_entries = {"cluster": [], "voxel": []}
    filtered_method_runs: dict[str, list[FeatRun]] = {}
    for method in methods:
        info = method_info[method]
        runs = [r for r in info["runs"] if run_key(r) in matched_keys]
        filtered_method_runs[method] = runs
        matched_counts[method] = len(runs)

        if not runs:
            print(f"No matched FEAT runs found for method {method}")
            continue

        if args.jobs > 1:
            from multiprocessing import Pool

            chunksize = max(1, len(runs) // (args.jobs * 4))
            with Pool(
                processes=args.jobs,
                initializer=_init_worker,
                initargs=(info["mask"], info["total_voxels"], args.voxel_z),
            ) as pool:
                for row in tqdm(
                    pool.imap_unordered(_evaluate_run_worker, runs, chunksize=chunksize),
                    total=len(runs),
                    desc=f"{method} (jobs={args.jobs})",
                ):
                    rows.append(row)
                    if args.fail_on_missing and (row["cluster_voxels"] is None or row["voxel_voxels"] is None):
                        raise FileNotFoundError(f"Missing inputs for {row['feat_dir']}")
        else:
            for run in tqdm(runs, desc=f"{method} (serial)"):
                row = evaluate_run(run, info["mask"], info["total_voxels"], voxel_z_thresh=args.voxel_z)
                rows.append(row)
                if args.fail_on_missing and (row["cluster_voxels"] is None or row["voxel_voxels"] is None):
                    raise FileNotFoundError(f"Missing inputs for {run.feat_dir}")

    if not rows:
        raise SystemExit("No runs processed")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows to {output_csv}")

    with open(counts_txt, "w", encoding="utf-8") as f:
        f.write(f"Matched subjects: {len(matched_subjects)}\n")
        f.write(f"Matched runs (total across methods): {len(matched_keys) * len(methods)}\n")
        for method in methods:
            f.write(f"{method} matched runs: {matched_counts.get(method, 0)}\n")
    print(f"Wrote counts to {counts_txt}")

    summary = (
        df.groupby(["method", "fwhm"])[["cluster_fp_pct", "voxel_fp_pct"]]
        .mean()
        .reset_index()
        .sort_values(["fwhm", "method"])
    )
    print("\nMean FP percentages by method and FWHM:")
    print(summary.to_string(index=False))

    if not args.skip_fpr_maps:
        maps_dir = Path(output_root) / "feat" / "fpr_maps"
        for method, runs in filtered_method_runs.items():
            if not runs:
                continue
            print(f"\nBuilding voxelwise FPR volumes for {method} ({len(runs)} runs)")
            mask_path = method_info[method]["mask_path"]

            # Overall (all FWHM) map per method
            cluster_counts, cluster_n, voxel_counts, voxel_n, mask_bool, mask_img = accumulate_fpr_maps(
                runs, mask_path, voxel_z_thresh=args.voxel_z
            )
            write_fpr_volume(cluster_counts, cluster_n, mask_img, maps_dir / f"{method}_cluster_fpr.nii.gz")
            write_fpr_volume(voxel_counts, voxel_n, mask_img, maps_dir / f"{method}_voxel_fpr.nii.gz")
            cluster_pct = (cluster_counts.astype(np.float32) / float(cluster_n) * 100.0) if cluster_n else np.zeros_like(cluster_counts, dtype=np.float32)
            voxel_pct = (voxel_counts.astype(np.float32) / float(voxel_n) * 100.0) if voxel_n else np.zeros_like(voxel_counts, dtype=np.float32)
            save_stat_map_png(cluster_pct, mask_img, maps_dir / f"{method}_cluster_fpr.png", f"{method} cluster FPR (all)")
            save_stat_map_png(voxel_pct, mask_img, maps_dir / f"{method}_voxel_fpr.png", f"{method} voxel FPR (all)")
            boxplot_entries["cluster"].append((f"{method}-all", cluster_pct[mask_bool]))
            boxplot_entries["voxel"].append((f"{method}-all", voxel_pct[mask_bool]))

            # Per-FWHM maps
            runs_by_fwhm: dict[str, list[FeatRun]] = {}
            for r in runs:
                runs_by_fwhm.setdefault(r.fwhm, []).append(r)
            for fwhm, f_runs in sorted(runs_by_fwhm.items(), key=lambda kv: float(kv[0])):
                sub_dir = maps_dir / method / f"fwhm-{fwhm}"
                cluster_counts, cluster_n, voxel_counts, voxel_n, mask_bool, mask_img = accumulate_fpr_maps(
                    f_runs, mask_path, voxel_z_thresh=args.voxel_z
                )
                write_fpr_volume(cluster_counts, cluster_n, mask_img, sub_dir / "cluster_fpr.nii.gz")
                write_fpr_volume(voxel_counts, voxel_n, mask_img, sub_dir / "voxel_fpr.nii.gz")
                cluster_pct = (cluster_counts.astype(np.float32) / float(cluster_n) * 100.0) if cluster_n else np.zeros_like(cluster_counts, dtype=np.float32)
                voxel_pct = (voxel_counts.astype(np.float32) / float(voxel_n) * 100.0) if voxel_n else np.zeros_like(voxel_counts, dtype=np.float32)
                save_stat_map_png(cluster_pct, mask_img, sub_dir / "cluster_fpr.png", f"{method} cluster FPR fwhm-{fwhm}")
                save_stat_map_png(voxel_pct, mask_img, sub_dir / "voxel_fpr.png", f"{method} voxel FPR fwhm-{fwhm}")
                boxplot_entries["cluster"].append((f"{method}-fwhm{fwhm}", cluster_pct[mask_bool]))
                boxplot_entries["voxel"].append((f"{method}-fwhm{fwhm}", voxel_pct[mask_bool]))
            print("Saved voxelwise FPR volumes and plots to", maps_dir)

        # Boxplots comparing FPR distributions across methods and smoothing levels
        save_boxplots(
            boxplot_entries["cluster"],
            maps_dir / "fpr_cluster_boxplots.png",
            "Clusterwise FPR distributions",
        )
        save_boxplots(
            boxplot_entries["voxel"],
            maps_dir / "fpr_voxel_boxplots.png",
            "Voxelwise FPR distributions",
        )


if __name__ == "__main__":
    main()
