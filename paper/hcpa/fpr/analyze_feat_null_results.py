import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy import ndimage, stats
from .paths import load_config, PACKAGE_ROOT, DEFAULT_CONFIG_PATH


def build_mask(mask_path: str | None, sample_img: str) -> np.ndarray:
    if mask_path:
        mask_img = nib.load(mask_path)
        return mask_img.get_fdata().astype(bool)
    sample = nib.load(sample_img)
    data = sample.get_fdata()
    return np.isfinite(data) & (data != 0)


def find_feat_outputs(output_root: str, method: str, fwhm: int):
    pattern = os.path.join(
        output_root,
        "feat",
        method,
        f"fwhm-{fwhm}",
        "sub-*",
        "dir-*_run-*",
        "design-*.feat",
        "stats",
        "zstat1.nii.gz",
    )
    return sorted(glob.glob(pattern))


def compute_clusters(z_img: str, mask: np.ndarray, cluster_ps: list[float], two_sided: bool):
    data = nib.load(z_img).get_fdata(dtype=np.float32)
    if data.shape != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image {data.shape}")
    results = []
    for p_val in cluster_ps:
        z_thr = stats.norm.isf(p_val / 2) if two_sided else stats.norm.isf(p_val)
        supra = np.abs(data) >= z_thr if two_sided else data >= z_thr
        supra = np.logical_and(supra, mask)
        labeled, n_clusters = ndimage.label(supra)
        sizes = ndimage.sum(supra, labeled, index=range(1, n_clusters + 1)) if n_clusters else []
        max_size = float(np.max(sizes)) if len(sizes) else 0.0
        results.append({
            "cluster_forming_p": p_val,
            "cluster_forming_z": float(z_thr),
            "n_clusters": int(n_clusters),
            "max_cluster_size": max_size,
            "detected": int(n_clusters > 0),
        })
    return results


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Summarize individual FEAT null runs for FPR estimation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to fpr_config.yaml")
    parser.add_argument("--method", choices=["csmooth", "gaussian"], required=True)
    parser.add_argument("--fwhm", type=int, required=True)
    parser.add_argument("--mask", help="Optional mask override")
    parser.add_argument("--output", help="Optional CSV output path")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output_root = config["paths"]["output_root"] or str(PACKAGE_ROOT / "output")
    cluster_ps = config["group"].get("cluster_forming_ps", [0.01, 0.001])
    two_sided = bool(config["group"].get("two_sided", True))

    z_paths = find_feat_outputs(output_root, args.method, args.fwhm)
    if not z_paths:
        raise SystemExit("No FEAT outputs found; run run_feat_null_firstlevel.py first")

    sample_mask_img = z_paths[0]
    mask = build_mask(args.mask or config["paths"].get("mask_path"), sample_mask_img)

    records = []
    for z_path in tqdm(z_paths):
        m = re.search(r"(sub-[^/]+)/dir-([^/]+)_run-([^/]+)/design-(\d+).feat", z_path)
        if not m:
            print(f"Skipping {z_path}; cannot parse metadata")
            continue
        subj, direction, run, design_id = m.group(1), m.group(2), m.group(3), int(m.group(4))
        clusters = compute_clusters(z_path, mask, cluster_ps, two_sided)
        for c in clusters:
            records.append({
                "subject": subj,
                "dir": direction,
                "run": run,
                "design_id": design_id,
                "method": args.method,
                "fwhm": args.fwhm,
                **c,
            })

    out_csv = args.output or os.path.join(output_root, "feat", args.method, f"fwhm-{args.fwhm}", "feat_null_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)
    print(f"Wrote {len(records)} rows to {out_csv}")


if __name__ == "__main__":
    main()
