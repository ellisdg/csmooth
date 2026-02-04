import argparse
import glob
import json
import os
import re
import yaml
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy import ndimage, stats


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gather_first_level_maps(output_root: str, method: str, fwhm: int, design_filter: set | None):
    pattern = os.path.join(
        output_root,
        "first_level",
        method,
        f"fwhm-{fwhm}",
        "sub-*",
        "dir-*_run-*",
        "design-*_zmap.nii.gz",
    )
    mapping = {}
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        design_match = re.search(r"design-(\d+)_zmap.nii.gz", basename)
        if not design_match:
            continue
        design_id = int(design_match.group(1))
        if design_filter and design_id not in design_filter:
            continue
        subj = path.split(os.sep)[-3]
        mapping.setdefault(subj, []).append((design_id, path))
    return mapping


def build_mask(mask_path: str | None, sample_img: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mask_path:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)
        return mask_data, mask_img.affine, mask_img.header
    sample = nib.load(sample_img)
    data = sample.get_fdata()
    mask_data = np.isfinite(data) & (data != 0)
    return mask_data, sample.affine, sample.header


def compute_group_z(data_matrix: np.ndarray) -> np.ndarray:
    n_subj = data_matrix.shape[1]
    mean = data_matrix.mean(axis=1)
    std = data_matrix.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = np.where(std > 0, mean / (std / np.sqrt(n_subj)), 0.0)
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=n_subj - 1)
    z_vals = stats.norm.isf(p_vals / 2)
    z_vals = np.nan_to_num(z_vals, nan=0.0, posinf=0.0, neginf=0.0)
    return z_vals.astype(np.float32)


def max_cluster(z_vol: np.ndarray, mask: np.ndarray, z_thr: float, two_sided: bool) -> tuple[float, int]:
    supra = np.abs(z_vol) >= z_thr if two_sided else z_vol >= z_thr
    supra = np.logical_and(supra, mask)
    labeled, n_clusters = ndimage.label(supra)
    if n_clusters == 0:
        return 0.0, 0
    sizes = ndimage.sum(supra, labeled, index=range(1, n_clusters + 1))
    max_size = float(np.max(sizes)) if sizes.size else 0.0
    return max_size, int(n_clusters)


def perm_cluster_threshold(data_matrix: np.ndarray, mask: np.ndarray, z_thr: float, two_sided: bool, n_perm: int, alpha: float, rng: np.random.Generator) -> float:
    if n_perm <= 0:
        return 0.0
    n_subj = data_matrix.shape[1]
    max_sizes = []
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n_subj)
        perm_data = data_matrix * signs
        z_perm = compute_group_z(perm_data)
        z_vol = np.zeros(mask.shape, dtype=np.float32)
        z_vol[mask] = z_perm
        max_size, _ = max_cluster(z_vol, mask, z_thr, two_sided)
        max_sizes.append(max_size)
    return float(np.percentile(max_sizes, 100 * (1 - alpha))) if max_sizes else 0.0


def load_vectorized(path: str, mask_idx: np.ndarray, cache: dict) -> np.ndarray:
    if path in cache:
        return cache[path]
    data = nib.load(path).get_fdata(dtype=np.float32)
    vec = data[mask_idx]
    cache[path] = vec
    if len(cache) > 256:
        cache.pop(next(iter(cache)))
    return vec


def main():
    parser = argparse.ArgumentParser(description="Permutation-based group null test to estimate FPR.")
    parser.add_argument("--config", required=True, help="Path to fpr_config.yaml")
    parser.add_argument("--method", choices=["csmooth", "gaussian"], required=True, help="Smoothing method")
    parser.add_argument("--fwhm", type=int, required=True, help="FWHM to analyze")
    parser.add_argument("--design-id", type=int, help="Restrict to a specific design id")
    parser.add_argument("--group-size", type=int, help="Override group size")
    parser.add_argument("--n-groups", type=int, help="Override number of group draws")
    parser.add_argument("--cluster-ps", type=float, nargs="*", help="Override cluster-forming p-values")
    parser.add_argument("--n-perm", type=int, help="Override number of permutations")
    parser.add_argument("--mask", help="Optional mask override")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--output-root", help="Override output root")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = args.output_root or config["paths"]["output_root"]
    design_filter = {args.design_id} if args.design_id is not None else None
    group_size = args.group_size or config["group"].get("group_size", 20)
    n_groups = args.n_groups or config["group"].get("n_groups", 500)
    cluster_ps = args.cluster_ps or config["group"].get("cluster_forming_ps", [0.01, 0.001])
    cluster_fwer = config["group"].get("cluster_fwer", 0.05)
    n_perm = args.n_perm if args.n_perm is not None else config["group"].get("n_perm", 500)
    two_sided = bool(config["group"].get("two_sided", True))
    rng = np.random.default_rng(args.seed or config["group"].get("seed", 2025))

    mapping = gather_first_level_maps(output_root, args.method, args.fwhm, design_filter)
    subjects = sorted(mapping.keys())
    if len(subjects) < group_size:
        raise SystemExit(f"Not enough subjects ({len(subjects)}) for group size {group_size}")

    sample_img = mapping[subjects[0]][0][1]
    mask_data, affine, header = build_mask(args.mask or config["paths"].get("mask_path"), sample_img)
    mask_idx = mask_data.astype(bool)

    records = []
    cache: dict[str, np.ndarray] = {}

    for iter_idx in tqdm(range(n_groups)):
        chosen_subj = rng.choice(subjects, size=group_size, replace=False)
        chosen_files = []
        design_ids = []
        for subj in chosen_subj:
            designs = mapping[subj]
            design_id, path = designs[rng.integers(0, len(designs))]
            chosen_files.append(path)
            design_ids.append(design_id)

        data_matrix = np.column_stack([load_vectorized(p, mask_idx, cache) for p in chosen_files])
        z_vals = compute_group_z(data_matrix)
        z_vol = np.zeros(mask_data.shape, dtype=np.float32)
        z_vol[mask_idx] = z_vals

        for p_val in cluster_ps:
            z_thr = stats.norm.isf(p_val / 2) if two_sided else stats.norm.isf(p_val)
            max_size, n_clusters = max_cluster(z_vol, mask_data, z_thr, two_sided)
            perm_thr = perm_cluster_threshold(data_matrix, mask_data, z_thr, two_sided, n_perm, cluster_fwer, rng)
            detected = max_size >= max(perm_thr, 0.0) if n_perm > 0 else n_clusters > 0
            records.append({
                "iteration": iter_idx,
                "method": args.method,
                "fwhm": args.fwhm,
                "group_size": group_size,
                "cluster_forming_p": p_val,
                "cluster_forming_z": z_thr,
                "cluster_fwer": cluster_fwer,
                "n_perm": n_perm,
                "two_sided": two_sided,
                "max_cluster_size": max_size,
                "perm_size_threshold": perm_thr,
                "n_clusters": n_clusters,
                "detected": int(detected),
                "subjects": ";".join(chosen_subj),
                "design_ids": ";".join(str(d) for d in design_ids),
            })

    out_dir = os.path.join(output_root, "group_stats", args.method, f"fwhm-{args.fwhm}")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "group_null_results.csv")
    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)
    print(f"Wrote {len(records)} rows to {out_csv}")


if __name__ == "__main__":
    main()
