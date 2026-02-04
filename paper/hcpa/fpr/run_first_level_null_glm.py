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
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_run_info(path: str) -> dict:
    basename = os.path.basename(path)
    match = re.search(r"(sub-[^_]+)_task-rest_dir-([^_]+)_run-([^_]+)_", basename)
    if not match:
        raise ValueError(f"Cannot parse run info from {basename}")
    return {
        "subject": match.group(1),
        "dir": match.group(2),
        "run": match.group(3),
    }


def iter_smoothed_runs(method: str, fwhm: int, config: dict, subjects_filter=None):
    base_dir = config["paths"][f"{method}_dir"]
    subjects_glob = config["paths"].get("subjects_glob", "sub-*")
    pattern = os.path.join(
        base_dir,
        subjects_glob,
        "func",
        f"*_task-rest*_space-MNI152NLin2009cAsym_desc-{method}_fwhm-{fwhm}_bold.nii.gz",
    )
    for path in sorted(glob.glob(pattern)):
        info = parse_run_info(path)
        if subjects_filter and info["subject"] not in subjects_filter:
            continue
        yield path, info


def design_paths(design_root: str, info: dict, design_id: int | None):
    run_dir = os.path.join(design_root, info["subject"], f"dir-{info['dir']}_run-{info['run']}")
    if design_id is None:
        return sorted(glob.glob(os.path.join(run_dir, "design-*.tsv")))
    return [os.path.join(run_dir, f"design-{design_id:03d}.tsv")]


def load_tr(img_path: str) -> float:
    meta_path = img_path.replace(".nii.gz", ".json")
    tr = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        tr = metadata.get("RepetitionTime")
    if tr is None:
        tr = nib.load(img_path).header.get_zooms()[3]
    return float(tr)


def fit_and_save(fmri_path: str, design_tsv: str, info: dict, config: dict, mask_path: str | None, output_root: str, overwrite: bool) -> None:
    design_idx = int(os.path.basename(design_tsv).split("-")[1].split(".")[0])
    run_out_dir = os.path.join(
        output_root,
        "first_level",
        info["method"],
        f"fwhm-{info['fwhm']}",
        info["subject"],
        f"dir-{info['dir']}_run-{info['run']}",
    )
    os.makedirs(run_out_dir, exist_ok=True)
    z_path = os.path.join(run_out_dir, f"design-{design_idx:03d}_zmap.nii.gz")
    eff_path = os.path.join(run_out_dir, f"design-{design_idx:03d}_effect.nii.gz")
    if not overwrite and os.path.exists(z_path) and os.path.exists(eff_path):
        return

    tr = load_tr(fmri_path)
    img = nib.load(fmri_path)
    n_scans = img.shape[-1]
    frame_times = np.arange(n_scans) * tr
    events = pd.read_csv(design_tsv, sep="\t")
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model=config["glm"].get("hrf_model", "glover"),
        drift_model=config["glm"].get("drift_model", "cosine"),
        high_pass=config["glm"].get("high_pass", 0.008),
        oversampling=config["glm"].get("oversampling", 50),
    )

    flm = FirstLevelModel(
        t_r=tr,
        mask_img=mask_path,
        smoothing_fwhm=None,
        hrf_model=config["glm"].get("hrf_model", "glover"),
        drift_model=config["glm"].get("drift_model", "cosine"),
        high_pass=config["glm"].get("high_pass", 0.008),
        noise_model="ar1",
        standardize=False,
        signal_scaling=False,
        minimize_memory=True,
    )
    flm = flm.fit(fmri_path, events)
    z_map = flm.compute_contrast("stim", output_type="z_score")
    eff_map = flm.compute_contrast("stim", output_type="effect_size")
    z_map.to_filename(z_path)
    eff_map.to_filename(eff_path)

    meta = {
        "subject": info["subject"],
        "dir": info["dir"],
        "run": info["run"],
        "method": info["method"],
        "fwhm": info["fwhm"],
        "design_id": design_idx,
        "tr": tr,
        "n_scans": n_scans,
        "fmri_path": fmri_path,
        "design_tsv": design_tsv,
        "mask_path": mask_path,
    }
    with open(os.path.join(run_out_dir, f"design-{design_idx:03d}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run first-level null GLMs on smoothed resting fMRI.")
    parser.add_argument("--config", required=True, help="Path to fpr_config.yaml")
    parser.add_argument("--method", choices=["csmooth", "gaussian"], required=True, help="Smoothing method")
    parser.add_argument("--fwhm", type=int, required=True, help="FWHM to process")
    parser.add_argument("--design-id", type=int, help="Optional design id (process only this design)")
    parser.add_argument("--subjects", nargs="*", help="Optional subject filter (e.g., sub-001)")
    parser.add_argument("--mask", help="Optional mask override")
    parser.add_argument("--output-root", help="Override output root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    mask_path = args.mask or config["paths"].get("mask_path")
    output_root = args.output_root or config["paths"]["output_root"]

    subjects_filter = set(args.subjects) if args.subjects else None

    runs = list(iter_smoothed_runs(args.method, args.fwhm, config, subjects_filter))
    if not runs:
        raise SystemExit("No smoothed runs found for given parameters")

    for fmri_path, info in tqdm(runs):
        info["method"] = args.method
        info["fwhm"] = args.fwhm
        for design_tsv in design_paths(os.path.join(output_root, "designs"), info, args.design_id):
            if not os.path.exists(design_tsv):
                print(f"Missing design file {design_tsv}, skipping")
                continue
            try:
                fit_and_save(fmri_path, design_tsv, info, config, mask_path, output_root, args.overwrite)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed for {fmri_path} with design {design_tsv}: {exc}")


if __name__ == "__main__":
    main()
