import argparse
import glob
import json
import os
import re
import subprocess
from typing import Optional

from .paths import load_config, PACKAGE_ROOT, DEFAULT_CONFIG_PATH

PLACEHOLDERS = {
    "fmri_filename": "fmri_filename",
    "output_directory": "output_directory",
    "smoothing_fwhm": "smoothing_fwhm",
    "total_volumes": "total_volumes",
    "total_voxels": "total_voxels"
}

VOXELS_PER_VOLUME = 1082035


def parse_fmri_metadata(path: str) -> dict:
    basename = os.path.basename(path)
    pattern = re.compile(
        r"(?P<subject>sub-[^_]+)_task-rest_dir-(?P<dir>[^_]+)_run-(?P<run>[^_]+)_.*desc-(?P<method>[^_]+)_fwhm-(?P<fwhm>[0-9]+(?:\\.[0-9]+)?)"
    )
    match = pattern.search(basename)
    if not match:
        raise ValueError(f"Cannot parse metadata from {basename}")
    return match.groupdict()


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
        info = parse_fmri_metadata(path)
        if subjects_filter and info["subject"] not in subjects_filter:
            continue
        yield path, info


def read_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def replace_ext_nii_json(path: str) -> str:
    return re.sub(r"\.nii\.gz$", ".json", path)


def find_clean_json(fmri_path: str, config: dict, method: str) -> str:
    candidates = [replace_ext_nii_json(fmri_path)]
    base_dir = config["paths"].get(f"{method}_dir")
    cleaned_dir = config["paths"].get("cleaned_dir")
    if cleaned_dir and base_dir and os.path.commonpath([os.path.abspath(fmri_path), os.path.abspath(base_dir)]) == os.path.abspath(base_dir):
        rel = os.path.relpath(fmri_path, base_dir)
        candidates.append(replace_ext_nii_json(os.path.join(cleaned_dir, rel)))
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Could not find cleaned JSON for {fmri_path}; tried: {candidates}")


def read_scrubbed_volumes(clean_json_path: str) -> int:
    with open(clean_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    try:
        return int(meta["NumberOfVolumesAfterScrubbing"])
    except KeyError as exc:
        raise KeyError(f"NumberOfVolumesAfterScrubbing missing in {clean_json_path}") from exc


def write_fsf(template_text: str, fmri_path: str, output_dir: str, total_volumes: int, fsf_path: str) -> None:
    os.makedirs(os.path.dirname(fsf_path), exist_ok=True)
    text = template_text
    text = text.replace(PLACEHOLDERS["fmri_filename"], fmri_path)
    text = text.replace(PLACEHOLDERS["output_directory"], output_dir)
    text = text.replace(PLACEHOLDERS["smoothing_fwhm"], str(0))
    text = text.replace(PLACEHOLDERS["total_volumes"], str(total_volumes))
    text = text.replace(PLACEHOLDERS["total_voxels"], str(total_volumes * VOXELS_PER_VOLUME))
    with open(fsf_path, "w", encoding="utf-8") as f:
        f.write(text)


def feat_output_exists(feat_dir: str) -> bool:
    return os.path.exists(os.path.join(feat_dir, "stats", "zstat1.nii.gz"))


def run_feat(fsf_path: str, container: str, log_dir: Optional[str], dry_run: bool) -> None:
    os.makedirs(log_dir or "logs", exist_ok=True)
    if container:
        cmd = [
            "apptainer",
            "exec",
            "-B",
            "/data",
            "-B",
            "/data2",
            container,
            "feat",
            fsf_path,
        ]
    else:
        cmd = ["feat", fsf_path]
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Generate and run FEAT first-level models for a single smoothed run.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to fpr_config.yaml")
    parser.add_argument("--fmri-filename", required=True, help="Path to the smoothed fMRI NIfTI")
    parser.add_argument("--design-id", type=int, help="Optional design id to select design-XYZ.tsv")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing FEAT outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running FEAT")
    args = parser.parse_args(argv)

    fmri_path = os.path.abspath(args.fmri_filename)
    info = parse_fmri_metadata(fmri_path)

    config = load_config(args.config)
    template_path = config["feat"]["fsf_template"]
    container = config["feat"]["container"]
    log_dir = config["feat"].get("log_dir")
    output_base = config["paths"].get("output_root") or str(PACKAGE_ROOT / "output")
    design_root = os.path.join(output_base, "designs")
    output_root = os.path.join(output_base, "feat", info["method"], f"fwhm-{info['fwhm']}")

    template_text = read_template(template_path)
    run_design_dir = os.path.join(design_root, info["subject"], f"dir-{info['dir']}_run-{info['run']}")
    if not os.path.isdir(run_design_dir):
        raise SystemExit(f"No designs for {fmri_path}")

    designs = sorted(glob.glob(os.path.join(run_design_dir, "design-*.tsv")))
    if args.design_id is not None:
        designs = [d for d in designs if f"design-{args.design_id:03d}" in d]
    if not designs:
        raise SystemExit(f"No design files found in {run_design_dir}")

    clean_json = find_clean_json(fmri_path, config, info["method"])
    total_volumes = read_scrubbed_volumes(clean_json)

    for design_tsv in designs:
        design_id = os.path.basename(design_tsv).split("-")[1].split(".")[0]
        feat_dir = os.path.join(
            output_root,
            "fsl",
            info["subject"],
            f"dir-{info['dir']}_run-{info['run']}",
            f"design-{design_id}.feat",
        )
        fsf_path = feat_dir.replace(".feat", ".fsf")
        if feat_output_exists(feat_dir) and not args.overwrite:
            continue
        write_fsf(
            template_text=template_text,
            fmri_path=fmri_path,
            output_dir=feat_dir,
            total_volumes=total_volumes,
            fsf_path=fsf_path,
        )
        run_feat(fsf_path, container=container, log_dir=log_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
