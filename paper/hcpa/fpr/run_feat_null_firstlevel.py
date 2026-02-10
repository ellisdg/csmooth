from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from typing import Optional
from pathlib import Path
import yaml

# Embed minimal config loader so this script can run standalone (copied from paths.py)
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "fpr_config.yaml"


def resolve_rel_path(value: Optional[str], base_dir: Path) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def load_config(config_path: Optional[str | Path] = None) -> dict:
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

PLACEHOLDERS = {
    "fmri_filename": "fmri_filename",
    "output_directory": "output_directory",
    "smoothing_fwhm": "smoothing_fwhm",
    "total_volumes": "total_volumes",
    "total_voxels": "total_voxels",
    "brain_mask_filename": "brain_mask_filename"
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


def _smoothed_to_clean_basename(basename: str) -> str:
    """Convert a smoothed filename basename to the cleaned ('desc-preproc') JSON basename.

    Examples:
    sub-..._desc-csmooth_fwhm-12_bold.nii.gz -> sub-..._desc-preproc_bold.json
    sub-..._desc-gaussian_fwhm-6_bold.nii.gz -> sub-..._desc-preproc_bold.json
    """
    # Replace the desc and optional fwhm part up to _bold.nii.gz with desc-preproc_bold.json
    new = re.sub(r"_desc-[^_]+(?:_fwhm-[0-9]+(?:\\.[0-9]+)?)?_bold\.nii\.gz$", "_desc-preproc_bold.json", basename)
    # If substitution didn't match the expected pattern, just replace extension
    if new == basename:
        new = re.sub(r"\.nii\.gz$", ".json", basename)
    return new


def find_clean_json(fmri_path: str, config: dict, method: str) -> str:
    # First try a .json next to the smoothed file (same dir)
    candidates = [replace_ext_nii_json(fmri_path)]
    base_dir = config["paths"].get(f"{method}_dir")
    cleaned_dir = config["paths"].get("cleaned_dir")
    # If the smoothed file is under the expected method directory, map it to the cleaned_dir
    if cleaned_dir and base_dir and os.path.commonpath([os.path.abspath(fmri_path), os.path.abspath(base_dir)]) == os.path.abspath(base_dir):
        rel = os.path.relpath(fmri_path, base_dir)
        rel_dir = os.path.dirname(rel)
        basename = os.path.basename(fmri_path)
        clean_basename = _smoothed_to_clean_basename(basename)
        candidates.append(os.path.join(cleaned_dir, rel_dir, clean_basename))
    # Also try mapping by replacing just the method-specific tokens in the full path
    if cleaned_dir:
        # Replace method directory segment with cleaned_dir and convert basename
        try:
            # If base_dir is a prefix, build a path using cleaned_dir
            rel = os.path.relpath(fmri_path, base_dir) if base_dir else None
            if rel:
                clean_path = os.path.join(cleaned_dir, os.path.dirname(rel), _smoothed_to_clean_basename(os.path.basename(fmri_path)))
                if clean_path not in candidates:
                    candidates.append(clean_path)
        except Exception:
            pass
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


def read_brain_mask_filename_from_json(clean_json_path: str) -> Optional[str]:
    with open(clean_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("BrainMaskFilename")


def write_fsf(template_text: str, fmri_path: str, output_dir: str, total_volumes: int, fsf_path: str,
              brain_mask_filename: str) -> None:
    os.makedirs(os.path.dirname(fsf_path), exist_ok=True)
    text = template_text
    text = text.replace(PLACEHOLDERS["fmri_filename"], fmri_path)
    text = text.replace(PLACEHOLDERS["output_directory"], output_dir)
    text = text.replace(PLACEHOLDERS["smoothing_fwhm"], str(0))
    text = text.replace(PLACEHOLDERS["total_volumes"], str(total_volumes))
    text = text.replace(PLACEHOLDERS["total_voxels"], str(total_volumes * VOXELS_PER_VOLUME))
    text = text.replace(PLACEHOLDERS["brain_mask_filename"], brain_mask_filename)
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
    output_root = os.path.join(output_base, "feat", info["method"], f"fwhm-{info['fwhm']}")

    template_text = read_template(template_path)

    clean_json = find_clean_json(fmri_path, config, info["method"])
    total_volumes = read_scrubbed_volumes(clean_json)
    fmriprep_brain_mask_filename = read_brain_mask_filename_from_json(clean_json)
    # we're actually going to ignore the fmriprep brainmask

    if info["method"] == "csmooth":
        brain_mask_filename = "/data2/david.ellis/public/HCPA/code/fpr/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
    elif info["method"] == "gaussian":
        brain_mask_filename = "/data2/david.ellis/public/HCPA/code/fpr/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask_resampled.nii.gz"
    else:
        raise ValueError(f"Unknown method {info['method']} for brain mask selection")
    print(f"Running FEAT for {fmri_path}")
    print(f"Brain mask: {brain_mask_filename}")
    print(f"Total volumes after scrubbing: {total_volumes}")
    print(f"Clean JSON: {clean_json}")
    print(f"Output FEAT directory: {output_root}")

    feat_dir = os.path.join(
        output_root,
        "fsl",
        info["subject"],
        f"dir-{info['dir']}_run-{info['run']}.feat",
    )
    fsf_path = feat_dir.replace(".feat", ".fsf")
    if feat_output_exists(feat_dir) and not args.overwrite:
        print(f"FEAT output exists at {feat_dir}, skipping (use --overwrite to force)")
        return
    write_fsf(
        template_text=template_text,
        fmri_path=fmri_path,
        output_dir=feat_dir,
        total_volumes=total_volumes,
        fsf_path=fsf_path,
        brain_mask_filename=brain_mask_filename
    )
    run_feat(fsf_path, container=container, log_dir=log_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
