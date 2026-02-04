import argparse
import glob
import os
import re
import shutil
import subprocess
import yaml
from typing import Optional


PLACEHOLDERS = {
    "fmri_filename": "fmri_filename",
    "output_directory": "output_directory",
    "smoothing_fwhm": "smoothing_fwhm",
    "events_txt": "events_txt",
}


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


def read_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_fsf(template_text: str, fmri_path: str, events_path: str, output_dir: str, smoothing: float, fsf_path: str) -> None:
    os.makedirs(os.path.dirname(fsf_path), exist_ok=True)
    text = template_text
    text = text.replace(PLACEHOLDERS["fmri_filename"], fmri_path)
    text = text.replace(PLACEHOLDERS["output_directory"], output_dir)
    text = text.replace(PLACEHOLDERS["smoothing_fwhm"], str(smoothing))
    text = text.replace(PLACEHOLDERS["events_txt"], events_path)
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


def main():
    parser = argparse.ArgumentParser(description="Generate and run FEAT first-level models for null designs.")
    parser.add_argument("--config", required=True, help="Path to fpr_config.yaml")
    parser.add_argument("--method", choices=["csmooth", "gaussian"], required=True)
    parser.add_argument("--fwhm", type=int, required=True)
    parser.add_argument("--design-id", type=int, help="Restrict to a single design id")
    parser.add_argument("--subjects", nargs="*", help="Optional subject filter (e.g., sub-001)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing FEAT outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running FEAT")
    args = parser.parse_args()

    config = load_config(args.config)
    template_path = config["feat"]["fsf_template"]
    container = config["feat"]["container"]
    smoothing = config["feat"].get("smoothing_fwhm", 0)
    log_dir = config["feat"].get("log_dir")
    design_root = os.path.join(config["paths"]["output_root"], "designs")
    output_root = os.path.join(config["paths"]["output_root"], "feat", args.method, f"fwhm-{args.fwhm}")

    template_text = read_template(template_path)
    subjects_filter = set(args.subjects) if args.subjects else None

    runs = list(iter_smoothed_runs(args.method, args.fwhm, config, subjects_filter))
    if not runs:
        raise SystemExit("No smoothed runs found for given parameters")

    for fmri_path, info in runs:
        run_design_dir = os.path.join(design_root, info["subject"], f"dir-{info['dir']}_run-{info['run']}")
        if not os.path.isdir(run_design_dir):
            print(f"No designs for {fmri_path}, skipping")
            continue
        designs = sorted(glob.glob(os.path.join(run_design_dir, "design-*.tsv")))
        if args.design_id is not None:
            designs = [d for d in designs if f"design-{args.design_id:03d}" in d]
        for design_tsv in designs:
            design_id = os.path.basename(design_tsv).split("-")[1].split(".")[0]
            feat_dir = os.path.join(
                output_root,
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
                events_path=design_tsv,
                output_dir=feat_dir,
                smoothing=smoothing,
                fsf_path=fsf_path,
            )
            run_feat(fsf_path, container=container, log_dir=log_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
