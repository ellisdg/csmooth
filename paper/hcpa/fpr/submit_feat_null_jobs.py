from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List
import re
import json
import yaml

# Embed minimal config loader so this script can run standalone (copied from paths.py)
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "fpr_config.yaml"


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


def parse_fmri_metadata(path: str) -> dict:
    """Extract subject, dir, run, method, fwhm from a smoothed filename.

    Example basename:
    sub-HCA6010538_task-rest_dir-AP_run-1_space-MNI152NLin2009cAsym_desc-csmooth_fwhm-6_bold.nii.gz
    """
    basename = os.path.basename(path)
    pattern = re.compile(
        r"(?P<subject>sub-[^_]+)_task-rest_dir-(?P<dir>[^_]+)_run-(?P<run>[^_]+)_.*desc-(?P<method>[^_]+)_fwhm-(?P<fwhm>[0-9]+(?:\\.[0-9]+)?)"
    )
    m = pattern.search(basename)
    if not m:
        raise ValueError(f"Cannot parse run info from {basename}")
    return m.groupdict()


def feat_output_exists(feat_dir: str) -> bool:
    return os.path.exists(os.path.join(feat_dir, "stats", "zstat1.nii.gz"))


def discover_runs(base_dir: str, method: str, fwhms: Iterable[str], subjects_glob: str) -> List[str]:
    runs: List[str] = []
    for fwhm in fwhms:
        pattern = os.path.join(
            base_dir,
            subjects_glob,
            "func",
            f"*_task-rest*_space-MNI152NLin2009cAsym_desc-{method}_fwhm-{fwhm}_bold.nii.gz",
        )
        runs.extend(glob.glob(pattern))
    return sorted(set(runs))


def build_feat_dir(output_base: str, info: dict) -> str | None:
    return os.path.join(
        output_base,
        "feat",
        info["method"],
        f"fwhm-{info['fwhm']}",
        "fsl",
        info["subject"],
        f"dir-{info['dir']}_run-{info['run']}.feat",
    )


def submit_job(sbatch_script: str, fmri_path: str, args: argparse.Namespace) -> None:
    cmd = [
        "sbatch",
        sbatch_script,
        "--config",
        args.config,
        "--fmri-filename",
        fmri_path,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Submit FEAT null first-level jobs across smoothed runs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to fpr_config.yaml")
    parser.add_argument("--method", choices=["csmooth", "gaussian"], default="csmooth", help="Smoothing method")
    parser.add_argument("--fwhm", nargs="*", help="Optional list of FWHM values to include")
    parser.add_argument("--subjects", nargs="*", help="Optional subject filter (e.g., sub-001)")
    parser.add_argument("--overwrite", action="store_true", help="Submit even if outputs exist")
    parser.add_argument("--skip-existing", action="store_true", help="Skip submission when output feat exists")
    parser.add_argument("--sbatch-script", help="Path to sbatch wrapper script")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    base_dir = config["paths"][f"{args.method}_dir"]
    subjects_glob = config["paths"].get("subjects_glob", "sub-*")
    fwhms = args.fwhm or [str(v) for v in config.get("smoothing", {}).get("fwhm_values", [])]
    if not fwhms:
        raise SystemExit("No FWHM values provided")

    sbatch_script = args.sbatch_script or os.path.join(os.path.dirname(__file__), "run_feat_null_firstlevel.sbatch")
    output_base = config["paths"].get("output_root") or str(PACKAGE_ROOT / "output")

    runs = discover_runs(base_dir=base_dir, method=args.method, fwhms=fwhms, subjects_glob=subjects_glob)
    if args.subjects:
        runs = [r for r in runs if parse_fmri_metadata(r)["subject"] in set(args.subjects)]
    if not runs:
        raise SystemExit("No matching smoothed runs found")

    for fmri_path in runs:
        info = parse_fmri_metadata(fmri_path)
        feat_dir = build_feat_dir(output_base, info)
        if args.skip_existing and feat_dir and feat_output_exists(feat_dir):
            continue
        submit_job(sbatch_script, fmri_path, args)


if __name__ == "__main__":
    main()
