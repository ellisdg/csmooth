from __future__ import annotations

import argparse
import glob
import os
import subprocess
from pathlib import Path
from typing import Iterable, List
import re
import yaml
import time
import getpass
import shutil

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


def count_matching_jobs(job_name: str, user: str) -> int:
    """Count queued jobs for `user` whose job name matches `job_name` using Slurm squeue.

    Uses squeue -u <user> -h --name <job_name> and counts non-empty output lines.
    Returns 0 if squeue is not available or on error.
    """
    try:
        out = subprocess.check_output(["squeue", "-u", user, "-h", "--name", job_name], text=True)
        lines = [l for l in out.splitlines() if l.strip()]
        return len(lines)
    except Exception:
        return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Submit FEAT null first-level jobs across smoothed runs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to fpr_config.yaml")
    parser.add_argument(
        "--method",
        choices=["csmooth", "gaussian"],
        nargs="*",
        default=["csmooth", "gaussian"],
        help="Smoothing method(s). By default both 'csmooth' and 'gaussian' are used."
    )
    parser.add_argument("--fwhm", nargs="*", help="Optional list of FWHM values to include")
    parser.add_argument("--subjects", nargs="*", help="Optional subject filter (e.g., sub-001)")
    parser.add_argument("--overwrite", action="store_true", help="Submit even if outputs exist")
    parser.add_argument("--skip-existing", action="store_true", help="Skip submission when output feat exists")
    parser.add_argument("--sbatch-script", help="Path to sbatch wrapper script")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting")
    # Monitoring options
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor job queue and pause submissions when queued jobs with matching job name exceed threshold",
    )
    parser.add_argument(
        "--monitor-threshold",
        type=int,
        default=300,
        help="Number of queued jobs to allow before pausing submissions (default: 300)",
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=60,
        help="Seconds to wait between queue checks when monitoring (default: 60)",
    )
    parser.add_argument(
        "--job-name",
        help=(
            "Job name substring to match when monitoring the queue. If omitted, the sbatch script basename "
            "will be used as the job-name pattern."
        ),
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)
    subjects_glob = config["paths"].get("subjects_glob", "sub-*")
    fwhms = args.fwhm or [str(v) for v in config.get("smoothing", {}).get("fwhm_values", [])]
    if not fwhms:
        raise SystemExit("No FWHM values provided")

    sbatch_script = args.sbatch_script or os.path.join(os.path.dirname(__file__), "run_feat_null_firstlevel.sbatch")
    output_base = config["paths"].get("output_root") or str(PACKAGE_ROOT / "output")

    # Ensure args.method is a list (argparse with nargs='*' always produces a list)
    methods = args.method

    user = getpass.getuser()
    # Determine job name pattern used for monitoring
    job_name_pattern = args.job_name or os.path.splitext(os.path.basename(sbatch_script))[0]

    any_runs_found = False
    for method in methods:
        base_dir = config["paths"].get(f"{method}_dir")
        if base_dir is None:
            print(f"Warning: configuration does not contain path for '{method}_dir'; skipping method {method}")
            continue
        runs = discover_runs(base_dir=base_dir, method=method, fwhms=fwhms, subjects_glob=subjects_glob)
        if args.subjects:
            runs = [r for r in runs if parse_fmri_metadata(r)["subject"] in set(args.subjects)]
        if not runs:
            print(f"No matching smoothed runs found for method '{method}'")
            continue
        any_runs_found = True

        for fmri_path in runs:
            info = parse_fmri_metadata(fmri_path)
            feat_dir = build_feat_dir(output_base, info)
            if args.skip_existing and feat_dir and feat_output_exists(feat_dir):
                continue
            submit_job(sbatch_script, fmri_path, args)

            # If monitoring enabled, wait until the number of queued jobs with matching job name
            # falls below the specified threshold before continuing submissions.
            if args.monitor:
                while True:
                    try:
                        nq = count_matching_jobs(job_name_pattern, user)
                    except Exception:
                        nq = 0
                    if nq < args.monitor_threshold:
                        break
                    print(
                        f"Job queue ({job_name_pattern}) has {nq} jobs for user {user} - sleeping for {args.monitor_interval}s"
                    )
                    time.sleep(args.monitor_interval)

    if not any_runs_found:
        raise SystemExit("No matching smoothed runs found for any requested method")


if __name__ == "__main__":
    main()
