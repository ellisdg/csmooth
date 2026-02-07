import argparse
import glob
import os
import subprocess
from typing import Iterable, List

from .paths import DEFAULT_CONFIG_PATH, load_config, PACKAGE_ROOT
from .run_feat_null_firstlevel import parse_fmri_metadata, feat_output_exists


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


def build_feat_dir(output_base: str, info: dict ) -> str | None:
    return os.path.join(
        output_base,
        "feat",
        info["method"],
        f"fwhm-{info['fwhm']}",
        "fsl",
        info["subject"],
        f"dir-{info['dir']}_run-{info['run']}.feat"
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
    if args.design_id is not None:
        cmd.extend(["--design-id", f"{args.design_id}"])
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
    parser.add_argument("--skip-existing", action="store_true", help="Skip submission when output feat exists for the selected design")
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
