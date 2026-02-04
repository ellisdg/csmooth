import argparse
import math
import os
import re
import glob
import yaml
import json
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm


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


def load_metadata(json_path: str) -> dict:
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_designs_for_run(img_path: str, config: dict, output_dir: str, n_designs: int, seed: int) -> None:
    info = parse_run_info(img_path)
    metadata = load_metadata(img_path.replace(".nii.gz", ".json"))
    img = nib.load(img_path)
    tr = metadata.get("RepetitionTime", img.header.get_zooms()[3])
    n_vols = metadata.get("NumberOfVolumesAfterScrubbing", img.shape[-1])
    duration_sec = tr * n_vols
    events_per_min = config["designs"].get("events_per_minute", 1.5)
    min_dur = config["designs"].get("min_duration", 1.0)
    max_dur = config["designs"].get("max_duration", 3.0)
    jitter_low, jitter_high = config["designs"].get("jitter_range_sec", [0.0, 0.0])
    n_events = max(1, int(math.ceil(events_per_min * duration_sec / 60.0)))

    run_out_dir = os.path.join(
        output_dir,
        info["subject"],
        f"dir-{info['dir']}_run-{info['run']}",
    )
    os.makedirs(run_out_dir, exist_ok=True)

    base_seed = seed + hash((info["subject"], info["dir"], info["run"])) % 10_000
    rng = np.random.default_rng(base_seed)

    for design_idx in range(n_designs):
        onset_max = max(0.0, duration_sec - max_dur)
        onsets = rng.uniform(0.0, onset_max, size=n_events)
        durations = rng.uniform(min_dur, max_dur, size=n_events)
        jitters = rng.uniform(jitter_low, jitter_high, size=n_events)
        onsets = np.maximum(0.0, onsets + jitters)
        df = pd.DataFrame({
            "onset": np.round(onsets, 3),
            "duration": np.round(durations, 3),
            "trial_type": "stim",
            "amplitude": 1.0,
        })
        df.sort_values("onset", inplace=True)
        out_path = os.path.join(run_out_dir, f"design-{design_idx:03d}.tsv")
        df.to_csv(out_path, sep="\t", index=False)
        meta = {
            "tr": tr,
            "n_volumes": int(n_vols),
            "duration_sec": float(duration_sec),
            "n_events": int(n_events),
            "seed": int(base_seed + design_idx),
        }
        with open(out_path.replace(".tsv", ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate random null designs per resting run (Eklund-style).")
    parser.add_argument("--config", required=True, help="Path to fpr_config.yaml")
    parser.add_argument("--output-dir", help="Override output directory for designs")
    parser.add_argument("--n-designs", type=int, help="Override number of designs per run")
    parser.add_argument("--subjects", nargs="*", help="Optional list of subject IDs (e.g., sub-001) to include")
    args = parser.parse_args()

    config = load_config(args.config)
    cleaned_dir = config["paths"]["cleaned_dir"]
    subjects_glob = config["paths"].get("subjects_glob", "sub-*")
    output_dir = args.output_dir or os.path.join(config["paths"]["output_root"], "designs")
    n_designs = args.n_designs or config["designs"].get("n_designs_per_run", 25)
    seed = config["designs"].get("seed", 1337)

    pattern = os.path.join(
        cleaned_dir,
        subjects_glob,
        "func",
        "*_task-rest*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    run_paths = sorted(glob.glob(pattern))
    if args.subjects:
        keep = set(args.subjects)
        run_paths = [p for p in run_paths if parse_run_info(p)["subject"] in keep]

    print(f"Found {len(run_paths)} runs to process. Writing designs to {output_dir}")
    for path in tqdm(run_paths):
        try:
            compute_designs_for_run(path, config, output_dir, n_designs, seed)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed for {path}: {exc}")


if __name__ == "__main__":
    main()
