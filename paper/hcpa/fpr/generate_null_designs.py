import argparse
import os
import re
import glob
import json
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from .paths import load_config, PACKAGE_ROOT, DEFAULT_CONFIG_PATH


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
    min_dur = config["designs"].get("min_duration", 10.0)
    max_dur = config["designs"].get("max_duration", 30.0)

    run_out_dir = os.path.join(
        output_dir,
        info["subject"],
        f"dir-{info['dir']}_run-{info['run']}",
    )
    os.makedirs(run_out_dir, exist_ok=True)

    base_seed = seed + hash((info["subject"], info["dir"], info["run"])) % 10_000
    rng = np.random.default_rng(base_seed)

    for design_idx in range(n_designs):
        onset = 0.0
        is_stim = True
        events = []
        while onset < duration_sec:
            block_len = float(rng.uniform(min_dur, max_dur))
            block_len = min(block_len, max(0.0, duration_sec - onset))
            if block_len <= 0:
                break
            if is_stim:
                if block_len < min_dur and events:
                    new_dur = events[-1]["duration"] + block_len
                    events[-1]["duration"] = round(min(new_dur, max_dur), 3)
                else:
                    events.append({
                        "onset": round(onset, 3),
                        "duration": round(block_len, 3),
                        "trial_type": "stim",
                        "amplitude": 1.0,
                    })
            onset += block_len
            is_stim = not is_stim
        df = pd.DataFrame(events)
        out_path = os.path.join(run_out_dir, f"design-{design_idx:03d}.tsv")
        df.to_csv(out_path, sep="\t", index=False)
        meta = {
            "tr": tr,
            "n_volumes": int(n_vols),
            "duration_sec": float(duration_sec),
            "n_events": int(len(events)),
            "seed": int(base_seed + design_idx),
            "min_duration": float(min_dur),
            "max_duration": float(max_dur),
        }
        with open(out_path.replace(".tsv", ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Generate random null designs per resting run (Eklund-style).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to fpr_config.yaml")
    parser.add_argument("--output-dir", help="Override output directory for designs")
    parser.add_argument("--n-designs", type=int, help="Override number of designs per run")
    parser.add_argument("--subjects", nargs="*", help="Optional list of subject IDs (e.g., sub-001) to include")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    cleaned_dir = config["paths"]["cleaned_dir"]
    subjects_glob = config["paths"].get("subjects_glob", "sub-*")
    output_dir = args.output_dir or os.path.join(config["paths"]["output_root"] or str(PACKAGE_ROOT / "output"), "designs")
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
