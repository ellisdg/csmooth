"""
This script takes the cortical surfaces from an example subject and builds the graph representations at
different voxel sizes for that subject.
fMRI data is then simulated such that there is a single active cortical region, but there is a significant amount
of white noise.
The data is then smoothed using the csmooth algorithm at each resolution without resampling,
and the results are compared to see how the grid resolution affects the smoothing results.
The accuracy of the smoothing is evaluated based on how well the smoothed data recovers the original active region
as measured by metrics such as sensitivity, specificity, and Dice coefficient.
The regression coefficients are estimated using OLS, as the focus is on the effect of grid resolution rather than
the estimation method, and the only noise present is white noise.
The smallest voxel size is 0.5mm isotropic, and the largest is 4mm isotropic, with increments of 0.5mm.
"""

# paper/simulations/grid_resolution_effect.py
import os
import csv
import numpy as np

from csmooth.smooth import smooth_images, save_labelmap
from csmooth.graph import create_graph
from csmooth.components import identify_connected_components


def _frange(start: float, stop: float, step: float):
    x = start
    # include stop with tolerance for float error
    while x <= stop + 1e-9:
        yield round(x, 10)
        x += step


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: boolean arrays of same shape
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    return sens, spec, dice, tp, fp, tn, fn


def simulate_volume(mask: np.ndarray, signal_amplitude: float, noise_std: float, rng: np.random.Generator):
    """
    Simulate a single 3D volume (no time dimension).
    - active voxels: signal_amplitude + N(0, noise_std)
    - inactive voxels: N(0, noise_std)
    Returns a 1D raveled array of length Nvox.
    """
    active = mask.astype(bool)
    nvox = active.size
    vol = rng.normal(0.0, noise_std, size=(nvox,)).astype(np.float32)
    vol[active] += signal_amplitude
    return vol


def run_grid_resolution_experiment(
    aparc_file: str,
    brain_mask_file: str,
    pial_l_file: str,
    pial_r_file: str,
    white_l_file: str,
    white_r_file: str,
    ground_truth_parcellation_label: int,
    fwhm: float,
    signal_amplitude: float,
    noise_std: float,
    output_dir: str,
    voxel_sizes=None,
    random_seed: int = 0,
):
    """
    Runs the experiment and returns a list of dict rows for CSV.
    This function is structured to be unit-testable by monkeypatching csmooth calls.
    Assumes all csmooth functions succeed and output directories exist or can be created.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    if voxel_sizes is None:
        voxel_sizes = list(_frange(0.5, 4.0, 0.5))

    rows = []

    for voxel_size in voxel_sizes:
        # Build graph representation at this voxel size.
        # Call create_graph with positional args (assume the function accepts these inputs)
        create_graph_args = (
            aparc_file,
            brain_mask_file,
            pial_l_file,
            pial_r_file,
            white_l_file,
            white_r_file,
            voxel_size,
        )
        graph, labelmap, meta = create_graph(*create_graph_args)  # type: ignore

        # labelmap expected to encode aparc labels per voxel in graph space
        label_arr = np.asarray(labelmap, dtype=np.int32)
        gt_mask = np.asarray(label_arr == ground_truth_parcellation_label).ravel()

        # Simulate a single 3D volume (signal + noise) and use it as the map to smooth.
        tmap = simulate_volume(gt_mask, signal_amplitude, noise_std, rng)

        # Smooth without resampling (operate on this graph/grid)
        # use positional args for smooth_images
        smoothed = smooth_images(graph, tmap, fwhm)  # type: ignore

        # Threshold: use 75th percentile of smoothed OR t>2.0, whichever is larger
        p75 = float(np.percentile(smoothed, 75))
        thr = max(2.0, p75)
        pred_mask = smoothed > thr

        sens, spec, dice, tp, fp, tn, fn = compute_binary_metrics(gt_mask, pred_mask)

        # Save predicted mask as a labelmap file for later inspection
        out_labelmap = os.path.join(output_dir, f"pred_mask_{voxel_size}mm.nii.gz")
        # call save_labelmap positionally; assume meta contains needed affine/metadata
        save_labelmap(out_labelmap, pred_mask.reshape(label_arr.shape), meta)  # type: ignore

        row = {
            "voxel_size_mm": float(voxel_size),
            "fwhm_mm": float(fwhm),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "dice": float(dice),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "n_voxels": int(gt_mask.size),
            "n_active_voxels": int(np.sum(gt_mask)),
            "out_labelmap": out_labelmap,
        }
        rows.append(row)

    # Write CSV summary
    if rows:
        csv_path = os.path.join(output_dir, "grid_resolution_effect_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows


def main():
    aparc_file = os.path.abspath("./sub-MSC06_desc-aparcaseg_dseg.nii.gz")
    brain_mask_file = os.path.abspath("./sub-MSC06_desc-brain_mask.nii.gz")
    pial_l_file = os.path.abspath("./sub-MSC06_hemi-L_pial.surf.gii")
    pial_r_file = os.path.abspath("./sub-MSC06_hemi-R_pial.surf.gii")
    white_l_file = os.path.abspath("./sub-MSC06_hemi-L_white.surf.gii")
    white_r_file = os.path.abspath("./sub-MSC06_hemi-R_white.surf.gii")

    ground_truth_parcellation_label = 1035
    fwhm = 6.0
    signal_amplitude = 1.0
    noise_std = 2.0

    output_dir = os.path.abspath("./grid_resolution_effect_outputs")

    run_grid_resolution_experiment(
        aparc_file=aparc_file,
        brain_mask_file=brain_mask_file,
        pial_l_file=pial_l_file,
        pial_r_file=pial_r_file,
        white_l_file=white_l_file,
        white_r_file=white_r_file,
        ground_truth_parcellation_label=ground_truth_parcellation_label,
        fwhm=fwhm,
        signal_amplitude=signal_amplitude,
        noise_std=noise_std,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
