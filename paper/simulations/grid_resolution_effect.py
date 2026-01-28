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
The smallest voxel size is 1mm isotropic, and the largest is 4mm isotropic, with increments of 1mm.
"""

# paper/simulations/grid_resolution_effect.py
import os
import csv
import numpy as np
import nibabel as nib
from tqdm import tqdm

from nilearn.image import resample_to_img

from csmooth.smooth import (smooth_images, save_labelmap, load_reference_image, process_mask, select_nodes,
                            find_optimal_tau, smooth_component)
from csmooth.graph import create_graph
from csmooth.components import identify_connected_components
from csmooth.affine import adjust_affine_spacing, resample_data_to_affine


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


def apply_estimated_gaussian_smoothing(signal_data, edge_src, edge_dst, edge_distances, labels, sorted_labels,
                                       unique_nodes, fwhm, low_memory=False):
    """
    Apply estimated Gaussian smoothing to a list of images based on the provided graph structure.
    Finds the optimal tau for each of the five largest components in the graph (background, wm left/right, gm left/right).
    The rest of the components are smoothed with the mean tau of the five largest components.

    :param edge_src: Source nodes of the graph edges.
    :param edge_dst: Destination nodes of the graph edges.
    :param edge_distances: Distances of the graph edges.
    :param labels: Labels for each node in the graph.
    :param sorted_labels: Sorted list of labels by size of components.
    :param unique_nodes: Unique nodes in the graph.
    :param fwhm: Target full width at half maximum in mm for smoothing.
    :param low_memory: If True, use low memory mode. This will reduce memory usage but may increase runtime.
    :return: None
    """
    # Estimate optimal tau for each component to achieve the target fwhm
    main_labels = sorted_labels[:5]
    taus = list()
    initial_tau = 2 * fwhm
    for label in tqdm(main_labels, desc="Finding optimal taus", unit="component"):
        _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_distances=edge_distances,
            labels=labels,
            label=label,
            unique_nodes=unique_nodes)
        _tau = find_optimal_tau(fwhm=fwhm,
                                edge_src=_edge_src,
                                edge_dst=_edge_dst,
                                edge_distances=_edge_distances,
                                shape=(len(_nodes),),
                                initial_tau=initial_tau)
        taus.append(_tau)
        # update initial tau to the last estimated tau
        initial_tau = _tau

    if signal_data.ndim == 3:
        signal_data = signal_data[..., None]

    _shape = signal_data.shape

    signal_data = signal_data.reshape(-1, signal_data.shape[-1])
    # For a single timepoint, use low-memory mode so smoothing is applied per 1D vector.
    if signal_data.shape[1] == 1:
        low_memory = True
    # smooth the background data with the mean estimated tau
    smooth_component(
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_distances=edge_distances,
        signal_data=signal_data,
        labels=labels,
        label=sorted_labels[5:],  # smooth the rest of the components with the mean tau
        unique_nodes=unique_nodes,
        tau=np.mean(taus),  # Use the mean tau for all components
        low_memory=low_memory
    )
    # smooth each main component with the estimated tau
    for label, _tau in zip(main_labels, taus):
        smooth_component(edge_src=edge_src,
                         edge_dst=edge_dst,
                         edge_distances=edge_distances,
                         signal_data=signal_data,
                         labels=labels,
                         label=label,
                         unique_nodes=unique_nodes,
                         tau=_tau,
                         low_memory=low_memory)

    return signal_data



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
    timepoints: int = 1,
):
    """
    Runs the experiment and returns a list of dict rows for CSV.
    This function is structured to be unit-testable by monkeypatching csmooth calls.
    Assumes all csmooth functions succeed and output directories exist or can be created.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    # Load input images once (assume aparc and brain mask are in same space)
    aparc_img = nib.load(aparc_file)
    brain_mask_image = nib.load(brain_mask_file)

    if voxel_sizes is None:
        voxel_sizes = list(_frange(1, 4.0, 1))[::-1]  # 4mm to 1mm

    rows = []

    # surfaces list passed to create_graph
    surface_files = [pial_l_file, pial_r_file, white_l_file, white_r_file]

    for voxel_size in voxel_sizes:
        # Build graph representation at this voxel size.
        # resample brain mask to desired voxel size
        _affine = adjust_affine_spacing(brain_mask_image.affine,
                                        np.asarray(voxel_size))
        reference_data = resample_data_to_affine(brain_mask_image.get_fdata(),
                                                 target_affine=_affine,
                                                 original_affine=brain_mask_image.affine,
                                                 interpolation="nearest")
        _shape = reference_data.shape[:3]
        reference_image = nib.Nifti1Image(reference_data, _affine)

        mask_array = process_mask(brain_mask_file, reference_image, mask_dilation=3)
        # create_graph expects (mask_array, image_affine, surface_files)
        edge_src, edge_dst, edge_distances = create_graph(mask_array, reference_image.affine, surface_files)

        labels, sorted_labels, unique_nodes = identify_connected_components(edge_src, edge_dst)

        output_labelmap = os.path.join(output_dir, f"labelmap_{voxel_size}mm.nii.gz")
        save_labelmap(output_labelmap, reference_image.shape, reference_image.affine, labels, sorted_labels,
                      unique_nodes)


        # labelmap expected to encode aparc labels per voxel in graph space
        aparg_img_resampled = resample_to_img(aparc_img, reference_image, interpolation='nearest', force_resample=True)
        aparc_data = np.asarray(aparg_img_resampled.get_fdata())
        label_arr = np.asarray(aparc_data, dtype=np.int32)
        gt_mask = np.asarray(label_arr == ground_truth_parcellation_label).ravel()

        # Simulate a single 3D volume (signal + noise) and use it as the map to smooth.
        tmap = simulate_volume(gt_mask, signal_amplitude, noise_std, rng)

        # set voxels outside the brain mask to zero
        tmap[~mask_array.ravel()] = 0.0

        # Smooth without resampling (operate on this graph/grid)
        tmap_3d = tmap.reshape(label_arr.shape).astype(np.float32)

        # save original tmap for reference
        original_tmap_img = nib.Nifti1Image(tmap_3d, reference_image.affine)
        original_tmap_path = os.path.join(output_dir, f"original_tmap_{voxel_size}mm.nii.gz")
        nib.save(original_tmap_img, original_tmap_path)

        smoothed = apply_estimated_gaussian_smoothing(
            signal_data=tmap_3d,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_distances=edge_distances,
            labels=labels,
            sorted_labels=sorted_labels,
            unique_nodes=unique_nodes,
            fwhm=fwhm,
            low_memory=False
        ).ravel()

        # save smoothed tmap for reference
        smoothed_tmap_img = nib.Nifti1Image(smoothed.reshape(label_arr.shape), reference_image.affine)
        smoothed_tmap_path = os.path.join(output_dir, f"smoothed_tmap_{voxel_size}mm.nii.gz")
        nib.save(smoothed_tmap_img, smoothed_tmap_path)

        # Threshold: 0.5; above is active
        thr = 0.5
        pred_mask = smoothed > thr

        sens, spec, dice, tp, fp, tn, fn = compute_binary_metrics(gt_mask, pred_mask)

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
            "out_labelmap": output_labelmap,
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
    file_dir = os.path.dirname(os.path.abspath(__file__))
    aparc_file = os.path.join(file_dir, "./sub-MSC06_desc-aparcaseg_dseg.nii.gz")
    brain_mask_file = os.path.join(file_dir, "./sub-MSC06_desc-brain_mask.nii.gz")
    pial_l_file = os.path.join(file_dir, "./sub-MSC06_hemi-L_pial.surf.gii")
    pial_r_file = os.path.join(file_dir, "./sub-MSC06_hemi-R_pial.surf.gii")
    white_l_file = os.path.join(file_dir, "./sub-MSC06_hemi-L_white.surf.gii")
    white_r_file = os.path.join(file_dir, "./sub-MSC06_hemi-R_white.surf.gii")

    ground_truth_parcellation_label = 1035
    fwhm = 6.0
    signal_amplitude = 1.0
    noise_std = 2.0

    output_dir = os.path.join(file_dir, "./grid_resolution_effect_outputs")

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
