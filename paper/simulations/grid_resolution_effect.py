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
import argparse
import matplotlib.pyplot as plt

from nilearn.image import resample_to_img

from paper.sensory.archive.plot_stat_maps import plot_mri_with_contours
from paper.sensory.archive.plot_stat_maps import plot_multiple_stat_maps


from csmooth.smooth import (
    save_labelmap,
    process_mask,
    select_nodes,
    find_optimal_tau,
    smooth_component,
)
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


def _roc_auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC via rank statistics (handles ties)."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty_like(sorted_scores, dtype=float)

    i = 0
    rank = 1
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (rank + (rank + (j - i)))
        ranks[i:j + 1] = avg_rank
        rank += (j - i + 1)
        i = j + 1

    ranks_full = np.empty_like(ranks)
    ranks_full[order] = ranks
    sum_ranks_pos = float(np.sum(ranks_full[y]))
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _pr_auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute PR AUC using trapezoid integration over the precision-recall curve."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    tp = 0
    fp = 0
    prev_score = None
    precisions = []
    recalls = []

    for yi, si in zip(y_sorted, s_sorted):
        if prev_score is None:
            prev_score = si
        if si != prev_score:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / n_pos
            precisions.append(precision)
            recalls.append(recall)
            prev_score = si
        if yi:
            tp += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_pos
    precisions.append(precision)
    recalls.append(recall)

    recalls = np.array([0.0] + recalls, dtype=float)
    precisions = np.array([1.0] + precisions, dtype=float)
    # use numpy.trapezoid (preferred) to compute area under precision-recall curve
    try:
        auc_pr = float(np.trapezoid(precisions, recalls))
    except Exception:
        auc_pr = float(np.trapz(precisions, recalls))
    return auc_pr


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Robust Pearson correlation that returns nan when constant or empty."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    # restrict to finite values where both are finite
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")
    x1 = x[mask]
    y1 = y[mask]
    if np.allclose(x1, x1[0]) or np.allclose(y1, y1[0]):
        return float("nan")
    # Pearson via covariance
    xm = x1 - x1.mean()
    ym = y1 - y1.mean()
    denom = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    if denom == 0:
        return float("nan")
    r = float(np.sum(xm * ym) / denom)
    return r


def simulate_volume(mask: np.ndarray, signal_amplitude: float, noise_std: float, rng: np.random.Generator):
    """Simulate a binary ROI volume (kept for backwards compatibility).

    NOTE: The main experiment now uses a continuous GT field + volume-aware noise.

    Simulate a single 3D volume (no time dimension).
    - active voxels: signal_amplitude + N(0, noise_std)
    - inactive voxels: N(0, noise_std)
    Returns a 1D raveled array of length Nvox.
    """
    active = mask.astype(bool)
    nvox = active.size
    vol = rng.normal(0.0, noise_std, size=(nvox,)).astype(float)
    vol[active] += signal_amplitude
    return vol


def _voxel_centers_mm(shape3, affine):
    """Return voxel-center world coordinates (mm) for a 3D grid."""
    ijk = np.indices(shape3).reshape(3, -1).T.astype(np.float64)
    ijk_h = np.c_[ijk, np.ones((ijk.shape[0], 1), dtype=np.float64)]
    xyz = (affine @ ijk_h.T).T[:, :3]
    return xyz


def make_continuous_gaussian_gt_from_aparc(
    aparc_img: nib.Nifti1Image,
    label: int,
    center_ijk: tuple[int, int, int],
    fwhm_mm: float,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Create a continuous Gaussian GT field confined to a single aparc label.

    The Gaussian is defined in world (mm) space using voxel-center coordinates but is
    computed on the native grid of aparc_img so the center_ijk is unambiguous.
    """
    aparc_data = np.asarray(aparc_img.get_fdata(), dtype=np.int32)
    if aparc_data.ndim != 3:
        raise ValueError(f"Expected 3D aparc image, got shape={aparc_data.shape}")

    cx, cy, cz = (int(center_ijk[0]), int(center_ijk[1]), int(center_ijk[2]))
    if not (0 <= cx < aparc_data.shape[0] and 0 <= cy < aparc_data.shape[1] and 0 <= cz < aparc_data.shape[2]):
        raise ValueError(
            f"center_ijk={center_ijk} is out of bounds for aparc shape={aparc_data.shape}"
        )

    mask = (aparc_data == int(label))

    center_h = np.asarray([cx, cy, cz, 1.0], dtype=np.float64)
    center_mm = (aparc_img.affine @ center_h)[:3]

    sigma_mm = float(fwhm_mm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    xyz = _voxel_centers_mm(aparc_data.shape[:3], aparc_img.affine)
    d2 = np.sum((xyz - center_mm[None, :]) ** 2, axis=1)
    g = np.exp(-0.5 * d2 / (sigma_mm ** 2)).astype(float).reshape(aparc_data.shape[:3])

    gt_field = (g * mask.astype(float)).astype(float)
    gt_field_img = nib.Nifti1Image(gt_field, aparc_img.affine)
    gt_mask_img = nib.Nifti1Image(mask.astype(np.uint8), aparc_img.affine)
    return gt_field_img, gt_mask_img


def noise_std_for_voxel_volume(
    noise_std_ref: float,
    voxel_volume_mm3: float,
    ref_volume_mm3: float = 1.0,
) -> float:
    """Scale per-voxel noise so noise power is comparable across voxel volumes."""
    v = float(voxel_volume_mm3)
    v0 = float(ref_volume_mm3)
    if v <= 0 or v0 <= 0:
        raise ValueError("voxel volumes must be positive")
    return float(noise_std_ref) * np.sqrt(v0 / v)


def _map_aparc_center_to_t1_z(aparc_img: nib.Nifti1Image, t1_img: nib.Nifti1Image,
                              center_ijk: tuple[int, int, int]) -> int:
    """Map an aparc voxel index to the nearest T1 axial slice index."""
    cx, cy, cz = (int(center_ijk[0]), int(center_ijk[1]), int(center_ijk[2]))
    center_h = np.asarray([cx, cy, cz, 1.0], dtype=np.float64)
    center_mm = (aparc_img.affine @ center_h)[:3]
    t1_aff_inv = np.linalg.inv(t1_img.affine)
    t1_h = t1_aff_inv @ np.asarray([center_mm[0], center_mm[1], center_mm[2], 1.0], dtype=np.float64)
    z_t1 = int(round(float(t1_h[2])))
    z_t1 = max(0, min(t1_img.shape[2] - 1, z_t1))
    return z_t1


def _centered_slice_block(center_z: int, n_slices: int, block_len: int) -> list[int]:
    """Return a contiguous axial slice block centered on center_z, clamped to volume bounds."""
    if block_len <= 1:
        return [max(0, min(n_slices - 1, int(center_z)))]
    half = block_len // 2
    start = max(0, int(center_z) - half)
    end = min(n_slices, start + int(block_len))
    start = max(0, end - int(block_len))
    return list(range(start, end))


def _plot_value_range(data: np.ndarray, mask: np.ndarray | None = None,
                      lower_q: float = 70.0, upper_q: float = 99.0, thr_q: float = 85.0) -> tuple[float, float, float]:
    """Compute vmin/vmax/threshold for stat map plotting using robust percentiles."""
    vals = data[mask] if mask is not None else data
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0, 0.5
    pos = vals[vals > 0]
    use = pos if pos.size > 0 else np.abs(vals)
    vmin = float(np.percentile(use, lower_q))
    vmax = float(np.percentile(use, upper_q))
    thr = float(np.percentile(use, thr_q))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(use.min())
        vmax = float(use.max()) if use.size else vmin + 1.0
    if not np.isfinite(thr):
        thr = vmin
    return vmin, vmax, thr


def _plot_value_range_multi(arrays: list[np.ndarray], mask: np.ndarray | None = None,
                            lower_q: float = 70.0, upper_q: float = 99.0, thr_q: float = 85.0) -> tuple[float, float, float]:
    """Compute shared vmin/vmax/threshold across multiple arrays."""
    if not arrays:
        return 0.0, 1.0, 0.5
    vals = []
    for arr in arrays:
        _vals = arr[mask] if mask is not None else arr
        _vals = _vals[np.isfinite(_vals)]
        if _vals.size:
            vals.append(_vals)
    if not vals:
        return 0.0, 1.0, 0.5
    all_vals = np.concatenate(vals)
    pos = all_vals[all_vals > 0]
    use = pos if pos.size > 0 else np.abs(all_vals)
    vmin = float(np.percentile(use, lower_q))
    vmax = float(np.percentile(use, upper_q))
    thr = float(np.percentile(use, thr_q))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(use.min())
        vmax = float(use.max()) if use.size else vmin + 1.0
    if not np.isfinite(thr):
        thr = vmin
    return vmin, vmax, thr


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
    *,
    # New controls for continuous GT + comparable noise
    gt_center_ijk: tuple[int, int, int] = (70, 157, 147),
    gt_fwhm_mm: float | None = None,
    threshold_quantile_within_gt: float = 0.85,
    reference_voxel_volume_mm3: float = 1.0,
    gt_amplitude: float = 2.0,
    # Plotting controls
    t1w_file: str | None = None,
    plot_outputs: bool = True,
    plot_block_len: int = 7,
):
    """
    Runs the experiment and returns a list of dict rows for CSV.
    This function is structured to be unit-testable by monkeypatching csmooth calls.
    Assumes all csmooth functions succeed and output directories exist or can be created.

    Continuous GT mode:
      - GT is a 3D Gaussian field defined in aparc space and confined to the given label.
      - GT is resampled (linear) into each target grid.

    Noise comparability:
      - noise_std is interpreted as the *per-voxel std at reference_voxel_volume_mm3* (default 1mm^3).
      - Per-voxel noise is scaled by sqrt(Vref / V).

    Thresholding:
      - For evaluation, GT and prediction masks are defined by a quantile threshold within the GT label domain
        to avoid dependence on absolute amplitude scales.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    # Load input images once (assume aparc and brain mask are in same space)
    aparc_img = nib.load(aparc_file)
    brain_mask_image = nib.load(brain_mask_file)
    t1_img = nib.load(t1w_file) if t1w_file else None

    if gt_fwhm_mm is None:
        gt_fwhm_mm = float(fwhm)

    # Build GT once in aparc space. (center_ijk is in aparc voxel indices.)
    gt_field_base_img, gt_label_mask_base_img = make_continuous_gaussian_gt_from_aparc(
        aparc_img=aparc_img,
        label=int(ground_truth_parcellation_label),
        center_ijk=gt_center_ijk,
        fwhm_mm=float(gt_fwhm_mm),
    )

    # Increase GT magnitude by the requested amplitude factor
    if float(gt_amplitude) != 1.0:
        gt_data = np.asarray(gt_field_base_img.get_fdata(), dtype=float) * float(gt_amplitude)
        gt_field_base_img = nib.Nifti1Image(gt_data, gt_field_base_img.affine)

    nib.save(gt_field_base_img, os.path.join(output_dir, "gt_field_base.nii.gz"))
    nib.save(gt_label_mask_base_img, os.path.join(output_dir, "gt_label_mask_base.nii.gz"))

    if voxel_sizes is None:
        voxel_sizes = list(_frange(1, 4.0, 1))[::-1]  # 4mm to 1mm

    rows = []

    plot_dir = os.path.join(output_dir, "plots")
    if plot_outputs and t1_img is not None:
        os.makedirs(plot_dir, exist_ok=True)

    combined_items = []
    combined_arrays = []

    # surfaces list passed to create_graph
    surface_files = [pial_l_file, pial_r_file, white_l_file, white_r_file]

    plot_surfaces = None
    if plot_outputs and t1_img is not None:
        plot_surfaces = [
            (white_l_file, "b"),
            (white_r_file, "b"),
            (pial_l_file, "r"),
            (pial_r_file, "r"),
        ]

    for voxel_size in voxel_sizes:
        # Build graph representation at this voxel size.
        # resample brain mask to desired voxel size
        _affine = adjust_affine_spacing(brain_mask_image.affine,
                                        np.asarray(voxel_size, dtype=np.float64))
        reference_data = resample_data_to_affine(brain_mask_image.get_fdata(),
                                                 target_affine=_affine,
                                                 original_affine=brain_mask_image.affine,
                                                 interpolation="nearest")
        reference_image = nib.Nifti1Image(reference_data, _affine)

        mask_array = process_mask(brain_mask_file, reference_image, mask_dilation=3)
        # create_graph expects (mask_array, image_affine, surface_files)
        edge_src, edge_dst, edge_distances = create_graph(mask_array, reference_image.affine, surface_files)

        labels, sorted_labels, unique_nodes = identify_connected_components(edge_src, edge_dst)

        output_labelmap = os.path.join(output_dir, f"labelmap_{voxel_size}mm.nii.gz")
        save_labelmap(output_labelmap, reference_image.shape, reference_image.affine, labels, sorted_labels,
                      unique_nodes)

        # Resample continuous GT field and label mask into this grid.
        gt_field_res = resample_to_img(gt_field_base_img, reference_image, interpolation="continuous", force_resample=True)
        gt_mask_res = resample_to_img(gt_label_mask_base_img, reference_image, interpolation="nearest", force_resample=True)

        gt_field = np.asarray(gt_field_res.get_fdata(), dtype=float)
        gt_label_mask = np.asarray(gt_mask_res.get_fdata(), dtype=np.uint8).astype(bool)

        # Restrict signal support to GT label and inside the (dilated) brain mask.
        domain_mask = gt_label_mask & mask_array.astype(bool)
        gt_field = gt_field * domain_mask.astype(float)

        gt_field_img = nib.Nifti1Image(gt_field, reference_image.affine)
        gt_field_path = os.path.join(output_dir, f"gt_field_{voxel_size}mm.nii.gz")
        nib.save(gt_field_img, gt_field_path)

        # Compute volume-aware noise std for this grid.
        zooms = reference_image.header.get_zooms()[:3]
        voxel_volume = float(np.prod(zooms))
        noise_std_this = noise_std_for_voxel_volume(
            noise_std_ref=float(noise_std),
            voxel_volume_mm3=voxel_volume,
            ref_volume_mm3=float(reference_voxel_volume_mm3),
        )

        # Simulate map: continuous GT + white noise (everywhere in mask), then zero outside mask.
        noise = rng.normal(0.0, noise_std_this, size=gt_field.shape).astype(float)
        tmap_3d = (float(signal_amplitude) * gt_field + noise).astype(float)
        tmap_3d[~mask_array.astype(bool)] = 0.0

        # Preserve raw map before smoothing (smoothing is in-place)
        raw_tmap_3d = tmap_3d.copy()

        # save original tmap for reference
        original_tmap_img = nib.Nifti1Image(raw_tmap_3d, reference_image.affine)
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

        smoothed_3d = smoothed.reshape(reference_image.shape[:3])

        # save smoothed tmap for reference
        smoothed_tmap_img = nib.Nifti1Image(smoothed_3d, reference_image.affine)
        smoothed_tmap_path = os.path.join(output_dir, f"smoothed_tmap_{voxel_size}mm.nii.gz")
        nib.save(smoothed_tmap_img, smoothed_tmap_path)

        if plot_outputs and t1_img is not None and plot_surfaces is not None:
            center_z = _map_aparc_center_to_t1_z(aparc_img, t1_img, gt_center_ijk)
            slices = _centered_slice_block(center_z, t1_img.shape[2], plot_block_len)

            gt_vmin, gt_vmax, gt_thr = _plot_value_range(gt_field, domain_mask)
            raw_vmin, raw_vmax, raw_thr = _plot_value_range(raw_tmap_3d, domain_mask)
            sm_vmin, sm_vmax, sm_thr = _plot_value_range(smoothed_3d, domain_mask)

            gt_fig = plot_mri_with_contours(
                mri_fname=t1w_file,
                surfaces=plot_surfaces,
                orientation="axial",
                slices=slices,
                show=False,
                slices_as_subplots=True,
                stat_map_fname=gt_field_path,
                stat_map_cmap="hot",
                stat_map_alpha=0.9,
                stat_map_threshold=gt_thr,
                stat_map_vmin=gt_vmin,
                stat_map_vmax=gt_vmax,
                surface_thickness=0.25,
                colorbar=True,
            )
            gt_png = os.path.join(plot_dir, f"gt_{voxel_size}mm.png")
            gt_fig.savefig(gt_png, dpi=300)
            plt.close(gt_fig)

            raw_fig = plot_mri_with_contours(
                mri_fname=t1w_file,
                surfaces=plot_surfaces,
                orientation="axial",
                slices=slices,
                show=False,
                slices_as_subplots=True,
                stat_map_fname=original_tmap_path,
                stat_map_cmap="hot",
                stat_map_alpha=0.9,
                stat_map_threshold=raw_thr,
                stat_map_vmin=raw_vmin,
                stat_map_vmax=raw_vmax,
                surface_thickness=0.25,
                colorbar=True,
            )
            raw_png = os.path.join(plot_dir, f"raw_{voxel_size}mm.png")
            raw_fig.savefig(raw_png, dpi=300)
            plt.close(raw_fig)

            sm_fig = plot_mri_with_contours(
                mri_fname=t1w_file,
                surfaces=plot_surfaces,
                orientation="axial",
                slices=slices,
                show=False,
                slices_as_subplots=True,
                stat_map_fname=smoothed_tmap_path,
                stat_map_cmap="hot",
                stat_map_alpha=0.9,
                stat_map_threshold=sm_thr,
                stat_map_vmin=sm_vmin,
                stat_map_vmax=sm_vmax,
                surface_thickness=0.25,
                colorbar=True,
            )
            sm_png = os.path.join(plot_dir, f"smoothed_{voxel_size}mm.png")
            sm_fig.savefig(sm_png, dpi=300)
            plt.close(sm_fig)

            # Combined plot: single slice, shared color scale, GT + raw + smoothed
            single_slice = [int(center_z)]
            combo_vmin, combo_vmax, combo_thr = _plot_value_range_multi(
                [gt_field, raw_tmap_3d, smoothed_3d],
                domain_mask,
            )
            combo_fig = plot_multiple_stat_maps(
                mri_fname=t1w_file,
                surfaces=plot_surfaces,
                stat_map_fnames=[gt_field_path, original_tmap_path, smoothed_tmap_path],
                slices=single_slice,
                orientation="axial",
                width=512,
                slices_as_subplots=True,
                stat_map_cmap="hot",
                stat_map_alpha=0.9,
                stat_map_threshold=combo_thr,
                mri_alpha=0.9,
                surface_alpha=0.9,
                stat_map_interpolation="nearest",
                surface_thickness=0.25,
                show=False,
                stat_map_vmin=combo_vmin,
                stat_map_vmax=combo_vmax,
                colorbar=True,
            )
            combo_png = os.path.join(plot_dir, f"combined_single_slice_{voxel_size}mm.png")
            combo_fig.savefig(combo_png, dpi=300)
            plt.close(combo_fig)

            combined_items.append(
                {
                    "voxel_size": float(voxel_size),
                    "gt_path": gt_field_path,
                    "raw_path": original_tmap_path,
                    "sm_path": smoothed_tmap_path,
                }
            )
            combined_arrays.append(np.where(domain_mask, gt_field, np.nan))
            combined_arrays.append(np.where(domain_mask, raw_tmap_3d, np.nan))
            combined_arrays.append(np.where(domain_mask, smoothed_3d, np.nan))

        # Define GT/pred masks by a quantile within the GT-label domain (more comparable across voxel sizes).
        gt_field_vec = gt_field.ravel()
        domain_idx = np.flatnonzero(domain_mask.ravel())
        if domain_idx.size == 0:
            gt_mask = np.zeros_like(gt_field_vec, dtype=bool)
            pred_mask = np.zeros_like(gt_field_vec, dtype=bool)
            roc_auc = float("nan")
            pr_auc = float("nan")
            pearson_r = float("nan")
            mae = float("nan")
            roc_auc_raw = float("nan")
            pr_auc_raw = float("nan")
            pearson_r_raw = float("nan")
            mae_raw = float("nan")
            sens_raw = float("nan")
            spec_raw = float("nan")
            dice_raw = float("nan")
            tp_raw = fp_raw = tn_raw = fn_raw = 0
        else:
            q = float(threshold_quantile_within_gt)
            gt_thr = float(np.quantile(gt_field_vec[domain_idx], q))
            gt_mask = np.zeros_like(gt_field_vec, dtype=bool)
            gt_mask[domain_idx] = gt_field_vec[domain_idx] >= gt_thr

            pred_thr = float(np.quantile(smoothed[domain_idx], q))
            pred_mask = np.zeros_like(gt_field_vec, dtype=bool)
            pred_mask[domain_idx] = smoothed[domain_idx] >= pred_thr

            roc_auc = _roc_auc_from_scores(gt_mask[domain_idx], smoothed[domain_idx])
            pr_auc = _pr_auc_from_scores(gt_mask[domain_idx], smoothed[domain_idx])
            # Continuous metrics: Pearson correlation and MAE between GT continuous field and smoothed values
            gt_vals = gt_field_vec[domain_idx]
            sm_vals = smoothed[domain_idx]
            pearson_r = _pearsonr(gt_vals, sm_vals)
            mae = float(np.mean(np.abs(gt_vals - sm_vals))) if gt_vals.size > 0 else float("nan")

            # Raw/no-smoothing metrics
            raw_vals = raw_tmap_3d.ravel()[domain_idx]
            roc_auc_raw = _roc_auc_from_scores(gt_mask[domain_idx], raw_vals)
            pr_auc_raw = _pr_auc_from_scores(gt_mask[domain_idx], raw_vals)
            pearson_r_raw = _pearsonr(gt_vals, raw_vals)
            mae_raw = float(np.mean(np.abs(gt_vals - raw_vals))) if gt_vals.size > 0 else float("nan")

            pred_thr_raw = float(np.quantile(raw_vals, q))
            pred_mask_raw = np.zeros_like(gt_field_vec, dtype=bool)
            pred_mask_raw[domain_idx] = raw_vals >= pred_thr_raw
            sens_raw, spec_raw, dice_raw, tp_raw, fp_raw, tn_raw, fn_raw = compute_binary_metrics(gt_mask, pred_mask_raw)

        sens, spec, dice, tp, fp, tn, fn = compute_binary_metrics(gt_mask, pred_mask)

        row = {
            "voxel_size_mm": float(voxel_size),
            "fwhm_mm": float(fwhm),
            "gt_fwhm_mm": float(gt_fwhm_mm),
            "gt_center_ijk": str(tuple(int(x) for x in gt_center_ijk)),
            "gt_amplitude": float(gt_amplitude),
            "threshold_quantile_within_gt": float(threshold_quantile_within_gt),
            "noise_std_ref": float(noise_std),
            "reference_voxel_volume_mm3": float(reference_voxel_volume_mm3),
            "voxel_volume_mm3": float(voxel_volume),
            "noise_std_this": float(noise_std_this),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "dice": float(dice),
            "pearson_r": float(pearson_r),
            "mae": float(mae),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "sensitivity_raw": float(sens_raw),
            "specificity_raw": float(spec_raw),
            "dice_raw": float(dice_raw),
            "pearson_r_raw": float(pearson_r_raw),
            "mae_raw": float(mae_raw),
            "tp_raw": int(tp_raw),
            "fp_raw": int(fp_raw),
            "tn_raw": int(tn_raw),
            "fn_raw": int(fn_raw),
            "n_voxels": int(gt_field_vec.size),
            "n_domain_voxels": int(domain_idx.size),
            "n_active_voxels": int(np.sum(gt_mask)),
            "out_labelmap": output_labelmap,
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "roc_auc_raw": float(roc_auc_raw),
            "pr_auc_raw": float(pr_auc_raw),
        }
        rows.append(row)

    if plot_outputs and t1_img is not None and plot_surfaces is not None and combined_items:
        combined_items = sorted(combined_items, key=lambda item: item["voxel_size"])
        n_cols = len(combined_items)
        fig, axs = plt.subplots(3, n_cols, figsize=(3.0 * n_cols, 8.5))
        if n_cols == 1:
            axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

        center_z = _map_aparc_center_to_t1_z(aparc_img, t1_img, gt_center_ijk)
        slices = [int(center_z)]
        shared_vmin, shared_vmax, shared_thr = _plot_value_range_multi(combined_arrays)

        for col, item in enumerate(combined_items):
            voxel_label = f"{item['voxel_size']}mm"

            plot_mri_with_contours(
                mri_fname=t1w_file,
                surfaces=plot_surfaces,
                orientation="axial",
                slices=slices,
                show=False,
                slices_as_subplots=False,
                stat_map_fname=item["gt_path"],
                stat_map_cmap="hot",
                stat_map_alpha=0.9,
                stat_map_threshold=shared_thr,
                stat_map_vmin=shared_vmin,
                stat_map_vmax=shared_vmax,
                surface_thickness=0.25,
                colorbar=False,
                ax=axs[0, col],
            )
            axs[0, col].set_title(f"GT {voxel_label}")

            plot_mri_with_contours(
                mri_fname=t1w_file,
                surfaces=plot_surfaces,
                orientation="axial",
                slices=slices,
                show=False,
                slices_as_subplots=False,
                stat_map_fname=item["raw_path"],
                stat_map_cmap="hot",
                stat_map_alpha=0.9,
                stat_map_threshold=shared_thr,
                stat_map_vmin=shared_vmin,
                stat_map_vmax=shared_vmax,
                surface_thickness=0.25,
                colorbar=False,
                ax=axs[1, col],
            )
            axs[1, col].set_title(f"Raw {voxel_label}")

            plot_mri_with_contours(
                mri_fname=t1w_file,
                surfaces=plot_surfaces,
                orientation="axial",
                slices=slices,
                show=False,
                slices_as_subplots=False,
                stat_map_fname=item["sm_path"],
                stat_map_cmap="hot",
                stat_map_alpha=0.9,
                stat_map_threshold=shared_thr,
                stat_map_vmin=shared_vmin,
                stat_map_vmax=shared_vmax,
                surface_thickness=0.25,
                colorbar=(col == n_cols - 1),
                ax=axs[2, col],
            )
            axs[2, col].set_title(f"Smoothed {voxel_label}")

        combined_grid_path = os.path.join(plot_dir, "combined_all_voxels.png")
        fig.savefig(combined_grid_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

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
    t1w_file = os.path.join(file_dir, "./sub-MSC06_desc-preproc_T1w.nii.gz")
    aparc_file = os.path.join(file_dir, "./sub-MSC06_desc-aparcaseg_dseg.nii.gz")
    brain_mask_file = os.path.join(file_dir, "./sub-MSC06_desc-brain_mask.nii.gz")
    pial_l_file = os.path.join(file_dir, "./sub-MSC06_hemi-L_pial.surf.gii")
    pial_r_file = os.path.join(file_dir, "./sub-MSC06_hemi-R_pial.surf.gii")
    white_l_file = os.path.join(file_dir, "./sub-MSC06_hemi-L_white.surf.gii")
    white_r_file = os.path.join(file_dir, "./sub-MSC06_hemi-R_white.surf.gii")

    parser = argparse.ArgumentParser(description="Run grid resolution GT simulation")
    parser.add_argument("--gt-amplitude", type=float, default=2.0, help="Amplitude multiplier for GT field")
    parser.add_argument(
        "--gt-center",
        type=int,
        nargs=3,
        default=(66, 129, 145),
        metavar=("I", "J", "K"),
        help="GT center voxel index in aparc space (i j k)",
    )
    parser.add_argument(
        "--gt-fwhm",
        type=float,
        default=6.0,
        help="GT Gaussian FWHM in mm (overrides default fwhm)",
    )
    parser.add_argument(
        "--infer-label-from-gt-center",
        action="store_true",
        help="Infer aparc label from the GT center voxel index",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(file_dir, "./grid_resolution_effect_outputs"),
        help="Base output directory; a run-specific subfolder will be created",
    )
    args = parser.parse_args()

    ground_truth_parcellation_label = 1035
    fwhm = 6.0
    signal_amplitude = 1.0
    noise_std = 2.0

    gt_center_ijk = tuple(int(v) for v in args.gt_center)

    if args.infer_label_from_gt_center:
        aparc_img = nib.load(aparc_file)
        aparc_data = np.asarray(aparc_img.get_fdata(), dtype=np.int32)
        i, j, k = gt_center_ijk
        if not (0 <= i < aparc_data.shape[0] and 0 <= j < aparc_data.shape[1] and 0 <= k < aparc_data.shape[2]):
            raise ValueError(f"gt_center {gt_center_ijk} is out of bounds for aparc shape={aparc_data.shape}")
        inferred_label = int(aparc_data[i, j, k])
        ground_truth_parcellation_label = inferred_label

    def _fmt_float(value: float) -> str:
        return f"{value:.3g}".replace(".", "p")

    run_id = (
        f"fwhm{_fmt_float(fwhm)}_"
        f"gtfwhm{_fmt_float(args.gt_fwhm)}_"
        f"gt{gt_center_ijk[0]}-{gt_center_ijk[1]}-{gt_center_ijk[2]}_"
        f"amp{_fmt_float(args.gt_amplitude)}"
    )
    output_dir = os.path.join(args.output_dir, run_id)

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
        gt_amplitude=args.gt_amplitude,
        t1w_file=t1w_file,
        gt_center_ijk=gt_center_ijk,
        gt_fwhm_mm=float(args.gt_fwhm),
    )


if __name__ == "__main__":
    main()
