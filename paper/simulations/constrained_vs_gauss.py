"""
This script simulates a single active cortical region on the aparc labels, adds Gaussian noise,
then compares constrained graph smoothing against volumetric Gaussian smoothing at target FWHM values.
Accuracy is evaluated with Dice/sensitivity/specificity and score-based metrics within the GT region.
"""

import os
import csv
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from nilearn.image import resample_to_img, smooth_img

from paper.sensory.archive.plot_stat_maps import plot_mri_with_contours
from paper.sensory.archive.plot_stat_maps import plot_multiple_stat_maps

from paper.simulations.grid_resolution_effect import (
    compute_binary_metrics,
    _roc_auc_from_scores,
    _pr_auc_from_scores,
    _pearsonr,
    _plot_value_range,
    _plot_value_range_multi,
    _map_aparc_center_to_t1_z,
    _centered_slice_block,
    _compute_crop_bounds_from_stat_map,
)

from csmooth.smooth import (
    save_labelmap,
    process_mask,
    select_nodes,
    find_optimal_tau,
    smooth_component,
)
from csmooth.graph import create_graph
from csmooth.components import identify_connected_components


def estimate_component_taus(
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_distances: np.ndarray,
    labels: np.ndarray,
    sorted_labels: np.ndarray,
    unique_nodes: np.ndarray,
    fwhm: float,
) -> tuple[list[int], list[float]]:
    main_labels = [int(lbl) for lbl in sorted_labels[:5]]
    taus: list[float] = []
    initial_tau = 2.0 * float(fwhm)
    for label in tqdm(main_labels, desc="Finding optimal taus", unit="component"):
        _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_distances=edge_distances,
            labels=labels,
            label=label,
            unique_nodes=unique_nodes,
        )
        _tau = find_optimal_tau(
            fwhm=float(fwhm),
            edge_src=_edge_src,
            edge_dst=_edge_dst,
            edge_distances=_edge_distances,
            shape=(len(_nodes),),
            initial_tau=initial_tau,
        )
        taus.append(float(_tau))
        initial_tau = float(_tau)
    return main_labels, taus


def apply_constrained_smoothing(
    signal_data: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_distances: np.ndarray,
    labels: np.ndarray,
    sorted_labels: np.ndarray,
    unique_nodes: np.ndarray,
    main_labels: list[int],
    taus: list[float],
    low_memory: bool = False,
) -> np.ndarray:
    """Apply constrained smoothing with precomputed taus per component."""
    if signal_data.ndim == 3:
        signal_data = signal_data[..., None]

    signal_data = signal_data.reshape(-1, signal_data.shape[-1])
    if signal_data.shape[1] == 1:
        low_memory = True

    smooth_component(
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_distances=edge_distances,
        signal_data=signal_data,
        labels=labels,
        label=sorted_labels[5:],
        unique_nodes=unique_nodes,
        tau=float(np.mean(taus)) if taus else 0.0,
        low_memory=low_memory,
    )

    for label, _tau in zip(main_labels, taus):
        smooth_component(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_distances=edge_distances,
            signal_data=signal_data,
            labels=labels,
            label=int(label),
            unique_nodes=unique_nodes,
            tau=float(_tau),
            low_memory=low_memory,
        )

    return signal_data


def _compute_gt_and_domain_masks(
    gt_field: np.ndarray,
    domain_mask: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, float]:
    gt_vec = gt_field.ravel()
    domain_idx = np.flatnonzero(domain_mask.ravel())
    if domain_idx.size == 0:
        return np.zeros_like(gt_vec, dtype=bool), float("nan")
    thr = float(np.quantile(gt_vec[domain_idx], float(quantile)))
    gt_mask = np.zeros_like(gt_vec, dtype=bool)
    gt_mask[domain_idx] = gt_vec[domain_idx] >= thr
    return gt_mask, thr


def _compute_pred_mask(
    pred_values: np.ndarray,
    domain_mask: np.ndarray,
    pred_threshold_quantile: float,
) -> tuple[np.ndarray, float]:
    pred_vec = pred_values.ravel()
    domain_idx = np.flatnonzero(domain_mask.ravel())
    if domain_idx.size == 0:
        return np.zeros_like(pred_vec, dtype=bool), float("nan")
    pred_thr = float(np.quantile(pred_vec[domain_idx], float(pred_threshold_quantile)))
    pred_mask = np.zeros_like(pred_vec, dtype=bool)
    pred_mask[domain_idx] = pred_vec[domain_idx] >= pred_thr
    return pred_mask, pred_thr


def _std_in_mask(values: np.ndarray, mask: np.ndarray) -> float:
    vals = values[mask]
    if vals.size == 0:
        return float("nan")
    return float(np.std(vals))


def _active_counts_by_tissue(
    pred_mask: np.ndarray,
    wm_mask: np.ndarray,
    gm_mask: np.ndarray,
    other_mask: np.ndarray,
) -> tuple[int, int, int]:
    return int(pred_mask[wm_mask].sum()), int(pred_mask[gm_mask].sum()), int(pred_mask[other_mask].sum())


def _compute_method_metrics(
    gt_mask: np.ndarray,
    domain_mask: np.ndarray,
    pred_values: np.ndarray,
    pred_threshold_quantile: float | None,
) -> dict:
    pred_vec = pred_values.ravel()
    domain_idx = np.flatnonzero(domain_mask.ravel())
    if domain_idx.size == 0:
        return {
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "dice": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "pearson_r": float("nan"),
            "mae": float("nan"),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "pred_thr": float("nan"),
            "pred_mask": np.zeros_like(pred_vec, dtype=bool),
        }

    # If pred_threshold_quantile is None, caller should have set it to an effective numeric
    # value (1 - prevalence). Here, defend in case it is still None by computing from gt
    # prevalence within the domain.
    if pred_threshold_quantile is None:
        # compute prevalence from gt_mask within domain and set quantile = 1 - prevalence
        prevalence = float(np.mean(gt_mask[domain_idx].astype(float))) if domain_idx.size else 0.0
        pred_threshold_quantile = float(1.0 - prevalence)

    pred_mask, pred_thr = _compute_pred_mask(pred_values, domain_mask, pred_threshold_quantile)

    sens, spec, dice, tp, fp, tn, fn = compute_binary_metrics(gt_mask, pred_mask)

    roc_auc = _roc_auc_from_scores(gt_mask[domain_idx], pred_vec[domain_idx])
    pr_auc = _pr_auc_from_scores(gt_mask[domain_idx], pred_vec[domain_idx])
    gt_vals = gt_mask.astype(float)[domain_idx]
    pearson_r = _pearsonr(gt_vals, pred_vec[domain_idx])
    mae = float(np.mean(np.abs(gt_vals - pred_vec[domain_idx]))) if domain_idx.size else float("nan")

    return {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "dice": float(dice),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "pearson_r": float(pearson_r),
        "mae": float(mae),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "pred_thr": float(pred_thr),
        "pred_mask": pred_mask,
    }


def _label_center_ijk(aparc_img: nib.Nifti1Image, label: int) -> tuple[int, int, int]:
    data = np.asarray(aparc_img.get_fdata(), dtype=np.int32)
    coords = np.argwhere(data == int(label))
    if coords.size == 0:
        raise ValueError(f"Label {label} not found in aparc")
    center = coords[len(coords) // 2]
    return int(center[0]), int(center[1]), int(center[2])


def run_constrained_vs_gauss_experiment_single_region(
    aparc_img: nib.Nifti1Image,
    brain_mask_image: nib.Nifti1Image,
    t1_img: nib.Nifti1Image | None,
    surface_files: list[str],
    label: int,
    center_ijk: tuple[int, int, int],
    fwhm_list: list[float],
    amplitude: float,
    noise_std: float,
    output_dir: str,
    mask_array: np.ndarray,
    gm_mask: np.ndarray,
    wm_mask: np.ndarray,
    other_mask: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_distances: np.ndarray,
    labels: np.ndarray,
    sorted_labels: np.ndarray,
    unique_nodes: np.ndarray,
    optimal_taus_by_fwhm: dict[float, list[float]] | None = None,
    random_seed: int | None = None,
    pred_threshold_quantile: float | None = None,
    overwrite_volumes: bool = False,
    save_images: bool = True,
    plot_outputs: bool = False,
) -> tuple[list[dict], dict[float, list[float]]]:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng() if random_seed is None else np.random.default_rng(int(random_seed))

    if optimal_taus_by_fwhm is None:
        optimal_taus_by_fwhm = {}

    aparc_res = resample_to_img(aparc_img, brain_mask_image, interpolation="nearest", force_resample=True)
    aparc_data = np.asarray(aparc_res.get_fdata(), dtype=np.int32)
    domain_mask = mask_array.astype(bool)

    gt_mask_3d = (aparc_data == int(label)) & domain_mask
    gt_field = gt_mask_3d.astype(float) * float(amplitude)

    gt_field_img = nib.Nifti1Image(gt_field, brain_mask_image.affine)
    gt_mask_img = nib.Nifti1Image(gt_mask_3d.astype(np.uint8), brain_mask_image.affine)

    if save_images:
        nib.save(gt_field_img, os.path.join(output_dir, "gt_field.nii.gz"))
        nib.save(gt_mask_img, os.path.join(output_dir, "gt_label_mask.nii.gz"))

    noise = rng.normal(0.0, float(noise_std), size=gt_field.shape).astype(float)
    raw_tmap_3d = (gt_field + noise).astype(float)
    raw_tmap_3d[~domain_mask] = 0.0

    raw_path = os.path.join(output_dir, "raw_tmap.nii.gz")
    # If raw volume exists and overwrite is False, load it; otherwise use generated and save if requested
    if os.path.exists(raw_path) and not overwrite_volumes:
        tmp_img = nib.load(raw_path)
        tmp_arr = np.asarray(tmp_img.get_fdata())
        if tmp_arr.shape == gt_field.shape:
            raw_tmap_img = tmp_img
            raw_tmap_3d = np.asarray(raw_tmap_img.get_fdata(), dtype=float)
        else:
            print(f"Existing raw volume {raw_path} has shape {tmp_arr.shape} != expected {gt_field.shape}; recomputing.")
            raw_tmap_img = nib.Nifti1Image(raw_tmap_3d, brain_mask_image.affine)
            if save_images:
                nib.save(raw_tmap_img, raw_path)
    else:
        raw_tmap_img = nib.Nifti1Image(raw_tmap_3d, brain_mask_image.affine)
        if save_images:
            nib.save(raw_tmap_img, raw_path)

    gt_mask = gt_mask_3d.ravel()
    wm_mask_flat = wm_mask.ravel()
    gm_mask_flat = gm_mask.ravel()
    other_mask_flat = other_mask.ravel()

    # If pred_threshold_quantile is None, compute an effective numeric quantile equal to
    # 1 - prevalence (prevalence computed from GT within the domain). This becomes the
    # default behaviour requested by the user.
    domain_idx_global = np.flatnonzero(domain_mask.ravel())
    if pred_threshold_quantile is None:
        prevalence = float(np.mean(gt_mask[domain_idx_global].astype(float))) if domain_idx_global.size else 0.0
        pred_threshold_quantile = float(1.0 - prevalence)

    gt_vmin, gt_vmax, gt_thr_plot = _plot_value_range(gt_field, domain_mask)
    raw_vmin, raw_vmax, _ = _plot_value_range(raw_tmap_3d, domain_mask)

    rows: list[dict] = []
    plot_dir = os.path.join(output_dir, "plots")
    combined_maps: list[np.ndarray] = []
    combined_paths: list[str] = []
    combined_thresholds: list[float] = []
    no_thr = float("-inf")
    crop_bounds = None
    if plot_outputs and t1_img is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plot_surfaces = [
            (surface_files[2], "b"),
            (surface_files[3], "b"),
            (surface_files[0], "r"),
            (surface_files[1], "r"),
        ]
        center_z = _map_aparc_center_to_t1_z(aparc_img, t1_img, center_ijk)
        slices = _centered_slice_block(center_z, t1_img.shape[2], 7)
        crop_bounds = _compute_crop_bounds_from_stat_map(
            stat_map_img=gt_field_img,
            t1_img=t1_img,
            slice_index=int(slices[len(slices) // 2]),
            threshold=float(gt_thr_plot),
            padding=10,
        )
        gt_path = os.path.join(output_dir, "gt_field.nii.gz")
        raw_path = os.path.join(output_dir, "raw_tmap.nii.gz")
        combined_maps = [gt_field, raw_tmap_3d]
        combined_paths = [gt_path, raw_path]
        combined_thresholds = [gt_thr_plot, float("nan")]
    else:
        plot_surfaces = None
        slices = None

    multi_fwhm = len(fwhm_list) > 1

    for fwhm in fwhm_list:
        fwhm = float(fwhm)
        fwhm_dir = os.path.join(output_dir, f"fwhm_{int(round(fwhm))}mm")
        os.makedirs(fwhm_dir, exist_ok=True)

        if fwhm not in optimal_taus_by_fwhm:
            main_labels, taus = estimate_component_taus(
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_distances=edge_distances,
                labels=labels,
                sorted_labels=sorted_labels,
                unique_nodes=unique_nodes,
                fwhm=fwhm,
            )
            optimal_taus_by_fwhm[fwhm] = taus
        else:
            main_labels = [int(lbl) for lbl in sorted_labels[:5]]
            taus = optimal_taus_by_fwhm[fwhm]

        constrained_path = os.path.join(fwhm_dir, "constrained_tmap.nii.gz")
        gaussian_path = os.path.join(fwhm_dir, "gaussian_tmap.nii.gz")

        # Constrained: load if exists and not overwriting, else compute and save (if enabled)
        if os.path.exists(constrained_path) and not overwrite_volumes:
            tmp_img = nib.load(constrained_path)
            tmp_arr = np.asarray(tmp_img.get_fdata())
            if tmp_arr.shape == raw_tmap_3d.shape:
                constrained = np.asarray(tmp_img.get_fdata(), dtype=float)
            else:
                print(f"Existing constrained volume {constrained_path} has shape {tmp_arr.shape} != expected {raw_tmap_3d.shape}; recomputing.")
                constrained = apply_constrained_smoothing(
                    signal_data=raw_tmap_3d.copy(),
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    edge_distances=edge_distances,
                    labels=labels,
                    sorted_labels=sorted_labels,
                    unique_nodes=unique_nodes,
                    main_labels=main_labels,
                    taus=taus,
                    low_memory=False,
                ).reshape(raw_tmap_3d.shape)
                if save_images:
                    constrained_img = nib.Nifti1Image(constrained, brain_mask_image.affine)
                    nib.save(constrained_img, constrained_path)
        else:
            constrained = apply_constrained_smoothing(
                signal_data=raw_tmap_3d.copy(),
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_distances=edge_distances,
                labels=labels,
                sorted_labels=sorted_labels,
                unique_nodes=unique_nodes,
                main_labels=main_labels,
                taus=taus,
                low_memory=False,
            ).reshape(raw_tmap_3d.shape)
            if save_images:
                constrained_img = nib.Nifti1Image(constrained, brain_mask_image.affine)
                nib.save(constrained_img, constrained_path)

        # Gaussian: load if exists and not overwriting, else compute and save (if enabled)
        if os.path.exists(gaussian_path) and not overwrite_volumes:
            tmp_img = nib.load(gaussian_path)
            tmp_arr = np.asarray(tmp_img.get_fdata())
            if tmp_arr.shape == raw_tmap_3d.shape:
                gaussian = np.asarray(tmp_img.get_fdata(), dtype=float)
            else:
                print(f"Existing gaussian volume {gaussian_path} has shape {tmp_arr.shape} != expected {raw_tmap_3d.shape}; recomputing.")
                gaussian_img = smooth_img(raw_tmap_img, fwhm=fwhm)
                gaussian = np.asarray(gaussian_img.get_fdata(), dtype=float)
                gaussian[~domain_mask] = 0.0
                if save_images:
                    gaussian_img = nib.Nifti1Image(gaussian, brain_mask_image.affine)
                    nib.save(gaussian_img, gaussian_path)
        else:
            gaussian_img = smooth_img(raw_tmap_img, fwhm=fwhm)
            gaussian = np.asarray(gaussian_img.get_fdata(), dtype=float)
            gaussian[~domain_mask] = 0.0
            if save_images:
                gaussian_img = nib.Nifti1Image(gaussian, brain_mask_image.affine)
                nib.save(gaussian_img, gaussian_path)

        raw_metrics = _compute_method_metrics(gt_mask, domain_mask, raw_tmap_3d, pred_threshold_quantile)
        constrained_metrics = _compute_method_metrics(gt_mask, domain_mask, constrained, pred_threshold_quantile)
        gaussian_metrics = _compute_method_metrics(gt_mask, domain_mask, gaussian, pred_threshold_quantile)

        raw_active_wm, raw_active_gm, raw_active_other = _active_counts_by_tissue(
            raw_metrics["pred_mask"], wm_mask_flat, gm_mask_flat, other_mask_flat
        )
        constrained_active_wm, constrained_active_gm, constrained_active_other = _active_counts_by_tissue(
            constrained_metrics["pred_mask"], wm_mask_flat, gm_mask_flat, other_mask_flat
        )
        gaussian_active_wm, gaussian_active_gm, gaussian_active_other = _active_counts_by_tissue(
            gaussian_metrics["pred_mask"], wm_mask_flat, gm_mask_flat, other_mask_flat
        )

        row = {
            "label": int(label),
            "center_ijk": str(tuple(int(x) for x in center_ijk)),
            "fwhm_mm": float(fwhm),
            "amplitude": float(amplitude),
            "noise_std": float(noise_std),
            "pred_threshold_quantile": float(pred_threshold_quantile) if pred_threshold_quantile is not None else float("nan"),
            "raw_pred_threshold": float(raw_metrics["pred_thr"]),
            "constrained_pred_threshold": float(constrained_metrics["pred_thr"]),
            "gaussian_pred_threshold": float(gaussian_metrics["pred_thr"]),
            "raw_sensitivity": float(raw_metrics["sensitivity"]),
            "raw_specificity": float(raw_metrics["specificity"]),
            "raw_dice": float(raw_metrics["dice"]),
            "raw_roc_auc": float(raw_metrics["roc_auc"]),
            "raw_pr_auc": float(raw_metrics["pr_auc"]),
            "raw_pearson_r": float(raw_metrics["pearson_r"]),
            "raw_mae": float(raw_metrics["mae"]),
            "raw_tp": int(raw_metrics["tp"]),
            "raw_fp": int(raw_metrics["fp"]),
            "raw_tn": int(raw_metrics["tn"]),
            "raw_fn": int(raw_metrics["fn"]),
            "raw_active_wm": int(raw_active_wm),
            "raw_active_gm": int(raw_active_gm),
            "raw_active_other": int(raw_active_other),
            "raw_std_all": _std_in_mask(raw_tmap_3d, domain_mask),
            "raw_std_wm": _std_in_mask(raw_tmap_3d, wm_mask),
            "raw_std_gm": _std_in_mask(raw_tmap_3d, gm_mask),
            "raw_std_other": _std_in_mask(raw_tmap_3d, other_mask),
            "constrained_sensitivity": float(constrained_metrics["sensitivity"]),
            "constrained_specificity": float(constrained_metrics["specificity"]),
            "constrained_dice": float(constrained_metrics["dice"]),
            "constrained_roc_auc": float(constrained_metrics["roc_auc"]),
            "constrained_pr_auc": float(constrained_metrics["pr_auc"]),
            "constrained_pearson_r": float(constrained_metrics["pearson_r"]),
            "constrained_mae": float(constrained_metrics["mae"]),
            "constrained_tp": int(constrained_metrics["tp"]),
            "constrained_fp": int(constrained_metrics["fp"]),
            "constrained_tn": int(constrained_metrics["tn"]),
            "constrained_fn": int(constrained_metrics["fn"]),
            "constrained_active_wm": int(constrained_active_wm),
            "constrained_active_gm": int(constrained_active_gm),
            "constrained_active_other": int(constrained_active_other),
            "constrained_std_all": _std_in_mask(constrained, domain_mask),
            "constrained_std_wm": _std_in_mask(constrained, wm_mask),
            "constrained_std_gm": _std_in_mask(constrained, gm_mask),
            "constrained_std_other": _std_in_mask(constrained, other_mask),
            "gaussian_sensitivity": float(gaussian_metrics["sensitivity"]),
            "gaussian_specificity": float(gaussian_metrics["specificity"]),
            "gaussian_dice": float(gaussian_metrics["dice"]),
            "gaussian_roc_auc": float(gaussian_metrics["roc_auc"]),
            "gaussian_pr_auc": float(gaussian_metrics["pr_auc"]),
            "gaussian_pearson_r": float(gaussian_metrics["pearson_r"]),
            "gaussian_mae": float(gaussian_metrics["mae"]),
            "gaussian_tp": int(gaussian_metrics["tp"]),
            "gaussian_fp": int(gaussian_metrics["fp"]),
            "gaussian_tn": int(gaussian_metrics["tn"]),
            "gaussian_fn": int(gaussian_metrics["fn"]),
            "gaussian_active_wm": int(gaussian_active_wm),
            "gaussian_active_gm": int(gaussian_active_gm),
            "gaussian_active_other": int(gaussian_active_other),
            "gaussian_std_all": _std_in_mask(gaussian, domain_mask),
            "gaussian_std_wm": _std_in_mask(gaussian, wm_mask),
            "gaussian_std_gm": _std_in_mask(gaussian, gm_mask),
            "gaussian_std_other": _std_in_mask(gaussian, other_mask),
        }
        rows.append(row)

        if plot_outputs and t1_img is not None and plot_surfaces is not None and slices is not None:
            stat_paths = [gt_path, raw_path, constrained_path, gaussian_path]
            # compute per-map vmin/vmax for plotting color scale, but use the metric-computed
            # prediction thresholds (pred_thr) for contouring / binary overlays so that plots
            # match the threshold used to compute dice/sensitivity/etc.
            con_vmin, con_vmax, _ = _plot_value_range(constrained, domain_mask)
            gau_vmin, gau_vmax, _ = _plot_value_range(gaussian, domain_mask)

            # Use thresholds that were used to compute the positive predictions / metrics
            raw_thr = float(raw_metrics.get("pred_thr", float("nan")))
            con_thr = float(constrained_metrics.get("pred_thr", float("nan")))
            gau_thr = float(gaussian_metrics.get("pred_thr", float("nan")))

            raw_thr_for_plot = raw_thr if np.isfinite(raw_thr) else float("-inf")

            # For the combined multi-map plot, prefer the mean of available pred thresholds so
            # the shared contour roughly reflects the prediction threshold used by the metrics.
            thr_values = [v for v in (raw_thr, con_thr, gau_thr) if v is not None and not np.isnan(v)]
            if thr_values:
                shared_thr = float(np.mean(thr_values))
                shared_vmin, shared_vmax, _ = _plot_value_range_multi(
                    [gt_field, raw_tmap_3d, constrained, gaussian],
                    domain_mask,
                )
            else:
                # fallback to previous shared computation if pred thresholds aren't available
                shared_vmin, shared_vmax, shared_thr = _plot_value_range_multi(
                    [gt_field, raw_tmap_3d, constrained, gaussian],
                    domain_mask,
                )

            combo_fig = plot_multiple_stat_maps(
                mri_fname=t1_img.get_filename(),
                surfaces=plot_surfaces,
                stat_map_fnames=stat_paths,
                slices=[int(slices[len(slices) // 2])],
                orientation="axial",
                width=512,
                slices_as_subplots=True,
                stat_map_cmap="hot",
                stat_map_alpha=0.9,
                # Provide per-map thresholds so each map's contour reflects the
                # threshold used to compute metrics (pred_thr). The color scale
                # (vmin/vmax) is shared so constrained and gaussian use the same cmap range.
                stat_map_thresholds=[gt_thr_plot, raw_thr_for_plot, con_thr, gau_thr],
                mri_alpha=0.9,
                surface_alpha=0.9,
                stat_map_interpolation="nearest",
                surface_thickness=0.25,
                show=False,
                stat_map_vmin=shared_vmin,
                stat_map_vmax=shared_vmax,
                colorbar=True,
                crop_map_fname=gt_path,
                crop_stat_map_threshold=gt_thr_plot,
                crop_padding=10,
            )
            combo_png = os.path.join(plot_dir, f"combined_fwhm_{int(round(fwhm))}mm.png")
            combo_fig.savefig(combo_png, dpi=300)
            plt.close(combo_fig)

            for name, path, vmin, vmax, thr in [
                ("gt", gt_path, gt_vmin, gt_vmax, gt_thr_plot),
                ("raw", raw_path, raw_vmin, raw_vmax, raw_thr_for_plot),
                ("constrained", constrained_path, con_vmin, con_vmax, con_thr),
                ("gaussian", gaussian_path, gau_vmin, gau_vmax, gau_thr),
            ]:
                fig = plot_mri_with_contours(
                    mri_fname=t1_img.get_filename(),
                    surfaces=plot_surfaces,
                    orientation="axial",
                    slices=slices,
                    show=False,
                    slices_as_subplots=True,
                    stat_map_fname=path,
                    stat_map_cmap="hot",
                    stat_map_alpha=0.9,
                    stat_map_threshold=thr,
                    stat_map_vmin=vmin,
                    stat_map_vmax=vmax,
                    surface_thickness=0.25,
                    colorbar=True,
                    crop_bounds=crop_bounds,
                )
                out_png = os.path.join(plot_dir, f"{name}_fwhm_{int(round(fwhm))}mm.png")
                fig.savefig(out_png, dpi=300)
                plt.close(fig)

            # Track combined inputs for the final multi-map figure
            combined_maps = [gt_field, raw_tmap_3d, constrained, gaussian]
            combined_paths = [gt_path, raw_path, constrained_path, gaussian_path]
            combined_thresholds = [gt_thr_plot, raw_thr_for_plot, con_thr, gau_thr]

    if multi_fwhm and plot_outputs and t1_img is not None and plot_surfaces is not None and slices is not None and len(combined_paths) > 2:
        combined_vmin, combined_vmax, _ = _plot_value_range_multi(combined_maps, domain_mask)
        combined_fig = plot_multiple_stat_maps(
            mri_fname=t1_img.get_filename(),
            surfaces=plot_surfaces,
            stat_map_fnames=combined_paths,
            slices=[int(slices[len(slices) // 2])],
            orientation="axial",
            width=1024,
            slices_as_subplots=True,
            stat_map_cmap="hot",
            stat_map_alpha=0.9,
            stat_map_thresholds=combined_thresholds,
            mri_alpha=0.9,
            surface_alpha=0.9,
            stat_map_interpolation="nearest",
            surface_thickness=0.25,
            show=False,
            stat_map_vmin=combined_vmin,
            stat_map_vmax=combined_vmax,
            colorbar=True,
            crop_map_fname=combined_paths[0],
            crop_stat_map_threshold=gt_thr_plot,
            crop_padding=10,
        )
        combo_all_png = os.path.join(plot_dir, "combined_all_fwhm.png")
        combined_fig.savefig(combo_all_png, dpi=300)
        plt.close(combined_fig)

    return rows, optimal_taus_by_fwhm


def run_constrained_vs_gauss_experiment(
    aparc_file: str,
    dseg_file: str,
    brain_mask_file: str,
    t1w_file: str,
    pial_l_file: str,
    pial_r_file: str,
    white_l_file: str,
    white_r_file: str,
    fwhm_list: list[float],
    amplitude: float,
    noise_std: float,
    output_dir: str,
    label_region: int | None = None,
    random_seed: int | None = None,
    pred_threshold_quantile: float | None = None,
    overwrite_volumes: bool = False,
    save_images: bool = True,
    plot_outputs: bool = False,
) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)

    aparc_img = nib.load(aparc_file)
    dseg_img = nib.load(dseg_file)
    brain_mask_image = nib.load(brain_mask_file)
    t1_img = nib.load(t1w_file) if t1w_file else None

    mask_array = process_mask(brain_mask_file, brain_mask_image, mask_dilation=3).astype(bool)

    dseg_res = resample_to_img(dseg_img, brain_mask_image, interpolation="nearest", force_resample=True)
    dseg_data = np.asarray(dseg_res.get_fdata(), dtype=np.int32)
    gm_mask = np.isin(dseg_data, (3, 42)) & mask_array
    wm_mask = np.isin(dseg_data, (2, 41)) & mask_array
    other_mask = mask_array & (dseg_data == 0)

    surface_files = [pial_l_file, pial_r_file, white_l_file, white_r_file]
    edge_src, edge_dst, edge_distances = create_graph(mask_array, brain_mask_image.affine, surface_files)
    labels, sorted_labels, unique_nodes = identify_connected_components(edge_src, edge_dst)

    labelmap_path = os.path.join(output_dir, "labelmap.nii.gz")
    if save_images:
        save_labelmap(labelmap_path, brain_mask_image.shape, brain_mask_image.affine, labels, sorted_labels, unique_nodes)

    if label_region is not None:
        label_regions = [int(label_region)]
    else:
        aparc_data = np.asarray(aparc_img.get_fdata(), dtype=np.int32)
        labels_unique = np.unique(aparc_data)
        label_regions = [int(lbl) for lbl in labels_unique if 1000 <= int(lbl) <= 2999]
        print(f"Running constrained vs Gaussian experiment on {len(label_regions)} aparc labels...")

    rows: list[dict] = []
    optimal_taus_by_fwhm: dict[float, list[float]] = {}

    for idx, label in enumerate(label_regions):
        center_ijk = _label_center_ijk(aparc_img, int(label))
        region_dir = os.path.join(output_dir, f"label_{int(label)}")
        seed = None if random_seed is None else int(random_seed) + idx

        region_rows, optimal_taus_by_fwhm = run_constrained_vs_gauss_experiment_single_region(
            aparc_img=aparc_img,
            brain_mask_image=brain_mask_image,
            t1_img=t1_img,
            surface_files=surface_files,
            label=int(label),
            center_ijk=center_ijk,
            fwhm_list=fwhm_list,
            amplitude=amplitude,
            noise_std=noise_std,
            output_dir=region_dir,
            mask_array=mask_array,
            gm_mask=gm_mask,
            wm_mask=wm_mask,
            other_mask=other_mask,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_distances=edge_distances,
            labels=labels,
            sorted_labels=sorted_labels,
            unique_nodes=unique_nodes,
            optimal_taus_by_fwhm=optimal_taus_by_fwhm,
            random_seed=seed,
            pred_threshold_quantile=pred_threshold_quantile,
            overwrite_volumes=overwrite_volumes,
            save_images=save_images,
            plot_outputs=plot_outputs,
        )
        rows.extend(region_rows)

    csv_path = os.path.join(output_dir, "constrained_vs_gauss_summary.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run constrained vs Gaussian smoothing simulation")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Amplitude of GT within the label region")
    parser.add_argument("--noise-std", type=float, default=1.0, help="Std-dev of added Gaussian noise")
    parser.add_argument("--label-region", type=int, default=None,
                        help="Aparc label for active region; if unset, all 1000-2999 labels are run")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for sampling noise")
    parser.add_argument(
        "--fwhm-list",
        type=float,
        nargs="+",
        default=[6.0],
        help="List of target FWHM values in mm",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=None,
        help="Quantile over brain-mask voxels for defining active voxels. If omitted, the code will use 1 - prevalence (prevalence computed from the GT within the brain mask)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (defaults under this script folder)",
    )
    parser.add_argument("--save-plots", action="store_true", help="Whether to save comparison plots")
    parser.add_argument("--no-save-volumes", action="store_true",
                        help="Whether to skip saving NIfTI outputs and plots")
    parser.add_argument(
        "--overwrite-volumes",
        action="store_true",
        help="If set, recompute and overwrite any existing output volumes (raw/constrained/gaussian). Otherwise existing volumes are loaded.",
    )
    return parser.parse_args()


def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    t1w_file = os.path.join(file_dir, "data/sub-MSC06_desc-preproc_T1w.nii.gz")
    aparc_file = os.path.join(file_dir, "data/sub-MSC06_desc-aparcaseg_dseg.nii.gz")
    dseg_file = os.path.join(file_dir, "data/sub-MSC06_desc-aseg_dseg.nii.gz")
    brain_mask_file = os.path.join(file_dir, "data/sub-MSC06_desc-brain_mask.nii.gz")
    pial_l_file = os.path.join(file_dir, "data/sub-MSC06_hemi-L_pial.surf.gii")
    pial_r_file = os.path.join(file_dir, "data/sub-MSC06_hemi-R_pial.surf.gii")
    white_l_file = os.path.join(file_dir, "data/sub-MSC06_hemi-L_white.surf.gii")
    white_r_file = os.path.join(file_dir, "data/sub-MSC06_hemi-R_white.surf.gii")

    args = parse_args()

    output_dir = args.output_dir or os.path.join(file_dir, "./constrained_vs_gauss_outputs")

    run_constrained_vs_gauss_experiment(
        aparc_file=aparc_file,
        dseg_file=dseg_file,
        brain_mask_file=brain_mask_file,
        t1w_file=t1w_file,
        pial_l_file=pial_l_file,
        pial_r_file=pial_r_file,
        white_l_file=white_l_file,
        white_r_file=white_r_file,
        fwhm_list=[float(v) for v in args.fwhm_list],
        amplitude=float(args.amplitude),
        noise_std=float(args.noise_std),
        output_dir=output_dir,
        label_region=args.label_region,
        random_seed=args.random_seed,
        pred_threshold_quantile=args.threshold_quantile,
        overwrite_volumes=bool(args.overwrite_volumes),
        save_images=not args.no_save_volumes,
        plot_outputs=bool(args.save_plots) and not args.no_save_volumes,
    )


if __name__ == "__main__":
    main()
