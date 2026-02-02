"""paper/simulations/fwhm_estimation_validation.py

Validate graph-based FWHM estimation on smoothed 3D noise volumes.
Workflow:
1) generate 3D Gaussian noise within the brain mask
2) smooth with nilearn Gaussian kernels at FWHM 1..15 mm
3) estimate FWHM on two graphs: "baseline" (unpruned 26-neighbor) and "constrained" (surface-pruned)
4) report GM/WM subsets and save CSV + plot of target vs estimated FWHM
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img, smooth_img

from csmooth.affine import adjust_affine_spacing, resample_data_to_affine
from csmooth.components import identify_connected_components
from csmooth.fwhm import estimate_fwhm
from csmooth.graph import (
    compute_edge_coordinates,
    compute_edge_distances,
    compute_edge_real_world_coordinates,
    mask2graph,
)
from paper.simulations.graph_connection_voxelsize_effect import GraphData, _build_graph_for_voxel_size


def _paths(base_dir: str) -> dict[str, str]:
    """Shared dataset paths reused from constrained_vs_gauss."""
    return {
        "t1": os.path.join(base_dir, "data/sub-MSC06_desc-preproc_T1w.nii.gz"),
        "aparc": os.path.join(base_dir, "data/sub-MSC06_desc-aparcaseg_dseg.nii.gz"),
        "dseg": os.path.join(base_dir, "data/sub-MSC06_desc-aseg_dseg.nii.gz"),
        "brain_mask": os.path.join(base_dir, "data/sub-MSC06_desc-brain_mask.nii.gz"),
        "pial_l": os.path.join(base_dir, "data/sub-MSC06_hemi-L_pial.surf.gii"),
        "pial_r": os.path.join(base_dir, "data/sub-MSC06_hemi-R_pial.surf.gii"),
        "white_l": os.path.join(base_dir, "data/sub-MSC06_hemi-L_white.surf.gii"),
        "white_r": os.path.join(base_dir, "data/sub-MSC06_hemi-R_white.surf.gii"),
    }


def _build_graph(brain_mask_img: nib.Nifti1Image, surfaces: tuple[str, str, str, str]) -> GraphData:
    """Return constrained 1mm graph using standard surface-based pruning."""
    return _build_graph_for_voxel_size(brain_mask_img=brain_mask_img, voxel_size_mm=1.0, surfaces=surfaces)


def _build_unpruned_graph(mask_array: np.ndarray, affine: np.ndarray, voxel_size_mm: float) -> GraphData:
    """Return unpruned 26-neighbor graph on the provided mask/affine."""
    edge_src, edge_dst = mask2graph(mask_array.astype(int))
    edge_src_3d, edge_dst_3d = compute_edge_coordinates(edge_src, edge_dst, mask_array.shape)
    edge_src_xyz, edge_dst_xyz = compute_edge_real_world_coordinates(edge_src_3d, edge_dst_3d, affine)
    edge_distances = compute_edge_distances(edge_src_xyz, edge_dst_xyz)
    labels, sorted_labels, unique_nodes = identify_connected_components(edge_src, edge_dst)
    return GraphData(
        voxel_size_mm=float(voxel_size_mm),
        affine=np.asarray(affine),
        mask_3d=np.asarray(mask_array).astype(bool),
        unique_nodes=np.asarray(unique_nodes).astype(int),
        edge_src=np.asarray(edge_src).astype(int),
        edge_dst=np.asarray(edge_dst).astype(int),
        edge_distances=np.asarray(edge_distances).astype(float),
        labels=np.asarray(labels).astype(int),
        sorted_labels=np.asarray(sorted_labels).astype(int),
    )


def _resample_tissue_masks(dseg_img: nib.Nifti1Image, target_mask: np.ndarray, target_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_img = nib.Nifti1Image(target_mask.astype(np.int16), target_affine)
    dseg_res = resample_to_img(dseg_img, target_img, interpolation="nearest", force_resample=True)
    dseg_data = np.asarray(dseg_res.get_fdata(), dtype=np.int32)
    gm_mask = np.isin(dseg_data, (3, 42)) & target_mask
    wm_mask = np.isin(dseg_data, (2, 41)) & target_mask
    return gm_mask, wm_mask


def _estimate_on_graph(graph: GraphData, smoothed_img: nib.Nifti1Image, node_mask: Optional[np.ndarray]) -> tuple[float, int, int]:
    data = np.asarray(smoothed_img.get_fdata(), dtype=float)
    signal_flat = data.ravel()
    base_mask = graph.mask_3d.ravel().astype(bool)
    mask_flat = base_mask if node_mask is None else (base_mask & node_mask.ravel())
    finite_mask = np.isfinite(signal_flat)
    mask_flat &= finite_mask
    if not np.any(mask_flat):
        return float("nan"), 0, 0

    nodes = np.flatnonzero(mask_flat)
    node_id = {node: i for i, node in enumerate(nodes)}

    edge_mask = mask_flat[graph.edge_src] & mask_flat[graph.edge_dst]
    if not np.any(edge_mask):
        return float("nan"), int(nodes.size), 0

    sub_src = graph.edge_src[edge_mask]
    sub_dst = graph.edge_dst[edge_mask]
    sub_dist = graph.edge_distances[edge_mask]

    mapped_src = np.fromiter((node_id[s] for s in sub_src), dtype=int)
    mapped_dst = np.fromiter((node_id[d] for d in sub_dst), dtype=int)
    signal_vec = signal_flat[nodes]
    try:
        fwhm = estimate_fwhm(edge_src=mapped_src, edge_dst=mapped_dst, edge_distances=sub_dist, signal_data=signal_vec)
    except Exception:
        fwhm = float("nan")
    return float(fwhm), int(nodes.size), int(sub_src.size)


def _plot_results(rows: list[dict], out_png: str) -> None:
    plt.figure(figsize=(8, 6))
    methods = sorted({(r["method"], r["tissue"]) for r in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 1)))
    for color, (method, tissue) in zip(colors, methods):
        xs = [r["fwhm_target"] for r in rows if r["method"] == method and r["tissue"] == tissue]
        ys = [r["fwhm_estimate"] for r in rows if r["method"] == method and r["tissue"] == tissue]
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        plt.plot(xs, ys, marker="o", label=f"{method} {tissue}", color=color)
    max_fwhm = max([r["fwhm_target"] for r in rows]) if rows else 0
    plt.plot([0, max_fwhm], [0, max_fwhm], linestyle="--", color="black", label="identity")
    plt.xlabel("Target FWHM (mm)")
    plt.ylabel("Estimated FWHM (mm)")
    plt.title("FWHM estimation on Gaussian-smoothed noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def run_simulation(fwhm_values: Iterable[float], output_dir: str, random_seed: Optional[int]) -> tuple[list[dict], str, str]:
    os.makedirs(output_dir, exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = _paths(base_dir)

    brain_mask_img = nib.load(paths["brain_mask"])
    zooms = brain_mask_img.header.get_zooms()[:3]
    if not np.allclose(zooms, (1.0, 1.0, 1.0), atol=1e-3):
        target_affine = adjust_affine_spacing(brain_mask_img.affine, np.array([1.0, 1.0, 1.0]))
        mask_resampled = resample_data_to_affine(
            data=brain_mask_img.get_fdata(),
            target_affine=target_affine,
            original_affine=brain_mask_img.affine,
            interpolation="continuous",
        )
        brain_mask_img = nib.Nifti1Image(mask_resampled, target_affine)

    dseg_img = nib.load(paths["dseg"])

    g_constrained = _build_graph(
        brain_mask_img=brain_mask_img,
        surfaces=(paths["pial_l"], paths["pial_r"], paths["white_l"], paths["white_r"]),
    )

    mask_array = g_constrained.mask_3d.astype(bool)

    g_baseline = _build_unpruned_graph(mask_array=mask_array, affine=g_constrained.affine, voxel_size_mm=g_constrained.voxel_size_mm)

    gm_mask, wm_mask = _resample_tissue_masks(dseg_img=dseg_img, target_mask=mask_array, target_affine=g_baseline.affine)

    rng = np.random.default_rng(seed=random_seed)
    noise_data = rng.normal(loc=0.0, scale=1.0, size=mask_array.shape)
    noise_data[~mask_array] = 0.0
    noise_img = nib.Nifti1Image(noise_data, g_baseline.affine)

    rows: list[dict] = []
    for fwhm in fwhm_values:
        smoothed = smooth_img(noise_img, fwhm=float(fwhm))

        for method_name, graph in (("baseline", g_baseline), ("constrained", g_constrained)):
            for tissue_name, tissue_mask in (
                ("all", None),
                ("gm", gm_mask),
                ("wm", wm_mask),
            ):
                f_est, n_nodes, n_edges = _estimate_on_graph(graph=graph, smoothed_img=smoothed, node_mask=tissue_mask)
                rows.append(
                    {
                        "fwhm_target": float(fwhm),
                        "method": method_name,
                        "tissue": tissue_name,
                        "fwhm_estimate": float(f_est),
                        "n_nodes": int(n_nodes),
                        "n_edges": int(n_edges),
                    }
                )

    csv_path = os.path.join(output_dir, "fwhm_estimation_validation.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    plot_path = os.path.join(output_dir, "fwhm_estimation_validation.png")
    if rows:
        _plot_results(rows, plot_path)
    return rows, csv_path, plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FWHM estimation on Gaussian-smoothed noise")
    parser.add_argument("--fwhm-min", type=float, default=1.0, help="Minimum FWHM (inclusive)")
    parser.add_argument("--fwhm-max", type=float, default=15.0, help="Maximum FWHM (inclusive)")
    parser.add_argument("--fwhm-step", type=float, default=1.0, help="FWHM step")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for CSV and plot")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise generation")
    # graph pruning uses the standard 1mm grid; no parent voxel graph is built
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(base_dir, "./fwhm_estimation_validation_outputs")
    fwhm_values = np.arange(float(args.fwhm_min), float(args.fwhm_max) + 1e-3, float(args.fwhm_step))
    rows, csv_path, plot_path = run_simulation(
        fwhm_values=fwhm_values,
        output_dir=output_dir,
        random_seed=args.seed,
    )
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
