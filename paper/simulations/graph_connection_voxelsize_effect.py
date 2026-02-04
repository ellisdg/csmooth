"""paper/simulations/graph_connection_voxelsize_effect.py

Experiment: does the *voxel size used to construct graph connections* change the
*effective smoothing* (in GM), even when smoothing happens on a 1mm grid?

We compare two graphs defined on the same 1mm node set:

1) Baseline: native 1mm graph connections (whatever `csmooth.graph.create_graph` produces)
2) Pruned: same 1mm nodes, but edges are pruned using connectivity from a 3mm “parent” graph.

Parent mapping:
- Each 1mm node is assigned to a 3mm parent node by nearest voxel-center in mm space.
- An edge (u,v) in the 1mm graph is kept iff:
    - parent(u) == parent(v)  (intra-parent edges always allowed)
      OR
    - there is an edge between parent(u) and parent(v) in the 3mm graph.

Then we simulate pure noise on the 1mm grid and smooth it on:
- baseline 1mm graph
- pruned 1mm graph

Primary outcome:
- GM standard deviation after smoothing (per connected GM component, and GM-weighted average)

Notes on validity:
- This isolates *graph topology* effects while holding the node locations/resolution fixed.
- It does NOT attempt to model acquisition downsampling; it’s a targeted ablation.
- Using pure noise is appropriate here because the output variance is directly tied to
  the smoothing operator’s energy/spread.

"""

from __future__ import annotations

import os
import csv
import argparse
from dataclasses import dataclass

import numpy as np
import nibabel as nib

from csmooth.graph import create_graph
from csmooth.components import identify_connected_components
from csmooth.smooth import select_nodes, find_optimal_tau
from csmooth.heat import heat_kernel_smoothing
from csmooth.affine import adjust_affine_spacing, resample_data_to_affine
from csmooth.fwhm import estimate_fwhm


@dataclass(frozen=True)
class GraphData:
    voxel_size_mm: float
    affine: np.ndarray
    mask_3d: np.ndarray
    unique_nodes: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_distances: np.ndarray
    labels: np.ndarray
    sorted_labels: np.ndarray


def _voxel_centers_mm(shape3: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    ijk = np.indices(shape3).reshape(3, -1).T.astype(np.float64)
    ijk_h = np.c_[ijk, np.ones((ijk.shape[0], 1), dtype=np.float64)]
    xyz = (affine @ ijk_h.T).T[:, :3]
    return xyz


def _build_graph_for_voxel_size(
    brain_mask_img: nib.Nifti1Image,
    voxel_size_mm: float,
    surfaces: tuple[str, str, str, str] | None,
) -> GraphData:
    """Build graph at a given voxel size using the same core pipeline as the main simulation."""

    orig_affine = brain_mask_img.affine.copy()
    target_affine = adjust_affine_spacing(orig_affine, float(voxel_size_mm))

    # resample the mask data array to the target affine
    mask_array = brain_mask_img.get_fdata()
    mask_array = resample_data_to_affine(data=mask_array,
                                         target_affine=target_affine,
                                         original_affine=orig_affine,
                                         interpolation="continuous")
    mask_array = mask_array > 0.5
    affine = target_affine

    # `create_graph` depends on surface files in the main simulation; keep that pathway.
    if surfaces is None:
        raise ValueError("surfaces must be provided (pial/white) to build the graph")

    pial_l_file, pial_r_file, white_l_file, white_r_file = surfaces
    edge_src, edge_dst, edge_distances = create_graph(
        mask_array=mask_array,
        image_affine=affine,
        surface_files=(pial_l_file, pial_r_file, white_l_file, white_r_file)
    )

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
        sorted_labels=np.asarray(sorted_labels).astype(int)
    )


def _build_parent_mapping_1mm_to_3mm(g1: GraphData, g3: GraphData) -> np.ndarray:
    """Return array parent_of_node sized (max_node_id_1mm+1) with parent node ids (3mm) or -1."""

    max1 = int(max(g1.edge_src.max(initial=0), g1.edge_dst.max(initial=0), g1.unique_nodes.max(initial=0)))
    parent = np.full(max1 + 1, -1, dtype=int)

    # voxel centers for both grids
    xyz1 = _voxel_centers_mm(tuple(int(x) for x in g1.mask_3d.shape), g1.affine)
    xyz3 = _voxel_centers_mm(tuple(int(x) for x in g3.mask_3d.shape), g3.affine)

    # Only consider nodes that exist in the graphs (mask voxels)
    nodes1 = g1.unique_nodes.astype(int)
    nodes3 = g3.unique_nodes.astype(int)

    xyz1_nodes = xyz1[nodes1]
    xyz3_nodes = xyz3[nodes3]

    # If there are no 3mm nodes, leave parent mapping as all -1
    if xyz3_nodes.size == 0:
        return parent

    # For 3mm nodes, build a nearest-neighbor lookup.
    # Avoid scipy dependency: brute force is okay for this targeted script, but we can do chunking.
    # Complexity: O(N1*N3). On typical masks this can be heavy; we chunk across 1mm nodes.
    # If this becomes a bottleneck, we can switch to sklearn NearestNeighbors or scipy.spatial.cKDTree.
    chunk = 5000
    for i0 in range(0, xyz1_nodes.shape[0], chunk):
        i1 = min(i0 + chunk, xyz1_nodes.shape[0])
        x = xyz1_nodes[i0:i1]
        # squared distances to all 3mm nodes
        d2 = np.sum((x[:, None, :] - xyz3_nodes[None, :, :]) ** 2, axis=2)
        nn = np.argmin(d2, axis=1)
        parent[nodes1[i0:i1]] = nodes3[nn]

    return parent


def _parent_adjacency_set(g3: GraphData) -> set[tuple[int, int]]:
    # store undirected pairs (min,max)
    s: set[tuple[int, int]] = set()
    for u, v in zip(g3.edge_src.astype(int), g3.edge_dst.astype(int)):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        s.add((a, b))
    return s


def prune_1mm_edges_using_3mm_connectivity(g1: GraphData, g3: GraphData, parent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pruned (edge_src, edge_dst, edge_distances) on 1mm node set."""

    parent_adj = _parent_adjacency_set(g3)

    src = g1.edge_src.astype(int)
    dst = g1.edge_dst.astype(int)

    psrc = parent[src]
    pdst = parent[dst]

    # If either endpoint has no parent mapping, drop
    ok = (psrc >= 0) & (pdst >= 0)

    # Keep intra-parent edges
    keep = ok & (psrc == pdst)

    # Keep edges whose parents are adjacent in 3mm
    cand = ok & (psrc != pdst)
    if np.any(cand):
        a = np.minimum(psrc[cand], pdst[cand])
        b = np.maximum(psrc[cand], pdst[cand])
        # membership check against the parent adjacency set (iterative but clear)
        keep_adj = np.zeros(a.shape, dtype=bool)
        for i_idx, (ai, bi) in enumerate(zip(a.tolist(), b.tolist())):
            keep_adj[i_idx] = (int(ai), int(bi)) in parent_adj
        keep[cand] = keep_adj

    pr_src = src[keep]
    pr_dst = dst[keep]
    pr_dist = g1.edge_distances[keep]

    return pr_src, pr_dst, pr_dist


def _gm_components_from_probseg(
    g: GraphData,
    gm_prob_img: nib.Nifti1Image,
    gm_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gm_component_labels, gm_nodes) using component-average GM probability."""
    from nilearn.image import resample_img

    gm_res = resample_img(gm_prob_img, target_affine=g.affine, target_shape=g.mask_3d.shape, interpolation="continuous")
    gm_prob = np.asarray(gm_res.get_fdata(), dtype=float).ravel()

    gm_component_labels = []
    gm_nodes = []
    for lbl in np.unique(g.labels):
        nodes = g.unique_nodes[g.labels == lbl]
        if nodes.size == 0:
            continue
        comp_probs = gm_prob[nodes]
        if np.mean(comp_probs[np.isfinite(comp_probs)]) > float(gm_threshold):
            gm_component_labels.append(int(lbl))
            gm_nodes.append(nodes)

    if gm_nodes:
        gm_nodes_arr = np.concatenate(gm_nodes)
    else:
        gm_nodes_arr = np.array([], dtype=int)

    return np.asarray(gm_component_labels, dtype=int), gm_nodes_arr


def _estimate_tau_for_fwhm(g: GraphData, fwhm_mm: float, n_components: int = 5) -> dict[int, float]:

    tau_by_label: dict[int, float] = {}
    for lbl in g.sorted_labels[:n_components]:
        _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(
            edge_src=g.edge_src,
            edge_dst=g.edge_dst,
            edge_distances=g.edge_distances,
            labels=g.labels,
            label=lbl,
            unique_nodes=g.unique_nodes,
        )
        tau = find_optimal_tau(
            fwhm=fwhm_mm,
            edge_src=_edge_src,
            edge_dst=_edge_dst,
            edge_distances=_edge_distances,
            shape=(len(_nodes),)
        )
        tau_by_label[int(lbl)] = float(tau)

    # fill remaining labels with average tau (consistent with grid_resolution_effect.py)
    if tau_by_label:
        default_tau = float(np.mean(list(tau_by_label.values())))
    else:
        default_tau = float("nan")

    for lbl in np.unique(g.labels):
        if int(lbl) not in tau_by_label:
            tau_by_label[int(lbl)] = default_tau

    return tau_by_label


def _smooth_noise_on_graph(g: GraphData, tau_by_label: dict[int, float], noise: np.ndarray) -> np.ndarray:
    if noise.ndim != 1:
        raise ValueError(f"Expected 1D noise vector, got shape={noise.shape}")

    sm = noise.copy()

    # smooth each component separately (as in the main script)
    for lbl in np.unique(g.labels):
        lbl_int = int(lbl)
        nodes = g.unique_nodes[g.labels == lbl]
        if nodes.size == 0:
            continue
        tau = float(tau_by_label.get(lbl_int, float("nan")))
        if not np.isfinite(tau):
            continue

        # select and renumber edges/nodes for this component (local indexing)
        _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(
            edge_src=g.edge_src,
            edge_dst=g.edge_dst,
            edge_distances=g.edge_distances,
            labels=g.labels,
            label=lbl_int,
            unique_nodes=g.unique_nodes,
        )

        if _nodes.size == 0:
            continue

        # extract local 1D signal for the nodes in this component
        local_signal = sm[_nodes]

        # apply heat kernel smoothing which expects a 1D signal for single-timepoint
        smoothed_local = heat_kernel_smoothing(
            edge_src=_edge_src,
            edge_dst=_edge_dst,
            edge_distances=_edge_distances,
            signal_data=local_signal,
            tau=tau,
        )

        # smoothed_local should be same shape as local_signal; write back into global vector
        sm[_nodes] = smoothed_local

    return sm


def _estimate_fwhm_for_top_components(
    g: GraphData,
    smoothed: np.ndarray,
    n_components: int = 5,
) -> list[tuple[int, float]]:
    results: list[tuple[int, float]] = []
    if smoothed.ndim != 1:
        raise ValueError(f"Expected 1D smoothed vector, got shape={smoothed.shape}")

    for lbl in g.sorted_labels[:n_components]:
        _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(
            edge_src=g.edge_src,
            edge_dst=g.edge_dst,
            edge_distances=g.edge_distances,
            labels=g.labels,
            label=lbl,
            unique_nodes=g.unique_nodes,
        )
        if _nodes.size < 2 or _edge_src.size == 0:
            results.append((int(lbl), float("nan")))
            continue

        comp_signal = smoothed[_nodes]
        finite_mask = np.isfinite(comp_signal)
        if np.sum(finite_mask) < 2:
            results.append((int(lbl), float("nan")))
            continue

        edge_mask = finite_mask[_edge_src] & finite_mask[_edge_dst]
        if not np.any(edge_mask):
            results.append((int(lbl), float("nan")))
            continue

        try:
            fwhm = estimate_fwhm(
                edge_src=_edge_src[edge_mask],
                edge_dst=_edge_dst[edge_mask],
                edge_distances=_edge_distances[edge_mask],
                signal_data=comp_signal,
            )
        except Exception:
            fwhm = float("nan")
        results.append((int(lbl), float(fwhm)))

    return results


def _gm_std_summary(values: np.ndarray, gm_component_labels: np.ndarray, labels: np.ndarray, unique_nodes: np.ndarray) -> float:
    """Return weighted mean std across GM components (weights = component size)."""
    if unique_nodes is None or unique_nodes.size == 0:
        return float("nan")

    weights = []
    stds = []

    for lbl in gm_component_labels:
        nodes = unique_nodes[labels == lbl]
        if nodes.size == 0:
            continue
        v = values[nodes]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        weights.append(int(nodes.size))
        stds.append(float(np.std(v)))

    if weights:
        w = np.asarray(weights, dtype=float)
        sarr = np.asarray(stds, dtype=float)
        return float(np.sum(w * sarr) / np.sum(w))

    return float("nan")


def _scenario_result(
    graph_name: str,
    g_smooth: GraphData,
    tau_by_label: dict[int, float],
    gm_component_labels: np.ndarray,
    eval_labels: np.ndarray,
    eval_unique_nodes: np.ndarray,
    noise: np.ndarray,
    n_fwhm_components: int = 5,
) -> tuple[dict, float, list[tuple[int, float]]]:
    """Smooth noise on a graph and return scenario-level metrics."""
    smoothed = _smooth_noise_on_graph(g_smooth, tau_by_label, noise)
    gm_weighted_std = _gm_std_summary(smoothed, gm_component_labels, eval_labels, eval_unique_nodes)
    fwhm_top = _estimate_fwhm_for_top_components(g_smooth, smoothed, n_components=n_fwhm_components)
    tau_vals = np.asarray([v for v in tau_by_label.values() if np.isfinite(v)], dtype=float)
    scenario = {
        "graph": graph_name,
        "graph_voxel_size_mm": float(g_smooth.voxel_size_mm),
        "graph_n_nodes": int(g_smooth.unique_nodes.size),
        "graph_n_edges": int(g_smooth.edge_src.size),
        "gm_weighted_std": float(gm_weighted_std),
        "tau_mean": float(np.mean(tau_vals)) if tau_vals.size else float("nan"),
        "tau_std": float(np.std(tau_vals)) if tau_vals.size else float("nan"),
    }
    return scenario, gm_weighted_std, fwhm_top


def plot_axial_comparison(
    panels: list[tuple[str, np.ndarray, GraphData]],
    surfaces: tuple[str, str, str, str] | None = None,
    slice_index: int | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (12.0, 4.0),
) -> "matplotlib.figure.Figure":
    """Plot side-by-side axial slices of smoothed images with optional surface overlays.

    panels: list of (title, smoothed_flat_array, GraphData). Each smoothed_flat_array should be a
            1D array that can be reshaped to the graph's mask shape (ravel order).
    surfaces: optional tuple of 4 surface file paths (pial_l, pial_r, white_l, white_r).
    slice_index: axial slice index (k). If None, uses the center axial slice of the first panel.

    Returns a matplotlib Figure.
    """
    # Local imports to avoid forcing heavy deps at module import time
    import matplotlib.pyplot as plt
    import nibabel as nib

    if not panels:
        raise ValueError("panels must contain at least one (title, array, GraphData) tuple")

    # determine slice index from first graph if not provided
    _, arr0, g0 = panels[0]
    vol_shape = tuple(int(x) for x in g0.mask_3d.shape)
    if slice_index is None:
        slice_index = int(vol_shape[2] // 2)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    # prepare surface vertices
    surf_vertices: list[np.ndarray | None] = [None, None, None, None]
    if surfaces is not None:
        for i, surf_path in enumerate(surfaces):
            if surf_path is None:
                surf_vertices[i] = None
                continue
            try:
                gi = nib.load(surf_path)
                # Gifti: vertices often in .darrays[0].data
                v = None
                if hasattr(gi, "darrays") and getattr(gi, "darrays"):
                    try:
                        v = np.asarray(gi.darrays[0].data, dtype=float)
                    except Exception:
                        v = None
                # fallback: some readers expose .get_arrays_from_intent
                if v is None:
                    try:
                        arrs = gi.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
                        if arrs:
                            v = np.asarray(arrs[0].data, dtype=float)
                    except Exception:
                        v = None
                surf_vertices[i] = v
            except Exception:
                surf_vertices[i] = None

    im = None
    for ax, (title, smoothed_flat, g) in zip(axes, panels):
        # reshape to volume and get axial slice
        vol = np.asarray(smoothed_flat, dtype=float).reshape(tuple(int(x) for x in g.mask_3d.shape))
        slice_img = vol[:, :, slice_index]

        # rotate for a more conventional radiological-ish view
        im = ax.imshow(
            np.rot90(slice_img),
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_axis_off()

        # overlay surface vertices projected into voxel space for this graph affine
        inv_aff = np.linalg.inv(np.asarray(g.affine))
        for v in surf_vertices:
            if v is None:
                continue
            try:
                hom = np.c_[v, np.ones((v.shape[0], 1), dtype=float)]
                ijk = (inv_aff @ hom.T).T[:, :3]
                k_idx = np.round(ijk[:, 2]).astype(int)
                mask_k = k_idx == int(slice_index)
                if not np.any(mask_k):
                    continue
                xs = ijk[mask_k, 0]
                ys = ijk[mask_k, 1]
                # imshow used np.rot90 -> map coordinates accordingly
                w = g.mask_3d.shape[0]
                plotted_x = ys
                plotted_y = (w - 1 - xs)
                ax.scatter(plotted_x, plotted_y, s=1.5, c="r", alpha=0.8, linewidths=0)
            except Exception:
                # skip any surface on failure
                continue

    # add a colorbar if we plotted anything
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
        cbar.set_label("smoothed value")
    fig.tight_layout()
    return fig


def run_experiment(
    t1w_file: str,
    brain_mask_file: str,
    gm_prob_file: str,
    wm_prob_file: str,
    pial_l_file: str,
    pial_r_file: str,
    white_l_file: str,
    white_r_file: str,
    fwhm_mm: float,
    noise_std: float,
    random_seed: int | None,
    output_csv: str,
    gm_threshold: float = 0.5,
    wm_threshold: float = 0.5,
    tau_fixed: float = 13.0,
):
    rng = np.random.default_rng(None if random_seed is None else int(random_seed))

    brain_mask_img = nib.load(brain_mask_file)
    gm_prob_img = nib.load(gm_prob_file)
    wm_prob_img = nib.load(wm_prob_file)

    surfaces = (pial_l_file, pial_r_file, white_l_file, white_r_file)

    g3 = _build_graph_for_voxel_size(brain_mask_img, voxel_size_mm=3.0, surfaces=surfaces)
    g1 = _build_graph_for_voxel_size(brain_mask_img, voxel_size_mm=1.0, surfaces=surfaces)

    parent = _build_parent_mapping_1mm_to_3mm(g1, g3)
    pr_src, pr_dst, pr_dist = prune_1mm_edges_using_3mm_connectivity(g1, g3, parent)

    if pr_src.size == 0:
        labels_p = np.array([], dtype=int)
        sorted_labels_p = np.array([], dtype=int)
        unique_nodes_p = np.array([], dtype=int)
    else:
        labels_p, sorted_labels_p, unique_nodes_p = identify_connected_components(pr_src, pr_dst)

    # Build a "pruned" graph object with same nodes/mask/affine as 1mm
    g1p = GraphData(
        voxel_size_mm=g1.voxel_size_mm,
        affine=g1.affine,
        mask_3d=g1.mask_3d,
        unique_nodes=unique_nodes_p,
        edge_src=pr_src,
        edge_dst=pr_dst,
        edge_distances=pr_dist,
        labels=labels_p,
        sorted_labels=sorted_labels_p,
    )

    # Noise lives on the 1mm grid
    nvox = int(np.prod(g1.mask_3d.shape))
    noise = rng.normal(0.0, float(noise_std), size=(nvox,)).astype(float)
    noise[~g1.mask_3d.ravel()] = np.nan

    # Compute tau (separately) for baseline and pruned graphs
    tau1_est = _estimate_tau_for_fwhm(g1, fwhm_mm=float(fwhm_mm))
    tau1p_est = _estimate_tau_for_fwhm(g1p, fwhm_mm=float(fwhm_mm))

    # Fixed tau maps reuse the same value for all components
    tau1_fixed = {int(lbl): float(tau_fixed) for lbl in np.unique(g1.labels)}
    tau1p_fixed = {int(lbl): float(tau_fixed) for lbl in np.unique(g1p.labels)}

    gm_component_labels, _ = _gm_components_from_probseg(g1, gm_prob_img, gm_threshold=float(gm_threshold))
    wm_component_labels, _ = _gm_components_from_probseg(g1, wm_prob_img, gm_threshold=float(wm_threshold))

    baseline, gm_std_1, fwhm_top_1 = _scenario_result(
        graph_name="original",
        g_smooth=g1,
        tau_by_label=tau1_est,
        gm_component_labels=gm_component_labels,
        eval_labels=g1.labels,
        eval_unique_nodes=g1.unique_nodes,
        noise=noise,
    )
    pruned, gm_std_1p, fwhm_top_1p = _scenario_result(
        graph_name="pruned",
        g_smooth=g1p,
        tau_by_label=tau1p_est,
        gm_component_labels=gm_component_labels,
        eval_labels=g1.labels,
        eval_unique_nodes=g1.unique_nodes,
        noise=noise,
    )
    mixed, gm_std_mixed, fwhm_top_mixed = _scenario_result(
        graph_name="pruned_original_tau",
        g_smooth=g1p,
        tau_by_label=tau1_est,
        gm_component_labels=gm_component_labels,
        eval_labels=g1.labels,
        eval_unique_nodes=g1.unique_nodes,
        noise=noise,
    )

    # Fixed-tau smoothing on graphs (baseline/pruned) using supplied tau_fixed
    baseline_fixed, gm_std_1_fixed, fwhm_top_1_fixed = _scenario_result(
        graph_name="original",
        g_smooth=g1,
        tau_by_label=tau1_fixed,
        gm_component_labels=gm_component_labels,
        eval_labels=g1.labels,
        eval_unique_nodes=g1.unique_nodes,
        noise=noise,
    )
    pruned_fixed, gm_std_1p_fixed, fwhm_top_1p_fixed = _scenario_result(
        graph_name="pruned",
        g_smooth=g1p,
        tau_by_label=tau1p_fixed,
        gm_component_labels=gm_component_labels,
        eval_labels=g1.labels,
        eval_unique_nodes=g1.unique_nodes,
        noise=noise,
    )

    # Assemble CSV with one row per scenario
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = [
        "experiment",
        "graph",
        "fwhm_mm",
        "tau_fixed",
        "noise_std",
        "random_seed",
        "gm_threshold",
        "wm_threshold",
        "graph_voxel_size_mm",
        "graph_n_nodes",
        "graph_n_edges",
        "n_edges_1mm",
        "n_edges_1mm_pruned",
        "gm_weighted_std",
        "gm_weighted_fwhm",
        "wm_weighted_std",
        "wm_weighted_fwhm",
        "gm_std_ratio_to_original",
        "tau_mean",
        "tau_std",
    ]
    for i in range(1, 6):
        fieldnames.append(f"fwhm_top{i}_label")
        fieldnames.append(f"fwhm_top{i}_value")

    rows = []
    def _component_weighted_fwhm(g: GraphData, smoothed: np.ndarray, component_labels: np.ndarray) -> float:
        if component_labels.size == 0:
            return float("nan")
        fwhms = []
        weights = []
        for lbl in component_labels:
            _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(
                edge_src=g.edge_src,
                edge_dst=g.edge_dst,
                edge_distances=g.edge_distances,
                labels=g.labels,
                label=int(lbl),
                unique_nodes=g.unique_nodes,
            )
            if _nodes.size < 2 or _edge_src.size == 0:
                continue
            comp_signal = smoothed[_nodes]
            finite_mask = np.isfinite(comp_signal)
            edge_mask = finite_mask[_edge_src] & finite_mask[_edge_dst]
            if not np.any(edge_mask):
                continue
            try:
                fwhm_val = estimate_fwhm(
                    edge_src=_edge_src[edge_mask],
                    edge_dst=_edge_dst[edge_mask],
                    edge_distances=_edge_distances[edge_mask],
                    signal_data=comp_signal,
                )
            except Exception:
                continue
            fwhms.append(float(fwhm_val))
            weights.append(int(_nodes.size))
        if weights:
            w = np.asarray(weights, dtype=float)
            f = np.asarray(fwhms, dtype=float)
            return float(np.sum(w * f) / np.sum(w))
        return float("nan")

    scenarios = [
        ("estimated_fwhm", baseline, tau1_est, gm_std_1, fwhm_top_1, g1, noise),
        ("estimated_fwhm", pruned, tau1p_est, gm_std_1p, fwhm_top_1p, g1p, noise),
        ("estimated_fwhm", mixed, tau1_est, gm_std_mixed, fwhm_top_mixed, g1p, noise),
        ("fixed_tau", baseline_fixed, tau1_fixed, gm_std_1_fixed, fwhm_top_1_fixed, g1, noise),
        ("fixed_tau", pruned_fixed, tau1p_fixed, gm_std_1p_fixed, fwhm_top_1p_fixed, g1p, noise),
    ]

    # Prepare containers to hold panels for plotting per experiment type
    plot_panels: dict[str, list[tuple[str, np.ndarray, GraphData]]] = {"estimated_fwhm": [], "fixed_tau": []}

    for experiment, scenario, tau_map, gm_std_val, fwhm_top, graph_obj, smoothed_noise in scenarios:
        # derive GM/WM weighted FWHM and WM weighted std
        smoothed_signal = _smooth_noise_on_graph(graph_obj, tau_map, smoothed_noise)
        gm_fwhm_weighted = _component_weighted_fwhm(graph_obj, smoothed_signal, gm_component_labels)
        wm_fwhm_weighted = _component_weighted_fwhm(graph_obj, smoothed_signal, wm_component_labels)
        wm_std_weighted = _gm_std_summary(smoothed_signal, wm_component_labels, graph_obj.labels, graph_obj.unique_nodes)

        # store panel (title, flattened signal, graph) for later plotting
        panel_title = f"{scenario['graph']}_{experiment}"
        plot_panels.setdefault(experiment, []).append((panel_title, smoothed_signal, graph_obj))

        row = {
            "experiment": experiment,
            "graph": scenario["graph"],
            "fwhm_mm": float(fwhm_mm),
            "tau_fixed": float(tau_fixed),
            "noise_std": float(noise_std),
            "random_seed": ("" if random_seed is None else int(random_seed)),
            "gm_threshold": float(gm_threshold),
            "wm_threshold": float(wm_threshold),
            "graph_voxel_size_mm": scenario["graph_voxel_size_mm"],
            "graph_n_nodes": scenario["graph_n_nodes"],
            "graph_n_edges": scenario["graph_n_edges"],
            "n_edges_1mm": int(g1.edge_src.size),
            "n_edges_1mm_pruned": int(pr_src.size),
            "gm_weighted_std": gm_std_val,
            "gm_weighted_fwhm": gm_fwhm_weighted,
            "wm_weighted_std": wm_std_weighted,
            "wm_weighted_fwhm": wm_fwhm_weighted,
            "gm_std_ratio_to_original": float(
                gm_std_val / gm_std_1
            ) if np.isfinite(gm_std_1) and gm_std_1 != 0 else float("nan"),
            "tau_mean": scenario["tau_mean"],
            "tau_std": scenario["tau_std"],
        }
        for i in range(1, 6):
            lbl = fwhm_top[i - 1][0] if len(fwhm_top) >= i else ""
            val = fwhm_top[i - 1][1] if len(fwhm_top) >= i else float("nan")
            row[f"fwhm_top{i}_label"] = lbl
            row[f"fwhm_top{i}_value"] = float(val) if val != "" else float("nan")
        rows.append(row)

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # After assembling rows, create comparison plots per experiment type
    try:
        for exp_type, panels in plot_panels.items():
            if not panels:
                continue
            try:
                fig = plot_axial_comparison(panels, surfaces=surfaces)
                out_png = os.path.join(os.path.dirname(output_csv), f"comparison_{exp_type}.png")
                # Use tight bbox and dpi for publication-quality
                fig.savefig(out_png, dpi=300, bbox_inches="tight")
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass
            except Exception:
                # don't fail the entire experiment run if plotting fails; continue
                continue
    except Exception:
        # defensive: ignore plotting issues
        pass

    return rows


def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(file_dir, "data")

    parser = argparse.ArgumentParser(description="Graph connection voxel-size ablation (1mm nodes, 3mm connectivity pruning)")
    parser.add_argument("--fwhm", type=float, default=6.0)
    parser.add_argument("--noise-std", type=float, default=2.0)
    parser.add_argument("--gm-threshold", type=float, default=0.5)
    parser.add_argument("--wm-threshold", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--tau-fixed", type=float, default=13.0)
    parser.add_argument("--output-csv", type=str, default=os.path.join(file_dir, "graph_connection_voxelsize_effect.csv"))

    args = parser.parse_args()

    t1w_file = os.path.join(data_dir, "sub-MSC06_desc-preproc_T1w.nii.gz")
    brain_mask_file = os.path.join(data_dir, "sub-MSC06_desc-brain_mask.nii.gz")
    gm_prob_file = os.path.join(data_dir, "sub-MSC06_label-GM_probseg.nii.gz")
    wm_prob_file = os.path.join(data_dir, "sub-MSC06_label-WM_probseg.nii.gz")
    pial_l_file = os.path.join(data_dir, "sub-MSC06_hemi-L_pial.surf.gii")
    pial_r_file = os.path.join(data_dir, "sub-MSC06_hemi-R_pial.surf.gii")
    white_l_file = os.path.join(data_dir, "sub-MSC06_hemi-L_white.surf.gii")
    white_r_file = os.path.join(data_dir, "sub-MSC06_hemi-R_white.surf.gii")

    run_experiment(
        t1w_file=t1w_file,
        brain_mask_file=brain_mask_file,
        gm_prob_file=gm_prob_file,
        wm_prob_file=wm_prob_file,
        pial_l_file=pial_l_file,
        pial_r_file=pial_r_file,
        white_l_file=white_l_file,
        white_r_file=white_r_file,
        fwhm_mm=float(args.fwhm),
        noise_std=float(args.noise_std),
        random_seed=None if args.random_seed is None else int(args.random_seed),
        output_csv=str(args.output_csv),
        gm_threshold=float(args.gm_threshold),
        wm_threshold=float(args.wm_threshold),
        tau_fixed=float(args.tau_fixed),
    )


if __name__ == "__main__":
    main()

