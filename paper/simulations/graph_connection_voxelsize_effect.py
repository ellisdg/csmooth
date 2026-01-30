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
from csmooth.smooth import process_mask, select_nodes, find_optimal_tau, smooth_component
from csmooth.affine import adjust_affine_spacing


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

    target_affine = adjust_affine_spacing(brain_mask_img.affine, float(voxel_size_mm))

    # We resample the mask using `process_mask` which is what the main script does.
    mask_3d, affine = process_mask(
        brain_mask_img=brain_mask_img,
        target_affine=target_affine,
        resample_interpolation="nearest",
    )

    # `create_graph` depends on surface files in the main simulation; keep that pathway.
    if surfaces is None:
        raise ValueError("surfaces must be provided (pial/white) to build the graph")

    pial_l_file, pial_r_file, white_l_file, white_r_file = surfaces
    edge_src, edge_dst, edge_distances = create_graph(
        mask_3d=mask_3d,
        affine=affine,
        pial_l_file=pial_l_file,
        pial_r_file=pial_r_file,
        white_l_file=white_l_file,
        white_r_file=white_r_file,
    )

    unique_nodes = select_nodes(mask_3d)
    # Connected components on the graph
    labels, _ = identify_connected_components(unique_nodes, edge_src, edge_dst)

    return GraphData(
        voxel_size_mm=float(voxel_size_mm),
        affine=np.asarray(affine),
        mask_3d=np.asarray(mask_3d).astype(bool),
        unique_nodes=np.asarray(unique_nodes).astype(int),
        edge_src=np.asarray(edge_src).astype(int),
        edge_dst=np.asarray(edge_dst).astype(int),
        edge_distances=np.asarray(edge_distances).astype(float),
        labels=np.asarray(labels).astype(int),
    )


def _build_parent_mapping_1mm_to_3mm(g1: GraphData, g3: GraphData) -> np.ndarray:
    """Return array parent_of_node sized (max_node_id_1mm+1) with parent node ids (3mm) or -1."""

    max1 = int(max(g1.edge_src.max(initial=0), g1.edge_dst.max(initial=0), g1.unique_nodes.max(initial=0)))
    parent = np.full(max1 + 1, -1, dtype=int)

    # voxel centers for both grids
    xyz1 = _voxel_centers_mm(tuple(g1.mask_3d.shape), g1.affine)
    xyz3 = _voxel_centers_mm(tuple(g3.mask_3d.shape), g3.affine)

    # Only consider nodes that exist in the graphs (mask voxels)
    nodes1 = g1.unique_nodes.astype(int)
    nodes3 = g3.unique_nodes.astype(int)

    xyz1_nodes = xyz1[nodes1]
    xyz3_nodes = xyz3[nodes3]

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
        # vectorized membership check by converting to structured array and using np.isin
        pairs = np.core.records.fromarrays([a, b], names="a,b", formats="i4,i4")
        parent_pairs = np.array(list(parent_adj), dtype=[("a", "i4"), ("b", "i4")])
        keep_adj = np.isin(pairs, parent_pairs)
        keep[cand] = keep_adj

    pr_src = src[keep]
    pr_dst = dst[keep]
    pr_dist = g1.edge_distances[keep]

    return pr_src, pr_dst, pr_dist


def _gm_nodes_from_probseg(g: GraphData, gm_prob_img: nib.Nifti1Image, gm_threshold: float = 0.2) -> np.ndarray:
    # resample gm_prob to the graph grid
    from nilearn.image import resample_img

    gm_res = resample_img(gm_prob_img, target_affine=g.affine, target_shape=g.mask_3d.shape, interpolation="continuous")
    gm_prob = np.asarray(gm_res.get_fdata(), dtype=float)

    gm_mask = (gm_prob >= float(gm_threshold)) & g.mask_3d
    gm_nodes = select_nodes(gm_mask)
    return np.asarray(gm_nodes).astype(int)


def _estimate_tau_for_fwhm(g: GraphData, fwhm_mm: float, n_components: int = 5) -> dict[int, float]:
    # identify component labels (biggest first)
    unique_labels, counts = np.unique(g.labels, return_counts=True)
    order = np.argsort(-counts)
    top_labels = unique_labels[order][: int(n_components)]

    tau_by_label: dict[int, float] = {}
    for lbl in top_labels:
        comp_nodes = g.unique_nodes[g.labels == lbl]
        tau = find_optimal_tau(
            component_nodes=comp_nodes,
            edge_src=g.edge_src,
            edge_dst=g.edge_dst,
            edge_distances=g.edge_distances,
            fwhm_mm=float(fwhm_mm),
        )
        tau_by_label[int(lbl)] = float(tau)

    # fill remaining labels with average tau (consistent with grid_resolution_effect.py)
    if tau_by_label:
        default_tau = float(np.mean(list(tau_by_label.values())))
    else:
        default_tau = float("nan")

    for lbl in unique_labels:
        if int(lbl) not in tau_by_label:
            tau_by_label[int(lbl)] = default_tau

    return tau_by_label


def _smooth_noise_on_graph(g: GraphData, tau_by_label: dict[int, float], noise: np.ndarray) -> np.ndarray:
    sm = np.full(noise.shape, np.nan, dtype=float)

    # smooth each component separately (as in the main script)
    for lbl in np.unique(g.labels):
        lbl_int = int(lbl)
        nodes = g.unique_nodes[g.labels == lbl]
        if nodes.size == 0:
            continue
        tau = float(tau_by_label.get(lbl_int, float("nan")))
        if not np.isfinite(tau):
            continue

        out = smooth_component(
            component_nodes=nodes,
            edge_src=g.edge_src,
            edge_dst=g.edge_dst,
            edge_distances=g.edge_distances,
            values=noise,
            tau=tau,
        )
        sm[nodes] = out[nodes]

    return sm


def _gm_std_summary(values: np.ndarray, gm_nodes: np.ndarray, labels: np.ndarray, unique_nodes: np.ndarray) -> tuple[float, dict[int, float]]:
    """Return (weighted mean std across GM components, per-component std)."""

    gm_set = set(gm_nodes.tolist())
    gm_label_ids = []
    for lbl in np.unique(labels):
        nodes = unique_nodes[labels == lbl]
        if nodes.size == 0:
            continue
        # if any node in this component is GM, treat as GM component
        if any(int(n) in gm_set for n in nodes[: min(nodes.size, 1000)]):
            # cheap check; refined below
            pass

    # Identify GM components precisely
    gm_node_mask = np.isin(unique_nodes, gm_nodes)
    gm_component_labels = np.unique(labels[gm_node_mask]) if np.any(gm_node_mask) else np.array([], dtype=int)

    per: dict[int, float] = {}
    weights = []
    stds = []

    for lbl in gm_component_labels:
        nodes = unique_nodes[labels == lbl]
        v = values[nodes]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        s = float(np.std(v))
        per[int(lbl)] = s
        weights.append(int(nodes.size))
        stds.append(s)

    if weights:
        w = np.asarray(weights, dtype=float)
        sarr = np.asarray(stds, dtype=float)
        weighted = float(np.sum(w * sarr) / np.sum(w))
    else:
        weighted = float("nan")

    return weighted, per


def run_experiment(
    t1w_file: str,
    brain_mask_file: str,
    gm_prob_file: str,
    pial_l_file: str,
    pial_r_file: str,
    white_l_file: str,
    white_r_file: str,
    fwhm_mm: float,
    noise_std: float,
    random_seed: int | None,
    output_csv: str,
    gm_threshold: float = 0.2,
):
    rng = np.random.default_rng(None if random_seed is None else int(random_seed))

    brain_mask_img = nib.load(brain_mask_file)
    gm_prob_img = nib.load(gm_prob_file)

    surfaces = (pial_l_file, pial_r_file, white_l_file, white_r_file)

    g3 = _build_graph_for_voxel_size(brain_mask_img, voxel_size_mm=3.0, surfaces=surfaces)
    g1 = _build_graph_for_voxel_size(brain_mask_img, voxel_size_mm=1.0, surfaces=surfaces)

    parent = _build_parent_mapping_1mm_to_3mm(g1, g3)
    pr_src, pr_dst, pr_dist = prune_1mm_edges_using_3mm_connectivity(g1, g3, parent)

    # Build a "pruned" graph object with same nodes/mask/affine as 1mm
    g1p = GraphData(
        voxel_size_mm=g1.voxel_size_mm,
        affine=g1.affine,
        mask_3d=g1.mask_3d,
        unique_nodes=g1.unique_nodes,
        edge_src=pr_src,
        edge_dst=pr_dst,
        edge_distances=pr_dist,
        labels=identify_connected_components(g1.unique_nodes, pr_src, pr_dst)[0].astype(int),
    )

    # Noise lives on the 1mm grid
    nvox = int(np.prod(g1.mask_3d.shape))
    noise = rng.normal(0.0, float(noise_std), size=(nvox,)).astype(float)
    noise[~g1.mask_3d.ravel()] = np.nan

    # Compute tau (separately) for baseline and pruned graphs
    tau1 = _estimate_tau_for_fwhm(g1, fwhm_mm=float(fwhm_mm))
    tau1p = _estimate_tau_for_fwhm(g1p, fwhm_mm=float(fwhm_mm))

    sm1 = _smooth_noise_on_graph(g1, tau1, noise)
    sm1p = _smooth_noise_on_graph(g1p, tau1p, noise)

    gm_nodes = _gm_nodes_from_probseg(g1, gm_prob_img, gm_threshold=float(gm_threshold))

    gm_std_1, gm_std_per_1 = _gm_std_summary(sm1, gm_nodes, g1.labels, g1.unique_nodes)
    gm_std_1p, gm_std_per_1p = _gm_std_summary(sm1p, gm_nodes, g1p.labels, g1p.unique_nodes)

    # Write one-row CSV + optional per-component breakdown columns
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    all_labels = sorted(set(gm_std_per_1.keys()) | set(gm_std_per_1p.keys()))
    fieldnames = [
        "fwhm_mm",
        "noise_std",
        "random_seed",
        "gm_threshold",
        "n_nodes_1mm",
        "n_edges_1mm",
        "n_edges_1mm_pruned",
        "gm_weighted_std_1mm",
        "gm_weighted_std_1mm_pruned",
        "gm_std_ratio_pruned_over_baseline",
    ]
    for lbl in all_labels:
        fieldnames.append(f"gm_std_comp_{lbl}_1mm")
        fieldnames.append(f"gm_std_comp_{lbl}_1mm_pruned")

    row = {
        "fwhm_mm": float(fwhm_mm),
        "noise_std": float(noise_std),
        "random_seed": ("" if random_seed is None else int(random_seed)),
        "gm_threshold": float(gm_threshold),
        "n_nodes_1mm": int(g1.unique_nodes.size),
        "n_edges_1mm": int(g1.edge_src.size),
        "n_edges_1mm_pruned": int(pr_src.size),
        "gm_weighted_std_1mm": float(gm_std_1),
        "gm_weighted_std_1mm_pruned": float(gm_std_1p),
        "gm_std_ratio_pruned_over_baseline": float(gm_std_1p / gm_std_1) if np.isfinite(gm_std_1) and gm_std_1 != 0 else float("nan"),
    }
    for lbl in all_labels:
        row[f"gm_std_comp_{lbl}_1mm"] = float(gm_std_per_1.get(lbl, float("nan")))
        row[f"gm_std_comp_{lbl}_1mm_pruned"] = float(gm_std_per_1p.get(lbl, float("nan")))

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)

    return row


def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(file_dir, "data")

    parser = argparse.ArgumentParser(description="Graph connection voxel-size ablation (1mm nodes, 3mm connectivity pruning)")
    parser.add_argument("--fwhm", type=float, default=6.0)
    parser.add_argument("--noise-std", type=float, default=2.0)
    parser.add_argument("--gm-threshold", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-csv", type=str, default=os.path.join(file_dir, "graph_connection_voxelsize_effect.csv"))

    args = parser.parse_args()

    t1w_file = os.path.join(data_dir, "sub-MSC06_desc-preproc_T1w.nii.gz")
    brain_mask_file = os.path.join(data_dir, "sub-MSC06_desc-brain_mask.nii.gz")
    gm_prob_file = os.path.join(data_dir, "sub-MSC06_label-GM_probseg.nii.gz")
    pial_l_file = os.path.join(data_dir, "sub-MSC06_hemi-L_pial.surf.gii")
    pial_r_file = os.path.join(data_dir, "sub-MSC06_hemi-R_pial.surf.gii")
    white_l_file = os.path.join(data_dir, "sub-MSC06_hemi-L_white.surf.gii")
    white_r_file = os.path.join(data_dir, "sub-MSC06_hemi-R_white.surf.gii")

    run_experiment(
        t1w_file=t1w_file,
        brain_mask_file=brain_mask_file,
        gm_prob_file=gm_prob_file,
        pial_l_file=pial_l_file,
        pial_r_file=pial_r_file,
        white_l_file=white_l_file,
        white_r_file=white_r_file,
        fwhm_mm=float(args.fwhm),
        noise_std=float(args.noise_std),
        random_seed=None if args.random_seed is None else int(args.random_seed),
        output_csv=str(args.output_csv),
        gm_threshold=float(args.gm_threshold),
    )


if __name__ == "__main__":
    main()

