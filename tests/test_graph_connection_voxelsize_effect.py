import numpy as np
import pytest

from paper.simulations.graph_connection_voxelsize_effect import (
    GraphData,
    _voxel_centers_mm,
    _build_parent_mapping_1mm_to_3mm,
    _parent_adjacency_set,
    prune_1mm_edges_using_3mm_connectivity,
)
import paper.simulations.graph_connection_voxelsize_effect as gve


def test_voxel_centers_mm_identity_affine():
    # 2x1x1 grid with identity affine -> centers at (0,0,0) and (1,0,0)
    shape = (2, 1, 1)
    affine = np.eye(4)
    centers = _voxel_centers_mm(shape, affine)
    assert centers.shape == (2, 3)
    assert np.allclose(centers[0], [0.0, 0.0, 0.0])
    assert np.allclose(centers[1], [1.0, 0.0, 0.0])


def test_voxel_centers_mm_non_identity_affine():
    # scaling x by 2 -> centers at 0 and 2
    shape = (2, 1, 1)
    affine = np.diag([2.0, 1.0, 1.0, 1.0])
    centers = _voxel_centers_mm(shape, affine)
    assert centers.shape == (2, 3)
    assert np.allclose(centers[0], [0.0, 0.0, 0.0])
    assert np.allclose(centers[1], [2.0, 0.0, 0.0])


def _make_graphdata(nodes, edges, affine, shape):
    """Helper to create a minimal GraphData for tests.
    nodes: list/array of node indices present
    edges: list of (src, dst) pairs
    affine: 4x4 affine
    shape: 3-tuple for mask shape
    """
    nodes = np.asarray(nodes, dtype=int)
    if len(edges) == 0:
        edge_src = np.array([], dtype=int)
        edge_dst = np.array([], dtype=int)
        edge_distances = np.array([], dtype=float)
    else:
        edge_src = np.asarray([e[0] for e in edges], dtype=int)
        edge_dst = np.asarray([e[1] for e in edges], dtype=int)
        # dummy distances
        edge_distances = np.ones(edge_src.shape, dtype=float)

    # labels: simple single component labeling where each unique node maps to label 0
    labels = np.zeros(int(nodes.max(initial=0) + 1), dtype=int)
    sorted_labels = np.array([0], dtype=int)

    mask_3d = np.zeros(shape, dtype=bool)
    # Mark all voxels as present so unique node ids refer to indices into flattened array
    mask_3d.ravel()[:] = False
    # For safety, if nodes correspond to flattened indices within shape, set them True
    for n in nodes:
        if 0 <= n < mask_3d.size:
            mask_3d.ravel()[n] = True

    return GraphData(
        voxel_size_mm=1.0,
        affine=np.asarray(affine),
        mask_3d=mask_3d,
        unique_nodes=nodes,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_distances=edge_distances,
        labels=labels,
        sorted_labels=sorted_labels,
    )


def test_build_parent_mapping_simple():
    # g1: two nodes at x=0 and x=1 (shape (2,1,1))
    affine1 = np.eye(4)
    g1 = _make_graphdata(nodes=[0, 1], edges=[], affine=affine1, shape=(2, 1, 1))

    # g3: single node at x=0 (shape (1,1,1))
    affine3 = np.eye(4) * 1.0
    g3 = _make_graphdata(nodes=[0], edges=[], affine=affine3, shape=(1, 1, 1))

    parent = _build_parent_mapping_1mm_to_3mm(g1, g3)
    # parent array should have length max1+1 == 2 and both map to node 0
    assert parent.shape[0] >= 2
    assert parent[0] == 0
    assert parent[1] == 0


def test_build_parent_mapping_scaled_affine():
    # g1 nodes at 0,1,2 (identity affine)
    affine1 = np.eye(4)
    g1 = _make_graphdata(nodes=[0, 1, 2], edges=[], affine=affine1, shape=(3, 1, 1))

    # g3 nodes with affine scaling on x so centers are at 0 and 2
    affine3 = np.diag([2.0, 1.0, 1.0, 1.0])
    g3 = _make_graphdata(nodes=[0, 1], edges=[], affine=affine3, shape=(2, 1, 1))

    parent = _build_parent_mapping_1mm_to_3mm(g1, g3)
    # Expect mapping: node0->0, node1->0 (tie goes to first), node2->1
    assert parent[0] == 0
    assert parent[1] == 0
    assert parent[2] == 1


def test_parent_adjacency_set_and_prune():
    # Build a 1mm graph with 4 nodes and three edges: (0,1), (1,2), (2,3)
    affine = np.eye(4)
    g1 = _make_graphdata(nodes=[0, 1, 2, 3], edges=[(0, 1), (1, 2), (2, 3)], affine=affine, shape=(4, 1, 1))

    # Build a 3mm graph whose nodes will be parents: let's pick parent node ids 10 and 11
    # For parent adjacency tests we just need edges between parent ids
    g3_with_adj = _make_graphdata(nodes=[10, 11], edges=[(10, 11)], affine=affine, shape=(12, 1, 1))
    g3_without_adj = _make_graphdata(nodes=[10, 11], edges=[], affine=affine, shape=(12, 1, 1))

    # Create a parent mapping manually: nodes 0,1 -> 10 ; nodes 2,3 -> 11
    max1 = 3
    parent = np.full(max1 + 1, -1, dtype=int)
    parent[0] = 10
    parent[1] = 10
    parent[2] = 11
    parent[3] = 11

    # If parent adjacency set does not include (10,11), prune should only keep intra-parent edges
    pr_src, pr_dst, pr_dist = prune_1mm_edges_using_3mm_connectivity(g1, g3_without_adj, parent)
    # Expect kept edges: (0,1) and (2,3) only
    kept = set(zip(pr_src.tolist(), pr_dst.tolist()))
    assert (0, 1) in kept
    assert (2, 3) in kept
    assert (1, 2) not in kept

    # If parent adjacency set includes (10,11), prune should keep the cross-parent edge too
    pr_src2, pr_dst2, pr_dist2 = prune_1mm_edges_using_3mm_connectivity(g1, g3_with_adj, parent)
    kept2 = set(zip(pr_src2.tolist(), pr_dst2.tolist()))
    assert (0, 1) in kept2
    assert (2, 3) in kept2
    assert (1, 2) in kept2


def test_parent_adjacency_set_ignores_selfloops_and_duplicates():
    affine = np.eye(4)
    # Create g3 with duplicate/reverse edges and a self-loop
    e = [(10, 11), (11, 10), (10, 10)]
    g3 = _make_graphdata(nodes=[10, 11], edges=e, affine=affine, shape=(12, 1, 1))
    s = _parent_adjacency_set(g3)
    # Should contain single undirected pair (10,11) and ignore self-loop
    assert (10, 11) in s
    assert (11, 10) not in s
    assert (10, 10) not in s
    assert len(s) == 1


def test_prune_with_selfloops_and_reverse_edges():
    affine = np.eye(4)
    # 1mm graph with nodes 0 and 1 and edges including reverse and self-loop
    g1 = _make_graphdata(nodes=[0, 1], edges=[(0, 1), (1, 0), (1, 1)], affine=affine, shape=(2, 1, 1))
    # 3mm graph adjacency present between 10 and 11
    g3 = _make_graphdata(nodes=[10, 11], edges=[(10, 11)], affine=affine, shape=(12, 1, 1))
    parent = np.full(2, -1, dtype=int)
    parent[0] = 10
    parent[1] = 11

    pr_src, pr_dst, pr_dist = prune_1mm_edges_using_3mm_connectivity(g1, g3, parent)
    kept = set(zip(pr_src.tolist(), pr_dst.tolist()))
    # All edges should be kept: (0,1), (1,0) cross-parent, and (1,1) intra-parent
    assert (0, 1) in kept
    assert (1, 0) in kept
    assert (1, 1) in kept


def test_prune_handles_missing_parents():
    # Edge where one node has no parent (-1) should be dropped
    affine = np.eye(4)
    g1 = _make_graphdata(nodes=[0, 1], edges=[(0, 1)], affine=affine, shape=(2, 1, 1))
    g3 = _make_graphdata(nodes=[10], edges=[], affine=affine, shape=(11, 1, 1))
    parent = np.full(2, -1, dtype=int)
    parent[0] = 10
    parent[1] = -1

    pr_src, pr_dst, pr_dist = prune_1mm_edges_using_3mm_connectivity(g1, g3, parent)
    # No edges should be kept because second node has no parent
    assert pr_src.size == 0
    assert pr_dst.size == 0
    assert pr_dist.size == 0


def test_build_parent_mapping_with_no_parents():
    # g1 has nodes but g3 has no nodes -> all parents should be -1
    affine1 = np.eye(4)
    g1 = _make_graphdata(nodes=[0, 1], edges=[], affine=affine1, shape=(2, 1, 1))
    g3 = _make_graphdata(nodes=[], edges=[], affine=affine1, shape=(0, 1, 1))
    parent = _build_parent_mapping_1mm_to_3mm(g1, g3)
    assert parent.shape[0] >= 2
    assert np.all(parent[:2] == -1)


def test_prune_with_no_edges_returns_empty():
    affine = np.eye(4)
    g1 = _make_graphdata(nodes=[0, 1], edges=[], affine=affine, shape=(2, 1, 1))
    g3 = _make_graphdata(nodes=[10], edges=[], affine=affine, shape=(11, 1, 1))
    parent = np.full(2, 10, dtype=int)
    pr_src, pr_dst, pr_dist = prune_1mm_edges_using_3mm_connectivity(g1, g3, parent)
    assert pr_src.size == 0
    assert pr_dst.size == 0
    assert pr_dist.size == 0


def test_smooth_noise_on_graph_applies_component_smoothing(monkeypatch):
    # Build simple GraphData: 3 nodes, edges between 0-1 only; two components: {0,1}, {2}
    unique_nodes = np.array([0, 1, 2], dtype=int)
    edge_src = np.array([0], dtype=int)
    edge_dst = np.array([1], dtype=int)
    edge_dist = np.array([1.0], dtype=float)
    labels = np.array([0, 0, 1], dtype=int)  # per-unique-node labels
    sorted_labels = np.array([0, 1], dtype=int)
    mask_3d = np.zeros((3, 1, 1), dtype=bool)
    mask_3d.ravel()[:] = True

    g = GraphData(
        voxel_size_mm=1.0,
        affine=np.eye(4),
        mask_3d=mask_3d,
        unique_nodes=unique_nodes,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_distances=edge_dist,
        labels=labels,
        sorted_labels=sorted_labels,
    )

    # noise vector length = nvox (mask size product)
    noise = np.array([10.0, 20.0, 30.0], dtype=float)

    # monkeypatch heat_kernel_smoothing to add 5 to each local value
    def fake_heat(edge_src, edge_dst, edge_distances, signal_data, tau):
        return signal_data + 5.0

    monkeypatch.setattr(gve, "heat_kernel_smoothing", fake_heat)

    tau_map = {0: 1.0, 1: float("nan")}  # only label 0 smoothing active
    out = gve._smooth_noise_on_graph(g, tau_map, noise)
    # nodes 0 and 1 should have been increased by +5, node 2 unchanged
    assert out[0] == pytest.approx(15.0)
    assert out[1] == pytest.approx(25.0)
    assert out[2] == pytest.approx(30.0)


def test_estimate_fwhm_for_top_components_happy_path(monkeypatch):
    # Create a bare GraphData with one sorted label
    unique_nodes = np.array([0, 1], dtype=int)
    edge_src = np.array([0], dtype=int)
    edge_dst = np.array([1], dtype=int)
    edge_dist = np.array([1.0], dtype=float)
    labels = np.array([0, 0], dtype=int)
    sorted_labels = np.array([0], dtype=int)
    g = GraphData(1.0, np.eye(4), np.ones((2, 1, 1), dtype=bool), unique_nodes, edge_src, edge_dst, edge_dist, labels, sorted_labels)

    # fake select_nodes to return local arrays
    def fake_select_nodes(edge_src, edge_dst, edge_distances, labels, label, unique_nodes):
        return np.array([0]), np.array([1]), np.array([1.0]), np.array([0, 1])

    monkeypatch.setattr(gve, "select_nodes", fake_select_nodes)

    # fake estimate_fwhm to return predictable number
    monkeypatch.setattr(gve, "estimate_fwhm", lambda edge_src, edge_dst, edge_distances, signal_data: 3.14)

    smoothed = np.array([0.5, 1.5], dtype=float)
    res = gve._estimate_fwhm_for_top_components(g, smoothed, n_components=1)
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0][0] == 0
    assert res[0][1] == pytest.approx(3.14)

    # shape check: passing non-1D should raise
    with pytest.raises(ValueError):
        gve._estimate_fwhm_for_top_components(g, np.zeros((2, 2)), n_components=1)


def test_gm_std_summary_weighted_average():
    # unique_nodes length 3, labels per unique node
    unique_nodes = np.array([0, 1, 2], dtype=int)
    labels = np.array([0, 0, 1], dtype=int)
    values = np.array([1.0, 2.0, 3.0], dtype=float)
    gm_component_labels = np.array([0, 1], dtype=int)

    # For label 0, nodes [0,1] std = 0.5; label1 nodes [2] std = 0.0
    # weights = [2,1] -> weighted mean = (2*0.5 + 1*0.0)/3 = 1/3
    out = gve._gm_std_summary(values, gm_component_labels, labels, unique_nodes)
    assert out == pytest.approx((2 * 0.5 + 1 * 0.0) / 3)

    # empty unique_nodes returns nan
    assert np.isnan(gve._gm_std_summary(values, gm_component_labels, labels, np.array([], dtype=int)))


def test_scenario_result_composes_subfunctions(monkeypatch):
    # Minimal GraphData for scenario: two nodes, one edge
    unique_nodes = np.array([0, 1], dtype=int)
    edge_src = np.array([0], dtype=int)
    edge_dst = np.array([1], dtype=int)
    edge_dist = np.array([1.0], dtype=float)
    labels = np.array([0, 0], dtype=int)
    sorted_labels = np.array([0], dtype=int)
    g = GraphData(1.0, np.eye(4), np.ones((2, 1, 1), dtype=bool), unique_nodes, edge_src, edge_dst, edge_dist, labels, sorted_labels)

    # monkeypatch subfunctions
    monkeypatch.setattr(gve, "_smooth_noise_on_graph", lambda g_smooth, tau_by_label, noise: np.array([1.0, 2.0]))
    monkeypatch.setattr(gve, "_gm_std_summary", lambda values, gm_labels, labels, unique_nodes: 0.42)
    monkeypatch.setattr(gve, "_estimate_fwhm_for_top_components", lambda g_smooth, smoothed, n_components=5: [(0, 2.5)])

    tau_map = {0: 1.0}
    gm_labels = np.array([], dtype=int)
    scenario, gm_std, fwhm_top, smoothed = gve._scenario_result("test", g, tau_map, gm_labels, labels, unique_nodes, np.array([0.0, 0.0]))
    assert scenario["graph"] == "test"
    assert scenario["graph_n_nodes"] == 2
    assert scenario["gm_weighted_std"] == pytest.approx(0.42)
    assert fwhm_top == [(0, 2.5)]
    assert smoothed.shape == (2,)


def test_scenario_result_integration_monkeypatched(monkeypatch):
    # Build a minimal graph with one component and known nodes
    unique_nodes = np.array([0, 1, 2], dtype=int)
    edge_src = np.array([0, 1], dtype=int)
    edge_dst = np.array([1, 2], dtype=int)
    edge_dist = np.array([1.0, 1.0], dtype=float)
    labels = np.array([0, 0, 0], dtype=int)
    sorted_labels = np.array([0], dtype=int)
    g = GraphData(1.0, np.eye(4), np.ones((3, 1, 1), dtype=bool), unique_nodes, edge_src, edge_dst, edge_dist, labels, sorted_labels)

    # Patch subroutines to deterministic values
    monkeypatch.setattr(gve, "_smooth_noise_on_graph", lambda g_smooth, tau_by_label, noise: np.array([1.0, 2.0, 3.0]))
    monkeypatch.setattr(gve, "_gm_std_summary", lambda values, gm_labels, labels, unique_nodes: 0.5)
    monkeypatch.setattr(gve, "_estimate_fwhm_for_top_components", lambda g_smooth, smoothed, n_components=5: [(0, 4.0)])

    tau_map = {0: 2.0}
    gm_labels = np.array([0], dtype=int)
    scenario, gm_std, fwhm_top, smoothed = gve._scenario_result(
        graph_name="integration",
        g_smooth=g,
        tau_by_label=tau_map,
        gm_component_labels=gm_labels,
        eval_labels=labels,
        eval_unique_nodes=unique_nodes,
        noise=np.zeros(3, dtype=float),
    )

    assert scenario["graph"] == "integration"
    assert scenario["graph_n_nodes"] == 3
    assert scenario["graph_n_edges"] == 2
    assert scenario["gm_weighted_std"] == pytest.approx(0.5)
    assert scenario["tau_mean"] == pytest.approx(2.0)
    assert fwhm_top == [(0, 4.0)]
    assert smoothed.shape == (3,)


def test_parent_mapping_spatial_nearest_with_spacing():
    # 1mm grid: 4 voxels along x; 3mm grid: 2 voxels spaced 2mm apart via affine
    affine1 = np.eye(4)
    g1 = _make_graphdata(nodes=[0, 1, 2, 3], edges=[], affine=affine1, shape=(4, 1, 1))
    affine3 = np.diag([2.0, 1.0, 1.0, 1.0])
    g3 = _make_graphdata(nodes=[0, 1], edges=[], affine=affine3, shape=(2, 1, 1))

    parent = _build_parent_mapping_1mm_to_3mm(g1, g3)
    # centers: g1 x={0,1,2,3}; g3 x={0,2} -> nearest mapping 0->0,1->0,2->1,3->1
    assert parent[0] == 0
    assert parent[1] == 0
    assert parent[2] == 1
    assert parent[3] == 1


def test_pruning_respects_parent_adjacency_complex():
    affine = np.eye(4)
    # 1mm nodes and edges forming a chain plus a diagonal chord
    g1 = _make_graphdata(
        nodes=[0, 1, 2, 3, 4],
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],
        affine=affine,
        shape=(5, 1, 1)
    )

    # Parent graph with three parents: A=10, B=11, C=12; adjacency only A-B and B-C (no A-C)
    g3 = _make_graphdata(
        nodes=[10, 11, 12],
        edges=[(10, 11), (11, 12)],
        affine=affine,
        shape=(13, 1, 1)
    )

    # Map nodes: 0,1 -> A(10); 2 -> B(11); 3,4 -> C(12)
    parent = np.array([10, 10, 11, 12, 12], dtype=int)

    pr_src, pr_dst, pr_dist = prune_1mm_edges_using_3mm_connectivity(g1, g3, parent)
    kept = set(zip(pr_src.tolist(), pr_dst.tolist()))

    # Edges expected to keep:
    # (0,1) intra A; (1,2) across A-B (adjacent); (2,3) across B-C (adjacent);
    # (3,4) intra C. Edge (0,4) spans A-C which are NOT adjacent in parent graph and must drop.
    assert (0, 1) in kept
    assert (1, 2) in kept
    assert (2, 3) in kept
    assert (3, 4) in kept
    assert (0, 4) not in kept


def test_scenario_result_integration_monkeypatched(monkeypatch):
    # Build a minimal graph with one component and known nodes
    unique_nodes = np.array([0, 1, 2], dtype=int)
    edge_src = np.array([0, 1], dtype=int)
    edge_dst = np.array([1, 2], dtype=int)
    edge_dist = np.array([1.0, 1.0], dtype=float)
    labels = np.array([0, 0, 0], dtype=int)
    sorted_labels = np.array([0], dtype=int)
    g = GraphData(1.0, np.eye(4), np.ones((3, 1, 1), dtype=bool), unique_nodes, edge_src, edge_dst, edge_dist, labels, sorted_labels)

    # Patch subroutines to deterministic values
    monkeypatch.setattr(gve, "_smooth_noise_on_graph", lambda g_smooth, tau_by_label, noise: np.array([1.0, 2.0, 3.0]))
    monkeypatch.setattr(gve, "_gm_std_summary", lambda values, gm_labels, labels, unique_nodes: 0.5)
    monkeypatch.setattr(gve, "_estimate_fwhm_for_top_components", lambda g_smooth, smoothed, n_components=5: [(0, 4.0)])

    tau_map = {0: 2.0}
    gm_labels = np.array([0], dtype=int)
    scenario, gm_std, fwhm_top, smoothed = gve._scenario_result(
        graph_name="integration",
        g_smooth=g,
        tau_by_label=tau_map,
        gm_component_labels=gm_labels,
        eval_labels=labels,
        eval_unique_nodes=unique_nodes,
        noise=np.zeros(3, dtype=float),
    )

    assert scenario["graph"] == "integration"
    assert scenario["graph_n_nodes"] == 3
    assert scenario["graph_n_edges"] == 2
    assert scenario["gm_weighted_std"] == pytest.approx(0.5)
    assert scenario["tau_mean"] == pytest.approx(2.0)
    assert fwhm_top == [(0, 4.0)]
    assert smoothed.shape == (3,)


def test_parent_mapping_matches_bruteforce_large_random():
    rng = np.random.default_rng(0)
    # 1mm grid with 6000 nodes to exercise chunking; full mask
    shape1 = (20, 15, 20)  # 6000 voxels
    mask1 = np.ones(shape1, dtype=bool)
    unique_nodes1 = np.flatnonzero(mask1.ravel())
    g1 = GraphData(
        voxel_size_mm=1.0,
        affine=np.eye(4),
        mask_3d=mask1,
        unique_nodes=unique_nodes1,
        edge_src=np.array([], dtype=int),
        edge_dst=np.array([], dtype=int),
        edge_distances=np.array([], dtype=float),
        labels=np.zeros(mask1.size, dtype=int),
        sorted_labels=np.array([0], dtype=int),
    )

    # 3mm grid with coarser spacing; mask all voxels
    shape3 = (10, 8, 10)
    affine3 = np.diag([2.0, 2.0, 2.0, 1.0])
    mask3 = np.ones(shape3, dtype=bool)
    unique_nodes3 = np.flatnonzero(mask3.ravel())
    g3 = GraphData(
        voxel_size_mm=2.0,
        affine=affine3,
        mask_3d=mask3,
        unique_nodes=unique_nodes3,
        edge_src=np.array([], dtype=int),
        edge_dst=np.array([], dtype=int),
        edge_distances=np.array([], dtype=float),
        labels=np.zeros(mask3.size, dtype=int),
        sorted_labels=np.array([0], dtype=int),
    )

    parent = _build_parent_mapping_1mm_to_3mm(g1, g3)

    # Brute-force nearest using voxel centers
    xyz1 = gve._voxel_centers_mm(shape1, g1.affine)
    xyz3 = gve._voxel_centers_mm(shape3, g3.affine)
    d2 = ((xyz1[:, None, :] - xyz3[None, :, :]) ** 2).sum(axis=2)
    brute = np.argmin(d2, axis=1)

    # parent mapping should match brute for all nodes
    # KD-tree ties may differ from argmin ordering; ensure chosen parent achieves the min distance
    d2_parent = d2[np.arange(d2.shape[0]), parent]
    d2_min = d2.min(axis=1)
    assert np.allclose(d2_parent, d2_min)


def test_estimate_tau_for_fwhm_fill_average(monkeypatch):
    # Graph with labels [5,4,3,99]; sorted_labels picks first three; label 99 should get mean of others
    unique_nodes = np.array([0, 1, 2, 3], dtype=int)
    labels = np.array([5, 4, 3, 99], dtype=int)
    sorted_labels = np.array([5, 4, 3, 99], dtype=int)
    g = GraphData(
        voxel_size_mm=1.0,
        affine=np.eye(4),
        mask_3d=np.ones((4, 1, 1), dtype=bool),
        unique_nodes=unique_nodes,
        edge_src=np.array([], dtype=int),
        edge_dst=np.array([], dtype=int),
        edge_distances=np.array([], dtype=float),
        labels=labels,
        sorted_labels=sorted_labels,
    )

    # select_nodes returns dummy data sized to number of nodes in the label
    def fake_select_nodes(edge_src, edge_dst, edge_distances, labels, label, unique_nodes):
        _nodes = unique_nodes[labels == label]
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), _nodes

    monkeypatch.setattr(gve, "select_nodes", fake_select_nodes)
    monkeypatch.setattr(gve, "find_optimal_tau", lambda fwhm, edge_src, edge_dst, edge_distances, shape: fwhm + float(shape[0]))

    tau_map = gve._estimate_tau_for_fwhm(g, fwhm_mm=6.0, n_components=3)

    # For labels 5,4,3: tau = 6 + n_nodes_per_label (all 1) => 7. Mean = 7.
    assert tau_map[5] == pytest.approx(7.0)
    assert tau_map[4] == pytest.approx(7.0)
    assert tau_map[3] == pytest.approx(7.0)
    # Remaining label 99 should get the average of computed taus
    assert tau_map[99] == pytest.approx(7.0)

