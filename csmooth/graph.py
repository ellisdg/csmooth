import logging
import nibabel as nib
import numpy as np


def remove_intersecting_edges(edges_src, edges_dst, voxel_coords, triangles):
    """
    Remove edges that intersect with the surface. Use trimesh to check for intersections.
    :param edges_src: source nodes of the edges (numpy array)
    :param edges_dst: destination nodes of the edges (numpy array)
    :param voxel_coords: coordinates of the voxels (numpy array)
    :param triangles: triangles of the surface (numpy array)
    :return edges_src, edges_dst: the filtered edges
    """
    import trimesh

    # Create a mesh from the triangles
    mesh = trimesh.Trimesh(vertices=voxel_coords, faces=triangles)

    # Check for intersections
    directions = edges_dst - edges_src
    lengths = np.linalg.norm(directions, axis=1)
    directions_normalized = directions / lengths[:, np.newaxis]

    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    locations, indices_ray, indices_triangle = intersector.intersects_location(
        ray_origins=edges_src,
        ray_directions=directions_normalized,
        multiple_hits=False,
    )

    # compute the distances between edges_src and the intersection points
    edge_src_to_intersection = np.linalg.norm(edges_src[indices_ray] - locations, axis=1)
    retained_mask = np.ones(len(edges_src), dtype=bool)


    # Retain connections with lengths that are shorter than the distance to the surface intersection
    # (i.e. the edge does not intersect with the surface)
    retained_mask[indices_ray] = edge_src_to_intersection > lengths[indices_ray]
    logging.info(f"Starting edges: {len(edges_src)}")
    logging.info(f"Retained edges: {retained_mask.sum()}")
    logging.info(f"Removed edges: {(~retained_mask).sum()}")
    return retained_mask


def remove_edges_intersecting_surface_files(edges_src, edges_dst, surface_filenames, affine):
    logging.info("Removing edges intersecting with surfaces")
    edge_mask = np.ones(len(edges_src), dtype=bool)

    for surface_filename in surface_filenames:
        logging.info(f"Loading surface: {surface_filename}")
        surface = nib.load(surface_filename)
        coords, triangles = surface.agg_data()
        voxel_coords = np.linalg.solve(affine, np.hstack((coords, np.ones((coords.shape[0], 1)))).T).T[:, :3]
        _edge_mask = remove_intersecting_edges(edges_src[edge_mask], edges_dst[edge_mask], voxel_coords, triangles)
        edge_mask[edge_mask] = _edge_mask
    return edge_mask


def mask2graph(mask_array):
    """
    Convert a 3D binary array to a graph representation using 26-connectivity.
    :param mask_array: 3D binary numpy array
    :return: adjacency matrix in sparse format
    """
    indices = np.argwhere(mask_array)
    D, H, W = mask_array.shape
    offsets = np.array([[dz, dy, dx]
                        for dz in [-1, 0, 1]
                        for dy in [-1, 0, 1]
                        for dx in [-1, 0, 1]
                        if not (dz == dy == dx == 0)])
    num_offsets = offsets.shape[0]
    neighbor_coords = indices[:, None, :] + offsets[None, :, :]

    # Ensure the neighbor coordinates are within bounds
    valid_mask = ((neighbor_coords[..., 0] >= 0) & (neighbor_coords[..., 0] < D) &
                  (neighbor_coords[..., 1] >= 0) & (neighbor_coords[..., 1] < H) &
                  (neighbor_coords[..., 2] >= 0) & (neighbor_coords[..., 2] < W))

    neighbor_coords = neighbor_coords[valid_mask]

    # Convert 3D coordinates to linear indices
    linear_neighbors = (neighbor_coords[..., 0] * H * W +
                        neighbor_coords[..., 1] * W +
                        neighbor_coords[..., 2])

    # Repeat each source index 26 times and then apply valid mask
    linear_indices = indices[..., 0] * H * W + indices[..., 1] * W + indices[..., 2]

    linear_indices = np.repeat(linear_indices, num_offsets).reshape(-1, num_offsets)

    linear_indices = linear_indices[valid_mask]

    # Create a flattened version of the array
    flat_array = mask_array.flatten()

    # Determine which neighbors are non-zero
    active_neighbors_mask = flat_array[linear_neighbors] == 1

    # Get edges
    edges_src = linear_indices[active_neighbors_mask]
    edges_dst = linear_neighbors[active_neighbors_mask]

    return edges_src, edges_dst


def compute_edge_coordinates(edge_src, edge_dst, shape):
    """
    Transform the edges from 1D index to 3D indices.
    :param edge_src:
    :param edge_dst:
    :param shape:
    :return:
    """
    # transform edges from 1D index back into 3D indices
    edge_src_3d = np.asarray(np.unravel_index(edge_src, shape)).T
    edge_dst_3d = np.asarray(np.unravel_index(edge_dst, shape)).T
    return edge_src_3d, edge_dst_3d


def compute_edge_real_world_coordinates(edge_src_3d, edge_dst_3d, affine):
    """
    Transform the 3D indices of the edges to real world coordinates.
    :param edge_src_3d:
    :param edge_dst_3d:
    :param affine:
    :return:
    """
    # transform 3D indices to real world coordinates
    edge_src_xyz = np.concatenate((edge_src_3d, np.ones((edge_src_3d.shape[0], 1))), axis=1) @ affine.T
    edge_dst_xyz = np.concatenate((edge_dst_3d, np.ones((edge_dst_3d.shape[0], 1))), axis=1) @ affine.T
    return edge_src_xyz, edge_dst_xyz


def compute_edge_distances(edge_src_xyz, edge_dst_xyz):
    # compute the distances between the edges
    edge_distances = np.linalg.norm(edge_src_xyz - edge_dst_xyz, axis=1)
    return edge_distances


def create_graph(mask_array, affine, surface_files, surface_affine=None):
    # TODO: implement surface_affine handling to transform surface points into mask space
    D, H, W = mask_array.shape
    image_tensor = mask_array
    edge_src, edge_dst = mask2graph(image_tensor)

    edge_src_3d, edge_dst_3d = compute_edge_coordinates(edge_src, edge_dst, mask_array.shape)
    edge_src_xyz, edge_dst_xyz = compute_edge_real_world_coordinates(edge_src_3d, edge_dst_3d, affine)
    edge_distances = compute_edge_distances(edge_src_xyz, edge_dst_xyz)

    # transform edges from 1D index back into 3D indices
    edge_src_3d = np.asarray(np.unravel_index(edge_src, (D, H, W))).T
    edge_dst_3d = np.asarray(np.unravel_index(edge_dst, (D, H, W))).T

    edge_mask = remove_edges_intersecting_surface_files(edge_src_3d,
                                                        edge_dst_3d,
                                                        surface_files,
                                                        affine)

    edge_src = edge_src[edge_mask]
    edge_dst = edge_dst[edge_mask]
    edge_distances= edge_distances[edge_mask]

    return edge_src, edge_dst, edge_distances


def select_nodes(edge_src, edge_dst, edge_distances, labels, label, unique_nodes):
    """
    Select nodes in the graph that belong to a specific component.
    :param edge_src: numpy array of source nodes
    :param edge_dst: numpy array of destination nodes
    :param edge_distances: numpy array of distances between nodes
    :param labels: numpy array of labels for each node
    :param label: label of the component to select
    :param unique_nodes: numpy array of unique nodes in the graph
    :return:
    """
    _nodes = unique_nodes[np.isin(labels, label)]
    _edge_mask = np.isin(edge_src, _nodes) & np.isin(edge_dst, _nodes)
    _edge_src = edge_src[_edge_mask]
    _edge_dst = edge_dst[_edge_mask]
    # renumber the nodes to match the signal data
    _nodes_map = {node: i for i, node in enumerate(_nodes)}
    _edge_src = np.vectorize(_nodes_map.get)(_edge_src)
    _edge_dst = np.vectorize(_nodes_map.get)(_edge_dst)
    _edge_distances = edge_distances[_edge_mask]
    return _edge_src, _edge_dst, _edge_distances, _nodes
