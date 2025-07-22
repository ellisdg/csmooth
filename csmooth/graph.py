import time
import nibabel as nib
import numpy as np
import trimesh
from csmooth.utils import logger


def remove_intersecting_edges(edges_src, edges_dst, vertex_coords, triangles, tolerance=1e-2):
    """
    Remove edges that intersect with the surface. Use trimesh to check for intersections.
    :param edges_src: source nodes of the edges (n, 3) numpy array. Each row is the (i, j, k) coordinate of a node.
                      The nodes should be voxel coordinates in the same space as the vertex_coords.
    :param edges_dst: destination nodes of the edges (n, 3) numpy array. Each row is the (i, j, k) coordinate of a node.
                      The nodes should be voxel coordinates in the same space as the vertex_coords.
    :param vertex_coords: coordinates of the surface vertices as a (n, 3) numpy array with (i, j, k) coordinates.
                          The coordinates should be in the same space as the edges_src and edges_dst.
    :param triangles: triangles of the surface (numpy array)
    :param tolerance: tolerance for intersection check. Default is 1e-2 mm. This is added to the length of the edge
                     when checking for intersections. This is added to fix errors potentially caused by floating point
                     inaccuracies. Lowering this value may result in less edges being removed, but may also result in
                     components not being disconnected properly.

    :return edges_src, edges_dst: the filtered edges
    """
    # TODO: adjust tolerance based on whether the number of connected components from al lthe surfaces is as expected

    logger.info("Checking for edges intersecting with the surface...")

    start = time.time()

    # Create a mesh from the triangles
    mesh = trimesh.Trimesh(vertices=vertex_coords, faces=triangles)

    # Fix potential issues with the mesh to ensure it is watertight
    mesh.fill_holes()
    mesh.process(validate=True)


    # Check for intersections
    directions = edges_dst - edges_src
    lengths = np.linalg.norm(directions, axis=1)
    # Avoid division by zero for zero-length edges
    directions_normalized = np.divide(directions, lengths[:, np.newaxis],
                                      out=np.zeros_like(directions, dtype=float),
                                      where=lengths[:, np.newaxis] != 0)

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
    # Add a small tolerance to handle floating point inaccuracies
    retained_mask[indices_ray] = (lengths[indices_ray] + tolerance) < edge_src_to_intersection

    logger.info(f"Starting edges: {len(edges_src)}")
    logger.info(f"Retained edges: {retained_mask.sum()}")
    logger.info(f"Removed edges: {(~retained_mask).sum()}")
    logger.debug(f"Edge intersection checks took {time.time() - start:.2f} seconds")
    return retained_mask


def remove_edges_intersecting_surface_files(edges_src, edges_dst, surface_filenames):
    logger.info("Removing edges intersecting with surfaces")
    edge_mask = np.ones(len(edges_src), dtype=bool)

    for surface_filename in surface_filenames:
        logger.info(f"Loading surface: {surface_filename}")
        surface = nib.load(surface_filename)
        coords, triangles = surface.agg_data()
        _edge_mask = remove_intersecting_edges(edges_src[edge_mask], edges_dst[edge_mask], coords, triangles)
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


def create_graph(mask_array, image_affine, surface_files):
    """
    Create a graph from a binary mask, removing edges that intersect with surfaces.
    :param mask_array: 3D binary numpy array where non-zero values indicate the region of interest
    :param image_affine: affine of the mask image for transforming to real world coordinates
    :param surface_files: GITFTI files of the surfaces to remove edges intersecting.
                         Typically, these are the gray and white matter surfaces.
    :return:
    """

    image_tensor = mask_array
    edge_src, edge_dst = mask2graph(image_tensor)

    edge_src_3d, edge_dst_3d = compute_edge_coordinates(edge_src, edge_dst, mask_array.shape)
    edge_src_xyz, edge_dst_xyz = compute_edge_real_world_coordinates(edge_src_3d, edge_dst_3d, image_affine)
    edge_distances = compute_edge_distances(edge_src_xyz, edge_dst_xyz)

    edge_mask = remove_edges_intersecting_surface_files(edge_src_xyz[..., :3],
                                                        edge_dst_xyz[..., :3],
                                                        surface_files)

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
    :return edge_src: numpy array of source nodes for the selected component
    :return edge_dst: numpy array of destination nodes for the selected component
    :return edge_distances: numpy array of distances for the selected component
    :return nodes: numpy array of nodes in the selected component
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
