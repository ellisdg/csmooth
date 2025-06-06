import itertools
import time
from functools import partial

import nilearn.image
import numpy as np
import scipy
from scipy.sparse.csgraph import connected_components

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from pygsp import graphs
from pygsp import filters

import nibabel as nib
import os

from csmooth.graph import create_graph
from csmooth.matrix import create_adjacency_matrix
from csmooth.affine import adjust_affine_spacing, resample_data_to_shape, resample_data_to_affine



def compute_gaussian_kernel(src_node, edge_src, edge_dst, edge_distances, sigma, max_connections):
    # find all nodes that are within max_distance * sigma
    neighbor_nodes = [src_node]
    neighbor_distances = [0]
    for j in range(max_connections):
        # find all edges that stem any of the current neighbor nodes
        edges_idx = np.where(np.isin(edge_src, neighbor_nodes))[0]
        # nodes that are connected to the current neighbor nodes
        target_nodes = edge_dst[edges_idx]
        # distances of the edges that connect the current neighbor nodes to the target nodes
        target_distances = edge_distances[edges_idx]
        # remove all nodes that are already in the neighbor nodes
        novel_candidates = np.setdiff1d(target_nodes, neighbor_nodes)
        # novel target_nodes are likely to have more than one connection
        # I only want to add the shortest distance
        # so I need to find the minimum distance for each novel candidate
        novel_candidates_distances = np.zeros(len(novel_candidates))
        for k, novel_candidate in enumerate(novel_candidates):
            # find all edges that connect the current neighbor nodes to the novel candidate
            novel_edges_idx = np.where(target_nodes == novel_candidate)[0]
            # find the minimum distance of the edges
            novel_candidates_distances[k] = np.min(target_distances[novel_edges_idx])

        if len(novel_candidates) > 0:
            neighbor_nodes.extend(novel_candidates)
            neighbor_distances.extend(novel_candidates_distances)
    weights = np.exp(-np.array(neighbor_distances) ** 2 / (2 * sigma ** 2))
    # normalize the weights
    weights = weights / np.sum(weights)
    # return the neighbor nodes and the weights
    return src_node, np.asarray(neighbor_nodes), weights


def multiproc_gaussian_kernel(edge_src, edge_dst, edge_distances, sigma, max_connections, n_procs=20, chunk_size=None):
    _compute_kernel = partial(compute_gaussian_kernel,
                              edge_src=edge_src,
                              edge_dst=edge_dst,
                              edge_distances=edge_distances,
                              sigma=sigma,
                              max_connections=max_connections)
    if chunk_size is None:
        chunk_size = int(len(np.unique(edge_src)) / (n_procs * 100))
    # compute the kernel in parallel
    results = process_map(_compute_kernel, np.unique(edge_src), max_workers=n_procs,
                          chunksize=chunk_size, desc="Computing Gaussian kernel", unit="nodes")
    return results


def networkx_graph_distances(edge_src, edge_dst, edge_distances, cutoff=None, n_jobs=4):
    import networkx as nx
    import nx_parallel as nxp

    nx.config.backends.parallel.active= True
    nx.config.backends.parallel.n_jobs = n_jobs

    # create graph
    G = nx.Graph()
    unique_nodes = np.unique(np.concatenate((edge_src, edge_dst)))
    G.add_nodes_from(unique_nodes)
    G.add_weighted_edges_from(np.column_stack((edge_src, edge_dst, edge_distances)))
    G = nxp.ParallelGraph(G)

    result = dict(nxp.all_pairs_dijkstra_path_length(G, cutoff=cutoff))
    return result


def networkx_gaussian_kernels(edge_src, edge_dst, edge_distances, fwhm, cutoff=None, n_jobs=4):
    result = networkx_graph_distances(edge_src, edge_dst, edge_distances, cutoff=cutoff, n_jobs=n_jobs)
    # for each src node, compute the gaussian kernel based on the distances
    new_edge_src = list()
    new_edge_dst = list()
    new_edge_weights = list()
    for src_node in result.keys():

        neighbors_dict = result[src_node]
        _edge_dsts = list(neighbors_dict.keys())
        assert src_node in _edge_dsts
        distances = np.asarray(list(neighbors_dict.values()))
        # compute the gaussian kernel
        sigma = fwhm / np.sqrt(8 * np.log(2))
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
        # normalize the weights
        weights = weights / np.sum(weights)
        new_edge_src.extend([src_node] * len(distances))
        new_edge_dst.extend(list(neighbors_dict.keys()))
        new_edge_weights.extend(weights)

    return new_edge_src, new_edge_dst, new_edge_weights


def gaussian_smoothing(data, edge_src, edge_dst, edge_distances, fwhm, n_jobs=20):
    """
    Smooth a graph signal using a Gaussian kernel.
    :param data: fmri data of shape (n_voxels, n_timepoints)
    :param edge_src: numpy array containing the source nodes of the edges.
    Each node corresponds to a voxel in the fmri data.
    :param edge_dst: numpy array containing the destination nodes of the edges
    Each node corresponds to a voxel in the fmri data.
    :param edge_distances: numpy array containing the distances of the edges
    :param fwhm: Full-width at half maximum of the Gaussian kernel in mm.
     is the maximum distance to consider for smoothing.
    :return: smoothed data of shape (n_voxels, n_timepoints)
    """
    # don't worry about nodes that are more than 3 times sigma away from the source node
    cutoff = (fwhm/np.sqrt(8 * np.log(2))) * 3

    print("Computing Gaussian kernels...")
    start = time.time()
    edge_src, edge_dst, edge_weights = networkx_gaussian_kernels(edge_src, edge_dst, edge_distances, fwhm, cutoff=cutoff,
                                                                 n_jobs=n_jobs)
    end = time.time()
    run_time = (end - start) / 60
    print(f"Time taken for to compute Gaussian kernels: {run_time:.2f} minutes")

    print("Applying Gaussian smoothing...")
    start = time.time()

    adjacency_matrix, unique_nodes = create_adjacency_matrix(edge_src=edge_src, edge_dst=edge_dst, weights=edge_weights)
    smoothed_data = data.copy()
    for i in range(smoothed_data.shape[1]):
        smoothed_data[:, i] = adjacency_matrix @ data[:, i]

    end = time.time()
    run_time = (end - start) / 60
    print(f"Time taken to apply Gaussian smoothing: {run_time:.2f} minutes")
    return smoothed_data.squeeze()


def heat_kernel_smoothing(edge_src, edge_dst, edge_distances, signal_data, tau):


    # create graph
    adjacency, nodes = create_adjacency_matrix(edge_src, edge_dst, weights=1/edge_distances)
    G = graphs.Graph(adjacency)
    G.estimate_lmax()

    # check if tau is a scalar or an iterable
    if isinstance(tau, (int, float)):
        return _heat_kernel_smoothing(G, signal_data.copy(), nodes, tau)
    else:
        return [_heat_kernel_smoothing(G, signal_data.copy(), nodes, t) for t in tau]


def _heat_kernel_smoothing(G, signal_data, nodes, tau):
    # filter graph
    heat_filter = filters.Heat(G, tau=tau)
    signal_data[nodes] = heat_filter.filter(signal_data[nodes])
    return signal_data


def identify_connected_components(edge_src, edge_dst, edge_distances):
    """
    Identify connected components in a graph defined by edges and distances.
    :param edge_src:
    :param edge_dst:
    :param edge_distances:
    :return: labels: numpy array of labels for each node in the graph,
             sorted_labels: numpy array of sorted labels by size of components.
    """

    adjacency_matrix, unique_nodes = create_adjacency_matrix(edge_src, edge_dst, weights=edge_distances)
    n_components, labels = connected_components(csgraph=adjacency_matrix.tocsr(), directed=False, return_labels=True)
    sorted_labels = np.argsort([(labels == l).sum() for l in np.unique(labels)])[::-1]

    return labels, sorted_labels, unique_nodes


def _smooth_component(edge_src, edge_dst, edge_distances, signal_data, tau=None,
                      fwhm=None):
    """
    Smooth a single component of the graph signal.
    :param edge_src:
    :param edge_dst:
    :param edge_distances:
    :param signal_data:
    :param labels:
    :param label:
    :param unique_nodes:
    :param tau:
    :param fwhm:
    :return:
    """
    if tau is None and fwhm is None:
        raise ValueError("Must provide either tau or fwhm")
    if tau is not None and fwhm is not None:
        raise ValueError("Must provide either tau or fwhm, not both")
    if fwhm is not None:
        smoothed_data = gaussian_smoothing(data=signal_data,
                                           edge_src=edge_src,
                                           edge_dst=edge_dst,
                                           edge_distances=edge_distances,
                                           fwhm=fwhm)
    elif tau is not None:
        smoothed_data = heat_kernel_smoothing(edge_src=edge_src,
                                              edge_dst=edge_dst,
                                              edge_distances=edge_distances,
                                              signal_data=signal_data,
                                              tau=tau)
    else:
        raise ValueError("Must provide either tau or fwhm")
    return smoothed_data


def smooth_component(edge_src, edge_dst, edge_distances, signal_data, labels, label, unique_nodes,
                     smoothed_signal_data, tau=None, fwhm=None):
    _nodes = unique_nodes[np.isin(labels, label)]
    _edge_mask = np.isin(edge_src, _nodes) & np.isin(edge_dst, _nodes)
    _edge_src = edge_src[_edge_mask]
    _edge_dst = edge_dst[_edge_mask]
    # renumber the nodes to match the signal data
    _nodes_map = {node: i for i, node in enumerate(_nodes)}
    _edge_src = np.vectorize(_nodes_map.get)(_edge_src)
    _edge_dst = np.vectorize(_nodes_map.get)(_edge_dst)
    _edge_distances = edge_distances[_edge_mask]

    _smoothed_data = _smooth_component(edge_src=_edge_src,
                                       edge_dst=_edge_dst,
                                       edge_distances=_edge_distances,
                                       signal_data=signal_data[_nodes, :],
                                       tau=tau,
                                       fwhm=fwhm)
    smoothed_signal_data[_nodes, :] = _smoothed_data


def load_and_resample_image(in_file, resample_resolution):
    reference_image = nib.load(in_file)
    original_shape = reference_image.shape[:3]
    original_affine = reference_image.affine

    if resample_resolution is not None:
        affine = adjust_affine_spacing(reference_image.affine, np.asarray(resample_resolution)).numpy()
        reference_data = resample_data_to_affine(reference_image.get_fdata()[..., 0],
                                                 target_affine=affine,
                                                 original_affine=reference_image.affine)
        shape = reference_data.shape[:3]
        reference_image = nib.Nifti1Image(reference_data, affine)
    else:
        affine = original_affine
        shape = original_shape

    return reference_image, affine, shape, original_shape, original_affine


def process_mask(mask_file, reference_image, mask_dilation):
    if mask_file is not None:
        mask_image = nib.load(mask_file)
    else:
        mask_image = nib.Nifti1Image(np.ones(reference_image.shape[:3], dtype=np.uint8), reference_image.affine)

    mask_image = nilearn.image.resample_to_img(mask_image, reference_image,
                                               interpolation="nearest",
                                               force_resample=True)
    mask_array = mask_image.get_fdata() > 0.5

    if mask_dilation is not None and mask_file is not None:
        mask_array = scipy.ndimage.binary_dilation(mask_array, iterations=mask_dilation)

    return mask_array


def save_labelmap(output_labelmap, shape, affine, labels, sorted_labels, unique_nodes):
    os.makedirs(os.path.dirname(output_labelmap), exist_ok=True)
    labelmap = np.zeros(shape, dtype=np.uint32).flatten()

    for i, label in enumerate(sorted_labels):
        labelmap[unique_nodes[labels == label]] = i + 1

    labelmap_image = nib.Nifti1Image(labelmap.reshape(shape), affine)
    labelmap_image.to_filename(output_labelmap)


def smooth_components(edge_src, edge_dst, edge_distances, signal_data, labels, sorted_labels, unique_nodes, tau, fwhm):
    smoothed_signal_data = signal_data.copy()
    for label in tqdm(sorted_labels, desc="Smoothing components", unit="component"):
        smooth_component(edge_src, edge_dst, edge_distances, signal_data, labels, label, unique_nodes,
                         smoothed_signal_data, tau=tau, fwhm=fwhm)
    return smoothed_signal_data


def smooth_image(in_file, out_file, surface_files, tau=None, fwhm=None, output_labelmap=None, overwrite=True,
                 resample_resolution=None, mask_file=None, mask_dilation=3, multiproc=4):
    """
    Smooth an image using graph signal smoothing.
    :param in_file: Path to a Nifti file to be smoothed.
    :param out_file: Output filename to save the smoothed image.
    :param surface_files: List of surface filenames to use for edge pruning.
    :param tau: Value of tau to use for graph signal smoothing. Either tau or fwhm must be provided.
    :param fwhm: Value of FWHM to use for Gaussian smoothing. Either tau or fwhm must be provided.
    :param output_labelmap: Optional output labelmap filename to save the individual components that were smoothed. To disable, set to None.
    :param overwrite: Whether to overwrite existing output files.
    :param resample_resolution: Optional (x, y, z) resolution to resample the image to. If None, no resampling is done.
    Otherwise, the image is resampled to the specified resolution prior to formation of the graph and smoothing. After
    smoothing, the image is resampled back to the original resolution.
    :param mask_file: Optional filename of a mask to use for smoothing. This can speed up processing and reduce
     computational requirements, If None, no mask is used.
    :param mask_dilation: Optional number of voxels to dilate the mask by. This can help to include more voxels in the
        smoothing process. If None, no dilation is done. Mask dilation is done in the resampled image space. If no
        signal image resampling is done, the mask is dilated in the signal image space (not the mask image space).
        If no mask filename is provided, this parameter is ignored.
    :return:
    """

    reference_image = nib.load(in_file)
    original_shape = reference_image.shape[:3]
    original_affine = reference_image.affine

    if mask_file is not None:
        mask_image = nib.load(mask_file)
    else:
        mask_image = nib.Nifti1Image(np.ones(original_shape, dtype=np.uint8), reference_image.affine)

    if resample_resolution is not None:
        affine = adjust_affine_spacing(reference_image.affine, np.asarray(resample_resolution)).numpy()
        reference_data = resample_data_to_affine(reference_image.get_fdata()[..., 0],
                                                 target_affine=affine,
                                                 original_affine=reference_image.affine)
        shape = reference_data.shape[:3]
        reference_image = nib.Nifti1Image(reference_data, affine)
        del reference_data
    else:
        affine = original_affine
        shape = original_shape

    # force the mask to be in the same space as the reference image
    if not np.allclose(original_affine, affine, atol=1e-6):
        import warnings
        warnings.warn("Mask affine does not match input image affine.")


    mask_image = nilearn.image.resample_to_img(mask_image, reference_image,
                                               interpolation="nearest",
                                               force_resample=True)
    mask_array = mask_image.get_fdata() > 0.5

    if mask_dilation is not None and mask_file is not None:
        mask_array = scipy.ndimage.binary_dilation(mask_array, iterations=mask_dilation)

    edge_src, edge_dst, edge_distances = create_graph(mask_array, affine, surface_files=surface_files)

    labels, sorted_labels, unique_nodes = identify_connected_components(edge_src, edge_dst, edge_distances)

    if output_labelmap is not None:
        os.makedirs(os.path.dirname(output_labelmap), exist_ok=True)
        print("Saving labelmap to:", output_labelmap)

        # save select labels to a file
        labelmap = np.zeros(shape, dtype=np.uint32).flatten()

        for i, label in enumerate(sorted_labels):
            labelmap[unique_nodes[labels == label]] = i + 1

        labelmap_image = nib.Nifti1Image(labelmap.reshape(shape), affine)
        labelmap_image.to_filename(output_labelmap)


    tqdm.write(f"Processing filename: {in_file}, tau: {tau}, fwhm: {fwhm}")

    signal_image = nib.load(in_file)
    signal_data = signal_image.get_fdata()
    if signal_data.ndim == 3:
        signal_data = signal_data[..., None]

    if resample_resolution is not None:
        signal_data = resample_data_to_shape(signal_data, shape)

    signal_shape = signal_data.shape

    x, y, z, t = signal_shape

    signal_data = signal_data.reshape(x * y * z, t)
    smoothed_signal_data = signal_data.copy()

    for label in tqdm(sorted_labels, desc="Smoothing components", unit="component"):
        smooth_component(edge_src, edge_dst, edge_distances, signal_data, labels, label, unique_nodes,
                         smoothed_signal_data, tau=tau, fwhm=fwhm)
    smoothed_signal_data = smoothed_signal_data.reshape(x, y, z, t)

    if resample_resolution is not None:
        smoothed_signal_data = resample_data_to_shape(smoothed_signal_data, original_shape)
    smoothed_image = nib.Nifti1Image(smoothed_signal_data, signal_image.affine)

    print(f"Saving smoothed image to: {out_file}")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    smoothed_image.to_filename(out_file)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Smooth fMRI images using constrained smoothing.")
    parser.add_argument("in_file", type=str,
                        help="Input image to be smoothed. In most cases this should be a preprocessed fMRI image.")
    parser.add_argument("out_file", type=str,
                        help="Output smoothed image filename.")
    parser.add_argument("--surface_files", type=str, nargs='+', required=True,
                        help="List of surface files to use for edge pruning. Must be in GIFTI format.")
    parser.add_argument("--tau", type=float,
                        help="Tau value for heat kernel smoothing. Either --tau or --fwhm must be provided.")
    parser.add_argument("--fwhm", type=float,
                        help="FWHM value for Gaussian smoothing. Either --tau or --fwhm must be provided.")
    parser.add_argument("--mask_file", type=str, default=None,
                        help="Optional mask file to use for smoothing. "
                             "Must be in the same space and resolution as the input image. "
                             "If not provided, the whole image is used which increases computational requirements "
                             "and runtime.")
    parser.add_argument("--mask_dilation", type=int, default=3,
                        help="Number of voxels to dilate the mask by. "
                             "This can help make sure no parts of the brain are being eroniously excluded due to any "
                             "masking errors. "
                             "If None, no dilation is done. Default is 3.")
    #TODO: add option to use labelmap instead of mask_file
    parser.add_argument("--output_labelmap", type=str,
                        help="Optional output labelmap filename to save the individual components that were smoothed. "
                             "By default, this is saved to the '{output_basename}_components.nii.gz'. "
                             "To disable, set to None. ")
    parser.add_argument("--voxel_size", type=float, default=2.0,
                        help="Isotropic voxel size for resampling the image and mask prior to smoothing. "
                             "Smaller voxel sizes allow for a more continuous graph but increase computational "
                             "requirements and runtime. Default is 2.0 mm.")
    parser.add_argument("--multiproc", type=int, default=4,
                        help="Number of parallel processes to use for smoothing.")
    parser.add_argument("--no_overwrite", action='store_true',
                        help="If set, do not overwrite existing output files. Default is to overwrite.")
    args = parser.parse_args()

    # Validation to ensure either tau or fwhm is provided
    if args.tau is None and args.fwhm is None:
        parser.error("Either --tau or --fwhm must be provided.")
    if args.tau is not None and args.fwhm is not None:
        parser.error("Only one of --tau or --fwhm can be provided, not both.")

    return args


def main():
    args = parse_args()

    output_labelmap = args.output_labelmap
    if output_labelmap.lower() == "none":
        output_labelmap = None
    elif output_labelmap is None:
        output_labelmap = os.path.splitext(args.out_file)[0] + "_components.nii.gz"

    if os.path.exists(args.out_file):
        print(f"Output file {args.out_file} already exists.")
        if args.no_overwrite:
            print("Exiting. Use --no_overwrite to overwrite existing files.")
        else:
            print("Overwriting existing file.")

    smooth_image(in_file=args.in_file,
                 out_file=args.out_file,
                 surface_files=args.surface_files,
                 tau=args.tau,
                 fwhm=args.fwhm,
                 mask_file=args.mask_file,
                 mask_dilation=args.mask_dilation,
                 output_labelmap=output_labelmap,
                 overwrite=~args.no_overwrite,
                 resample_resolution=(args.voxel_size, args.voxel_size, args.voxel_size),)

    print("Smoothing complete.")


if __name__ == "__main__":
    main()
