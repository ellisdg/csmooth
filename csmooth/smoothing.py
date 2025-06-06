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


def smooth_image(filename, surface_filenames, tau=None, fwhm=None, output_filename=None, output_directory=None,
                 write_labelmap=True, overwrite=False, resample_resolution=None, mask_filename=None, mask_dilation=3):
    """
    Smooth an image using graph signal smoothing.
    :param filename: Can be a path to a Nifti file or a list of filenames.
    :param surface_filenames: List of surface filenames to use for edge pruning.
    :param tau: value(s) of tau to use for graph signal smoothing.
    :param fwhm: value(s) of FWHM to use for Gaussian smoothing. If None, no smoothing is applied.
    :param output_filename: Optional output filename to save the smoothed image. Must be None if multiple
    input filenames or taus are provided. Either output_filename or output_directory must be provided.
    :param output_directory: Optional output directory to save the smoothed image. Must be provided if multiple
    input filenames or taus are provided.
    :param write_labelmap: Whether to write the labelmap of the voxel locations and components that were smoothed.
    :param overwrite: Whether to overwrite existing output files.
    :param resample_resolution: Optional (x, y, z) resolution to resample the image to. If None, no resampling is done.
    Otherwise, the image is resampled to the specified resolution prior to formation of the graph and smoothing. After
    smoothing, the image is resampled back to the original resolution.
    :param mask_filename: Optional filename of a mask to use for smoothing. This can speed up processing and reduce
     computational requirements, If None, no mask is used.
    :param mask_dilation: Optional number of voxels to dilate the mask by. This can help to include more voxels in the
        smoothing process. If None, no dilation is done. Mask dilation is done in the resampled image space. If no
        signal image resampling is done, the mask is dilated in the signal image space (not the mask image space).
        If no mask filename is provided, this parameter is ignored.
    :return:
    """


    if output_filename is None and output_directory is None:
        raise ValueError("Must provide either output_filename or output_directory")

    if not isinstance(tau, list):
        tau = [tau]
    else:
        if output_directory is None:
            raise ValueError("Must provide output_directory if multiple tau values are provided")
        output_filename = None

    if isinstance(filename, list):
        image_filenames = filename
        if output_directory is None:
            raise ValueError("Must provide output_directory if multiple input filenames are provided")
        # set output_filename to None
        output_filename = None
    else:
        image_filenames = [filename]

    reference_image = nib.load(image_filenames[0])
    original_shape = reference_image.shape[:3]
    original_affine = reference_image.affine

    if mask_filename is not None:
        mask_image = nib.load(mask_filename)
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

    mask_image = nilearn.image.resample_to_img(mask_image, reference_image, interpolation="nearest", force_resample=True)
    mask_array = mask_image.get_fdata() > 0.5
    if mask_dilation is not None and mask_filename is not None:
        mask_array = scipy.ndimage.binary_dilation(mask_array, iterations=mask_dilation)

    edge_src, edge_dst, edge_distances = create_graph(mask_array, affine, surface_files=surface_filenames)

    labels, sorted_labels, unique_nodes = identify_connected_components(edge_src, edge_dst, edge_distances)
    select_labels = [*sorted_labels[:5], sorted_labels[5:]]

    if write_labelmap and output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        labelmap_filename = os.path.join(output_directory, "components_labelmap.nii.gz")
        print("Saving labelmap to:", labelmap_filename)

        # save select labels to a file
        labelmap = np.zeros(shape, dtype=np.uint32).flatten()
        for i, label in enumerate(sorted_labels):
            labelmap[unique_nodes[labels == label]] = i + 1
        labelmap_image = nib.Nifti1Image(labelmap.reshape(shape), affine)
        labelmap_image.to_filename(labelmap_filename)

    # TODO: figure out a way to do all image files in one pass

    for _image_filename, _tau, _fwhm in tqdm(itertools.product(image_filenames, tau, fwhm), desc="Processing files",
                                           unit="file"):
        tqdm.write(f"Processing filename: {_image_filename}, tau: {_tau}, fwhm: {_fwhm}")
        if output_filename is None and _tau is not None:
            _output_filename = os.path.join(output_directory,
                                            f"smoothed_image_tau-{_tau:.2f}",
                                            os.path.basename(_image_filename))
        elif output_filename is None and _fwhm is not None:
            _output_filename = os.path.join(output_directory,
                                            f"smoothed_image_fwhm-{_fwhm:.2f}",
                                            os.path.basename(_image_filename))
        else:
            _output_filename = output_filename
        if os.path.exists(_output_filename) and not overwrite:
            continue
        os.makedirs(os.path.dirname(_output_filename), exist_ok=True)
        signal_image = nib.load(_image_filename)
        signal_data = signal_image.get_fdata()
        if signal_data.ndim == 3:
            signal_data = signal_data[..., None]

        if resample_resolution is not None:
            signal_data = resample_data_to_shape(signal_data, shape)

        signal_shape = signal_data.shape

        x, y, z, t = signal_shape

        signal_data = signal_data.reshape(x * y * z, t)
        smoothed_signal_data = signal_data.copy()

        # TODO: smooth all components, I don't think it will had much time

        for label in tqdm(select_labels, desc="Smoothing components", unit="component"):
            smooth_component(edge_src, edge_dst, edge_distances, signal_data, labels, label, unique_nodes,
                             smoothed_signal_data, tau=_tau, fwhm=_fwhm)
        smoothed_signal_data = smoothed_signal_data.reshape(x, y, z, t)
        if resample_resolution is not None:
            smoothed_signal_data = resample_data_to_shape(smoothed_signal_data, original_shape)
        smoothed_image = nib.Nifti1Image(smoothed_signal_data, signal_image.affine)
        print(f"Saving smoothed image to: {_output_filename}")
        smoothed_image.to_filename(_output_filename)


def demo(multiproc=6):
    from multiprocessing import Pool
    subject_folders = sorted(glob.glob("/media/conda2/public/sensory/derivatives/fmriprep/sub-*"))
    if multiproc > 1:
        with Pool(multiproc) as pool:
            pool.map(smooth_image_wrapper, subject_folders)
    else:
        for subject_folder in subject_folders:
            smooth_image_wrapper(subject_folder)


if __name__ == "__main__":
    import os
    import glob
    def smooth_image_wrapper(subject_folder):
        if not os.path.isdir(subject_folder):
            return
        print("Processing subject folder:", subject_folder)
        subject = os.path.basename(subject_folder)
        image_filenames = glob.glob(os.path.join(subject_folder, f"func/{subject}*_space-T1w_desc-preproc_bold.nii.gz"))
        surface_filenames = [os.path.join(subject_folder, "anat", f"{subject}_hemi-L_pial.surf.gii"),
                             os.path.join(subject_folder, "anat", f"{subject}_hemi-R_pial.surf.gii"),
                             os.path.join(subject_folder, "anat", f"{subject}_hemi-L_white.surf.gii"),
                             os.path.join(subject_folder, "anat", f"{subject}_hemi-R_white.surf.gii")]
        output_directory = f"/media/conda2/public/sensory/derivatives/constrained_smoothing/{subject}"
        mask_filename = os.path.join(subject_folder, "anat", f"{subject}_desc-brain_mask.nii.gz")
        smooth_image(image_filenames, surface_filenames,
                     tau=None,
                     fwhm=[3, 6, 9, 12],
                     output_directory=output_directory, resample_resolution=(1, 1, 1),
                     mask_filename=mask_filename,
                     write_labelmap=True)
    demo(multiproc=False)
