import time
import logging
import numpy as np
import networkx as nx
import nx_parallel as nxp

from csmooth.matrix import create_adjacency_matrix


def networkx_graph_distances(edge_src, edge_dst, edge_distances, cutoff=None, n_jobs=4):

    nx.config.backends.parallel.active = True
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


def compute_gaussian_kernels(edge_src, edge_dst, edge_distances, fwhm, n_jobs=4):
    """
    Compute Gaussian kernels for smoothing a graph signal.
    :param edge_src:
    :param edge_dst:
    :param edge_distances:
    :param fwhm:
    :param n_jobs:
    :return:
    """
    # don't worry about nodes that are more than 3 times sigma away from the source node
    cutoff = (fwhm / np.sqrt(8 * np.log(2))) * 3

    logging.info("Computing Gaussian weights...")
    start = time.time()
    edge_src, edge_dst, edge_weights = networkx_gaussian_kernels(edge_src, edge_dst, edge_distances, fwhm,
                                                                 cutoff=cutoff,
                                                                 n_jobs=n_jobs)
    end = time.time()
    run_time = (end - start) / 60
    logging.info(f"Time taken to compute Gaussian weights: {run_time:.2f} minutes")
    return edge_src, edge_dst, edge_weights


def gaussian_smoothing(data, edge_src, edge_dst, edge_distances, fwhm, n_jobs=4):
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

    edge_src, edge_dst, edge_weights = compute_gaussian_kernels(edge_src=edge_src, edge_dst=edge_dst,
                                                        edge_distances=edge_distances, fwhm=fwhm, n_jobs=n_jobs)

    return apply_gaussian_smoothing(data=data, edge_src=edge_src, edge_dst=edge_dst, edge_weights=edge_weights)


def apply_gaussian_smoothing(data, edge_src, edge_dst, edge_weights):
    """
    :param data:
    :param edge_src:
    :param edge_dst:
    :param edge_weights:
    :return:
    """
    logging.info("Applying Gaussian smoothing...")
    start = time.time()

    adjacency_matrix, unique_nodes = create_adjacency_matrix(edge_src=edge_src, edge_dst=edge_dst, weights=edge_weights)
    smoothed_data = data.copy()
    for i in range(smoothed_data.shape[1]):
        smoothed_data[:, i] = adjacency_matrix @ data[:, i]

    end = time.time()
    run_time = (end - start) / 60
    logging.info(f"Time taken to apply Gaussian smoothing: {run_time:.2f} minutes")
    return smoothed_data.squeeze()
