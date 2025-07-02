import logging
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components

from csmooth.matrix import create_adjacency_matrix
from csmooth.graph import select_nodes
import time


def identify_connected_components(edge_src, edge_dst, edge_distances):
    """
    Identify connected components in a graph defined by edges and distances.
    :param edge_src:
    :param edge_dst:
    :param edge_distances:
    :return: labels: numpy array of labels for each node in the graph,
             sorted_labels: numpy array of sorted labels by size of components.
             unique_nodes: numpy array of unique nodes in the graph.
    """

    adjacency_matrix, unique_nodes = create_adjacency_matrix(edge_src, edge_dst, weights=edge_distances)
    n_components, labels = connected_components(csgraph=adjacency_matrix.tocsr(), directed=False, return_labels=True)
    sorted_labels = np.argsort([(labels == l).sum() for l in np.unique(labels)])[::-1]

    return labels, sorted_labels, unique_nodes


def check_for_bottlenecks(edge_src, edge_dst, edge_distances, labels, label, unique_nodes, sampling_fraction=0.001):
    """
    Check for bottlenecks in the graph defined by edges and distances.
    :param edge_src: numpy array of source nodes
    :param edge_dst: numpy array of destination nodes
    :param edge_distances: numpy array of distances between nodes
    :param labels: numpy array of labels for each node
    :param label: label to check for bottlenecks
    :param unique_nodes: numpy array of unique nodes in the graph
    :param sampling_fraction: fraction of edges to sample for betweenness centrality calculation
    :return: edge_src_bottleneck, edge_dst_bottleneck, edge_distances_bottleneck:
             numpy arrays of edges and distances that are bottlenecks.
    """
    logging.info('Checking for bottlenecks')
    start_time = time.time()
    # select nodes that belong to the specified component
    _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(edge_src, edge_dst, edge_distances, labels, label, unique_nodes)

    # create undirected networkx graph
    G  = nx.Graph()
    G.add_weighted_edges_from(zip(_edge_src, _edge_dst, _edge_distances))
    logging.debug(f"Graph is directed: {G.is_directed()}")

    # estimate the betweenness centrality of the graph edges
    betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=True,
                                                 k=int(len(G.nodes) * sampling_fraction))

    # find outliers with large betweenness centrality
    mean_bc = np.mean(list(betweenness.values()))
    std_bc = np.std(list(betweenness.values()))
    threshold = mean_bc + 3 * std_bc
    logging.debug(f"Betweenness centrality threshold: {threshold:.4f}")
    outliers = [edge for edge, bc in betweenness.items() if bc > threshold]

    logging.debug(f"Checked for outlier edges in {time.time() - start_time:.2f} seconds")
    logging.info(f"Found {len(outliers)} outliers in the graph for label {label} with high betweenness centrality")

    logging.info(f"Removing {len(outliers)} outliers from the graph")
    for edge in outliers:
        G.remove_edge(*edge)

    # check the number and size of connected components after removing outliers
    cc_size = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    logging.info(f"Number of connected components after removing outliers: {len(cc_size)} with sizes {cc_size}")

