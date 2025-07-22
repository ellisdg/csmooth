

import numpy as np
from scipy.sparse.csgraph import connected_components


from csmooth.matrix import create_adjacency_matrix


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

