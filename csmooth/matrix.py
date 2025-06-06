import numpy as np
from scipy.sparse import csr_matrix


def create_adjacency_matrix(edge_src, edge_dst, weights=None):
    unique_nodes = np.unique(np.concatenate((edge_src, edge_dst)))
    edge_src_short = np.searchsorted(unique_nodes, edge_src)
    edge_dst_short = np.searchsorted(unique_nodes, edge_dst)
    if weights is None:
        weights = np.ones(len(edge_src_short))
    adjacency_matrix = csr_matrix((weights, (edge_src_short, edge_dst_short)),
                                  shape=(len(unique_nodes), len(unique_nodes)))

    # check that the adjacency matrix is symmetric
    try:
        assert np.abs(np.sum(adjacency_matrix - adjacency_matrix.T)) < 1e-6
    except AssertionError:
        raise ValueError("Adjacency matrix is not symmetric")

    return adjacency_matrix, unique_nodes
