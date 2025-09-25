from csmooth.matrix import create_adjacency_matrix

from pygsp import graphs, filters


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
    heat_filter = filters.Heat(G, scale=tau)
    signal_data[nodes] = heat_filter.filter(signal_data[nodes])
    return signal_data
