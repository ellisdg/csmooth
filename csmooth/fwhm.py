import numpy as np

def estimate_fwhm(edge_src, edge_dst, edge_distances, signal_data):
    # compute the average inter-neighbor distance
    dv = np.mean(edge_distances)

    # compute the variance of differences in signal between neighbors
    # for each edge, compute the difference in signal
    diff = signal_data[edge_src] - signal_data[edge_dst]

    # compute the variance of the differences
    var_ds = np.var(diff)

    # compute variance of the signal
    var_s = np.var(signal_data[np.unique(np.concatenate((edge_src, edge_dst)))])

    # compute the FWHM
    tmp = 1 - var_ds/(2 * var_s)
    tmp = max(tmp, 1e-12)
    tmp = np.log(tmp)
    tmp = (-2 * np.log(2))/tmp
    fwhm = dv * np.sqrt(tmp)
    return fwhm

