import numpy as np
import logging
import time

from csmooth.fwhm import estimate_fwhm
from csmooth.heat import heat_kernel_smoothing


def graph_smoothing_with_gradient_descent(data, edge_src, edge_dst, edge_distances, fwhm,
                                          max_iterations=100, stop_threshold=0.01, initial_tua=None, learning_rate=1.0,
                                          decay_rate=0.99, random_seed=42):
    """
    Smooth a signal based on a graph targeting a specific fwhm.
    This function uses gradient descent to find the optimal smoothing parameter
    to achieve the target fwhm.
    First, a random noise image is generated. Then, the signal is smoothed using the initial tau.
    The fwhm of the smoothed signal is estimated. The gradient of the fwhm with respect to tau is computed.
    The tau is updated using the gradient and learning rate. This process is repeated until the fwhm is close
    enough to the target fwhm or the maximum number of iterations is reached.
    The final value of tau is used to smooth the original signal.
    :param data: signal data to be smoothed.
    :param edge_src: source nodes of the graph edges.
    :param edge_dst: destination nodes of the graph edges.
    :param edge_distances: distances of the graph edges.
    :param fwhm: target full width at half maximum in mm.
    :param max_iterations: maximum number of iterations for gradient descent.
    :param stop_threshold: threshold for stopping the gradient descent.
    :param initial_tua: initial value for the smoothing parameter. If None, 2*fwhm is used as the initial value.
    :param learning_rate: learning rate for gradient descent.
    :param decay_rate: decay rate for the learning rate.
    :param kwargs: see graph_signal_smoothing for the other parameters.
    :param random_seed: random seed for generating the random noise image. By default, 42.
    :return: smoothed signal
    """

    tua = find_optimal_tau(fwhm=fwhm, edge_src=edge_src, edge_dst=edge_dst,
                            edge_distances=edge_distances, shape=data.shape, initial_tua=initial_tua,
                            max_iterations=max_iterations, stop_threshold=stop_threshold,
                            learning_rate=learning_rate, decay_rate=decay_rate, random_seed=random_seed)
    smoothed_signal = heat_kernel_smoothing(signal_data=data, tau=tua, edge_src=edge_src,
                                            edge_dst=edge_dst, edge_distances=edge_distances)

    return smoothed_signal, tua

def find_optimal_tau(fwhm, edge_src, edge_dst, edge_distances, shape, initial_tua=None,
                     max_iterations=100, stop_threshold=0.005, learning_rate=3.0, decay_rate=0.99,
                     random_seed=42):
    start_time = time.time()
    if initial_tua is None:
        initial_tua = 2*fwhm
    tua = initial_tua

    np.random.seed(random_seed)
    random_noise = np.random.randn(*shape)
    mae = np.inf
    current_fwhm = np.nan
    for i in range(max_iterations):
        smoothed_noise = heat_kernel_smoothing(signal_data=random_noise, tau=tua, edge_src=edge_src,
                                               edge_dst=edge_dst, edge_distances=edge_distances)
        current_fwhm = estimate_fwhm(edge_src=edge_src, edge_dst=edge_dst,
                                     edge_distances=edge_distances, signal_data=smoothed_noise)
        logging.debug(f"Iteration {i}: current fwhm: {current_fwhm:.2f}, target fwhm: {fwhm:.2f} tau: {tua:.2f}")
        last_mae = mae
        mae = np.abs(current_fwhm - fwhm)
        if mae < stop_threshold:
            break

        gradient = (current_fwhm - fwhm) / mae
        tua -= learning_rate * gradient * mae
        learning_rate *= decay_rate

        if mae > last_mae:
            logging.warning(f"MAE increased from {last_mae:.4f} to {mae:.4f}. "
                            f"Halving the learning rate and resetting tau.")
            learning_rate /= 2.0
            tua = initial_tua

    end_time = time.time()
    logging.info(f"Optimal tau: {tua:.2f}, "
                 f"achieved fwhm: {current_fwhm:.2f}, "
                 f"target fwhm: {fwhm:.2f}, "
                 f"iterations: {i + 1}, "
                 f"time taken: {(end_time - start_time) / 60:.2f} minutes")
    return tua
