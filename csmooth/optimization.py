import numpy as np
import time
from csmooth.fwhm import estimate_fwhm
from csmooth.heat import heat_kernel_smoothing
from csmooth.utils import logger


def graph_smoothing_with_gradient_descent(data, edge_src, edge_dst, edge_distances, fwhm,
                                          max_iterations=100, stop_threshold=0.01, initial_tau=None, learning_rate=1.0,
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
    :param initial_tau: initial value for the smoothing parameter. If None, 2*fwhm is used as the initial value.
    :param learning_rate: learning rate for gradient descent.
    :param decay_rate: decay rate for the learning rate.
    :param random_seed: random seed for generating the random noise image. By default, 42.
    :param patience: number of iterations to wait before stopping if the fwhm does not improve.
    :return: smoothed signal
    """

    tua = find_optimal_tau(fwhm=fwhm, edge_src=edge_src, edge_dst=edge_dst,
                           edge_distances=edge_distances, shape=data.shape, initial_tau=initial_tau,
                           max_iterations=max_iterations, stop_threshold=stop_threshold,
                           learning_rate=learning_rate, decay_rate=decay_rate, random_seed=random_seed)
    smoothed_signal = heat_kernel_smoothing(signal_data=data, tau=tua, edge_src=edge_src,
                                            edge_dst=edge_dst, edge_distances=edge_distances)

    return smoothed_signal, tua

def find_optimal_tau(fwhm, edge_src, edge_dst, edge_distances, shape, initial_tau=None,
                     max_iterations=50, stop_threshold=0.005, learning_rate=3.0, decay_rate=0.99,
                     random_seed=42, error_threshold=0.5, patience=5):

    # TODO: add momentum to the gradient descent

    start_time = time.time()

    initial_patience = patience

    if initial_tau is None:
        initial_tau = 2 * fwhm
    tau = initial_tau

    np.random.seed(random_seed)
    random_noise = np.random.randn(*shape)
    mae = np.inf
    current_fwhm = np.nan
    best_tau = tau
    best_mae = np.inf
    best_fwhm = current_fwhm
    for i in range(max_iterations):
        smoothed_noise = heat_kernel_smoothing(signal_data=random_noise, tau=tau, edge_src=edge_src,
                                               edge_dst=edge_dst, edge_distances=edge_distances)
        current_fwhm = estimate_fwhm(edge_src=edge_src, edge_dst=edge_dst,
                                     edge_distances=edge_distances, signal_data=smoothed_noise)
        logger.debug(f"Iteration {i}: current fwhm: {current_fwhm:.2f}, target fwhm: {fwhm:.2f} tau: {tau:.2f}")
        previous_mae = mae
        mae = np.abs(current_fwhm - fwhm)
        if mae < best_mae:
            best_mae = mae
            best_tau = tau
            best_fwhm = current_fwhm
        if mae < stop_threshold:
            break
        if i == max_iterations - 1:
            logger.warning(f"Maximum iterations reached ({max_iterations}). ")
            break

        gradient = current_fwhm - fwhm
        previous_tau = tau
        tau -= learning_rate * gradient
        if tau <= 0:
            logger.info(f"Tau became non-positive. Reducing tau to half of previous value.")
            tau = previous_tau / 2
        elif mae > previous_mae:
            logger.warning(f"MAE increased from {previous_mae:.4f} to {mae:.4f}.")
            patience -= 1
            if patience <= 0:
                logger.warning(f"Patience of {initial_patience} iterations exceeded. Stopping gradient descent.")
                break
            else:
                logger.warning(f"Continuing gradient descent with reduced learning rate.")
            learning_rate /= 2
        else:
            learning_rate *= decay_rate
            patience = initial_patience  # reset patience if we made progress

    if mae > error_threshold:
        raise ValueError(f"Failed to achieve target fwhm within error threshold: {error_threshold:.3f}. "
                         f"Final fwhm: {current_fwhm:.2f}, target fwhm: {fwhm:.2f}, "
                         f"tau: {tau:.2f}.")

    end_time = time.time()

    logger.info(f"Gradient descent completed.")
    logger.info(f"Final tau: {best_tau:.2f}, "
                 f"achieved fwhm: {best_fwhm:.2f}, "
                 f"target fwhm: {fwhm:.2f}, "
                 f"time taken: {(end_time - start_time) / 60:.2f} minutes")
    return best_tau
