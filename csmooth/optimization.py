import numpy as np


from csmooth.fwhm import estimate_fwhm
from csmooth.heat import heat_kernel_smoothing


def graph_smoothing_with_gradient_descent(fwhm, max_iterations=100, stop_threshold=0.01, tau=None, learning_rate=1.0,
                                          decay_rate=0.99, **kwargs):
    """
    Smooth a signal based on a graph targeting a specific fwhm.
    This function uses gradient descent to find the optimal smoothing parameter
    to achieve the target fwhm.
    This can be used on an image with random noise to find a value for tau
    that approximates the desired fwhm.
    :param fwhm: target fwhm in mm.
    :param max_iterations: maximum number of iterations for gradient descent.
    :param stop_threshold: threshold for stopping the gradient descent.
    :param tau: initial value for the smoothing parameter.
    :param learning_rate: learning rate for gradient descent.
    :param decay_rate: decay rate for the learning rate.
    :param kwargs: see graph_signal_smoothing for the other parameters.
    :return: smoothed signal, optimal tau, and the fwhm of the smoothed signal.
    """
    if tau is None:
        tau = fwhm
    smoothed_signal = None
    for i in range(max_iterations):
        smoothed_signal = heat_kernel_smoothing(tau=tau, **kwargs)
        current_fwhm = estimate_fwhm(edge_src=kwargs["edge_src"], edge_dst=kwargs["edge_dst"],
                                     edge_distances=kwargs["edge_distances"], signal_data=smoothed_signal)
        print(f"Iteration {i}: current fwhm: {current_fwhm:.2f}, target fwhm: {fwhm:.2f} tau: {tau:.2f}")
        mae = np.abs(current_fwhm - fwhm)
        if mae < stop_threshold:
            break
        gradient = (current_fwhm - fwhm) / mae
        tau -= learning_rate * gradient * mae
        learning_rate *= decay_rate
    return smoothed_signal, tau, fwhm

