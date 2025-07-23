import numpy as np
from scipy.signal import convolve2d

def improved_pcnn_dynamic_alpha(image_gray, t=60):
    m, n = image_gray.shape
    S = image_gray.copy()
    S_mean = np.mean(S)
    alpha = np.log(1.0 / (S_mean + np.power(S, 0.25) + 1e-8))  # 避免log(0)
    U = np.zeros((m, n))
    E = np.ones((m, n)) * 0.5
    Y = np.zeros((m, n))
    fire_map = np.zeros((m, n))
    W_kernel = np.array([[0.05, 0.20, 0.04],
                         [0.21, 0.00, 0.21],
                         [0.04, 0.20, 0.05]])
    M_kernel = np.array([[0.10, 0.15, 0.10],
                         [0.15, 0.00, 0.15],
                         [0.10, 0.15, 0.10]])

    W = np.exp(-2 * alpha) * W_kernel
    M = np.exp(-3 * alpha) * M_kernel
    V = 1.0
    for n_iter in range(t):
        Y_prev = Y.copy()
        Q = np.exp(-(n_iter + 1) * alpha)
        F_link = convolve2d(Y_prev, W, mode='same', boundary='symm')
        F_mod = convolve2d(Y_prev, M, mode='same', boundary='symm')
        # U[n]
        U = np.exp(-alpha) * U + S * (1 + F_link + F_mod)
        # Y[n]
        Y = (U > E).astype(float)
        # E[n]
        E = np.exp(-3 * alpha) * (E + V) + Q + V * Y

        fire_map += Y
    fire_map = fire_map ** 6
    enhanced = (fire_map - fire_map.min()) / (fire_map.max() - fire_map.min() + 1e-8)

    return enhanced, fire_map
