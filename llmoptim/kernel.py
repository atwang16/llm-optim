import numpy as np
import ot
import torch


def compute_barycenter(row1, row2, reg=0.01, numItermax=1000, stopThr=1e-5):
    n = len(row1)
    x = np.arange(n)
    M = (x[:, None] - x[None, :]) ** 2
    P = np.vstack((row1, row2)).T
    barycenter = ot.bregman.barycenter_debiased(P, M, reg=reg, weights=np.array([0.5, 0.5]), method='sinkhorn', numItermax=numItermax, stopThr=stopThr)
    return barycenter


def fill_rows(P):
    non_zero_rows = np.where(np.sum(P, axis=1) > 0)[0]
    for i, row_idx in enumerate(non_zero_rows):
        if i != len(non_zero_rows) - 1:
            if non_zero_rows[i + 1] - row_idx > 1:
                # Populate with barycenter
                row1 = P[row_idx]
                row2 = P[non_zero_rows[i + 1]]
                barycenter = compute_barycenter(row1, row2)
                P[row_idx + 1:non_zero_rows[i + 1]] = barycenter
    return P