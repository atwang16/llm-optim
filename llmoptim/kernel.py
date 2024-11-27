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


def fill_rows(sparse_prob: np.ndarray):
    non_zero_rows = np.where(np.sum(sparse_prob, axis=1) > 0)[0]
    # Check if row 0 is all-zero
    if non_zero_rows[0] != 0:
        # Populate with barycenter
        row1 = np.ones_like(sparse_prob[0]) * 1e-12
        row2 = sparse_prob[non_zero_rows[0]]
        barycenter = compute_barycenter(row1, row2)
        sparse_prob[0:non_zero_rows[0]] = barycenter
    for i, row_idx in enumerate(non_zero_rows):
        if i != len(non_zero_rows) - 1:
            if non_zero_rows[i + 1] - row_idx > 1:
                # Populate with barycenter
                row1 = sparse_prob[row_idx]
                row2 = sparse_prob[non_zero_rows[i + 1]]
                barycenter = compute_barycenter(row1, row2)
                sparse_prob[row_idx + 1:non_zero_rows[i + 1]] = barycenter
    return sparse_prob
