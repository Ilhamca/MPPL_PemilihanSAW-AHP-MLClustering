import numpy as np


def compute_weights_and_consistency(comparison_matrix: np.ndarray):
    """Compute principal eigenvector weights and CI/CR for a pairwise comparison matrix.

    Returns (weights, lambda_max, CI, CR)
    """
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues.real[max_idx]
    principal = eigenvectors[:, max_idx].real
    weights = principal / principal.sum()

    n = comparison_matrix.shape[0]
    if n > 1:
        CI = (lambda_max - n) / (n - 1)
    else:
        CI = 0.0

    # Saaty's RI table for n up to 10
    RI = {1:0.0, 2:0.0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    RI_n = RI.get(n, None)
    if RI_n is None or RI_n == 0:
        CR = 0.0
    else:
        CR = CI / RI_n

    return weights, lambda_max, CI, CR
