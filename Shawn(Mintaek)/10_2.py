import numpy as np
import time

def house(A):
    """
    Householder algorithm for QR decomposition
    A: m x n matrix
    m >= n
    Returns W, R where W is lower tirangular and R is upper triangular
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    W = np.zeros((m, n))

    for k in range(n):
        x = A[k:, k].copy()
        e1 = np.zeros_like(x)
        e1[0] = 1
        if x[0] >= 0:
            s = 1.0
        else:
            s = -1.0

        v = x + s * np.linalg.norm(x) * e1
        v = v / np.linalg.norm(v)

        W[k:, k] = v
        A[k:, k:] -= 2 * np.outer(v, v @ A[k:, k:])
    R = np.triu(A[:n, :n])
    return W, R

def formQ(W):
    """
    Input: W matrix from householder algorithm
    Output: m x m orthogonal matrix Q
    """
    m, n = W.shape
    Q = np.eye(m)
    for k in range(n-1, -1, -1):
        v = W[k:, k]
        Q[k:, :] -= 2 * np.outer(v, v @ Q[k:, :])
    return Q

def classical_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.vdot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
        
    return Q, R

# Benchmark
sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

print("n, Householder time, CGS time")

for n in sizes:
    """
    Benchmarking Householder and Classical Gram-Schmidt algorithms for n x n matrices
    Compare the time taken and the orthogonality of the resulting Q matrices
    """
    A = np.random.uniform(size=(n, n))

    start = time.time()
    W, R = house(A)
    Q = formQ(W)
    house_time = time.time() - start

    start = time.time()
    Q_cgs, R_cgs = classical_gram_schmidt(A)
    cgs_time = time.time() - start

    print(f"{n}x{n}, householder algorithm: {house_time}s, CGS algorithm: {cgs_time}s")
    house_orth = np.linalg.norm(Q.T @ Q - np.eye(n))
    cgs_orth = np.linalg.norm(Q_cgs.T @ Q_cgs - np.eye(n))

    print("Householder orthogonality error:", house_orth)
    print("CGS orthogonality error:", cgs_orth)
