import numpy as np
def norm(v):
    return np.linalg.norm(v)
def normalize(v):
    return v / norm(v)

# def eig(A, iterations = 1000):
#     A = np.copy(A)
    

#     return eigenvalues, eigenvectors
def qe_decomposition(A):
    import numpy as np

def qr_decomposition(A):
    A = A.copy()
    m, n = A.shape

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        #Normalize v to get the j-th column of Q
        R[j, j] = norm(v)           
        Q[:, j] = v / R[j, j] if R[j, j] != 0 else v

    return Q, R

def qr_iteration(A, iterations=1000):
    if (A.T @ A != A @ A.T).any():
        raise ValueError("Matrix must be symmetric for QR iteration to converge to eigenvalues.")
    A = A.copy()
    Q_total = np.eye(A.shape[0])    #Identity matrix initially
    for _ in range(iterations):
        Q, R = qr_decomposition(A)
        A = R @ Q
        Q_total = Q_total @ Q   # = Q1 @ Q2 @ ... @ Qk, total rotation applied to original basis
    return A, Q_total           #Q_total contains the eigenvectors, A converges to a diagonal matrix with eigenvalues on the diagonal

def svd(A):
    # Step 1: Compute ATA
    AtA = A.T @ A
    # Step 2: Eigen-decomposition of ATA
    diag, eigvectors = qr_iteration(AtA)
    eigvals = np.maximum(np.diag(diag), 0)  # Ensure non-negative eigenvalues, since ATA has all non-negative eigenvalues
    Sigma = np.diag(np.sqrt(eigvals))
    V = eigvectors
    #Solve for U by using A = U * Sigma * V^T to get U = A * V * Sigma^-1   (since V is orthogonal, VT * V = I)
    sigma = np.sqrt(eigvals)   # 1D array

    Sigma = np.diag(sigma)      #2D diagonal matrix

    sigma_inv = np.zeros_like(sigma)#Copies shape of sigma, all zeros

    for i in range(len(sigma)):
        if sigma[i] > 1e-12:
            sigma_inv[i] = 1 / sigma[i]

    Sigma_inv = np.diag(sigma_inv)
    # print("Sigma_inv:\n", Sigma_inv)
    # print("Sigma:\n", Sigma)
    U = A @ V @ Sigma_inv
    return U, Sigma, V.T




if __name__ == "__main__":
    A = np.array([[1, 2], [3, 4]])
    Q,R = (qr_decomposition(A.T @ A)) #ATA
    # print("A:\n", A)
    # print("Q:\n", Q)
    # print("R:\n", R)
    qr_iteration(A.T @ A)
    # print("Eigenvalues of ATA:\n", [qr_iteration(A.T @ A)[0][i, i] for i in range(A.shape[1])])
    # print("Eigenvectors of ATA:\n", qr_iteration(A.T @ A)[1])
    U, Sigma, VT = svd(A)
    print("U:\n", U)
    print("Sigma:\n", Sigma)
    print("VT:\n", VT)
    print("Reconstructed A:\n", U @ Sigma @ VT)