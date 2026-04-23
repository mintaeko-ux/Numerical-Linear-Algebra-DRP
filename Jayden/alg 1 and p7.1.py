import numpy as np

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
    
# Define matrix A as given from Exericise 6.4
A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

Q1, R1 = classical_gram_schmidt(A)
Ar = Q1 @ R1
difference = A - Ar
frobenius_err = np.linalg.norm(difference, ord='fro')

print("Q1:", np.round(Q1, 4))
print("R1:", np.round(R1, 4))
print("Frobenius Norm of (A-QR):", np.round(frobenius_err, 4))

# Define matrix B as given in exercise 6.4 (part b)
B = np.array([
    [1, 2],
    [0, 1],
    [1, 0]
], dtype=float)

Q2, R2 = classical_gram_schmidt(B)
Br = Q2 @ R2
diffB = B - Br
frobenius_err_B = np.linalg.norm(diffB, ord='fro')

print("Q2:", np.round(Q2, 4))
print("R2:", np.round(R2, 4))
print("Frobenius Norm of (B-QR):", np.round(frobenius_err_B, 4))