from graph_43 import plot_svd
import numpy as np
A1 = np.array(
    [[1,2],
     [0,2]]
)
A2 = np.array(
    [[3,0],
     [0,-2]]
)
A3 = np.array(
    [[3,0],
     [0,-2]]
)
A4 = np.array(
    [[3,0],
     [0,-2]]
)
A5 = np.array(
    [[2,0],
     [0,3]]
)
A6 = np.array(
    [[0,2],
     [0,0],
     [0,0]]
)
A7 = np.array(
    [[1,1],
     [0,0]]
)
A8 = np.array(
    [[1,1],
     [1,1]]
)
for A in [A1, A2, A3, A4, A5, A6, A7, A8]:
    print("A:\n", A)
    plot_svd(A)