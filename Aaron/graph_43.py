import numpy as np
import matplotlib.pyplot as plt

from functions_43 import svd
# def plot_svd(A):
#     #Unit circle
#     t = np.linspace(0, 2*np.pi, 400)
#     circle = np.vstack((np.cos(t), np.sin(t)))

import numpy as np
import matplotlib.pyplot as plt

def plot_svd(A):

    # unit circle
    t = np.linspace(0, 2*np.pi, 400)
    x= np.cos(t)
    y = np.sin(t)
    circle = np.vstack((x, y))

    U, Sigma, VT = svd(A)
    # print("V:\n", VT.T)
    #Left singular vectors (columns of U)
    u1 = U[:, 0]
    u2 = U[:, 1]
    #Right singular vectors (columns of V)
    V = VT.T
    v1 = V[:, 0]
    v2 = V[:, 1]
    S_mat = np.diag(Sigma)
    # V^T applied to circle
    circle_v = VT @ circle

    # final ellipse: A x = U Σ V^T x
    ellipse = A @ circle

    def plot_vec(ax, v, label=None):
        ax.plot([0, v[0]], [0, v[1]], linewidth=2, label=label)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    #------------------------------------------------------------------------------------------------------

    #Graph 1: Unit circle and right singular vectors
    axs[0].plot(circle[0], circle[1], label="unit circle")

    plot_vec(axs[0], v1, "v1")
    plot_vec(axs[0], v2, "v2")
    # print("U:\n", U)

    axs[0].set_title("Right Singular Vectors (V)")
    axs[0].axis("equal")
    #------------------------------------------------------------------------------------------------------
    #Graph 2: Circle transformed by V^T with left singular vectors, stretched by singular values
    axs[1].plot(ellipse[0], ellipse[1])
    # print("Sigma:\n", Sigma)
    # print("u1:\n", u1)
    # print("u2:\n", u2)
    # print("s1*u1",Sigma[0]*u1)

    plot_vec(axs[1], Sigma[0,0]*u1, "σ1 u1")
    plot_vec(axs[1], Sigma[1,1]*u2, "σ2 u2")
    axs[1].set_title("Final: U Σ V^T  (unit circle stretched by A = U Σ V^T)")
    axs[1].axis("equal")

    axs[0].legend()
    axs[1].legend()
    axs[0].grid()
    axs[1].grid()
    fig.text(
    0.01, 0.99,
    f"A =\n{A}",
    fontsize=9,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    
    plt.show()
if __name__ == "__main__":
    A = np.array([[3, 1], [0, 2]])
    plot_svd(A)