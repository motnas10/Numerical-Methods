import numpy as np
import matplotlib.pyplot as plt
import random
from PolynomialInterpolation import PolynomialIntepolation as pi


if __name__ == "__main__":

    # EXERCISE 1
    f1 = lambda x: np.sin(1.3*x)
    bound = [0,2*np.pi]

    #f1 = lambda x: 1/(1+x**2)
    #bound = [-5,5]
    
    degree = 4
    
    nodes = np.linspace(bound[0], bound[1], degree+1)
    values = f1(nodes)

    
    #pol = solver.LagrangeInterpolation()
    k = 9
    error = []
    for d in range(1,10):
        solver = pi(bound, 100, d, f1)
        pol = solver.PiecewiseLagrangeInterpolation(1)
        solver.error()
        error.append(solver.err)



    fig, ax = plt.subplots(1,2)
    ax = ax.flatten()
    ax[0].plot(solver.grid, f1(solver.grid), "--", c='k')
    ax[0].plot(solver.grid, pol, c='b')
    ax[0].scatter(nodes, values, c='r')
    ax[0].scatter(nodes, np.zeros(len(nodes)), marker="x", c='r')
    ax[0].grid()
    ax[0].set_title(f"Lagrange Piecewise-Interpolation with {degree} nodes")
    
    
    ax[1].plot(np.arange(1,10), error)
    ax[1].set_xlabel("degree")
    ax[1].set_ylabel("error")
    ax[1].set_yscale("log")
    ax[1].grid()
    ax[1].set_title("Error")

    fig.tight_layout()

    plt.show()


