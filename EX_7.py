import numpy as np
import matplotlib.pyplot as plt
import random
from PolynomialInterpolation import PolynomialIntepolation as pi
from NumericalIntegration import NumericalIntegration as ni

if __name__ == "__main__":

    # EXERCISE 1
    f1 = lambda x: np.sin(x)
    bound = [0, np.pi]
    exx = 2
    n = 10

    grid = np.linspace(bound[0], bound[1], 1000)
    func = f1(grid)

    solver = ni(bound, n, f1)
    print(solver.Trapezoidal())

    fig, ax = plt.subplots()
    ax.plot(grid, func, c='k')
    ax.scatter(solver.nodes, solver.values, marker="x", c='r')
    ax.scatter(solver.nodes, np.zeros(len(solver.nodes)), marker="x", c='r')
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("function")
    ax.set_title("Function to integrate")
    ax.grid()


    n = 21
    fig, ax = plt.subplots(1,2)
    
    for m in [solver.Midpoint, solver.Trapezoidal, solver.CavalieriSimpson]:
        err = solver.error(n=n, exact=exx, method=m)
        ax[0].plot(np.arange(2,n), err[0])
        ax[0].grid()
        ax[0].set_xlabel("degree")
        ax[0].set_ylabel("Integral")
        ax[0].set_title("Integral value")
        ax[0].set_xbound(0,n)
        ax[0].legend(["Midpoint", "Trapezoidal", "Cavalieri-Simpson"])

        ax[1].plot(np.arange(2,n), err[1])
        ax[1].grid()
        ax[1].set_xlabel("degree")
        ax[1].set_ylabel("error")
        ax[1].set_title("Error")
        ax[1].set_xbound(0,n)
        ax[1].set_yscale("log")
        ax[1].legend(["Midpoint", "Trapezoidal", "Cavalieri-Simpson"])

    fig.tight_layout()
    



    plt.show()
