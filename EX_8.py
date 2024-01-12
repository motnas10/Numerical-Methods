import numpy as np
import matplotlib.pyplot as plt
import random
from PolynomialInterpolation import PolynomialIntepolation as pi
from NumericalIntegration import NumericalIntegration as ni
from ODEIntegration import ODEIntegration as ode


if __name__ == "__main__":

    # EXERCISE 1
    f = lambda t, y: y
    
    def f(t, y):
        a = y[0]
        y = np.array([a])
        return y
    
    y0 = np.array([0.5])
    t0 = 0
    T = 5
    dt = 0.1
    f_exact = lambda t: np.exp(t)

    

    # Numerical solution
    solver = ode(f, t0, y0, T, dt)
    t_FE, y_FE = solver.ForwardEuler()
    p_FE = solver.convergence_order(f_exact, method=solver.ForwardEuler)
    
    t_BE, y_BE = solver.BackwardEuler()
    p_BE = solver.convergence_order(f_exact, method=solver.BackwardEuler)
    
    t_H, y_H = solver.Heun()
    p_H = solver.convergence_order(f_exact, method=solver.Heun)
    
    t_CN, y_CN = solver.CranckNicholson()
    p_CN = solver.convergence_order(f_exact, method=solver.CranckNicholson)
    
    t = [t_FE, t_BE, t_H, t_CN]
    y = [y_FE, y_BE, y_H, y_CN]
    p = [p_FE, p_BE, p_H, p_CN]

    
    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    fig.suptitle("Numerical solution")
    ax[0].set_title("Forward Euler")
    ax[1].set_title("Backward Euler")
    ax[2].set_title("Heun")
    ax[3].set_title("Cranck-Nicholson")

    for i in range(4):
        for j in range(len(y[i])):
            ax[i].plot(t[i], y[i][j,:], c='k')
            ax[i].scatter(t[i], f_exact(t[i]), c='r', alpha=0.5, s=5)
            ax[i].grid()
            ax[i].set_xlabel("t")
            ax[i].set_ylabel("y")
            ax[i].legend(["Numerical", "Exact"])

    # ax[0].plot(t[0], y[0][0,:])
    # ax[0].plot(t[0], y[0][1,:])

    #ax[0].plot(t_FE, y_FE[0,:])
    #ax[0].plot(t[0], y[1,:])
    
    fig.tight_layout()

    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    fig.suptitle("Error")
    ax[0].set_title("Forward Euler")
    ax[1].set_title("Backward Euler")
    ax[2].set_title("Heun")
    ax[3].set_title("Cranck-Nicholson")

    for i in range(4):
        for j in range(len(y[i])):
            ax[i].plot(t[i], np.abs(y[i][j,:]-f_exact(t[i])), c='k')
            ax[i].grid()
            ax[i].set_xlabel("t")
            ax[i].set_ylabel("error")
            ax[i].set_yscale("log")
    
    fig.tight_layout()


    # fig, ax = plt.subplots(2,2)
    # ax = ax.flatten()
    # fig.suptitle("Convergence")
    # ax[0].set_title("Forward Euler")
    # ax[1].set_title("Backward Euler")
    # ax[2].set_title("Heun")
    # ax[3].set_title("Cranck-Nicholson")

    # for i in range(4):
    #     ax[i].scatter(np.arange(len(p[i])), p[i], c='k')
    #     ax[i].grid()
    #     ax[i].set_xlabel("degree")
    #     ax[i].set_ylabel("order")
    # fig.tight_layout()



    plt.show()
