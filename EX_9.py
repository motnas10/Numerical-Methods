import numpy as np
import matplotlib.pyplot as plt
import random
from PolynomialInterpolation import PolynomialIntepolation as pi
from NumericalIntegration import NumericalIntegration as ni
from ODEIntegration import ODEIntegration as ode


if __name__ == "__main__":

    # EXERCISE 1
    f = lambda t, y: -y*np.log(y)
    
    # def f(t, y):
    #     a = y[0]
    #     y = np.array([a])
    #     return y
    
    y0 = np.array([0.5])
    t0 = 0
    T = 5
    dt = 0.1
    f_exact = lambda t: np.exp(-np.exp((np.log(np.log(2))-t)))

    

    # Numerical solution
    solver = ode(f, t0, y0, T, dt)
    t_RK2, y_RK2 = solver.RungeKutta2()
    p_RK2 = solver.convergence_order(f_exact, method=solver.RungeKutta2)

    p_RK2 = np.zeros(len(t_RK2))
    for i in range(1,len(t_RK2)-1):
        #p_RK2[i-1] = abs(np.log(y_RK2[0,i-1]/y_RK2[0,i])/np.log(2))
        p_RK2[i-1] = abs(np.log((y_RK2[0,i-1]-y_RK2[0,i+1])/(y_RK2[0,i]-y_RK2[0,i+1]))/np.log(2))

    ddt = [0.1/(2**i) for i in range(6)]
    err_RK2 = []
    for dt in ddt:
        solver = ode(f, t0, y0, T, dt)
        t_RK2, y_RK2 = solver.RungeKutta2()
        err_RK2.append(solver.error(f_exact))

    solver = ode(f, t0, y0, T, dt)
    t_RK4, y_RK4 = solver.RungeKutta4()
    p_RK4 = solver.convergence_order(f_exact, method=solver.RungeKutta4)

    p_RK4 = np.zeros(len(t_RK4)-1)
    for i in range(1,len(t_RK4)-1):
        #p_RK4[i-1] = abs(np.log(y_RK4[0,i-1]/y_RK4[0,i])/np.log(2))
        p_RK4[i-1] = abs(np.log((y_RK4[0,i-1]-y_RK4[0,i+1])/(y_RK4[0,i]-y_RK4[0,i+1]))/np.log(2))

    err_RK4 = []
    for dt in ddt:
        solver = ode(f, t0, y0, T, dt)
        t_RK4, y_RK4 = solver.RungeKutta4()
        err_RK4.append(solver.error(f_exact))
    

    fig, ax = plt.subplots(2,3)
    ax = ax.flatten()
    ax[0].set_title("Runge-Kutta 2")
    ax[0].plot(t_RK2, y_RK2[0,:], c='k')
    ax[0].scatter(t_RK2, f_exact(t_RK2), c='r', alpha=0.5, s=5)
    ax[0].grid()
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("y")
    ax[0].legend(["Numerical solution", "Exact solution"])

    ax[3].set_title("Runge-Kutta 4")
    ax[3].plot(t_RK4, y_RK4[0,:], c='k')
    ax[3].scatter(t_RK4, f_exact(t_RK4), c='r', alpha=0.5, s=5)
    ax[3].grid()
    ax[3].set_xlabel("t")
    ax[3].set_ylabel("y")
    ax[3].legend(["Numerical solution", "Exact solution"])

    ax[1].set_title("ERROR - Runge-Kutta 2")
    ax[1].plot(ddt, err_RK2, c='k')
    ax[1].grid()
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")

    ax[4].set_title("ERROR - Runge-Kutta 4")
    ax[4].plot(ddt, err_RK4, c='k')
    ax[4].grid()
    ax[4].set_xscale("log")
    ax[4].set_yscale("log")

    ax[2].set_title("CONVERGENCE - Runge-Kutta 2")
    ax[2].plot(p_RK2[:-2], c='k')
    ax[2].grid()
    ax[2].set_ylim(0,2)
    #ax[2].set_xscale("log")
    #ax[2].set_yscale("log")

    ax[5].set_title("CONVERGENCE - Runge-Kutta 4")
    ax[5].plot(p_RK4[:-2], c='k')
    ax[5].grid()
    ax[5].set_ylim(0,2)
    #ax[5].set_xscale("log")
    #ax[5].set_yscale("log")

    fig.tight_layout()
    plt.show()


