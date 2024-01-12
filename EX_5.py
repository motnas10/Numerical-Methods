import numpy as np
import matplotlib.pyplot as plt
from RootFinding import RootFinding as rf





if __name__ == "__main__":

    # EXERCISE 1
    grid = np.linspace(-20,20,1000)

    f1 = lambda x: x**2 - x - 2
    df1 = lambda x: 2*x - 1

    f2 = lambda x: np.sqrt(x+2) - x

    bound = [0,20]

    solver = rf(f1, bound,df=df1, x0=1, x1=5)
    solver.Newton()
    root = solver.x
    convergence = solver.P
    print(f"Root: {root}")
    print(f"Convergence: {convergence}")
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(grid, f1(grid), c='k')
    ax[0,0].plot(grid, df1(grid), c='b')
    ax[0,0].grid()
    ax[0,0].set_title("function")
    ax[0,1].plot(root)
    ax[0,1].grid()
    ax[0,1].set_title("root")
    ax[1,0].plot(solver.error)
    ax[1,0].grid()
    ax[1,0].set_title("error")
    ax[1,1].plot(convergence)
    ax[1,1].grid()
    ax[1,1].set_title("convergence")
    
    fig.tight_layout()
    plt.show()