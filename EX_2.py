import numpy as np
import matplotlib.pyplot as plt
import random
from LinearSystemSolver import LinearSystemSolver



if __name__ == "__main__":
     
    # TEST FOR THOMAS ALGORTIHM
    print("THOMAS - TEST")
    
    size = 5
    d = np.array([1,1,1,1,1])
    dl = np.array([2,2,2,2])
    du = np.array([3,3,3,3])

    A = np.diag(d) + np.diag(dl, -1) + np.diag(du, 1)
    b = np.array([4,6,6,6,3])

    solver = LinearSystemSolver(A, b)
    sol = solver.Thomas()
        

    print(f"Matrix:\n{solver.M}")
    print(f"Note:\n{solver.b}")
    print(f"x solution:\n{sol}")
    print(f"Condition number:\n{solver.ConditionNumber()}")
    print(f"L matrix:\n{solver.L}")
    print(f"U matrix:\n{solver.U}")
    #print(f"iter:\n{solver.iter_count}")
    #print(f"error:\n{solver.error}")
    print(f"Residual:\n{solver.Residual()}")


    # TEST FOR JACOBI ALGORITHM
    print("JACOBI - TEST")
    
    A = np.array([[1,1,1,1],
                  [1,2,2,2],
                  [1,2,3,3],
                  [1,2,3,4]])
    
    b = np.array([4,7,9,10])

    x0 = np.array([2,2,2,2])

    solver = LinearSystemSolver(A, b)
    sol = solver.Jacobi(x0)
        

    print(f"Matrix:\n{A}")
    print(f"Note:\n{b}")
    print(f"x solution:\n{sol}")
    print(f"Condition number:\n{solver.ConditionNumber()}")
    print(f"iter:\n{solver.iter_count}")
    print(f"error:\n{solver.error}")
    print(f"Residual:\n{solver.Residual()}")

    """
    # EXERCIZE 2.2

    # -u''(x) = pi^2 * sen(pi*x)     for x in [0,1]
    # u(0) = u(1) = 0

    def f(x):
        return np.pi**2 * np.sin(np.pi*x)

    dx = 1.e-2
    x = np.arange(0, 1+dx, dx)

    A = np.array([[1,    0,   0],
                  [-dx,  1,   0],
                  [0,    -dx, 1]])
    
    b = np.array([0, 0, 0])

    u = np.zeros(len(x))

    for i in range(1, len(x)):
        b[0] = f(x[i])
        u[i] = LinearSystemSolver(A, b).Thomas()[2]

    
    fig, ax = plt.subplots()
    ax.plot(x, u, label="u(x)")

    plt.show()
    """
