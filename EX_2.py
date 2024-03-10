import numpy as np
import matplotlib.pyplot as plt
import random
from LinearSystems import LinearSystem




if __name__ == "__main__":
    """
    # TEST FOR THOMAS ALGORTIHM
    print("THOMAS - TEST")
    
    size = 5
    d = np.array([1,1,1,1,1])
    dl = np.array([2,2,2,2])
    du = np.array([3,3,3,3])

    A = np.diag(d) + np.diag(dl, -1) + np.diag(du, 1)
    b = np.array([4,6,6,6,3])

    solver = LinearSystem(A, b)
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

    """

    # TEST FOR JACOBI ALGORITHM
    print("JACOBI - TEST")
    
    A = np.array([[1,1,1,1],
                  [1,2,2,2],
                  [1,2,3,3],
                  [1,2,3,4]])
    
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]])

    b = np.array([4,7,9,10])
    
    b = np.array([15, 10, 10, 10])

    x0 = np.array([2,2,2,2])

    solver = LinearSystem(A, b)
    sol = solver.Jacobi(x0)
        

    print(f"Matrix:\n{A}")
    print(f"Note:\n{b}")
    print(f"x solution:\n{sol}")
    print(f"Condition number:\n{solver.ConditionNumber()}")
    print(f"iter:\n{solver.iter_count}")
    print(f"error:\n{solver.error}")
    print(f"Residual:\n{solver.Residual()}")
    
    """
    # TEST FOR GAUSS-SEIDEL ALGORITHM
    print("GAUSS-SEIDEL - TEST")

    A = np.array([[1,1,1,1],
                  [1,2,2,2],
                  [1,2,3,3],
                  [1,2,3,4]])
    
    b = np.array([4,7,9,10])

    x0 = np.array([2,2,2,2])

    solver = LinearSystem(A, b)
    sol = solver.GaussSeidel(x0)

    print(f"Matrix:\n{A}")
    print(f"Note:\n{b}")
    print(f"x solution:\n{sol}")
    print(f"Condition number:\n{solver.ConditionNumber()}")
    print(f"iter:\n{solver.iter_count}")
    print(f"error:\n{solver.error}")
    print(f"Residual:\n{solver.Residual()}")

    """

