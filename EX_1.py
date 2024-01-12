import numpy as np
import matplotlib.pyplot as plt
import random
from LinearSystemSolver import LinearSystemSolver



if __name__ == "__main__":
    """
    # systems to solve
    A1 = np.array([[1, -1,  4],
                [3,  1,  5],
                [1,  3, -1]])
        
    b1 = np.array([10, 15, 6])



    A2 = np.array([[1, -1,  2, -3],
                [2,  1,  0, -1],
                [0,  2,  1,  1],
                [2,  0,  1,  1]])
        
    b2 = np.array([0, 3, -3, 0])



    A3 = np.identity(5)
    b3 = np.array([1, 2, 3, 4, 5])


    # HILBERT MATRIX

    def hilbert_matrix(n):
        H = np.zeros((n, n))
        for i in range(1, n+1):
            for j in range(1, n+1):
                H[i-1, j-1] = 1 / (i + j - 1)
        return H

    #H = hilbert_matrix(5000)
    #bh = np.sum(H, axis=1)
    """

    # LOWER TRIANGULAR MATRIX
    
    print("LOEWER TRIANGULAR MATRIX - TEST")
    L = np.array([[1,0,0,0],
                  [1,1,0,0],
                  [1,1,1,0],
                  [1,1,1,1]])

    b = np.array([1,2,3,4])

    solver = LinearSystemSolver(L, b)
    sol = solver.L_system(L)

    print(f"Matrix:\n{L}")
    print(f"Note:\n{b}")
    print(f"x solution:\n{sol}")
    print(f"Condition number:\n{solver.ConditionNumber()}")
    print(f"L matrix:\n{solver.L}")
    print(f"U matrix:\n{solver.U}")
    #print(f"iter:\n{solver.iter_count}")
    #print(f"error:\n{solver.error}")
    print(f"Residual:\n{solver.Residual()}")

    # UPPER TRIANGULAR MATRIX
    
    print("UPPER TRIANGULAR MATRIX - TEST")
    U = np.array([[1,1,1,1],
                  [0,1,1,1],
                  [0,0,1,1],
                  [0,0,0,1]])

    b = np.array([4,3,2,1])

    solver = LinearSystemSolver(U, b)
    sol = solver.U_system(U)

    print(f"Matrix:\n{U}")
    print(f"Note:\n{b}")
    print(f"x solution:\n{sol}")
    print(f"Condition number:\n{solver.ConditionNumber()}")
    print(f"L matrix:\n{solver.L}")
    print(f"U matrix:\n{solver.U}")
    #print(f"iter:\n{solver.iter_count}")
    #print(f"error:\n{solver.error}")
    print(f"Residual:\n{solver.Residual()}")


    # GAUSSIAN ELIMINATION

    print("GAUSSIAN ELIMINATION - TEST")

    d = np.array([1,1,1,1,1])
    dl = np.array([2,2,2,2])
    du = np.array([3,3,3,3])

    A = np.diag(d) + np.diag(dl, -1) + np.diag(du, 1)
    b = np.array([4,6,6,6,3])

    solver = LinearSystemSolver(A, b)
    sol = solver.GaussElimination()
        

    print(f"Matrix:\n{A}")
    print(f"Note:\n{b}")
    print(f"x solution:\n{sol}")
    print(f"Condition number:\n{solver.ConditionNumber()}")
    print(f"L matrix:\n{solver.L}")
    print(f"U matrix:\n{solver.U}")
    #print(f"iter:\n{solver.iter_count}")
    #print(f"error:\n{solver.error}")
    print(f"Residual:\n{solver.Residual()}")


    # CHOLESKY

    print("CHOLESKY - TEST")

    A = np.array([[1,1,1,1],
                  [1,2,2,2],
                  [1,2,3,3],
                  [1,2,3,4]])
    
    b = np.array([4,7,9,10])

    solver = LinearSystemSolver(A, b)
    sol = solver.Cholesky()        


    print(f"Matrix:\n{solver.M}")
    print(f"Note:\n{b}")
    print(f"x solution:\n{sol}")
    print(f"Condition number:\n{solver.ConditionNumber()}")
    print(f"L matrix:\n{solver.L}")
    print(f"U matrix:\n{solver.U}")
    #print(f"iter:\n{solver.iter_count}")
    #print(f"error:\n{solver.error}")
    print(f"Residual:\n{solver.Residual()}")

    