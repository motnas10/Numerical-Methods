import numpy as np
import matplotlib.pyplot as plt
import random
from LinearSystemSolver import LinearSystemSolver as ls
from RootFinding import RootFinding as rf
from EigenvaluesDecomposition import EigenvaluesDecomposition as ed



if __name__ == "__main__":

    A = np.array([[4,-5],
                  [2, 3]])
    
    B = np.array([[0,11,-5],
                  [-2, 17,-7],
                  [-4,26,-10]])
    
    C = np.array([[1,0.5,-0.5,0.2],
                  [0.5,12,0.8,1],
                  [-0.5,0.8,-16,1],
                  [0.2,1,1,-4]])
    
    D = np.array([[2,0,0.5,-1],
                  [0.5,7,6.5,21],
                  [-2,1,12,-0.5],
                  [0,-0.5,0,18]])
    
    #H = np.hilbert(4)
    def Hilbert(n):
        H = np.zeros((n, n))
        for i in range(1, n+1):
            for j in range(1, n+1):
                H[i-1, j-1] = 1 / (i + j - 1)
        return H
    
    matrix = [A, B, C, D]
    """
    # EXERCISE 1
    
    for m in matrix:
        print()
        print(f"Matrix:\n{m}")
        print(f"Condition number:\n{ls(m, np.zeros(m.shape[0])).ConditionNumber()}")
        solver = ed(m, np.ones(m.shape[0]))
        #solver.GerschgorinCircle()
        lamb_max , autovec = solver.PowerMethod()
        
        print(f"Max lambda:\n{lamb_max}")
        print(f"Autovec:\n{autovec}")
        print()
        print(f"Correct lambda: {np.linalg.eig(m)[0]}")
        print(f"Correct autovec: {np.linalg.eig(m)[1]}")

    for n in [2,4,10,20,50]:
        H = Hilbert(n)
        print()
        print(f"Matrix dimension: {n}")
        print(f"Matrix:\n{H}")
        print(f"Condition number:\n{ls(H, np.zeros(H.shape[0])).ConditionNumber()}")
        solver = ed(H, np.ones(H.shape[0]))
        solver.GerschgorinCircle()
        lamb_max , autovec = solver.PowerMethod()
        print(f"Max lambda:\n{lamb_max}")
        print(f"Autovec:\n{autovec}")
        print()
        print(f"Correct lambda: {np.linalg.eig(H)[0]}")
        print(f"Correct autovec: {np.linalg.eig(H)[1]}")
    
    # EXERCISE 2

    for m in matrix:
        print()
        print(f"Matrix:\n{m}")
        print(f"Condition number:\n{ls(m, np.zeros(m.shape[0])).ConditionNumber()}")
        solver = ed(m, np.ones(m.shape[0]))
        #solver.GerschgorinCircle()
        lamb_min , autovec = solver.InversePowerMethod(sigma=0.1)
        print(f"Min lambda:\n{lamb_min}")
        print(f"Autovec:\n{autovec}")
        print()
        print(f"Correct lambda: {np.linalg.eig(m)[0]}")
        print(f"Correct autovec: {np.linalg.eig(m)[1]}")



    # EXERCISE 3

    for m in matrix:
        print()
        print(f"Matrix:\n{m}")
        print(f"Condition number:\n{ls(m, np.zeros(m.shape[0])).ConditionNumber()}")
        solver = ed(m, np.ones(m.shape[0]))
        #solver.GerschgorinCircle()
        T = solver.QRIteration()
        print(f"Eigenvalues:\n{T}")
        print(f"Autovec:\n{autovec}")
        print()
        print(f"Correct lambda: {np.linalg.eig(m)[0]}")
        print(f"Correct autovec: {np.linalg.eig(m)[1]}")
    """

    # EXERCISE 4

    E = np.array([[1,1,0,0],
                  [1,5,1,0],
                  [1,0,8,1],
                  [1,0,0,11]])
    
    solver = ed(E, np.ones(E.shape[0]))
    #solver.GerschgorinCircle()

    max_l = solver.PowerMethod()[0]
    min_l = solver.InversePowerMethod()[0]
    print()
    print(f"Matrix:\n{E}")
    print(f"Condition number:\n{ls(E, np.zeros(E.shape[0])).ConditionNumber()}")
    print(f"Max eigenvalue: {max_l}")
    print(f"Min eigenvalue: {min_l}")
    print(f"Correct lambda: {np.linalg.eig(E)[0]}")


