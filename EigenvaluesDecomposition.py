import numpy as np
import LinearSystemSolver as ls
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from copy import deepcopy



class EigenvaluesDecomposition():

    def __init__(self, matrix, x0, tol=1e-3, max_iter=100):
        self.n, self.m = matrix.shape
        self.M = matrix
        self.T = None
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.max_l = 0
        self.min_l = np.inf
        self.r = 0
        
    def PowerMethod(self,):
        """
        Power method for finding the dominant eigenvalue and eigenvector.
        """
        y = self.M @ self.x0
        x_new = y / np.linalg.norm(y)
        self.max_l = x_new.conj().T @ self.M @ x_new
        self.r = self.M @ x_new - self.max_l*x_new
        self.iter = 0

        while np.linalg.norm(self.r)/np.abs(self.max_l) > self.tol and self.iter < self.max_iter:
            y = self.M @ x_new
            x_new = y / np.linalg.norm(y)
            self.max_l = x_new.conj().T @ self.M @ x_new
            self.r = self.M @ x_new - self.max_l*x_new
            self.iter += 1

        return self.max_l, x_new


    def InversePowerMethod(self, sigma=0):
        """
        Inverse power method for finding the smallest eigenvalue and eigenvector.
        """
        M = deepcopy(self.M)
        self.M = self.M - sigma*np.eye(self.n)

        y = ls.LinearSystemSolver(self.M, self.x0).GaussElimination()
        x_new = y / np.linalg.norm(y)
        self.min_l = x_new.conj().T @ y
        self.r = self.M @ x_new - self.min_l*x_new
        self.iter = 0

        while np.linalg.norm(self.r)/np.abs(self.min_l) > self.tol and self.iter < self.max_iter:
            y = ls.LinearSystemSolver(self.M, x_new).GaussElimination()
            x_new = y / np.linalg.norm(y)
            self.min_l = x_new.conj().T @ y
            self.r = self.M @ x_new - self.min_l*x_new
            self.iter += 1
        
        self.M = M
        #self.min_l = 1/(self.min_l + sigma)
        self.min_l = 1/self.min_l + sigma
        return self.min_l, x_new


    def QRIteration(self):
        """
        QR iteration for finding the eigenvalues and eigenvectors.
        """
        T = self.M
        #_, Q, R = ls.LinearSystemSolver(T, self.x0).GramSchmidt()
        Q, R = np.linalg.qr(T)
        T = Q @ R
        T_new = R @ Q
        self.r = np.linalg.norm(T - T_new)
        self.iter = 0

        while self.r > self.tol and self.iter < self.max_iter:
            #_, Q, R = ls.LinearSystemSolver(T_new, self.x0).GramSchmidt()
            Q, R = np.linalg.qr(T_new)
            T = Q @ R
            T_new = R @ Q
            self.r = np.linalg.norm(T - T_new)
            self.iter += 1

        self.T = T_new
        return T_new
    
    ##################################################################

    def GerschgorinCircle(self, matrix=None):
        if isinstance(matrix, np.ndarray):
            self.M = matrix
            self.n, self.m = self.M.shape

        centers = np.diag(self.M)
        centers_x = np.real(centers)
        centers_y = np.imag(centers)
        centers_c = list(zip(centers_x, centers_y))
        
        radii = np.sum(np.abs(self.M), axis=1) - np.abs(centers)
        
        fig, ax = plt.subplots()
        for i in range(self.n):
            center = centers_c[i]
            circle = Circle((center[0], center[1]), radii[i], fill=True, alpha=0.5)
            ax.add_artist(circle)
        ax.set_aspect('equal')
        ax.grid()
        max = np.max(np.abs(centers)+radii)
        ax.set_xlim(-max-1, max+1)
        ax.set_ylim(-max-1, max+1)
        ax.set_xlabel('Real axis')
        ax.set_ylabel('Imaginary axis')
        plt.title('Gerschgorin circles')
        plt.show()



if __name__ == "__main__":

    # TEST FOR GERSCHGORIN CIRCLES
    print("GERSCHGORIN CIRCLES - TEST")
    A = np.array([[2,1+4j,0],
                  [1,10j,1],
                  [0,1,1]])
    
    solver = EigenvaluesDecomposition(A)
    solver.GerschgorinCircle()
