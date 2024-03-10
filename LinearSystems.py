import numpy as np
from copy import deepcopy



class LinearSystem():

    def __init__(self, A, b):
        self.M = deepcopy(A)
        self.n, self.m = np.shape(A)
        if self.n != self.m:
            return print("No square matrix provided")
        
        self.b = deepcopy(b)
        self.x = np.ones(len(b))
        self.Mult = np.zeros((self.n, self.n))
        self.U = np.zeros((self.n, self.n))
        self.L = np.zeros((self.n, self.n))

        self.iter_solution = []
        self.iter_error = []


    ########################################################################################
    # DIRECT METHODS FOR LINEAR SYSTEMS

    def L_system(self, L_mat=False, b_vec=False):
        """
        Method for lower triangular system
        ATTRIBUTES NEEDED:
        - M: lower triangular matrix
        - b: vector of known terms
        OUTPUT:
        - x: vector of solutions
        """
        if isinstance(L_mat, np.ndarray):
            self.L = L_mat
        if isinstance(b_vec, np.ndarray):
            self.b = b_vec
        
        # Forward substitution
        for i in range(0, self.n):
            self.x[i] = (self.b[i] - np.sum([self.L[i,k]*self.x[k] for k in range(i)])) / self.L[i,i]

        return self.x

    ########################################

    def U_system(self, U_mat=False, b_vec=False):
        """
        Method for upper triangular system
        ATTRIBUTES NEEDED:
        - M: upper triangular matrix
        - b: vector of known terms
        OUTPUT:
        - x: vector of solutions
        """
        if isinstance(U_mat, np.ndarray):
            self.U = U_mat
        if isinstance(b_vec, np.ndarray):
            self.b = b_vec
        
        # Backward substitution
        self.x[self.n-1] = self.b[self.n-1]/self.U[self.n-1,self.n-1]
        for i in range(self.n-2,-1,-1):
            self.x[i] = (self.b[i]-np.sum([self.U[i,k]*self.x[k] for k in range(i+1, self.n)]))/self.U[i,i]
        
        return self.x
    
    ########################################    

    def GaussElimination(self, piv=False, total=False):
        """
        Method for solving a linear system with Gauss Elimination.
        Partial or Total Pivoting can be performed.
        INPUTS:
        - piv: boolean, True if pivoting is performed
        - total: boolean, True if total pivoting is performed
        ATTRIBUTES NEEDED:
        - M: system matrix
        - b: vector of known terms
        OUTPUT:
        - x: vector of solutions
        """
        A = self.M
        b = self.b
        
        for k in range(self.n):
            # Check diagonal element != 0
            if not piv and A[k,k] == 0:
                raise ValueError("Zero pivot element encountered")
            
            # Perform partial pivoting if diag el == 0
            if piv and not total and A[k,k] == 0:
                self.P_partial(k)
            
            # Perform total pivoting if diag el == 0
            if piv and total and A[k,k] == 0:
                self.P_total(k)

            # Perform elimination
            for i in range(k+1, self.n):
                m = A[i,k] / A[k,k]
                self.Mult[i,k] = m
                b[i] -= m * b[k]
                for j in range(k, self.n): 
                    A[i,j] -= m * A[k,j]

        self.L = np.identity(self.n) + self.Mult
        self.U = A
        return self.U_system()
    
    def P_partial(self, k):
        """
        Method for partial pivoting.
        """
        A = self.M
        b = self.b

        # Find maximum pivot element in current column
        max_index = np.argmax(np.abs(A[k:, k])) + k

        # Swapping rows
        if max_index != k:
            A[[k, max_index]] = A[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]
    
    def P_total(self, k):
        """
        Method for total pivoting.
        """
        A = self.M
        b = self.b

        # Create an array to store the pivot indices
        pivots = np.arange(self.n)

        # Find the indices of the maximum element in the entire submatrix A[k:, k:]
        max_index = np.unravel_index(np.argmax(np.abs(A[k:, k:])), A[k:, k:].shape)
        # Adjust the indices to account for the offset
        max_index = (max_index[0] + k, max_index[1] + k)

        # Swap rows and columns if necessary
        if max_index[0] != k:
            A[[k, max_index[0]]] = A[[max_index[0], k]]
            b[[k, max_index[0]]] = b[[max_index[0], k]]

        if max_index[1] != k:
            A[:, [k, max_index[1]]] = A[:, [max_index[1], k]]

    ########################################

    def Cholesky(self):
        """
        Method for solving a linear system with Cholesky algorithm.
        ATTRIBUTES NEEDED:
        - M: system matrix
        - b: vector of known terms
        OUTPUT:
        - x: vector of solutions
        """
        self.L = np.zeros_like(self.M)        
        
        # Computing L matrix
        for i in range(self.n):
            for j in range(i+1):
                sum = np.sum([self.L[i,k]*self.L[j,k] for k in range(j+1)])
                if i==j:
                    self.L[i,i] = np.sqrt(self.M[i,i] - sum)
                else:
                    self.L[i,j] = (self.M[i,j] - sum)/self.L[j,j]
    
        # Compute solution
        b = deepcopy(self.b)
        self.U = deepcopy(self.L.T)
        self.b = deepcopy(self.L_system())
        self.x = deepcopy(self.U_system(self.U))
        self.b = b

        return self.x
    
    ########################################

    def Thomas(self):
        """
        Method for solving a linear system with Thomas algorithm.
        ATTRIBUTES NEEDED:
        - M: system matrix
        - b: vector of known terms
        OUTPUT:
        - x: vector of solutions
        """
        # Extracting diagonals from M
        a = np.diag(self.M)
        c = np.diag(self.M, k=1)
        b = np.diag(self.M, k=-1)
        # Initialize L diagonals
        Ld = np.ones(self.n)
        beta = np.ones(self.n-1)
        # Initialize U diagonals
        alfa = np.ones(self.n)
        gamma = np.ones(self.n-1)

        # LU DECOMPOSITION
        # - Gamma entries
        gamma[:] = c[:]
        # - Alfa and Beta entries
        for i in range(self.n-1):
            if i == 0:
                alfa[0] = a[0]
                beta[0] = b[0]/alfa[0]
            else:
                alfa[i] = a[i] - beta[i-1]*gamma[i-1]
                beta[i] = b[i]/alfa[i]
        
        alfa[-1] = a[-1] - beta[-1]*gamma[-1]

        # Fill L matrix
        np.fill_diagonal(self.L, Ld)
        np.fill_diagonal(self.L[1:], beta)
        # Fill U matrix
        np.fill_diagonal(self.U, alfa)
        np.fill_diagonal(self.U[:,1:], gamma)

        # Compute solution
        b = deepcopy(self.b)
        self.b = deepcopy(self.L_system())
        self.x = self.U_system()
        self.b = b
        return self.x
    
    ########################################################################################
    # ITERATIVE METHODS FOR LINEAR SYSTEMS

    def Jacobi(self, x0, tol=1e-3, max_iter=500):
        """
        Method for solving a linear system with Jacobi algorithm.
        INPUTS:
        - x0: initial guess
        - tol: tolerance
        - max_iter: maximum number of iterations
        ATTRIBUTES NEEDED:
        - M: system matrix
        - b: vector of known terms
        OUTPUT:
        - x: vector of best solutions
        """
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        
        # Compute the inverse of the diagonal matrix D
        D = np.diag(np.diag(self.M))
        D_inv = np.linalg.inv(D)

        # Jacobi iterative matrix
        J = self.M - D

        # Initialization
        x = x0
        x_new = D_inv @ (self.b - J @ x)
        self.iter_count = 0
        self.res = np.linalg.norm(self.b - np.dot(self.M, x))
        self.error = np.inf
        
        while self.error > tol and self.iter_count < max_iter:
            x = x_new.copy()
            x_new = J @ x + D_inv @ self.b
            self.res = np.linalg.norm(self.b - np.dot(self.M, x_new))           
            self.error = np.linalg.norm(x_new - x)
            self.iter_solution.append(x)
            self.iter_error.append(self.error)
            self.iter_count += 1

        self.x = x
        return x
    
    ########################################

    def GaussSeidel(self,  x0, tol=1e-3, max_iter=100):
        """
        Method for solving a linear system with Gauss-Seidel algorithm.
        INPUTS:
        - x0: initial guess
        - tol: tolerance
        - max_iter: maximum number of iterations
        ATTRIBUTES NEEDED:
        - M: system matrix
        - b: vector of known terms
        OUTPUT:
        - x: vector of best solutions
        """
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.rho = np.max(np.abs(np.linalg.eigvals(np.inv(np.tril(self.M)))))
        
        x = x0.astype(float)
        x_new = np.zeros_like(x)
        self.iter_count = 0
        self.error = np.inf

        while self.error > tol and self.iter_count < max_iter:
            for i in range(self.n):
                s1 = 0
                s2 = 0
                for j in range(self.n):
                    if j<i:
                        s1 += self.M[i][j] * x[j]
                    elif j>i:
                        s2 += self.M[i][j] * x[j]

                x_new[i] = (self.b[i] - s1 - s2) / self.M[i][i]

                """
                s1 = np.dot(self.M[i,:i], x[:i])
                s2 = np.dot(self.M[i,i+1:], x[i+1:])
                x_new[i] = (self.b[i] - s1 - s2) / self.M[i][i]
                """
            
            self.error = np.linalg.norm(x_new - x)
            x = x_new.copy()
            self.iter_solution.append(x)
            self.iter_error.append(self.error)
            self.iter_count += 1

        self.x = x
        return x
    
    ########################################

    def GramSchmidt(self):
        """
        Method for computing the QR decomposition of a matrix.
        """
        self.Q = np.zeros((self.m, self.n))
        self.R = np.zeros((self.n, self.n))

        for j in range(self.n):
            # q = a_j
            q = self.M[:,j]
            for i in range(j):
                # q_hat = a_j - SUM {q_j * (q_jT * a_j)}
                q -= np.dot(self.Q[:,i], self.M[:,j]) * self.Q[:,i]
                # set R matrix elements
                self.R[i,j] = np.dot(self.Q[:,i], self.M[:,j])
            # setting R matrix diagonal elements as norm of q_hat
            self.R[j,j] = np.linalg.norm(q)
            # setting Q matrix columns as q_hat / ||q_hat||
            self.Q[:,j] = q / np.linalg.norm(q)

        self.L = self.Q
        self.U = self.R
        
        self.b = deepcopy(self.L_system())
        self.x = self.U_system()

        return self.x, self.Q, self.R

    ########################################

    def Householder(self):
        """
        Method for computing the QR decomposition of a matrix.
        """
        self.Q = np.eye(self.m)
        self.R = self.M.copy()

        for k in range(self.n):
            # take k-th column from k-th row
            x = self.R[k:,k]
            # construct the first canonical vector
            # with first entry equal to the sign first entry of x
            e = np.zeros_like(x)
            e[0] = np.sign(x[0])
            # construct the v vector
            v = x + e * np.linalg.norm(x)
            v /= np.linalg.norm(v)
            # update R and Q matrices
            self.R[k:,k:] -= 2 * np.outer(v, np.dot(v, self.R[k:,k:]))
            self.Q[k:,:] -= 2 * np.outer(v, np.dot(v, self.Q[k:,:]))

        self.L = self.Q.T
        self.U = self.R
        self.b = deepcopy(self.L_system())
        self.x = self.U_system()

        return self.x, self.Q.T, self.R

    ########################################

    def GivenRotation(self):
        self.Q = np.eye(self.m)
        self.R = self.M.copy()

        for j in range(self.n):
            for i in range(j+1, self.m):
                if self.R[i,j] != 0:
                    r = np.hypot(self.R[j,j], self.R[i,j])
                    c = self.R[j,j] / r
                    s = -self.R[i,j] / r
                    G = np.array([[c, -s], [s, c]])
                    self.R[[j,i],j:] = G @ self.R[[j,i],j:]
                    self.Q[:,[j,i]] = self.Q[:,[j,i]] @ G.T
        
        self.L = self.Q.T
        self.U = self.R
        self.b = deepcopy(self.L_system())
        self.x = self.U_system()

        return self.x, self.Q.T, self.R
    
    ########################################

    def LeastSquaresNormal(self):
        """
        Method for computing the least squares solution with normal equations.
        """
        M = deepcopy(self.M)
        b = deepcopy(self.b)
        self.M = self.M.T @ self.M
        self.b = self.M.T @ self.b
        self.x = self.GaussElimination()
        self.M = M
        self.b = b

        return self.x

    def LeastSquaresQR(self):
        """
        Method for computing the least squares solution with QR decomposition.
        """
        M = deepcopy(self.M)
        b = deepcopy(self.b)
        Q, R = np.linalg.qr(self.M)
        self.M = R
        self.b = Q.T @ self.b
        self.x = self.GaussElimination()
        self.M = M
        self.b = b
        return self.x

    ################################################################################################
    # EVALUATION MATRIX

    def ConditionNumber(self):
        """
        Method for computing the condition number of the matrix.
        """
        return np.linalg.cond(self.M)

    def Residual(self):
        """
        Method for computing the residual.
        """
        print(self.b)
        return np.linalg.norm(self.b - np.dot(self.M, self.x))

    def IsDiagonallyDominant(self):
        """
        Method for checking if a matrix is diagonally dominant.
        """
        for i in range(self.n):
            if self.M[i,i] <= np.sum(np.abs(self.M[i,:])) - np.abs(self.M[i,i]):
                return print("Matrix NOT diagonally dominant")
        
        return print("Matrix diagonally dominant")
    
    def Get_Spectral_Radius(self, A):
        return np.max(np.abs(np.linalg.eigvals(A)))



