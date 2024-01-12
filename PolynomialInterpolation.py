import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.linear_model import LinearRegression
from copy import deepcopy



class PolynomialIntepolation():

    def __init__(self, bound, N, n, func):
        self.bound = bound # Interval bounds
        self.N = N # Number of points in the grid
        self.n = n # Degree of the polynomial
        self.n_pw = None # Degree of piecewise polynomial
        self.k = None # Degree of piecewise polynomial
        self.func = func # Function to interpolate

        self.nodes = np.linspace(self.bound[0], self.bound[1], self.n+1)
        self.nodes_pw = np.linspace(self.bound[0], self.bound[1], self.n+1)
        self.grid = np.linspace(self.bound[0], self.bound[1], self.N)
        self.grid_pw = np.linspace(self.bound[0], self.bound[1], self.N)
        try:
            self.values = self.func(self.nodes)
            self.m = 1
        except TypeError:
            self.values = self.func
            self.m = len(self.values)
        
        self.NP = np.ones((self.n,self.N)) # Nodal Polynomial
        self.LB = np.ones((self.n+1,self.N)) # Lagrange Basis
        self.HB = np.ones((self.m+1, self.n+1)) # Hermite Basis
        
        self.poly = np.zeros(self.N)
        self.poly_pw = np.zeros(self.N)

        self.err = None

    def NodalPolynomial(self, x=None):
        """
        Nodal polynomial.
        """
        if isinstance(x, np.ndarray):
            self.grid = x

        for i in range(1, self.n):
            self.NP[i,:] = np.prod([self.grid[:]-self.nodes[j] for j in range(i)])
        
        return self.NP

    ######################################################################

    def LagrangeBasis(self, x=None):
        """
        Lagrange basis polynomials.
        """
        if isinstance(x, np.ndarray):
            self.grid = x
        
        for i in range(self.n+1):
            self.LB[i,:] = np.prod([(self.grid[:]-self.nodes[j])/(self.nodes[i]-self.nodes[j]) for j in range(self.n+1) if j != i], axis=0)

        return self.LB

    def LagrangeInterpolation(self, x=None):
        """
        Lagrange interpolation.
        """
        if isinstance(x, np.ndarray):
            self.grid = x
        
        self.LagrangeBasis(x)
        #print(self.values)
        self.poly = np.sum([self.values[i]*self.LB[i] for i in range(self.n+1)], axis=0)
        return self.poly
    

    def PiecewiseLagrangeInterpolation(self, k, x=None):
        """
        Piecewise Lagrange interpolation.
        INPUTS:
        - k: number of subintervals
        """
        if isinstance(x, np.ndarray):
            self.grid = x

        # Iniitialize nodes vec for each interval
        #self.nodes = np.zeros(k+1)
        # Initialize a poly-vec to which concatenate the
        # values of poly in each interval
        self.poly_pw = np.zeros(2)
        # Set self.n to k in order to give the correct
        # poly degree to the LagrangeBasis method
        self.n_pw = self.n
        self.n = k
        self.LB = np.ones((self.n+1,self.N)) # Lagrange Basis
        # Cycle over the subintervals
        for i in range(len(self.nodes_pw)-1):
            # Genertae k+1 nodes for each subinterval
            self.nodes = np.linspace(self.nodes_pw[i], self.nodes_pw[i+1], self.n+1)
            # Generate the grid for each subinterval
            self.grid = np.linspace(self.nodes_pw[i], self.nodes_pw[i+1], self.N)
            self.grid_pw = np.concatenate((self.grid_pw, self.grid))
            # Compute the values of the function in the nodes
            # if a callable function is passed
            try:
                self.values = self.func(self.nodes)
                self.m = 1
            # otherwise set the values to the values
            # if a vector is passed
            except TypeError:
                self.values = self.func
                self.m = len(self.values)
            
            # Compute the LagrangeBase for each subinterval
            self.LagrangeBasis(x)
            # Compute the polynomial for each subinterval
            poly = np.sum([self.values[i]*self.LB[i] for i in range(self.n+1)], axis=0)
            # Concatenate the polynomial values
            self.poly_pw = np.concatenate((self.poly_pw, poly))
        
        # Remove the first two elements of the poly_pw vector
        
        self.poly = deepcopy(self.poly_pw[2:])
        self.grid = deepcopy(self.grid_pw[self.N:])
        return self.poly

    ######################################################################
    
    def HermiteBasis(self, x=None):
        """
        Hermite basis polynomials.
        """
        if isinstance(x, np.ndarray):
            self.grid = x
        
        for j in range(self.m):
            for i in range(self.n):
                prod = np.prod([((self.grid[:]-self.nodes[k])/(self.nodes[i]-self.nodes[k]))**(j+1) for k in range(self.n) if k != i], axis=0)
                self.HB[j, i] = (self.grid[:]-self.nodes[i])**j/(np.factorial(j)) * prod
        
        return self.HB

    def HermiteInterpolation(self, x=None):
        if isinstance(x, np.ndarray):
            self.grid = x

        self.HermiteBasis(x)
        self.poly = 0
        for i in range(self.n):
            for k in range(self.m):
                self.poly += self.values[k]*self.HB[k,i]
                
        return self.poly
    

    ######################################################################

    def error(self):
        """
        Error of the interpolation.
        """
        try:
            self.err = np.max(abs(self.func(self.grid)-self.poly))
        except TypeError:
            self.err =  np.max(abs(self.func-self.poly))
        
        return self.err

        