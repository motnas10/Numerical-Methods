import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp


class NumericalIntegration():

    def __init__(self, bound, m, func):
        self.bound = bound # interval bounds
        self.m = m # numbero of nodes
        self.nodes = np.linspace(self.bound[0], self.bound[1], self.m)
        self.func = func # function to integrate
        self.values = self.func(self.nodes)
        self.integral = None # integral value


    def Midpoint(self):
        """
        Midpoint rule.
        """
        self.integral = 0
        for i in range(self.m-1):
            dx = self.nodes[i+1] - self.nodes[i]
            mid = self.func((self.nodes[i] + self.nodes[i+1])/2)
            self.integral += dx*mid
        
        return self.integral

    def Trapezoidal(self):
        """
        Trapezoidal rule.
        """
        self.integral = 0
        for i in range(self.m-1):
            dx = self.nodes[i+1] - self.nodes[i]
            self.integral += dx*(self.values[i] + self.values[i+1])/2
        
        return self.integral
    

    def CavalieriSimpson(self):
        """
        Cavalieri-Simpson rule.
        """
        self.nodes = np.linspace(self.bound[0], self.bound[1], 2*self.m)
        self.values = self.func(self.nodes)
        sum_odd = np.sum(self.values[1:-1:2])
        sum_even = np.sum(self.values[2:-1:2])

        self.integral = 0

        # for i in range(0,2*self.m-2,2):
        #     dx = self.nodes[i+2] - self.nodes[i]
        #     self.integral += (dx/6)*(self.values[i] + 4*self.values[i+1] + self.values[i+2])
        
        dx = self.nodes[1] - self.nodes[0]
        self.integral = (dx/3)*(self.values[0] + 4*sum_odd + 2*sum_even + self.values[-1])

        return self.integral

    ####################################################################
    
    def error(self, exact, n=1, method=None):
        """
        Error.
        """
        self.integrals = []
        self.errors = []
        for i in range(2,n):
            self.__init__(self.bound, i, self.func)
            method()
            self.integrals.append(self.integral)
            self.errors.append(np.abs(self.integral - exact))

        return self.integrals, self.errors