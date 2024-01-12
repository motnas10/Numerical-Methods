import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class RootFinding():

    def __init__(self, f, boundary, x0, df=None, x1=None, tol=1e-6, max_iter=1000):
        self.f = f # funvtion
        self.df = df # derivative
        self.boundary = boundary # boundary
        self.x = [] # iterative roots
        self.x0 = x0 # initial guess
        self.x1 = x1   # initial guess
        self.tol = tol # tolerance
        self.max_iter = max_iter # max iterations
        self.iter = 0 # iterations
        self.error = [] # error
        self.P = [] # convergence
    
    def convergence(self):
        """
        Convergence of the root finding method.
        """
        for i in range(len(self.x)-2):
            p = np.log2(abs(self.x[i+1]-self.x[-1]))/np.log2(abs(self.x[i]-self.x[-1]))
            self.P.append(p)
            self.error.append(abs(self.x[i+1]-self.x[-1]))

    #########################################################################

    def Bisection(self):
        """
        Bisection method for finding the root of a function.
        """
        a, b = self.boundary
        K = np.log2((b-a)/self.tol)-1
        self.iter = 0
        while self.iter < K:
            x_new = (a+b)/2
            self.x.append(x_new)
            if np.abs(self.f(x_new))<self.tol:
                break
            elif self.f(x_new)*self.f(a) < 0:
                b = x_new
            else:
                a = x_new
            self.iter += 1

        self.convergence()

    
    def Newton(self):
        """
        Newton method for finding the root of a function.
        """
        self.iter = 0
        self.error = [np.inf]
        self.x.append(self.x0)

        while min(self.error) > self.tol and self.iter < self.max_iter:
            x = self.x[self.iter]
            x_new = x - self.f(x)/self.df(x)
            self.error.append(np.linalg.norm(x_new - x))
            self.x.append(x_new)
            self.iter += 1

        self.convergence()
    
    def Secant(self):
        """
        Secant method for finding the root of a function.
        """
        self.iter = 0
        self.error = [np.inf]
        self.x.append(self.x0)
        self.x.append(self.x1)

        while min(self.error) > self.tol and self.iter < self.max_iter:
            #print(self.x)
            x0 = self.x[self.iter]
            x1 = self.x[self.iter+1]
            x_new = x1 - self.f(x1)*(x1-x0)/(self.f(x1)-self.f(x0))
            self.x.append(x_new)
            self.error.append(np.linalg.norm(x_new - x1))
            self.iter += 1

        self.convergence()

    def Chord(self):
        """
        Chord method for finding the root of a function.
        """
        a, b = self.boundary
        chord = (b-a)/(self.f(b)-self.f(a))
        self.iter = 0
        self.error = [np.inf]
        self.x.append(self.x0)        

        while min(self.error) > self.tol and self.iter < self.max_iter:
            x = self.x[self.iter]
            x_new = x - self.f(x)*chord
            self.error.append(np.linalg.norm(x_new - x))
            self.x.append(x_new)
            self.iter += 1

        self.convergence()

    def FixedPoint(self, g):
        """
        Fixed point method for finding the root of a function.
        """
        self.iter = 0
        self.error = np.inf
        while self.error > self.tol and self.iter < self.max_iter:
            self.x = g(self.x)
            self.error = np.linalg.norm(self.f(self.x))
            self.iter += 1

        return self.x
    
    def RegulaFalsi(self):
        """
        Regula Falsi method for finding the root of a function.
        """
        a, b = self.boundary
        q = (b-a)/(self.f(b)-self.f(a))
        self.iter = 0
        self.error = [np.inf]
        self.x.append(self.x0)        

        while min(self.error) > self.tol and self.iter < self.max_iter:
            if self.error[-1] < 1.e-2:
                q = (self.x[self.iter]-self.x[self.iter-1])/(self.f(self.x[self.iter])-self.f(self.x[self.iter-1]))
            x = self.x[self.iter]
            x_new = x - self.f(x)*q
            self.error.append(np.linalg.norm(x_new - x))
            self.x.append(x_new)
            self.iter += 1
        
        self.convergence()
    
    def QuasiNewton(self,):
        pass


if __name__ == "__main__":

    pass