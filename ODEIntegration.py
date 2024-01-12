import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
import RootFinding as rf

# Feature

class ODEIntegration():

    def __init__(self, func, t0, y0, T, dt):
        self.func = func
        self.t0 = t0
        self.y0 = y0
        self.m = y0.shape[0]
        self.T = T
        self.dt = dt
        self.t = np.arange(self.t0, self.T, self.dt)
        self.y = np.zeros((self.m,len(self.t)))
        self.y[:,0] = self.y0
        self.method = self.ForwardEuler

    def restore(self):
        """
        Restore the initial conditions.
        """
        self.t = np.arange(self.t0, self.T, self.dt)
        self.y = np.zeros((self.m,len(self.t)))
        self.y[:,0] = self.y0

    #############################################

    def ForwardEuler(self):
        """
        Forward Euler method.
        """
        # reset all attributes
        self.restore()
        
        # Forward Euler iteration
        for i in range(len(self.t)-1):
            self.y[:,i+1] = self.y[:,i] + self.dt*self.func(self.t[i], self.y[:,i])
        
        return self.t, self.y
    
    def BackwardEuler(self):
        """
        Backward Euler method.
        """
        # reset all attributes
        self.restore()

        # Backward Euler iteration
        for i in range(len(self.t)-1):
            # set RootFinding parameters
            f = lambda x: x - self.y[:,i] - self.dt*self.func(self.t[i+1], x)
            a, b = -2*self.y[:,i], 2*self.y[:,i]
            y0 = self.y[:,i] + self.dt*self.func(self.t[i], self.y[:,i]) # set second point with EF

            # find the root and set as solution of i-th step
            solver = rf.RootFinding(f, (a,b), self.y[:,i], x1=y0, tol=1e-6)                
            solver.Secant()
            self.y[:,i+1] = solver.x[-1]
        
        return self.t, self.y
    
    def Heun(self):
        """
        Heun method.
        """
        # reset all attributes
        self.restore()

        # Heun iteration
        for i in range(len(self.t)-1):
            self.y[:,i+1] = self.y[:,i] + (self.dt/2)*(self.func(self.t[i], self.y[:,i]) + self.func(self.t[i+1], self.y[:,i] + self.dt*self.func(self.t[i], self.y[:,i])))
        
        return self.t, self.y
    
    def CranckNicholson(self):
        """
        Cranck-Nicholson method.
        """
        # reset all attributes
        self.restore()

        # Crank Nicholson iteration
        for i in range(len(self.t)-1):
            # set RootFinding parameters
            f = lambda x: x - self.y[:,i] - (self.dt/2)*(self.func(self.t[i], self.y[:,i]) + self.func(self.t[i+1], x))
            a, b = -2*self.y[:,i], 2*self.y[:,i]
            y0 = self.y[:,i] + self.dt*self.func(self.t[i], self.y[:,i]) # set second point with EF
            
            # find the root and set as solution of i-th step
            solver = rf.RootFinding(f, (a,b), self.y[:,i], x1=y0, tol=1e-6)
            solver.Secant()
            self.y[:,i+1] = solver.x[-1]
        
        return self.t, self.y

    def RungeKutta2(self):
        """
        Runge-Kutta 2nd order method.
        """
        # reset all attributes
        self.restore()

        # Runge-Kutta 2nd order iteration
        for i in range(len(self.t)-1):
            k1 = self.func(self.t[i],         self.y[:,i])
            k2 = self.func(self.t[i]+self.dt, self.y[:,i] + k1*self.dt)
            self.y[:,i+1] = self.y[:,i] + self.dt*k2
        
        return self.t, self.y

    def RungeKutta4(self):
        """
        Runge-Kutta-4 integration method of ODEs

        INPUTS:
        - ode: function with ODEs system that calculate dot-quantities y'=f(t,y)
        - time: np.array of evaluation time
        - state: np.array with quantities of the initial state
        - consts: tuple with system costants
        OUTPUTS:
        - new_state: np.array, same length of vector time,
                        with state quantities at each time
        """
        # reset all attributes
        self.restore()

        # Runge-Kutta 4th order iteration
        for i in range(len(self.t)-1):
            k1 = self.func(self.t[i],            self.y[:,i])
            k2 = self.func(self.t[i]+self.dt/2,  self.y[:,i] + k1*self.dt/2)
            k3 = self.func(self.t[i]+self.dt/2,  self.y[:,i] + k2*self.dt/2)
            k4 = self.func(self.t[i]+self.dt,    self.y[:,i] + k3*self.dt)
            
            self.y[:,i+1] = (self.y[:,i] + self.dt*(k1 + 2*k2 + 2*k3 + k4)/6)
        
        return self.t, self.y

    
    #############################################

    def error(self, exact):
        """
        Calculate the error.
        """
        return np.linalg.norm(self.y - exact(self.t))

    #############################################
    
    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.t, self.y)
        ax.grid()
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.set_title("ODE Integration")
        plt.show()

    #############################################

    def convergence_order(self, exact, method=ForwardEuler):
        """
        Calculate the convergence order.
        """
        # DT = self.dt
        # self.method = method
        # dt = np.array([0.2, 0.1, 0.05, 0.0025, 0.00125, 0.000625, 0.0003125])
        # p = []
        # for i in range(1, len(dt)):
        #     self.dt = dt[i]
        #     self.t = np.arange(self.t0, self.T, self.dt)
        #     self.y = np.zeros((self.m,len(self.t)))
        #     self.y[:,0] = self.y0
        #     self.method()
        #     err = self.error(exact)

        #     self.dt = dt[i-1]
        #     self.t = np.arange(self.t0, self.T, self.dt)
        #     self.y = np.zeros((self.m,len(self.t)))
        #     self.y[:,0] = self.y0
        #     self.method()
        #     err_star = self.error(exact)

        #     p.append(abs(np.log2(err/err_star)))
        
        # self.dt = DT

        p = np.zeros(len(self.y)-1)
        for i in range(1, len(self.y)):
            p[i-1] = np.log(self.y[0,i-1]/self.y[0,i],2)
        return p
    
    #############################################
