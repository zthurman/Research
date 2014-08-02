#!/usr/bin/env python
# Phase Space Strange Attractor Plot for the Lorenz Equations, 1.2D (fractal)

# imports:
from numpy import zeros, linspace, array
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt      # plt function defined for plotting, imported from matplotlib
import numpy as np
import pylab
from pylab import figure, show, xlabel, ylabel, title
from scipy.integrate import odeint

def lorenz_sys(t, q):        # define the Lorenz system by the t and q vector, q is a 3D vector
    x = q[0]      # x-dimension of the q-vector
    y = q[1]      # y-dimension of the q-vector
    z = q[2]      # z-dimension of the q-vector
    # sigma, rho and beta are global constants
    # f is the system of the combined elements of q and the global constants
    f = [sigma * (y - x),
         rho*x - y - x*z,
         x*y - beta*z]
    return f


ic = [1.0, 2.0, 1.0]    # initial conditions for the three elements of q
t0 = 0.0        # initial time
t1 = 100.0      # final time 
dt = 0.01       # timestep

sigma = 10.0     # constant in system
rho = 28.0      # constant in system
beta = 10.0/3     # constant in system

solver = ode(lorenz_sys)    # calling ode function to evaluate the Lorenz system

t = []        # empty time vector
soln = []      # empty solution vector
solver.set_initial_value(ic, t0)   # setting the intial conditions of the solver to the inits defined in ic and t0 
#solver.set_integrator('dop853'), #less fine grained solution
solver.set_integrator('dopri5')    # setting the Runge-Kutte algorithm to be used when solving the system

while solver.successful() and solver.t < t1:   # while statement defining execution parameters for the solver
    solver.integrate(solver.t + dt)
    t.append(solver.t)
    soln.append(solver.y)

t = array(t)
soln = array(soln)

fig = figure()
ax = Axes3D(fig)
ax.plot(soln[:,0], soln[:,1], soln[:,2])
title("Lorenz Equations, Strange Attractor")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
show()

def do_tplot():
    pylab.figure()
    tsp = pylab.arange(len(soln[:,0]))
    pylab.plot(tsp,soln[:,0])
    pylab.title("Time solution of Lorenz Equations")
    pylab.xlabel("Time")
    pylab.ylabel("X Dynamical Variable")
    pylab.show()
    return

print do_tplot()
