#!/usr/bin/env python
# A General Ordinary Differential Equation (ODE) Solver, applied to SHM

from __future__ import division
from scipy import *
import numpy as np
import pylab
import matplotlib as mp
from matplotlib import pyplot as plt  
import sys
import math as mt
from scipy.integrate import ode

# Function used for generating the  data, calls RK4 function and used in ODE_generate function

def generate(data_length, odes, state, parameters):
    data = np.zeros([state.shape[0], data_length])
    for i in xrange(1):
        state = rk4(odes, state, parameters)
    for i in xrange(data_length):
        state = rk4(odes, state, parameters)
        data[:, i] = state
    return data

# Function called in the generate function, takes ODEs function as an argument 

def rk4(odes, state, parameters, dt=0.01):
    k1 = dt * odes(state, parameters)
    k2 = dt * odes(state + 0.5 * k1, parameters)
    k3 = dt * odes(state + 0.5 * k2, parameters)
    k4 = dt * odes(state + k3, parameters)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Alternate RK4 function

"""def RK4(t0 = 0, x0 = np.array([1]), t1 = 5 , dt = 0.01, ng = None):  
    tsp = np.arange(t0, t1, dt)
    Nsize = np.size(tsp)
    X = np.empty((Nsize, np.size(x0)))
    X[0] = x0

    for i in range(1, Nsize):
        k1 = ng(X[i-1],tsp[i-1])
        k2 = ng(X[i-1] + dt/2*k1, tsp[i-1] + dt/2)
        k3 = ng(X[i-1] + dt/2*k2, tsp[i-1] + dt/2)
        k4 = ng(X[i-1] + dt*k3, tsp[i-1] + dt)
        X[i] = X[i-1] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return X"""

# Function that holds the initial conditions and parameters of the ODE

def ODEs((x_vec), (t)):
    omega_squared = 4;
    x,y = x_vec;
    return np.array([y, -omega_squared*x])

# Function that calls the generate function that generates data from the system

def ODE_generate(data_length):
    return generate(data_length, ODEs, \
            	    np.array([0,0]), xrange(data_length))

# Function that creates the phase plot for the system

"""def do_pplot():
    pylab.figure()
    data = ODE_generate(10000)   # how long the function is solved for
    pylab.plot(data[0,:], data[1,:])
    pylab.title("Phase Portrait - ODE")
    pylab.xlabel("X Dynamical Variable")
    pylab.ylabel("Y Dynamical Variable")
    pylab.show()

print do_pplot()"""

# Function that creates the time plot for the system

def do_tplot():
    pylab.figure()
    data = ODE_generate(100)   # how long the function is solved for
    pylab.plot(data[0])
    pylab.title("ODE Solution")
    pylab.xlabel("Time")
    pylab.ylabel("Numerically approximated solution")
    pylab.xlim(0,100)
    pylab.ylim(-1,1)
    pylab.show()

print do_tplot()
