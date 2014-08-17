#!/usr/bin/env python
# A General Ordinary Differential Equation (ODE) Solver

from __future__ import division
from scipy import *
import numpy as np
import pylab
import matplotlib as mp
from matplotlib import pyplot as plt  
import sys
import math as mt
from scipy.integrate import ode

def generate(data_length, odes, state, parameters):
    data = np.zeros([state.shape[0], data_length])
    for i in xrange(1):
        state = rk4(odes, state, parameters)
    for i in xrange(data_length):
        state = rk4(odes, state, parameters)
        data[:, i] = state
    return data

def rk4(odes, state, parameters, dt=0.01):
    k1 = dt * odes(state, parameters)
    k2 = dt * odes(state + 0.5 * k1, parameters)
    k3 = dt * odes(state + 0.5 * k2, parameters)
    k4 = dt * odes(state + k3, parameters)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def RK4(t0 = 0, x0 = np.array([1]), t1 = 5 , dt = 0.01, ng = None):  
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
    return X

def HR_odes((x,y,z), (a, b, c, d, r, s, I, xnot)):
    return np.array([y - a*(x**3) + (b*(x**2)) - z + I, \
                        c - d*(x**2) - y, \
                        r*(s*(x - xnot) - z)])
