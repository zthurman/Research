#!/usr/bin/env python
# PURPOSE: To solve a system of differential equations numerically in order
# to simulate neuron membrane potential over time using the Hindmarsh-Rose 
# system of differential equations. cheese
#
# Edits:      Who:        Nature of Edit:
# ======      ====        ===============
# 10/9/14     ZT      	  Addition of fft for single uncoupled HR
#			  
#
# Functions:
# ==========
# ~HR: function that sets up and calls the system of differential
# equations that simulates neuron membrane potential and recovery variable.
# ~HR2: function that sets up and calls the system of differential
# equations that simulates neuron membrane potential and recovery variable
# for two linearly coupled FN neurons.
# ~RK4: function that uses the Runge-Kutte method to numerically solve the
# system.
# ~do_pplot: function that calls RK4 to evaluate HR given the input parameters
# of the model to generate a phase plot for the system.
# ~do_tplot: function that calls RK4 to evaluate HR given the input parameters
# of the model to generate a membrane potential over time plot for the system.
# ~do_vec_pplot: function that calls RK4 to evaluate HR given the input parameters
# of the model to generate a vector phase plot for the system.
# ~do_p2plot: function that calls RK4 to evaluate HR2 given the input parameters
# of the model to generate a phase plot for two linearly coupled HR neurons.
# ~do_t2plot: function that calls RK4 to evaluate HR2 given the input parameters
# of the model to generate a plot of membrane potential over time for two linearly 
# coupled HR neurons.
# ~do_fftplot: function that calls RK4 to evaluate HR given the input parameters
# of the model to generate the power spectrum of the membrane potential signal.
#
#
# Variables:
# ==========
# ~a, b, c, d, r, s, I: give rise to biological behavior
# ~k: coupling constant
# ~t0 = 0: is the default initial time value, t1 = 5 is the default final time value, 
# ~dt = 0.01: is the default time step for the time range used by the solver
# ~x0: gives the initial conditions for the solver
# ~ng: is where you provide the function to be solved using the Runge-Kutta algorithm
# ~tsp: creates the time range for the solver
# ~Nsize: creates an integer the size of the time vector
# ~X: creates a vector to hold the solution of the function over time, note that the 
# ~X: vector is of the same dimension as the initial condition(s) given


#   Numerical simulation for Hindmarsh-Rose model
# Defined as: x(dot) = y-(a*x^3) + (b*x^2) - z + I
#             y(dot) = c - d*x^2 - y
#             z(dot) = r*(s*(x - x0) - z)

# Imports

from __future__ import division
from scipy import *
import numpy as np
import pylab
import matplotlib as mp
from matplotlib import pyplot as plt  
import sys
import math as mt
from scipy.integrate import ode

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

def HR(x,t, a = 1.0, b = 3.0, c = 1.0, d = 5.0, r = 0.006, s = 4.0, I = 1.3, xnot = -1.5):
    return np.array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + I, \
                        c - d*(x[0]**2) - x[1], \
                        r*(s*(x[0] - xnot) - x[2])])

def do_pplot():
    pylab.figure()
    X = RK4(x0 = np.array([3, 0, -1.2]), t1 = 100,dt = 0.02, ng = HR)
    pylab.plot(X[:,0], X[:,1])
    pylab.title("Phase Portrait - Hindmarsh-Rose")
    pylab.xlabel("Membrane Potential")
    pylab.ylabel("Membrane Recovery Variable")
    pylab.savefig('HRpplot.png')
    pylab.show()

print do_pplot()

def do_tplot():
    pylab.figure()
    X = RK4(x0 = np.array([3, 0, -1.2]), t1 = 100,dt = 0.02, ng = HR)
    t0 = 0
    t1 = 100
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.title("Membrane Potential over Time - single uncoupled HR neuron")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.savefig('HRtplot.png')
    pylab.show()
    return

print do_tplot()

def do_fftplot():
    X = RK4(x0 = np.array([3, 0, -1.2]), t1 = 100,dt = 0.02, ng = HR)
    Y = mean(X[:,0])
    X[:,0] = X[:,0] - Y
    fdata = X[:,0].size
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(fdata/2 - 1), time_step)
    idx = np.argsort(freqs)
    pylab.plot(freqs[idx], ps[idx])
    pylab.title("Power Spectrum of Membrane Potential Signal")
    pylab.xlabel("Frequency ~(kHz)")
    pylab.ylabel("Power")
    pylab.xlim(0,1)
    pylab.ylim(0,4e6)
    pylab.savefig('HRfftplot.png')
    pylab.show()
    return

print do_fftplot()

def HR2(x,t, a = 1.0, b = 3.0, c = 1.0, d = 5.0, r = 0.006, s = 4.0, I = 2.75, xnot = -1.5, k = 0.75):
    return np.array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + I + k*(x[3] - x[0]), \
                    c - d*(x[0]**2) - x[1], \
                    r*(s*(x[0] - xnot) - x[2]), \
                    x[4] - a*(x[3]**3) + (b*(x[3]**2)) - x[5] + I + k*(x[0] - x[3]), \
                    c - d*(x[3]**2) - x[4], \
                    r*(s*(x[3] - xnot) - x[5])])

def do_p2plot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01,0.01,0.01]), t1 = 600,dt = 0.01, ng = HR2)
    pylab.plot(X[:,1], X[:,0])
    pylab.plot(X[:,4], X[:,3])
    pylab.title("Phase Portrait")
    pylab.xlabel("Membrane Potential")
    pylab.ylabel("Membrane Recovery Variable")
    pylab.show()
    return

print do_p2plot()
    
def do_t2plot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01,0.01,0.01]), t1 = 600,dt = 0.01, ng = HR2)
    t0 = 0
    t1 = 600
    dt = 0.01
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.plot(tsp,X[:,3])
    pylab.title("Membrane Potential over Time")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential (mV)")
    pylab.show()
    return

print do_t2plot()
