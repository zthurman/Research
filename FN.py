#!/usr/bin/env python
# PURPOSE: To solve a system of differential equations numerically in order
# to simulate neuron membrane potential over time using the Fitzhugh-Nagumo 
# system of differential equations.
#
# Edits:      Who:        Nature of Edit:
# ======      ====        ===============
# 9/13/14     ZT      	  Modification of FFT to subtract DC signal component (peak at zero)
#			              also added outputting figures as png
#
# Functions:
# ==========
# ~FN: function that sets up and calls the system of differential
# equations that simulates neuron membrane potential and recovery variable.
# ~FN2: function that sets up and calls the system of differential
# equations that simulates neuron membrane potential and recovery variable
# for two linearly coupled FN neurons.
# ~RK4: function that uses the Runge-Kutte method to numerically solve the
# system.
# ~do_pplot: function that calls RK4 to evaluate FN given the input parameters
# of the model to generate a phase plot for the system.
# ~do_tplot: function that calls RK4 to evaluate FN given the input parameters
# of the model to generate a membrane potential over time plot for the system.
# ~do_vec_pplot: function that calls RK4 to evaluate FN given the input parameters
# of the model to generate a vector phase plot for the system.
# ~do_p2plot: function that calls RK4 to evaluate FN2 given the input parameters
# of the model to generate a phase plot for two linearly coupled FN neurons.
# ~do_t2plot: function that calls RK4 to evaluate FN2 given the input parameters
# of the model to generate a plot of membrane potential over time for two linearly 
# coupled ML neurons.
# ~do_fftplot: function that calls RK4 to evaluate FN given the input parameters
# of the model to generate the power spectrum of the membrane potential signal.
#
#
# Variables:
# ==========
# ~a, b, and c: give rise to biological behavior
# ~t0 = 0: is the default initial time value, t1 = 5 is the default final time value, 
# ~dt = 0.01: is the default time step for the time range used by the solver
# ~x0: gives the initial conditions for the solver
# ~ng: is where you provide the function to be solved using the Runge-Kutta algorithm
# ~tsp: creates the time range for the solver
# ~Nsize: creates an integer the size of the time vector
# ~X: creates a vector to hold the solution of the function over time, note that the 
# X vector is of the same dimension as the initial condition(s) given


#   Numerical simulation for FitzHugh-Nagumo model
# Defined as: x(dot) = c(x+r-x**3/3+z)
#             r(dot) = -(x-a+br)/c

# Imports:

from __future__ import division
from scipy import *
from numpy import *
import numpy as np
import pylab
import matplotlib as mp
import matplotlib.pyplot as plt
import sys
from numpy.fft import fft, fftfreq

# Original parameters from paper this Runge-Kutte solver was pulled from were: a = 0.1,b = 0.25,c = 0.5

# Defining the Fitzhugh-Nagumo system of x(dot) and r(dot)
# inputs a, b and c give biological behavior, I is the input stimulus
# x is the membrane potential variable, r is the membrane recovery variable

def FN(x,t,a = 0.75,b = 0.8,c = 3, I = -0.40):
    return np.array([c*(x[0]+ x[1]- x[0]**3/3 + I), \
                    -1/c*(x[0]- a + b*x[1])])

# Super-critical Hopf bifurcation for the system somewhere in the region of I = -0.39947 with I constant 

#   Using a Runge-Kutta 4th order differential equation solver:

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

def do_pplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = FN)
    pylab.plot(X[:,0], X[:,1])
    pylab.title("Phase Portrait - single uncoupled FN")
    pylab.xlabel("Membrane Potential")
    pylab.ylabel("Membrane Recovery Variable")
    pylab.savefig('FNpplot.png')
    pylab.show()
    return

print do_pplot()

k = 1
def vec_pplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = FN)
    plt.quiver(X[:,0], X[:,1], X[:,1], X[:,0], pivot='middle')      # quiver allows the plotting of vector fields in two dimensions
    plt.axis('equal')
    plt.title("Vector Field")
    plt.xlabel("Membrane Potential")
    plt.ylabel("Membrane Recovery Variable")
    
print vec_pplot()

def do_tplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = FN)
    t0 = 0
    t1 = 100
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.title("Membrane Potential over Time - single uncoupled FN neuron")
    pylab.xlabel("Time")
    pylab.savefig('FNtplot.png')
    pylab.show()
    return

print do_tplot()

#  Now for two coupled neurons

#  Create a function for two linearly coupled neurons with a coupling constant of k
# second neuron has no input stimulus because it's coupled to the first
# by varying the coupling constant can moderate the behavior of both neurons

def FN2(x,t,a = 0.75,b = 0.8,c = 3, I = -0.80, k = 0.75):
    return np.array([c*(x[0]+ x[1]- x[0]**3/3 + I + k*(x[2] - x[0])), \
                    -1/c*(x[0]- a + b*x[1]), \
                     c*(x[2]+ x[3]- x[2]**3/3 + k*(x[0] - x[2])),
                    -1/c*(x[2]- a + b*x[3])])

def do_p2plot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.02, ng = FN2)
    pylab.plot(X[:,1], X[:,0])
    pylab.plot(X[:,3], X[:,2])
    pylab.title("Phase Portrait - linearly coupled FN")
    pylab.xlabel("Membrane Potential")
    pylab.ylabel("Membrane Recovery Variable")
    pylab.savefig('FNp2plot.png')
    pylab.show()
    return

print do_p2plot()

def do_t2plot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.02, ng = FN2)
    t0 = 0
    t1 = 100
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.plot(tsp,X[:,2])
    pylab.title("Membrane Potential over Time - two linearly coupled FN neurons")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.savefig('FNt2plot.png')
    pylab.show()
    return

print do_t2plot()

#  Analyzing the relationship between input stimulus and firing frequency for the Fitzhugh-Nagumo model.
# Measure the diffference in the maxima for the firing neuron as a function of increasing input stimulus, FN.

def do_tplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = FN)
    t0 = 0
    t1 = 100
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.title("Membrane Potential over Time - single uncoupled neuron")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.xlim(0,100)
    pylab.ylim(-1.9,2.2)
    pylab.show()
    return

# print do_tplot()

X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = FN)  #define global X

# Compute the Fourier Transform of the neuron membrane potential over time

def do_fftplot():
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 100,dt = 0.02, ng = FN)
    Y = mean(X)    # determine DC component of signal
    X = X - Y      # subtract DC component from signal to get rid of peak at 0
    ps = np.abs(np.fft.fft(X[4:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(len(X[4:,0])/2 - 1), time_step)
    idx = np.argsort(freqs)
    pylab.plot(freqs[idx], ps[idx])
    pylab.title("Power Spectrum of Membrane Potential Signal - FN")
    pylab.xlabel("Frequency (kHz)")
    pylab.ylabel("Power")
    pylab.xlim(0,0.4)
    pylab.ylim(0,2e7)
    pylab.savefig('FNfftplot.png')
    pylab.show()
    return

print do_fftplot()
