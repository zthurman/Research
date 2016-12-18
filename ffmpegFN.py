#!/usr/bin/env python
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

# PURPOSE: To solve a system of differential equations numerically in order
# to simulate neuron membrane potential over time using the Fitzhugh-Nagumo 
# system of differential equations. Also, explore I parameter space of the
# model.
#
# Edits:      Who:        Nature of Edit:
# ======      ====        ===============
# 9/26/14     ZT      	  Fine tuning of plot values so it doesn't auto-scale
#			  
# Functions:
# ==========
# ~FNGen: generates the FN function for the system, takes all constants as input parameters
# ~newFN: newFN function that defines the system of coupled nonlinear differential eqns,
# taking inputs from FNGen as constants for the system
# ~RK4: function that uses the Runge-Kutte method to numerically solve the
# system
# ~gen_image: generates subplot for the system that includes: time plot, phase plot, and
# power spectrum
# ~main: increments the value of I between 0 to -1.6 for 600 frames making a ~20 second video
#
# Variables:
# ==========
# ~a, b, and c: give rise to biological behavior
# ~t0 = 0: is the default initial time value, t1 = 5 is the default final time value, 
# ~dt = 0.01: is the default time step for the time range used by the solver
# ~x0: gives the initial conditions for the RK4 solver
# ~ng: is where you provide the function to be solved using the Runge-Kutta algorithm
# ~tsp: creates the time range for the solver
# ~Nsize: creates an integer the size of the time vector
# ~X: creates a vector to hold the solution of the function over time, note that the 
# ~X vector is of the same dimension as the initial condition(s) given


#   Numerical simulation for FitzHugh-Nagumo model
# Defined as: x(dot) = c(x+r-x**3/3+z)
#             r(dot) = -(x-a+br)/c

# Defining the Fitzhugh-Nagumo system of x(dot) and r(dot)
# inputs a, b and c give biological behavior, I is the input stimulus
# x is the membrane potential variable, r is the membrane recovery variable

#   Three sub-plots for Fitzhugh-Nagumo: tplot, pplot, fftplot for ffmpegging a video of frames for visualization of 
# Super-critical Hopf bifurcation

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

def FNGen(a = 0.75, b = 0.8, c = 3, I = -0.5):
    def newFN(x,t,a = a, b = b, c = c, I = I):
        return np.array([c*(x[0]+ x[1]- x[0]**3/3 + I), \
                        -1/c*(x[0]- a + b*x[1])])
    return newFN
    
def gen_image(I, filename):
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 200,dt = 0.02, ng = FNGen(I=I))     #generate the data
    t0 = 0
    t1 = 200
    dt = 0.02
    
    tsp = np.arange(t0, t1, dt)
    Y = mean(X)    # determine DC component of signal
    X = X - Y      # subtract DC component from PS to get rid of peak at 0
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(len(X[:,0])/2 - 1), time_step)
    idx = np.argsort(freqs)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize=(20,5))  # define the figure with three subplots 

    ax1.plot(X[:,1], X[:,0])    # subplot 1, phase portrait of membrane potential and membrane recovery variable - FN
    ax1.set_title('Phase Portrait - FN')  # title for the first of the three sub-plots
    ax1.set_xlabel('Membrane Recovery Variable')      # label for the x-axis
    ax1.set_ylabel('Membrane Potential')      # label for the y-axis
    ax1.set_xlim(-1.15,2)
    ax1.set_ylim(-3,2)

    ax2.plot(tsp, X[:,0])    # subplot 1, time plot for FN
    ax2.set_title('Membrane Potential over Time - FN')  # title for the first of the three sub-plots
    ax2.set_xlabel('Time')      # label for the x-axis
    ax2.set_ylabel('Membrane Potential')      # label for the y-axis
    ax2.set_xlim(0,200)
    ax2.set_ylim(-2.5,2)

    ax3.plot(freqs[idx], ps[idx])    # subplot 1, time plot for FN
    ax3.set_title('Power Spectrum of Membrane Potential - FN')  # title for the first of the three sub-plots
    ax3.set_xlabel('Frequency (kHz)')      # label for the x-axis
    ax3.set_ylabel('Power')      # label for the y-axis
    ax3.set_xlim(0,0.6)
    ax3.set_ylim(0,9e7)
    pylab.savefig(filename)
    return

def main():
    I = 0
    for n in xrange(600):
	I = -1.6*(n/599)
	gen_image(I,'masterplots/masterplot{}.png'.format(n)) 			

if __name__ == '__main__':
    main()

# ffmpeg -r 25 -i masterplot%d.png masterplotFN.mp4
