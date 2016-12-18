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
# to simulate neuron membrane potential over time using the Izhikevich 
# system of differential equations. Also, explore I parameter space of the
# model.
#
# Edits:      Who:        Nature of Edit:
# ======      ====        ===============
# 11/2/14     ZT      	  Initial commit with standard documentation introduction
#			  
# Functions:
# ==========
# ~IzhiGen: generates the Izhikevich function for the system, takes all constants as input parameters
# ~newIzhi: newIzhi function that defines the system of coupled nonlinear differential eqns,
# taking inputs from IzhiGen as constants for the system
# ~RK4: function that uses the Runge-Kutte method to numerically solve the
# system
# ~gen_image: generates subplot for the system that includes: time plot, phase plot, and
# power spectrum
# ~main: increments the value of I between 0 to -1.6 for 600 frames making a ~20 second video
#
# Variables:
# ==========
# ~a, b, c, and d: give rise to biological behavior
# ~t0 = 0: is the default initial time value, t1 = 5 is the default final time value, 
# ~dt = 0.01: is the default time step for the time range used by the solver
# ~x0: gives the initial conditions for the RK4 solver
# ~ng: is where you provide the function to be solved using the Runge-Kutta algorithm
# ~tsp: creates the time range for the solver
# ~Nsize: creates an integer the size of the time vector
# ~X: creates a vector to hold the solution of the function over time, note that the 
# ~X vector is of the same dimension as the initial condition(s) given

# Izhikevich model
# Described by:

# vdot = 0.04*v^2 + 5*v + 140 - u + I
# udot = a*(b*v - u)
# where:
#   v(v>30) = c
#   u(v>30) = u + d

# Defining the Izhikevich system of v(dot) and u(dot)
# inputs a, b, c and d give biological behavior, I is the input stimulus
# x is the membrane potential variable, r is the membrane recovery variable

#   Three sub-plots for Izhikevich: tplot, pplot, fftplot for ffmpegging a video of frames for visualization of 
# Super-critical Hopf bifurcation

# Parameter ranges: I = 10
# ~regular spiking: a = 0.02, b = 0.2, c = -65, d = 2
# ~fast spiking: a = 0.1, b = 0.2, c = -65, d = 2
# ~bursting: a = 0.02, b = 0.2, c = -50, d = 2
# ~chattering:a = 0.02, b = 0.2, c = -55, d = 2

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

def IzhiGen(a = 0.02, b = 0.2, c = -65, d = 2, I = 10):
    def newIzhi(x,t, a = a, b = b, c = c, d = d, I = I):
	  if x[0] >= 30:
            x[0] = c
            x[1] = x[1] + d
    	  return np.array([0.04*(x[0]**2) + 5*x[0] + 140 - x[1] + I, \
                    	  a*(b*x[0] - x[1])])  
    return newIzhi

def gen_image(I, filename):
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 200,dt = 0.02, ng = IzhiGen(I=I))     #generate the data
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
    ax1.set_title('Phase Portrait - Izhikevich')  # title for the first of the three sub-plots
    ax1.set_xlabel('Membrane Recovery Variable')      # label for the x-axis
    ax1.set_ylabel('Membrane Potential')      # label for the y-axis
    #ax1.set_xlim(-1.15,2)
    #ax1.set_ylim(-3,2)

    ax2.plot(tsp, X[:,0])    # subplot 1, time plot for FN
    ax2.set_title('Membrane Potential over Time - Izhikevich')  # title for the first of the three sub-plots
    ax2.set_xlabel('Time')      # label for the x-axis
    ax2.set_ylabel('Membrane Potential')      # label for the y-axis
    ax2.set_xlim(0,200)
    ax2.set_ylim(-80,40)

    ax3.plot(freqs[idx], ps[idx])    # subplot 1, time plot for FN
    ax3.set_title('Power Spectrum of Membrane Potential - Izhikevich')  # title for the first of the three sub-plots
    ax3.set_xlabel('Frequency (kHz)')      # label for the x-axis
    ax3.set_ylabel('Power')      # label for the y-axis
    ax3.set_xlim(0,0.6)
    #ax3.set_ylim(0,9e7)
    pylab.savefig(filename)
    return

def main():
    I = 1
    for n in xrange(100):
	I = 12*(n/99)
	gen_image(I,'masterplots/masterplot{}.png'.format(n)) 			

if __name__ == '__main__':
    main()

# ffmpeg -r 25 -i masterplot%d.png masterplotIzhi.mp4
