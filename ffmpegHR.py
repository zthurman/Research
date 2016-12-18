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
import math

# PURPOSE: To solve a system of differential equations numerically in order
# to simulate neuron membrane potential over time using the Hindmarsh-Rose 
# system of differential equations. Also, explore I parameter space of the
# model.
#
# Edits:      Who:        Nature of Edit:
# ======      ====        ===============
# 9/27/14     ZT      	  Initial commit with standard documentation introduction
#			  
# Functions:
# ==========
# ~HRGen: generates the HR function for the system, takes all constants as input parameters
# ~newHR: newHR function that defines the system of coupled nonlinear differential eqns,
# taking inputs from HRGen as constants for the system
# ~RK4: function that uses the Runge-Kutte method to numerically solve the
# system
# ~gen_image: generates subplot for the system that includes: time plot, phase plot, and
# power spectrum
# ~main: increments the value of I between -10 to 10 for 600 frames making a ~20 second video
#
# Variables:
# ==========
# ~a = 1.0, b = 3.0, c = 1.0, d = 5.0, r = 0.006, s = 4.0, I = 2.5, xnot = -1.5
# ~t0 = 0: is the default initial time value, t1 = 5 is the default final time value, 
# ~dt = 0.01: is the default time step for the time range used by the solver
# ~x0: gives the initial conditions for the RK4 solver
# ~ng: is where you provide the function to be solved using the Runge-Kutta algorithm
# ~tsp: creates the time range for the solver
# ~Nsize: creates an integer the size of the time vector
# ~X: creates a vector to hold the solution of the function over time, note that the 
# ~X vector is of the same dimension as the initial condition(s) given

#   Numerical simulation for Hindmarsh-Rose model
# Defined as: x(dot) = y-(a*x^3) + (b*x^2) - z + I*(1+0.1*sin(wp.*t)
#             y(dot) = c - d*x^2 - y
#             z(dot) = r*(s*(x - x0) - z)

#   Three sub-plots for Hindmarsh-Rose: tplot, pplot, fftplot for ffmpegging a video of frames for visualization of 
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

def HRGen(a = 1.0, b = 3.0, c = 1.0, d = 5.0, r = 0.006, s = 4.0, I = 2.5, xnot = -1.5):
    def newHR(x,t, a=a, b=b, c=c, d=d, r=r, s=s, I=I, xnot=xnot):
        return np.array([x[1] - a*(x[0]**3) + (b*(x[0]**2)) - x[2] + I, \
                        c - d*(x[0]**2) - x[1], \
                        r*(s*(x[0] - xnot) - x[2])])
    return newHR

def gen_image(I, filename):
    X = RK4(x0 = np.array([3, 0, -1.2]), t1 = 800,dt = 0.02, ng = HRGen(I=I))     #generate the data
    t0 = 0
    t1 = 800
    dt = 0.02
    
    tsp = np.arange(t0, t1, dt)
    Y = mean(X)    # determine DC component of signal
    X = X - Y      # subtract DC component from PS to get rid of peak at 0
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(len(X[:,0])/2 - 1), time_step)
    idx = np.argsort(freqs)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize=(20,5))  # define the figure with three subplots 

    ax1.plot(X[:,1], X[:,0])    # subplot 1, phase portrait of membrane potential and membrane recovery variable - HR
    ax1.set_title('Phase Portrait - HR')  # title for the first of the three sub-plots
    ax1.set_xlabel('Membrane Recovery Variable')      # label for the x-axis
    ax1.set_ylabel('Membrane Potential')      # label for the y-axis
    ax1.set_xlim(-20,5)
    ax1.set_ylim(-2,7)

    ax2.plot(tsp, X[:,0])    # subplot 1, time plot for HR
    ax2.set_title('Membrane Potential over Time - HR')  # title for the first of the three sub-plots
    ax2.set_xlabel('Time')      # label for the x-axis
    ax2.set_ylabel('Membrane Potential')      # label for the y-axis
    ax2.set_xlim(100,800)
    ax2.set_ylim(-2,5)

    ax3.plot(freqs[idx], ps[idx])    # subplot 1, time plot for HR
    ax3.set_title('Power Spectrum of Membrane Potential - HR')  # title for the first of the three sub-plots
    ax3.set_xlabel('Frequency (~kHz)')      # label for the x-axis
    ax3.set_ylabel('Power')      # label for the y-axis
    ax3.set_xlim(0,0.6)
    ax3.set_ylim(0,1.5e8)
    pylab.savefig(filename)
    return

def main():
    I = -10
    for n in xrange(100):
	I = 10*(n/99)
	gen_image(I,'masterplots/masterplot{}.png'.format(n)) 

if __name__ == '__main__':
    main()

# ffmpeg -r 25 -i masterplot%d.png masterplotHR.mp4
