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
# to simulate neuron membrane potential over time using the Hodgkin-Huxley 
# system of differential equations. Also, explore I parameter space of the
# model.
#
# Edits:      Who:        Nature of Edit:
# ======      ====        ===============
# 10/11/14     ZT      	  Initial commit with standard documentation introduction
#			  
# Functions:
# ==========
# ~HHGen: generates the HH function for the system, takes all constants as input parameters
# ~newHH: newHH function that defines the system of coupled nonlinear differential eqns,
# taking inputs from HHGen as constants for the system
# ~RK4: function that uses the Runge-Kutte method to numerically solve the
# system
# ~gen_image: generates subplot for the system that includes: time plot, phase plot, and
# power spectrum
# ~main: increments the value of I between 2 to -50 for around 600 frames making a ~20 second video
#
# Variables:
# ==========
# ~g_K=36, g_Na=120, g_L=0.3, E_K=12, E_Na=-115, E_L=-10.613, C_m=1, I=membrane input stimulus
# ~t0 = 0: is the default initial time value, t1 = 5 is the default final time value, 
# ~dt = 0.01: is the default time step for the time range used by the solver
# ~x0: gives the initial conditions for the RK4 solver
# ~ng: is where you provide the function to be solved using the Runge-Kutta algorithm
# ~tsp: creates the time range for the solver
# ~Nsize: creates an integer the size of the time vector
# ~X: creates a vector to hold the solution of the function over time, note that the 
# ~X vector is of the same dimension as the initial condition(s) given

# Differential equation solver for the Hodgkins-Huxley neuron model.
# Original 1952 paper found here: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf
# According to their paper the system is expressed as:

# V(dot) = (-g_K*n^4(V - E_K) - g_Na*m^3*h(V - E_Na) - g_L(V - E_L) + I(t))/C_m
# n(dot) = alpha_n*(1 - n) - beta_n*n
# m(dot) = alpha_m*(1 - m) - beta_m*m
# h(dot) = alpha_h*(1 - h) - beta_h*h

# where:
# alpha_n = 0.01*(V + 10)/(e^(V + 10/10) - 1)
# beta_n = 0.125*e^(V/80)
# alpha_m = 0.1*(V + 25)/(e^(V + 25/10) - 1)
# beta_m = 4*e^(V/18)
# alpha_h = 0.07*e^(V/20)
# beta_h = 1/(e^(V + 30/10) - 1)

#   So if we want to do this one the same way that we've done Fitzhugh-Nagumo, Morris-Lecar and Hindmarsh-Rose 
# our model that goes in the function will be as follows:

# V(dot) = (-g_K*n^4(V - E_K) - g_Na*m^3*h(V - E_Na) - g_L(V - E_L) + I(t))/C_m
# n(dot) = 0.01*(V + 10)/(exp^(V + 10/10) - 1)*(1 - n) - 0.125*exp^(V/80)*n
# m(dot) = 0.1*(V + 25)/(exp^(V + 25/10) - 1)*(1 - m) - 4*exp^(V/18)*m
# h(dot) = 0.07*exp^(V/20)*(1 - h) - 1/(exp^(V + 30/10) - 1)*h

#   Three sub-plots for Hodgkin-Huxley: tplot, pplot, fftplot for ffmpegging a video of frames for visualization of 
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

def HHGen(g_K=36, g_Na=120, g_L=0.3, E_K=12, E_Na=-115, E_L=-10.613, C_m=1, I=-10):
    def newHH(x,t, g_K=g_K, g_Na=g_Na, g_L=g_L, E_K=E_K, E_Na=E_Na, E_L=E_L, C_m=C_m, I=I):
            alpha_n = (0.01*(x[0]+10))/(exp((x[0]+10)/10)-1)
	    beta_n = 0.125*exp(x[0]/80)
	    alpha_m = (0.1*(x[0]+25))/(exp((x[0]+25)/10)-1)
	    beta_m = 4*exp(x[0]/18)
	    alpha_h = (0.07*exp(x[0]/20))
	    beta_h = 1 / (exp((x[0]+30)/10)+1)
    	    return np.array([(g_K*(x[1]**4)*(x[0]-E_K) + g_Na*(x[2]**3)*x[3]*(x[0]-E_Na) + g_L*(x[0]-E_L) - I)*(-1/C_m), \
                            alpha_n*(1-x[1]) - beta_n*x[1], \
                            alpha_m*(1-x[2]) - beta_m*x[2], \
                            alpha_h*(1-x[3]) - beta_h*x[3]])
    return newHH

def gen_image(I, filename):
    X = RK4(x0 = np.array([0,0,0,0]), t1 = 100,dt = 0.01, ng = HHGen(I=I)) #generate the data
    t0 = 0
    t1 = 100
    dt = 0.01
    tsp = np.arange(t0, t1, dt)

    Y = mean(X[:,0])
    X[:,0] = X[:,0] - Y
    fdata = X[:,0].size
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(fdata/2 - 1), time_step)
    idx = np.argsort(freqs)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize=(20,5))  # define the figure with three subplots

    ax1.plot(X[:,1], X[:,0])    # subplot 1, phase portrait of membrane potential and membrane recovery variable - HH
    ax1.set_title('Phase Portrait - HH')  # title for the first of the three sub-plots
    ax1.set_xlabel('Potassium gating variable')      # label for the x-axis
    ax1.set_ylabel('Membrane Potential')      # label for the y-axis
    ax1.set_xlim(0,0.8)
    ax1.set_ylim(-90,30)

    ax2.plot(tsp, -X[:,0])    # subplot 1, time plot for HH
    ax2.set_title('Membrane Potential over Time - HH')  # title for the first of the three sub-plots
    ax2.set_xlabel('Time')      # label for the x-axis
    ax2.set_ylabel('Membrane Potential')      # label for the y-axis
    ax2.set_xlim(0,100)
    ax2.set_ylim(-40,100)

    ax3.plot(freqs[idx], ps[idx])    # subplot 1, time plot for HH
    ax3.set_title('Power Spectrum of Membrane Potential Signal - HH')  # title for the first of the three sub-plots
    ax3.set_xlabel('Frequency')      # label for the x-axis
    ax3.set_ylabel('Power')      # label for the y-axis
    ax3.set_xlim(0,0.6)
    ax3.set_ylim(0,1.4e10)
    pylab.savefig(filename)
    return

def main():
    I = 2
    for n in xrange(600):
	I = -50*(n/599)
	gen_image(I,'masterplots/masterplot{}.png'.format(n)) 

if __name__ == '__main__':
    main()

# ffmpeg -r 25 -i masterplot%d.png masterplotHH.mp4
