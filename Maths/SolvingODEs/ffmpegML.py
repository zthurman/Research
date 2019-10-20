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
import math as mt

# PURPOSE: To solve a system of differential equations numerically in order
# to simulate neuron membrane potential over time using the Morris-Lecar 
# system of differential equations. Also, explore Iapp parameter space of the
# model.
#
# Edits:      Who:        Nature of Edit:
# ======      ====        ===============
# 9/27/14     ZT      	  Initial commit with standard documentation introduction
#			  
# Functions:
# ==========
# ~MLGen: generates the ML function for the system, takes all constants as input parameters
# ~newML: newML function that defines the system of coupled nonlinear differential eqns,
# taking inputs from MLGen as constants for the system
# ~RK4: function that uses the Runge-Kutte method to numerically solve the
# system
# ~gen_image: generates subplot for the system that includes: time plot, phase plot, and
# power spectrum
# ~main: increments the value of Iapp between 78 to 230 for 600 frames making a ~20 second video
#
# Variables:
# ==========
# ~c = 20, vk = -84, gk = 8, vca = 120, gca = 4.4, vl = -60, gl = 2, phi = 0.04
# ~t0 = 0: is the default initial time value, t1 = 5 is the default final time value, 
# ~dt = 0.01: is the default time step for the time range used by the solver
# ~x0: gives the initial conditions for the RK4 solver
# ~ng: is where you provide the function to be solved using the Runge-Kutta algorithm
# ~tsp: creates the time range for the solver
# ~Nsize: creates an integer the size of the time vector
# ~X: creates a vector to hold the solution of the function over time, note that the 
# ~X vector is of the same dimension as the initial condition(s) given

#   Numerical simulation for Morris-Lecar model
# Defined as: v(dot) = (-gca*mss(v)*(v-vca) - gk*w*(v-vk) - gl*(v-vl) + Iapp)/c
#             w(dot) = (phi(wss - w))/tau
#  Where: c = 20, vk = -84, gk = 8, vca = 120, gca = 4.4, vl = -60, gl = 2, phi = 0.04
#         mss = 0.5[1 + tanh((v - v1)/v2)]
#         wss = 0.5[1 + tanh((v - v3)/v4)]
#         tau = 1/cosh((v - v3)*(2*v4))

#   Three sub-plots for Morris-Lecar: tplot, pplot, fftplot for ffmpegging a video of frames for visualization of 
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

def MLGen(c = 20,vk = -84,gk = 8,vca = 120,gca = 4.4,vl = -60,gl = 2,phi = 0.04,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,Iapp = 90):
    def newML(v,t, c=c, vk=vk, gk=gk, vca=vca, gca=gca, vl=vl, gl=gl, phi=phi, v1=v1, v2=v2, v3=v3, v4=v4, Iapp=Iapp):
        return np.array([(-gca*(0.5*(1 + mt.tanh((v[0] - v1)/v2)))*(v[0]-vca) - gk*v[1]*(v[0]-vk) - gl*(v[0]-vl) + Iapp), \
                        (phi*((0.5*(1 + mt.tanh((v[0] - v3)/v4))) - v[1]))/(1/mt.cosh((v[0] - v3)/(2*v4)))])
    return newML

def gen_image(Iapp, filename):
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 800,dt = 0.02, ng = MLGen(Iapp=Iapp))     #generate the data
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

    ax1.plot(X[:,1], X[:,0])    # subplot 1, phase portrait of membrane potential and membrane recovery variable - ML
    ax1.set_title('Phase Portrait - ML')  # title for the first of the three sub-plots
    ax1.set_xlabel('Membrane Recovery Variable')      # label for the x-axis
    ax1.set_ylabel('Membrane Potential')      # label for the y-axis
    #ax1.set_xlim(-0.9,1.25)
    #ax1.set_ylim(-2.5,2.5)

    ax2.plot(tsp, X[:,0])    # subplot 1, time plot for ML
    ax2.set_title('Membrane Potential over Time - ML')  # title for the first of the three sub-plots
    ax2.set_xlabel('Time')      # label for the x-axis
    ax2.set_ylabel('Membrane Potential')      # label for the y-axis
    #ax2.set_xlim(0,200)
    #ax2.set_ylim(-2.5,2.25)

    ax3.plot(freqs[idx], ps[idx])    # subplot 1, time plot for ML
    ax3.set_title('Power Spectrum of Membrane Potential - ML')  # title for the first of the three sub-plots
    ax3.set_xlabel('Frequency (kHz)')      # label for the x-axis
    ax3.set_ylabel('Power')      # label for the y-axis
    ax3.set_xlim(0,0.6)
    #ax3.set_ylim(0,3e7)
    pylab.savefig(filename)
    return

def main():
    Iapp = 78
    for n in xrange(600):
	Iapp = 230*(n/599)
	gen_image(Iapp,'masterplots/masterplot{}.png'.format(n)) 

if __name__ == '__main__':
    main()

# ffmpeg -r 25 -i masterplot%d.png masterplotML.mp4
