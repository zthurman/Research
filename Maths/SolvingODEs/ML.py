#!/usr/bin/env python
# PURPOSE: To solve a system of differential equations numerically in order
# to simulate neuron membrane potential over time using the Morris-Lecar 
# system of differential equations. cheese
#
# Edits:      Who:        Nature of Edit:
# ======      ====        ===============
# 9/27/14     ZT          Assorted modifications to documentation intro
#			    
#
# Functions:
# ==========
# ~ML: function that sets up and calls the system of differential
# equations that simulates neuron membrane potential and recovery variable.
# ~ML2: function that sets up and calls the system of differential
# equations that simulates neuron membrane potential and recovery variable
# for two linearly coupled ML neurons.
# ~RK4: function that uses the Runge-Kutte method to numerically solve the
# system.
# ~do_pplot: function that calls RK4 to evaluate ML given the input parameters
# of the model to generate a phase plot for the system.
# ~do_tplot: function that calls RK4 to evaluate ML given the input parameters
# of the model to generate a membrane potential over time plot for the system.
# ~do_vec_pplot: function that calls RK4 to evaluate ML given the input parameters
# of the model to generate a vector phase plot for the system.
# ~do_p2plot: function that calls RK4 to evaluate ML2 given the input parameters
# of the model to generate a phase plot for two linearly coupled ML neurons.
# ~do_t2plot: function that calls RK4 to evaluate ML2 given the input parameters
# of the model to generate a plot of membrane potential over time for two linearly 
# coupled ML neurons.
# ~do_fftplot: function that calls RK4 to evaluate ML given the input parameters
# of the model to generate the power spectrum of the membrane potential signal.
#
#
# Variables:
# ==========
# ~c = 20, vk = -84, gk = 8, vca = 120, gca = 4.4, vl = -60, gl = 2, phi = 0.04
# ~t0 = 0: is the default initial time value, t1 = 5 is the default final time value, 
# ~dt = 0.01: is the default time step for the time range used by the solver
# ~x0: gives the initial conditions for the solver
# ~ng: is where you provide the function to be solved using the Runge-Kutta algorithm
# ~tsp: creates the time range for the solver
# ~Nsize: creates an integer the size of the time vector
# ~X: creates a vector to hold the solution of the function over time, note that the 
# X vector is of the same dimension as the initial condition(s) given


#   Numerical simulation for Morris-Lecar model
# Defined as: v(dot) = (-gca*mss(v)*(v-vca) - gk*w*(v-vk) - gl*(v-vl) + Iapp)/c
#             w(dot) = (phi(wss - w))/tau
#  Where: c = 20, vk = -84, gk = 8, vca = 120, gca = 4.4, vl = -60, gl = 2, phi = 0.04
#         mss = 0.5[1 + tanh((v - v1)/v2)]
#         wss = 0.5[1 + tanh((v - v3)/v4)]
#         tau = 1/cosh((v - v3)*(2*v4))

# Imports:

from __future__ import division
from scipy import *
import numpy as np
import pylab
import matplotlib as mp
from matplotlib import pyplot as plt  
import sys
import math as mt
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Now for the actual Morris-Lecar model itself:
def ML(v,t,c = 20,vk = -84,gk = 8,vca = 130,gca = 4.4,vl = -60,gl = 2,phi = 0.04,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,Iapp = 80):
    return np.array([(-gca*(0.5*(1 + mt.tanh((v[0] - v1)/v2)))*(v[0]-vca) - gk*v[1]*(v[0]-vk) - gl*(v[0]-vl) + Iapp), \
                     (phi*((0.5*(1 + mt.tanh((v[0] - v3)/v4))) - v[1]))/(1/mt.cosh((v[0] - v3)/(2*v4)))])

# Super-critical Hopf bifurcation for the system somewhere in the region of Iapp = 79.352-79.353 with Iapp constant

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
    X = RK4(x0 = np.array([0,0]), t1 = 1000,dt = 0.1, ng = ML)
    pylab.plot(X[:,0], X[:,1])
    pylab.title("Phase Portrait - single uncoupled ML neuron")
    pylab.xlabel("Membrane Potential")
    pylab.ylabel("Membrane Recovery Variable")
    pylab.savefig('MLpplot.png')
    pylab.show()
    return

print do_pplot()

def do_tplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 1000,dt = 0.1, ng = ML)
    t0 = 0
    t1 = 1000
    dt = 0.1
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.title("Membrane Potential over Time - single uncoupled neuron")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.savefig('MLtplot.png')
    pylab.show()
    return

print do_tplot()

#  Now for two linearly coupled ML neurons

#  Create a function for two linearly coupled neurons with a coupling constant of k
# second neuron has no input stimulus because it's coupled to the first
# by varying the coupling constant can moderate the behavior of both neurons

def ML2(v,t,c = 20,vk = -84,gk = 8,vca = 120,gca = 4.4,vl = -60,gl = 2,phi = 0.04,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,Iapp = 125,k = 1.5):
    return np.array([(-gca*(0.5*(1 + mt.tanh((v[0] - v1)/v2)))*(v[0]-vca) - gk*v[1]*(v[0]-vk) - gl*(v[0]-vl) + Iapp + k*(v[2] - v[0])), \
                     (phi*((0.5*(1 + mt.tanh((v[0] - v3)/v4))) - v[1]))/(1/mt.cosh((v[0] - v3)/(2*v4))),
                     (-gca*(0.5*(1 + mt.tanh((v[2] - v1)/v2)))*(v[2]-vca) - gk*v[3]*(v[2]-vk) - gl*(v[2]-vl) + k*(v[0] - v[2])), 
                     (phi*((0.5*(1 + mt.tanh((v[2] - v3)/v4))) - v[3]))/(1/mt.cosh((v[2] - v3)/(2*v4)))])                             

def do_p2plot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 400,dt = 0.02, ng = ML2)
    pylab.plot(X[:,1], X[:,0])
    pylab.plot(X[:,3], X[:,2])
    pylab.title("Phase Portrait - two linearly coupled ML neurons")
    pylab.xlabel("Membrane Potential")
    pylab.ylabel("Membrane Recovery Variable")
    pylab.savefig('MLp2plot.png')
    pylab.show()
    return

print do_p2plot()
    
def do_t2plot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 400,dt = 0.02, ng = ML2)
    t0 = 0
    t1 = 400
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.plot(tsp,X[:,2])
    pylab.title("Membrane Potential over Time - two linearly coupled neurons")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.xlim(0,400)
    pylab.ylim(-75,80)
    pylab.savefig('MLt2plot.png')
    pylab.show()
    return

print do_t2plot()

#  Analyzing the relationship between input stimulus and firing frequency for the Morris-Lecar model.
# Measure the diffference in the maxima for the firing neuron as a function of increasing input stimulus, ML.

def do_tplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 800,dt = 0.1, ng = ML)
    t0 = 0
    t1 = 800
    dt = 0.1
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.title("Membrane Potential over Time - single uncoupled neuron")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.xlim(0,400)
    pylab.ylim(-60,80)
    pylab.show()
    return

#print do_tplot()

X = RK4(x0 = np.array([0.01,0.01]), t1 = 800,dt = 0.1, ng = ML) #define global X

# Compute the Power spectrum of the neuron membrane potential over time

def do_fftplot():
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 800,dt = 0.1, ng = ML)
    Y = mean(X)		# determine DC component of signal
    X = X - Y		# subtract DC component from signal to get rid of peak at 0
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(X.size/2 - 1), time_step)
    idx = np.argsort(freqs)
    pylab.plot(freqs[idx], ps[idx])
    pylab.title("Power Spectrum of Membrane Potential Signal")
    pylab.xlabel("Frequency (kHz)")
    pylab.ylabel("Power")
    pylab.xlim(0,0.4)
    pylab.ylim(0,2e10)
    pylab.savefig('MLfftplot.png')
    pylab.show()
    print X.size
    return

print do_fftplot()
