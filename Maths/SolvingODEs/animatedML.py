#!/usr/bin/env python
# PURPOSE: To solve a system of differential equations numerically in order
# to simulate neuron membrane potential over time using the Morris-Lecar 
# system of differential equations. This time animated.
#
# Edits:      Who:        Nature of Edit:
# ======      ====        ===============
# 9/12/14     ZT          Initial commit with standard documentation introduction
#
# Functions:
# ==========
# ~ML: function that sets up and calls the system of differential
# equations that simulates neuron membrane potential and recovery variable.
# ~RK4: function that uses the Runge-Kutte method to numerically solve the
# system.
# ~init: function that initializes the animation figure with a zero line
# ~animate: takes i as an argument and calls global RK4 and ML functions to
# solve the FN model and break it up into frames over time
# ~animation.FuncAnimation: animates the frames together using matplotlib
# ~anim.save: saves the animation as an MP4 video
#
#
# Variables:
# ==========
# ~c, vk, gk, vca, gca, vl, gl, phi, v1, v2, v3, v4, Iapp: give rise to biological behavior
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
import matplotlib.animation as animation

# Now for the actual Morris-Lecar model itself:
def ML(v,t,c = 20,vk = -84,gk = 8,vca = 130,gca = 4.4,vl = -60,gl = 2,phi = 0.04,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,Iapp = 80):
    return np.array([(-gca*(0.5*(1 + mt.tanh((v[0] - v1)/v2)))*(v[0]-vca) - gk*v[1]*(v[0]-vk) - gl*(v[0]-vl) + Iapp), \
                     (phi*((0.5*(1 + mt.tanh((v[0] - v3)/v4))) - v[1]))/(1/mt.cosh((v[0] - v3)/(2*v4)))])

# Super-critical Hopf bifurcation for the system somewhere in the region of Iapp = 79.352-79.353 with Iapp constant

#   Using a Runge-Kutta 4th order differential equation solver:
# t0 = 0 is the default initial time value, t1 = 5 is the default final time value, 
# dt = 0.01 is the default time step for the time range used by the solver
# x0 gives the initial conditions for the solver
# ng is where you provide the function to be solved using the Runge-Kutta algorithm
# tsp creates the time range for the solver
# Nsize creates an integer the size of the time vector
# X creates a vector to hold the solution of the function over time, note that the 
# X vector is of the same dimension as the initial condition(s) given

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

# Animation of the neuron firing over time

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 150), ylim=(-70, 70))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    global RK4, ML
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 1000,dt = 0.02, ng = ML)
    t0 = 0
    t1 = 1000
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    line.set_data(tsp-0.5*i, X[:,0])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=20, blit=True)
pylab.xlabel("Time")
pylab.ylabel("Single uncoupled ML Neuron")
pylab.title("Animation of super-threshold ML Neuron")
anim.save('basic_ML.mp4', fps=30)

#plt.show()
