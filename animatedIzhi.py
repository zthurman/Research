#!/usr/bin/env python
# Animated plot of a neuron over time

from __future__ import division
from scipy import *
import numpy as np
import pylab
import matplotlib as mp
from matplotlib import pyplot as plt  
import sys
import matplotlib.animation as animation

# Parameter ranges: I = 10
# ~fast spiking: a = 0.02, b = 0.2, c = -65, d = 2
# ~regular spiking: a = 0.1, b = 0.2, c = -65, d = 2
# ~bursting: a = 0.02, b = 0.2, c = -50, d = 2

# Supercritical Hopf with fast spiking parameters between: I = 3.77437 - 3.77438

# Defining the Izhikevich system of v(dot) and u(dot)
# inputs a, b, c and dgive biological behavior, I is the input stimulus
# v is the membrane potential variable, u is the membrane recovery variable

def Izhi(x,t, a = 0.02, b = 0.2, c = -65, d = 2, I = 10):
    if x[0] >= 30:
        x[0] = c
        x[1] = x[1] + d
    return np.array([0.04*(x[0]**2) + 5*x[0] + 140 - x[1] + I, \
                    a*(b*x[0] - x[1])])

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
ax = plt.axes(xlim=(0, 55.1), ylim=(-80, 40))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially by FuncAnimation
def animate(i):
    global RK4, Izhi
    X = RK4(x0 = np.array([0.01,0.01]), t1 = 500,dt = 0.02, ng = Izhi)
    t0 = 0
    t1 = 500
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    line.set_data(tsp-0.5*i, X[:,0])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=500, interval=20, blit=True)
pylab.xlabel("Time")
pylab.ylabel("Single uncoupled Izhikevich Neuron")
pylab.title("Animation of super-threshold Izhikevich Neuron - fast spiking")
anim.save('basic_Izhi_fastspiking.mp4', fps=30)

#plt.show()
