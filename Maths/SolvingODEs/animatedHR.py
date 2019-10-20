#!/usr/bin/env python
#   Numerical simulation for Hindmarsh-Rose model
# Defined as: x(dot) = y-(a*x^3) + (b*x^2) - z + I*(1+0.1*sin(wp.*t)
#             y(dot) = c - d*x^2 - y
#             z(dot) = r*(s*(x - x0) - z)

# Animated plot of a neuron over time

from __future__ import division
from scipy import *
import numpy as np
import pylab
import matplotlib as mp
from matplotlib import pyplot as plt  
import sys
import matplotlib.animation as animation

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

# Animation of the neuron firing over time

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 100), ylim=(-2, 3))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    global RK4, HR
    X = RK4(x0 = np.array([3, 0, -1.2]), t1 = 350,dt = 0.02, ng = HR)
    t0 = 0
    t1 = 350
    dt = 0.02
    tsp = np.arange(t0, t1, dt)
    line.set_data(tsp-0.5*i, X[:,0])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=500, interval=20, blit=True)
pylab.xlabel("Time")
pylab.ylabel("Single uncoupled HR Neuron")
pylab.title("Animation of super-threshold HR Neuron")
anim.save('basic_HR.mp4', fps=30)

#plt.show()
