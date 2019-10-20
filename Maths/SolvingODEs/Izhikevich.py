# Izhikevich model
# Described by: cheese

# vdot = 0.04*v^2 + 5*v + 140 - u + I
# udot = a*(b*v - u)
# where:
#   v(v>30) = c
#   u(v>30) = u - d

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

# Parameter ranges: I = 10
# ~fast spiking: a = 0.02, b = 0.2, c = -65, d = 2
# ~regular spiking: a = 0.1, b = 0.2, c = -65, d = 2
# ~bursting: a = 0.02, b = 0.2, c = -50, d = 2

def Izhi(x,t, a = 0.02, b = 0.2, c = -65, d = 2, I = 10):
    if x[0] >= 30:
        x[0] = c
        x[1] = x[1] + d
    return np.array([0.04*(x[0]**2) + 5*x[0] + 140 - x[1] + I, \
                    a*(b*x[0] - x[1])])

def do_pplot():
    pylab.figure()
    X = RK4(x0 = np.array([0,0]), t1 = 300,dt = 0.01, ng = Izhi)
    pylab.plot(X[:,1], X[:,0])
    pylab.title("Phase Portrait - Izhikevich")
    pylab.xlabel("Membrane Recovery Variable")
    pylab.ylabel("Membrane Potential")
    pylab.savefig('Izhipplot.png')
    pylab.show()
    return

print do_pplot()

def do_tplot():
    pylab.figure()
    X = RK4(x0 = np.array([0,0]), t1 = 300,dt = 0.01, ng = Izhi)
    t0 = 0
    t1 = 300
    dt = 0.01
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,0])
    pylab.title("Membrane Potential over Time - Izhikevich")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential (mV)")
    pylab.ylim(-80,40)
    pylab.savefig('Izhitplot.png')
    pylab.show()
    return

print do_tplot()

def do_fftplot():
    X = RK4(x0 = np.array([0,0]), t1 = 300,dt = 0.01, ng = Izhi)
    Y = mean(X)    # determine DC component of signal
    X = X - Y      # subtract DC component from signal to get rid of peak at 0
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(len(X[:,0])/2 - 1), time_step)
    idx = np.argsort(freqs)
    pylab.plot(freqs[idx], ps[idx])
    pylab.title("Power Spectrum of Membrane Potential Signal - Izhikevich")
    pylab.xlabel("Frequency")
    pylab.ylabel("Power")
    pylab.xlim(0,0.6)
    pylab.ylim(0,1.75e10)
    pylab.savefig('Izhifftplot.png')
    pylab.show()
    return

print do_fftplot()
