# Hodgkins-Huxley

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

def HH(x,t, g_K=36, g_Na=120, g_L=0.3, E_K=12, E_Na=-115, E_L=-10.613, C_m=1, I=-10):
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
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.01, ng = HH)
    pylab.plot(X[:,1], X[:,0])
    pylab.title("Phase Portrait - HH")
    pylab.xlabel("Potassium gating variable")
    pylab.ylabel("Membrane Potential")
    pylab.savefig("HHpplot.png")
    pylab.show()
    return

print do_pplot()

def do_tplot():
    pylab.figure()
    X = RK4(x0 = np.array([0.01,0.01,0.01,0.01]), t1 = 100,dt = 0.01, ng = HH)
    t0 = 0
    t1 = 100
    dt = 0.01
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,-X[:,0])
    pylab.title("Membrane Potential over Time - HH")
    pylab.xlabel("Time")
    pylab.ylabel("Membrane Potential")
    pylab.savefig("HHtplot.png")
    #pylab.xlim(0,400)
    #pylab.ylim(-5,35)
    pylab.show()
    return

print do_tplot()

def do_fftplot():
    X = RK4(x0 = np.array([0,0,0,0]), t1 = 100,dt = 0.01, ng = HH)
    Y = mean(X[:,0])
    X[:,0] = X[:,0] - Y
    fdata = X[:,0].size
    ps = np.abs(np.fft.fft(X[:,0]))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(int(fdata/2 - 1), time_step)
    idx = np.argsort(freqs)
    pylab.plot(freqs[idx], ps[idx])
    pylab.title("Power Spectrum of Membrane Potential Signal")
    pylab.xlabel("Frequency ~(kHz)")
    pylab.ylabel("Power")
    pylab.xlim(0,1)
    pylab.ylim(0,1e10)
    pylab.savefig('HHfftplot.png')
    pylab.show()
    return

print do_fftplot()
