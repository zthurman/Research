#  This code is for simulating the ballistics of a projectile fired vertically
# into the air given the coupled nonlinear system of differential equations 
# derived from Newton's second law.

# Expressed as:
# m*v_xdot = - c*v_x*sqrt{v_x^2 + v_y^2}
# m*v_ydot = m*g - c*v_y*sqrt{v_x^2 + v_y^2}

from __future__ import division
from scipy import *
import numpy as np
import pylab
import matplotlib as mp
from matplotlib import pyplot as plt  
import sys
import math as mt

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

def ballistix(x,t, gamma = 0.25, m = 0.028, D = 0.0056, g = 9.8):
    c = gamma*(D**2)
    return np.array([ -(c*sqrt(x[0]**2 + x[1]**2)*x[0])/m, \
                    -g - (c*sqrt(x[0]**2 + x[1]**2)*x[1])/m])

def do_tplot():
    X = RK4(x0 = np.array([0,444]), t1 = 200, dt = 0.1, ng = ballistix)
    t0 = 0
    t1 = 200
    dt = 0.1
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,1])
    pylab.title("Y velocity over time, .22")
    pylab.xlabel("Time")
    pylab.ylabel("Vertical velocity")
    pylab.savefig("22 Figure.png")
    pylab.show()
    return

print do_tplot()

def ballistix(x,t, gamma = 0.25, m = 0.159, D = 0.0091, g = 9.8):
    c = gamma*(D**2)
    return np.array([ -(c*sqrt(x[0]**2 + x[1]**2)*x[0])/m, \
                    -g - (c*sqrt(x[0]**2 + x[1]**2)*x[1])/m])

def do_tplot():
    X = RK4(x0 = np.array([0,429]), t1 = 200,dt = 0.1, ng = ballistix)
    t0 = 0
    t1 = 200
    dt = 0.1
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,1])
    pylab.title("Y velocity over time, .357")
    pylab.xlabel("Time")
    pylab.ylabel("Vertical velocity")
    pylab.savefig("357 Figure.png")
    pylab.show()
    return

print do_tplot()

def ballistix(x,t, gamma = 0.25, m = 0.252, D = 0.0114, g = 9.8):
    c = gamma*(D**2)
    return np.array([ -(c*sqrt(x[0]**2 + x[1]**2)*x[0])/m, \
                    -g - (c*sqrt(x[0]**2 + x[1]**2)*x[1])/m])

def do_tplot():
    X = RK4(x0 = np.array([0,309]), t1 = 200,dt = 0.1, ng = ballistix)
    t0 = 0
    t1 = 200
    dt = 0.1
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,1])
    pylab.title("Y velocity over time, .45")
    pylab.xlabel("Time")
    pylab.ylabel("Vertical velocity")
    pylab.savefig("45 Figure.png")
    pylab.show()
    return

print do_tplot()

def ballistix(x,t, gamma = 0.25, m = 0.710, D = 0.0127, g = 9.8):
    c = gamma*(D**2)
    return np.array([ -(c*sqrt(x[0]**2 + x[1]**2)*x[0])/m, \
                    -g - (c*sqrt(x[0]**2 + x[1]**2)*x[1])/m])

def do_tplot():
    X = RK4(x0 = np.array([0,900.2]), t1 = 400,dt = 0.1, ng = ballistix)
    t0 = 0
    t1 = 400
    dt = 0.1
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,1])
    pylab.title("Y velocity over time, .50 BMG")
    pylab.xlabel("Time")
    pylab.ylabel("Vertical velocity")
    pylab.ylim(-500,1000)
    pylab.savefig("50 BMG Figure.png")
    pylab.show()
    return

print do_tplot()

def ballistix(x,t, gamma = 0.25, m = 0.3, D = 0.0185, g = 9.8):
    c = gamma*(D**2)
    return np.array([ -(c*sqrt(x[0]**2 + x[1]**2)*x[0])/m, \
                    -g - (c*sqrt(x[0]**2 + x[1]**2)*x[1])/m])

def do_tplot():
    X = RK4(x0 = np.array([0,396.2]), t1 = 200, dt = 0.1, ng = ballistix)
    t0 = 0
    t1 = 200
    dt = 0.1
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,1])
    pylab.title("Y velocity over time, 12 Gauge")
    pylab.xlabel("Time")
    pylab.ylabel("Vertical velocity")
    pylab.savefig("12 Gauge Figure.png")
    pylab.show()
    return

print do_tplot()

def ballistix(x,t, gamma = 0.25, m = 4.672, D = 0.1173, g = 9.8):
    c = gamma*(D**2)
    return np.array([ -(c*sqrt(x[0]**2 + x[1]**2)*x[0])/m, \
                    -g - (c*sqrt(x[0]**2 + x[1]**2)*x[1])/m])

def do_tplot():
    X = RK4(x0 = np.array([0,452.93]), t1 = 120,dt = 0.1, ng = ballistix)
    t0 = 0
    t1 = 120
    dt = 0.1
    tsp = np.arange(t0, t1, dt)
    pylab.plot(tsp,X[:,1])
    pylab.title("Y velocity over time, 12 pounder")
    pylab.xlabel("Time")
    pylab.ylabel("Vertical velocity")
    pylab.savefig("12 pounder Figure.png")
    pylab.show()
    return

print do_tplot()
