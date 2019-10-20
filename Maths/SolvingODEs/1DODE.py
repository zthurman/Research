#!/usr/bin/env python
# 1D ODE, exponential function

from matplotlib import pyplot as plt      # plt function defined for plotting, imported from matplotlib
import numpy as np
from pylab import figure, show, xlabel, ylabel
from scipy.integrate import odeint
import pylab

a = 2

def f(y, t_out):
    return a*y

t_out = np.arange(0, 6, 0.1)
# print t_out

y0 = 1.0

y_out = odeint(f, y0, t_out)
y_out = y_out[:, 0]  # convert the returned 2D array to a 1D array
# print y_out

# Plot of the ODE

def do_tplot():
    pylab.figure()
    t_out = np.arange(0, 6, 0.1)
    y_out = odeint(f, y0, t_out)
    y_out = y_out[:, 0]  # convert the returned 2D array to a 1D array
    pylab.plot(t_out,y_out)
    pylab.title("Solution to ydot = a*y where a is constant")
    pylab.xlabel("Time")
    pylab.ylabel("y")
    pylab.show()
    return

print do_tplot()
