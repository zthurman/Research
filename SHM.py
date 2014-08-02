#!/usr/bin/env python
# 2nd Order ODE: Simple Harmonic Oscillator
from matplotlib import pyplot as plt      # plt function defined for plotting, imported from matplotlib
import numpy as np
from pylab import figure, show, xlabel, ylabel
from scipy.integrate import odeint
import pylab

omega_squared = 4     # This is the angular frequency of the oscillation 

# Next f is defined as a function of two variables, the x-vector and time

def f(x_vec, t):
    
    x,y = x_vec  # tuple unpacking, x_vec defined as a two dimensional vector made up of x and y
    
    return [y, -omega_squared*x]

y0 = [0.7, 0.5]    # initial conditions of the dependent y variable

t_out = np.arange(0, 5, 0.1)   # time vector, 0-5s with stepsize of 0.1

y_out = odeint(f, y0, t_out)   # y vector, numerical approximation to the solution of function f determined  #beginning at y0 over t0

plt.figure(figsize = (5,5))  # plot the function and define the figure size

fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize=(10,10))  # define the figure with three subplots 

xx, yy = y_out.T  # extract x and y cols

ax1.plot(xx, yy)    # subplot 1, phase portrait of xdot and ydot
ax1.set_title('Phase Portrait SHO')  # title for the first of the three sub-plots
ax1.set_xlabel('xdot')      # label for the x-axis
ax1.set_ylabel('ydot')      # label for the y-axis
#plt.sca(ax1)
#colorline(xx, yy, cmap='jet')
ax1.axis('scaled')    # autoscale the output  

ax2.plot(t_out, xx)    # subplot 2, x variable over time interval
ax2.set_title('xdot over time')  # title for the second of the three sub-plots
ax2.set_xlabel('time')      # label for the t-axis
ax2.set_ylabel('xdot')      # label for the y-axis
#plt.sca(ax2)
#colorline(t_output, xx, cmap='cool')
ax2.axis('scaled')    # autoscale the output

ax3.plot(t_out, yy)     # subplot 3, y variable over time interval
ax3.set_title('ydot over time')  # title for the third of the three sub-plots
ax3.set_xlabel('time')      # label for the t-axis
ax3.set_ylabel('ydot')      # label for the y-axis
ax3.axis('scaled')    # autoscale the output

plt.tight_layout()
plt.show()
