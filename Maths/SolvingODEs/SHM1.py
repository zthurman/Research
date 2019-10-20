#!/usr/bin/env python
# Alternative SHM Plotting method, slightly more differenter than SHM.py

from scipy.integrate import ode
from matplotlib import pyplot as plt      # plt function defined for plotting, imported from matplotlib
import numpy as np
from pylab import figure, show, xlabel, ylabel
import pylab

# Note: this alternative method requires the function be defined as f(t,y) as opposed to f(y,t) as done above
# Simple Harmonic Oscillator, 2D ODE

omega_squared = 4   # here again defining the angular frequency

def f(t, x_vec):      # define the function f(t,x_vec), note the flip, t is the first variable of the f function
    x, y = x_vec      # tuple unpacking, x_vec defined as a two dimensional vector made up of x and y
    
    return [y, -omega_squared*x]

t_final = 10.0    # final entry of the time vector for the region of f to be numerically solved
dt = 0.1          # timestep

y0 = [0.7, 0.5]   # initial conditions for the y variable of the x vector
t0 = 0.0          # initial time for the numerically obtained solution of f

y_result = []     # empty solution vector
t_output = []     # empty time vector

# Initialization:

backend = "dopri5"    # this is the version of the Runge-Kutte algorithm that's numerically approximating the solution

solver = ode(f)
solver.set_integrator(backend)  # nsteps=1
solver.set_initial_value(y0, t0)

y_result.append(y0)
t_output.append(t0)

while solver.successful() and solver.t < t_final:
    solver.integrate(solver.t + dt, step=1)
    
    y_result.append(solver.y)
    t_output.append(solver.t)
    
y_result = np.array(y_result)    # output of the solver, y-vector
t_output = np.array(t_output)    # time array output by the solver 

#print y_result

fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize=(10,10))

xx, yy = y_result.T  # extract x and y cols

ax1.plot(xx, yy)
#plt.sca(ax1)
ax1.set_title('Phase Portrait SHO')  # title for the first of the three sub-plots
ax1.set_xlabel('xdot')      # label for the x-axis
ax1.set_ylabel('ydot')      # label for the y-axis
#colorline(xx, yy, cmap='jet')
ax1.axis('scaled')

ax2.plot(t_output, xx)
#plt.sca(ax2)
ax2.set_title('xdot over time')  # title for the second of the three sub-plots
ax2.set_xlabel('time')      # label for the t-axis
ax2.set_ylabel('xdot')      # label for the y-axis
#colorline(t_output, xx, cmap='cool')
ax2.axis('scaled')

ax3.plot(t_output, yy)
#plt.sca(ax3)
ax3.set_title('ydot over time')  # title for the third of the three sub-plots
ax3.set_xlabel('time')      # label for the t-axis
ax3.set_ylabel('ydot')      # label for the y-axis
#colorline(t_output, yy)
ax3.axis('scaled')

plt.tight_layout()
plt.show()

# Plotting the vector field of a harmonic oscillator in phase space

# pylab.figure()
x = np.linspace(-1.0, 1.0, 10)   # create a linear space for the vector field
XX, YY = np.meshgrid(x, x)      # generate a mesh grid for this vector field

k = omega_squared
plt.figure()
plt.quiver(XX, YY, YY, -k*XX, pivot='middle')      # quiver allows the plotting of vector fields in two dimensions
pylab.title('Vector field for SHM')
pylab.xlabel('Xdot')
pylab.ylabel('Ydot')
plt.axis('equal')      # can use 'scaled' as input 
plt.show()

# Fancy Phase Space Vector Field for SHO

plt.streamplot(XX, YY, YY, -k*XX)   # allows fancy flowie arrows to indicate a cooler looking stream of ther field
plt.quiver(XX, YY, YY, -k*XX, pivot = 'middle')
pylab.title('Fancy Vector field for SHM')
pylab.xlabel('Xdot')
pylab.ylabel('Ydot')
plt.axis('equal')
plt.autoscale(True,'both',True)
plt.show()
