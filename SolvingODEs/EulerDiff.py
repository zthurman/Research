# Using Euler's method, we solve: dx/dt = 1 - x + t

# Imports:

import numpy as np
import scipy
import matplotlib.pylab as plt

# Initializing

dt = 0.2
tend = 5
x = 1
xEuler = x

for t in np.arange(dt,tend+dt,dt):
    x = x + np.dot(dt, (1 - x + t))
    xEuler = np.array(np.hstack((xEuler,x)))

t = np.arange(0,tend+dt,dt)
xreal = t + exp(-t)
#plt.plot(t,xreal)
plt.plot(t,(xEuler - x)/x)
