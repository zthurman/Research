# Using Second Order numerical approximation, we solve: dx/dt = 1 - x + t

# Imports:

import numpy as np
import scipy
import matplotlib.pylab as plt

dt = 0.2
tend = 5
x = 1
xSecond = x
for t in np.arange(dt,(tend)+(dt),dt):
    x = x + np.dot(dt, 1-x+t) + np.dot(dt**2, (1-x)/2)
    xSecond = np.array(np.hstack((xSecond, x)))

t = np.arange(0,tend+dt,dt)
#xreal = t + exp(-t)
#plt.plot(t,xreal)
plt.plot(t,(xSecond - x)/x)
