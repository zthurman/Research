#!/usr/bin/env python
## Complex Dynamics: Fractal Geometry cheese
## Exploration of a bifurcation diagram for the Logistic Equation as described by Verhulst

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

#  Define the function that will generate the bifurcation diagram for the logistic map, with variation
# of the growth rate parameter r.

def logisticmap():
    # Define the logistic equation:
    def logistic(r, x):
        return r*x*(1-x)
    
    n = 20000     # number of r value steps b/w initial and final value
    r = np.linspace(2.9, 4.0, n)     # range of r values
    iterations = 1500      # total number of iterations
    lastits = 100      # number of iterations used to generate map after transients have decayed out
    x = 1e-7 * np.ones(n)      # initial conditions of the logistic equation
    
    plt.figure()
    for i in range(iterations):
        x = logistic(r, x)
        if i >= (iterations - lastits):
            plt.plot(r, x, ',k', alpha=.06)
    plt.xlim(2.9, 4)
    plt.title('Bifurcation diagram - Logistic Map')
    plt.xlabel('r')
    plt.ylabel('$x_n$')
    pylab.savefig('logisticmap.png')
    
print logisticmap()
