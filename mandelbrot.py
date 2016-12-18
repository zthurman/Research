## Complex Dynamics: Fractal Geometry
## Exploration of a bifurcation diagram for the Mandelbrot Set cheese

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

# Define the function that will generate the logistic map

def mandelbrotmap():
    # Define the logistic equation:
    def mandelbrot(x, c):
        return x**2 + c
    
    n = 10000     # number of r value steps b/w initial and final value
    c = np.linspace(-2, 0.25, n)     # range of r values
    iterations = 1500      # total number of iterations
    last = 100      # number of iterations used to generate map after transients have decayed out
    x = 1e-7 * np.ones(n)      # initial conditions of the logistic equation
    
    plt.figure()
    for i in range(iterations):
        x = mandelbrot(x, c)
        if i >= (iterations - last):
            plt.plot(c, x, ',k', alpha=.04)
    #plt.xlim(2.9, 4)
    plt.title('Bifurcation diagram - Mandelbrot Set')
    plt.xlabel('c')
    plt.ylabel('$z_n$')
