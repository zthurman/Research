#!/usr/bin/env python

## Complex Dynamics: Fractal Geometry
## Exploration of a bifurcation diagram for the Mandelbrot Set

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
    def mandelbrot1(x, c):
        return x**2 + c
    
    n = 10000     # number of r value steps b/w initial and final value
    c = np.linspace(-2, 0.25, n)     # range of r values
    iterations = 1500      # total number of iterations
    last = 100     # number of iterations used to generate map after transients have decayed out
    x = 1e-7 * np.ones(n)      # initial conditions of the logistic equation
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,10))  # define the figure with two subplots 
    for i in range(iterations):
        x = mandelbrot1(x, c)
        if i >= (iterations - last):
            ax1.plot(c, x, ',k', alpha=.04)
            ax1.set_title('Bifurcation diagram - Mandelbrot Set')
            ax1.set_xlabel('c')
            ax1.set_ylabel('$z_n$')   # subplot 1, time plot for FN
    
    
    def mandelbrot( h,w, maxit=100):
         # Returns an image of the Mandelbrot fractal of size (h,w).
            
        y,x = ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
        c = x+y*1j
        z = c
        q = -1
        divtime = maxit + zeros(z.shape, dtype=int)
        
        for i in xrange(maxit):
            z  = z**2 +  c     #    c
            # c = -0.74591 + 0.11254j
            # c = -1.764 + 0.01j
            # c = -1.772 + 0.013j
            # c = -1.254024 + 0.046569j
            # c = -0.95 + 0.24387j
            # c = -0.925 + 0.26785j
            
            
            #z = (((z + -6 -5j)**2 + q - 1)/(2*z + q - 2))**2 # Model I in Models for Magnetism pg. 194
            #z =                                   # Model II in Models for Magnetism pg. 194
            diverge = z*conj(z) > 2**2            # who is diverging
            div_now = diverge & (divtime==maxit)  # who is diverging now
            divtime[div_now] = i                  # note when
            z[diverge] = 2                        # avoid diverging too much
        return divtime
    
    ax2.imshow(mandelbrot(1500,1500), cmap='bone_r')   # subplot 1, phase portrait of membrane potential and membrane recovery variable - FN
    ax2.contour(mandelbrot(1500,1500), colors='black', levels = xrange(12))
    ax2.set_title('Mandelbrot Set')  # title for the first of the three sub-plots
    ax2.set_xlabel('c')      # label for the x-axis
    ax2.set_ylabel('$z_n$')      # label for the y-axis
    #pylab.savefig('mandelbrot_subplot.png')
    
    #pylab.figure()
    #plt.plot(mandelbrot(1500,1500), c, x, ',k', alpha=.04)
    #plt.show()
    
print mandelbrotmap()
