#!/usr/bin/env python

from numpy import *
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib


def mandelbrot( h,w, maxit=50):
     # Returns an image of the Mandelbrot fractal of size (h,w). Lower the maxit count, the less detail in the
     # set. Increase maxit for more details.
        
    y,x = ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    q = -1
    divtime = maxit + zeros(z.shape, dtype=int)
    
    for i in xrange(maxit):
        z  = z**2 +  c   #    c
        # c = -0.74591 + 0.11254j
        # c = -1.764 + 0.01j
        # c = -1.772 + 0.013j
        # c = -1.254024 + 0.046569j
        # c = -0.95 + 0.24387j
        # c = -0.925 + 0.26785j
        # c = -0.1011 + 0.9563j
        # c = 0.001643721971153 - 0.822467633298876j
        
        
        
        #z = (((z + -6 -5j)**2 + q - 1)/(2*z + q - 2))**2 # Model I in Models for Magnetism pg. 194
        #z =                                   # Model II in Models for Magnetism pg. 194
        diverge = z*conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much
    return divtime


plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
pylab.imshow(mandelbrot(3000,3000), cmap='afmhot_r')
pylab.contour(mandelbrot(3000,3000), colors='black', levels=xrange(15))
# cmap = cubehelix, rainbow, prism, seismic_r, nipy_spectral_r, terrain_r, 
# afmhot, CMRmap_r, bone_r
pylab.title('The Mandelbrot Viewer')
pylab.xlabel('c')
pylab.ylabel('That stuff')
#pylab.savefig('mandelbrot_countoured.png')
pylab.show()
