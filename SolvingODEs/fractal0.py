#!/usr/bin/python

from numpy import *
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def mandelbrot( h,w, maxit=150):
     # Returns an image of the Mandelbrot fractal of size (h,w).
        
    y,x = ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    q = -1
    divtime = maxit + zeros(z.shape, dtype=int)
    
    for i in xrange(maxit):
        z  = z**2 +  0.32 + 0.043j     #    c
        #z = ((z**2 + q - 1)/(2*z + q - 2))**2 # Model I in Models for Magnetism pg. 194
        #z =                                   # Model II in Models for Magnetism pg. 194
        diverge = z*conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much
    return divtime

figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
#img = mpimg.imread('../_static/stinkbug.png')
#lum_img = img[:,:,0]
#plt.imshow(mandelbrot(800,800))
pylab.imshow(mandelbrot(1200,1200))
#imgplot = plt.imshow(mandelbrot(800,800))
#imgplot.set_clim(0.0,0.7)
pylab.title('The Mandelbrot Viewer')
pylab.xlabel('This stuff')
pylab.ylabel('That stuff')
pylab.show()
