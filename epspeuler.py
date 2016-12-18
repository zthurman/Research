# EPSP: http://en.wikipedia.org/wiki/Excitatory_postsynaptic_potential
# Excitatory postsynaptic potential

# Imports:

import numpy as np
import scipy
import matplotlib.pylab as plt

def EPSPEuler():
    # inits and contants
    c_m = 1
    g_L = 1
    tau_syn = 1
    E_syn = 10
    delta_t = 0.01
    g_syn = np.zeros([1200])
    I_syn = np.zeros([1200])
    v_m = np.zeros([1200])
    t = np.zeros([1200])
    g_syn[0] = 0
    I_syn[0] = 0
    v_m[0] = 0
    t[0] = 0

    # Numerical integration with Euler's method
    for i in np.arange(1, (10/delta_t)+1):
        t[int(i)-1] = t[int((i-1))-1]+delta_t
        if np.abs((t[int(i)-1]-1))<0.001:
            g_syn[int((i-1))-1] = 1
        
        g_syn[int(i)-1] = g_syn[int((i-1))-1]-np.dot((delta_t/tau_syn), g_syn[int((i-1))-1])
        I_syn[int(i)-1] = np.dot(g_syn[int(i)-1], v_m[int((i-1))-1]-E_syn)
        v_m[int(i)-1] = v_m[int((i-1))-1]-np.dot(np.dot((delta_t/c_m),g_L), v_m[int((i-1))-1]) \
                        -np.dot((delta_t/c_m), I_syn[int(i)-1])
        
    # plots
    plt.plot(t, v_m)
    plt.plot(t, (g_syn*5.), 'r--')
    plt.plot(t, (I_syn/5.), 'k:')
    plt.title("Membrane Potential over Time - EPSP")
    plt.xlabel("Time")
    plt.ylabel("Membrane Potential")
    plt.legend(['Membrane Potential','Conductance','Synaptic Current'])
    plt.savefig('EPSPtplot.png')
    return

print EPSPEuler()
