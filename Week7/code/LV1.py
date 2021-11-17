#!/usr/bin/env python3
"""Runs and plots Lotka-Volterra model for predator-prey system"""
__author__ = "Izie Wood (iw121@ic.ac.uk)"

############## imports ###############
import numpy as np
import scipy as sc
import matplotlib.pylab as p
import scipy.integrate as integrate

############# functions ###############
def dCR_dt(pops, t=0):
    """Returns growth rate of consumer and resource population
    at any given time step"""
    R = pops[0]
    C = pops[1]
    dRdt = r * R - a * R * C
    dCdt = -z * C + e * a * R * C

    return np.array([dRdt, dCdt])

#### assign parameter values
r = 1.
a = 0.1
z = 1.5
e = 0.75

#### Define the time vector 
t = np.linspace(0, 15, 1000)

#### set initial conditions for the two populations 
R0 = 10
C0 = 5
RC0 = np.array([R0, C0]) # convert to array to inpurt in dCR_dt

### numerically integrate this system forwards from those starting conditions
pops, infodict = integrate.odeint(dCR_dt, RC0, t, full_output=True)
infodict['message'] # return successful integration

### visualise results

# plot population density of Resources and Consumers over time
f1 = p.figure(1)

p.plot(t, pops[:,0], 'g-', label='Resource density') # Plot resource density
p.plot(t, pops[:,1], 'b-', label = 'Consumer density') # Plot consumer density
p.grid()
p.legend(loc='best')
p.xlabel('Time')
p.ylabel('Population density')
p.title('Consumer-Resource population dynamics')


# plot consumer density by Resource density
f2 = p.figure(2)
p.plot(pops[:,0], pops[:,1], 'ro')
p.grid()
p.xlabel('Resource density')
p.ylabel('Consumer density')
p.title('Consumer-Resource population dynamics')


## save figures to pdf 
f1.savefig('../results/LV_model.pdf')
f2.savefig('../results/LV_model1.pdf')

