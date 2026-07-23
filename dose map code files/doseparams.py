
import os
import numpy as np

SPATIAL_DIM = 2
SIMS_PER_CPU = 10
dE = -0.09 #not including the first energy step
ds = 0.005 #positive!
METHOD = "KZ"

NUM_CPUs = os.cpu_count() - 2 #Leave two free 
E0 = 62 #MeV
EPS_0 = 0.005 #determined from E0 - COME BACK
ALPHA =  2.633e-3
P = 1.735
EMIN = -dE
RHO = 1 #water is 1gcm^{-3} 

#For the domain D
range_allowance = 1.2
y_scaling = 0.5
origin=np.array([0.0, 0.0, 0.0])

def calcaulte_kappa():
    return 2 * 0.072**2 / (P * ALPHA**2 * E0 **(2*P))
KAPPA = calcaulte_kappa()





