
import os
import numpy as np

#For the spatial kernel method only
l=0.05
#l = 0.02 - this is what V uses for 100k sims
SIGMA = l/2

SPATIAL_DIM = 2
SIMS_PER_CPU = 2000
dE = -0.09 #not including the first energy step
ds = 0.005 #positive!
METHOD = "V"

NUM_CPUs = os.cpu_count() - 2 #Leave two free 
E0 = 62 #MeV
EMIN = 1.0 #The stopping power model diverges here
EPS_0 = 0.005 #determined from E0 - COME BACK
ALPHA =  2.633e-3
P = 1.735
RHO = 1 #water is 1gcm^{-3} 
straggling_severity = "Moderate"

#For the domain D
range_allowance = 1.2
y_scaling = 0.5
origin=np.array([0.0, 0.0, 0.0])
width_sdev_factor = 3 #how many times l should the width be, also depends on E0 ideally
energy_spread_percent = 1/100 #initial energy spread in % from Chronholm and Pryer 2026 
                              #we will interpret this as the % of the mean = standard dev 

#def calcaulte_kappa():
#    return 2 * 0.072**2 / (P * ALPHA**2 * E0 **(2*P))

def choose_kappa():
    """
    None, Light, Moderate, Strong
    """
    if straggling_severity == "None":
        return 0
    elif straggling_severity == "Light":
        return 1e-6
    elif straggling_severity == "Moderate":
        return 4e-5
    elif straggling_severity == "Strong":
        return 1e-3
KAPPA = choose_kappa()






