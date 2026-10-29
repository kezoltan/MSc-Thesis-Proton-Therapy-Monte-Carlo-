# July 2026 (updated October 2026)
# Standard parameters for reproducing results 

import numpy as np
from .core_params import *
import os

# Parameters organised by purpose
# Medium is water phantom 
# Be aware that many parameters can and should be modified, and have linked values 
#   e.g. higher energy -> longer range -> consider increasing side_len for runtime

#======SIMULATION PARAMETERS======

# SEED
master_seed_seq = np.random.SeedSequence(42)

# Number of (CPU) cores used in parallel
NUM_CPUS = max(1, os.cpu_count() - 2)

sim_num = NUM_CPUS * SIMS_PER_CPU

#======PHYSICAL PARAMETERS (BEAM AND MEDIUM DEPENDENT)======

# Mean initial proton energy
E0 = 62 

# Medium dependent parameters 

# for BK model
ALPHA =  2.633e-3
p = 1.735
# Density of medium
RHO = 1 

#======SPATIAL AND MESH PARAMETERS======

# Given the range, allow at least this factor more depth along beam axis
range_allowance = 1.6

# Second depth factor, default is 1.0
y_scaling = 1.0

# Set the origin explicitly
origin=np.array([0.0, 0.0, 0.0])

# The uniform distance between adjacent nodes along one coordinate direction
side_len=0.05 #0.02 - this is what track_length uses for 200k sims
len_reciprocal=1/side_len

#======NUMERICAL PROTON TRANSPORT PARAMETERS======

# Proton absorption energy
EMIN = 0.1 

T=E0-EMIN

# Angular diffusion parameter
EPS_0 = 0.005

straggling_severity = "Moderate" #None, Light, Moderate, Strong

# Gauss quadrature parameters (for SF dose computation only)
t1 = 1/np.sqrt(3)
quad_nodes = [-t1, t1]

# Spatial kernel width parameter (for SK dose computation only)
SIGMA = side_len*3/4 # Need large enough for smooth results

#======BEAM SAMPLING (NEEDS REWORK)======

width_sdev_factor = 3 #how many times side_len should the width be, also depends on E0 ideally
width_spread=True
if width_sdev_factor==0:
    width_spread=False

energy_spread_percent = 1/100 #initial energy spread in % from Chronholm and Pryer 2026
                                #we will interpret this as the % of the mean = standard dev
energy_spread=True
energy_sdev = 0#E0 * energy_spread_percent #convert to the standard dev
if energy_sdev == 0:
    energy_spread=False

#======MLMC RECOMMENDED PARAMETERS======

# The step level at which MLMC level 0 is initialised
MLMC_LEVEL_OFFSET = 5 #5-9

# The number of MLMC levels use in MLMC test
L_conv_test = 8

# The number of proton simulations used in MLMC test
N_conv_test = 200000 #minimum: 50k, ideal: >=100k 

# Maximum MLMC level before MLMC driver is forced to terminate
Lmax = 17 

# Initialising simulation number for MLMC driver
N0 = 50000 #need large enough to get a good variance estimate

# Base for geometric MLMC step sizes
base = 2

# Minimum MLMC level for MLMC driver
Lmin = 2 


#ds = 0.005*2 #positive!
#dE = -0.09