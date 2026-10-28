
import os
import numpy as np
from math import ceil, prod
import warnings

#Units:
#length = cm
#mass = g
#Beam axis: x

#=====================================================================================
#------------KEY PARAMETERS-----------------------------------------------------------
#=====================================================================================

file_path = "/tmp/CHANGE_THIS" #### SET AS USER
SPATIAL_DIM = 2 # 2 or 3
sampling_type= "mlmc" # "mc" or "mlmc"
sde_model = "energy" # "energy" or "track_length"
dose_method = "SF" # "SF" or "SK" for (shape function or spatial kernel)

#-----------MC PARAMETERS-----------------------

SIMS_PER_CPU = 1600 #1600

#-----------MLMC PARAMETERS---------------------

MLMC_LEVEL_OFFSET = 5 #5-9
L_conv_test = 8 #runtime caution if finest level exceeds 12
N_conv_test = 200000 #minimum: 50k, ideal: >=.100K (runtime caution)

Lmax = 17 #max mlmc level before termination
Eps = np.array([0.2, 0.1]) #np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2])
N0 = 50000 #need large enough to get a good variance estimate

#=====================================================================================
#-------------------------------------------------------------------------------------
#=====================================================================================

if dose_method=='SF':
    theta = 0.25 #proportion allocated to discretisation error
if dose_method=='SK':
    theta=0.5

#Fixed parameters:
base = 2
Lmin = 2 #min mlmc level

NUM_CPUS = os.cpu_count() - 2 #Leave one free
E0 = 62 #MeV
EMIN = 0.1 #The stopping power model diverges here
T=E0-EMIN
EPS_0 = 0.005
ALPHA =  2.633e-3
p = 1.735

def stopping_power(E, a=ALPHA, p=p):
    return 1/(a*p) * E**(1-p)

RHO = 1 #water is 1gcm^{-3}
side_len=0.05 #0.02 - this is what track_length uses for 200k sims
len_reciprocal=1/side_len
SIGMA = side_len*3/4  # side_len/4 was too small and gave rough images; side_len/2 also seemed small.

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
master_seed_seq = np.random.SeedSequence(42)

#=====================================================================================
#-------------------------------------------------------------------------------------
#=====================================================================================

#ds = 0.005*2 #positive!
#dE = -0.09

sim_num = NUM_CPUS * SIMS_PER_CPU
straggling_severity = "Moderate" #None, Light, Moderate, Strong
range_allowance = 1.6
y_scaling = 1.0
origin=np.array([0.0, 0.0, 0.0])

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

#For Gauss quad in dose method 1
t1 = 1/np.sqrt(3)
quad_nodes = [-t1, t1]

KAPPA = choose_kappa()

def choose_l():
    """
    Calculates side_len given the number of simulations N for the given dimension, for the FEM method.
    """
    N = SIMS_PER_CPU * NUM_CPUS
    factor = 0.8 #can play with this, have learnt 0.5 is too small
    return N**(-1/(2*SPATIAL_DIM)) * factor
def choose_Omega_0():
    if SPATIAL_DIM == 3:
        Omega0 = np.array([1.0, 0.0, 0.0]) #Must be on the unit sphere
    elif SPATIAL_DIM == 2:
        Omega0 = np.array([1.0, 0.0]) #Beam axis is the x axis, must be on the unit sphere
    return Omega0
def calculate_R0():
    """
    Analytical approx for the max range of protons
    """
    return ALPHA*E0**p

Omega0=choose_Omega_0()
RO=calculate_R0()

def nodes(side_len):
    """
    Defines the length dimensions of the cuboid D and gives the positions of all nodes in D
    The origin is at the corner of domain D's face where the beam originates, so all coordinates
    within D are positive.
    ALL nodes coordinates must be exactly divisible by the side length side_len

    Inputs:
            range_allowance, float in (1, 2): R0*range_allowance \approx length of D along beam axis
            y_scaling, float in (0, 1): length of D*y_scaling \approx the width and depth in the other two directions

    Returns: tuple X,Y,(Z): type(X) = np.array ; X.shape = (number of nodes along x axis, y sim, z sim)
                                                                including the endpoints!

    Notes: X[i,j,k] will retrieve the x coordinate of the ith,jth,kth node
           i,j,k=0,0,0 -> (x,y,z)=(0,-1,-1)

    """
    R0 = calculate_R0()
    range_upper = R0*range_allowance
    lower_x_bd = -9 #needs to be large enough to get hit w ~0% chance
                    #if varying parameters -- this will need to be adjusted

    #Make sure the lengths are divisible by side_len
    beam_axis_depth = ceil(range_upper / side_len)*side_len
    height = ceil(range_upper*y_scaling / side_len)*side_len #height from the origin lower corner

    #Calculate the total number of nodes in each direction
    nx = int(round(beam_axis_depth / side_len)) + 1
    ny = int(round(height / side_len)) + 1

    #Turn these into node positions with side_len
    xs = np.arange(lower_x_bd, nx) * side_len
    ys = np.arange(ny) * side_len

    if SPATIAL_DIM==3:
        zs = ys #the y and z directions are the same
        X,Y,Z=np.meshgrid(xs,ys,zs, indexing='ij')
        return X,Y,Z
    elif SPATIAL_DIM==2:
        X,Y=np.meshgrid(xs,ys, indexing='ij')
        return X,Y
    else:
        raise ValueError("Dimension (dim) must be either 2D or 3D.")

if SPATIAL_DIM==3:
    X_meshgrid, Y_meshgrid, Z_meshgrid = nodes(side_len)
    meshgrid = X_meshgrid, Y_meshgrid, Z_meshgrid
    nodes_array = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel(), Z_meshgrid.ravel()], axis=-1)
if SPATIAL_DIM==2:
    X_meshgrid, Y_meshgrid= nodes(side_len)
    meshgrid = X_meshgrid, Y_meshgrid
    nodes_array = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel()], axis=-1)

#Store this once
node_lookup = {tuple(np.rint(node / side_len).astype(int)): i for i, node in enumerate(nodes_array)}

def choose_X0(meshgrid):
    """
    Calculate length, depth and width of the cuboid domain. Assumes bounded below by zero in all dims.
    Then returns X0, the expected initial position of the Gaussian beam.

    meshgrid: tuple (X, Y, Z) indexing ij
    """
    domain_lower_bounds = np.zeros(SPATIAL_DIM)

    if SPATIAL_DIM==3:
        X_meshgrid, Y_meshgrid, Z_meshgrid = meshgrid
        upper_X = X_meshgrid[-1,-1,-1]
        upper_Y = Y_meshgrid[-1,-1,-1]
        lower_X = X_meshgrid[0,0,0]
        lower_Y = Y_meshgrid[0,0,0]
        lower_Z = Z_meshgrid[0,0,0]
        upper_Z = Z_meshgrid[-1,-1,-1] #hi

        domain_upper_bounds = np.array([upper_X, upper_Y, upper_Z])
        domain_lower_bounds = np.array([lower_X, lower_Y, lower_Z])

        y0 = (lower_Y + upper_Y)*1/2
        z0 = (lower_Z + upper_Z)*1/2
        X0 = np.array([0.0,y0,z0])

    elif SPATIAL_DIM ==2:
        X_meshgrid, Y_meshgrid = meshgrid
        upper_X = X_meshgrid[-1,-1]
        lower_X = X_meshgrid[0,0]
        upper_Y = Y_meshgrid[-1,-1]
        lower_Y = Y_meshgrid[0,0] #This file relies on this being 0

        domain_upper_bounds = np.array([upper_X, upper_Y])
        domain_lower_bounds = np.array([lower_X, lower_Y])

        y0 = (lower_Y + upper_Y)*1/2
        X0 = np.array([0.0,y0])

    return X0, domain_lower_bounds, domain_upper_bounds

X0_mean, domain_lower_bounds, domain_upper_bounds = choose_X0(meshgrid)
dose_shape = prod(X_meshgrid.shape) #More accurately this is a length but it works as a 1D shape
max_steps = sum(X_meshgrid.shape) #All boundaries, just needs to be large enough but relatively small

def verify_params():
    """
    Check that all parameters assigned in this file are valid and raise some potential warnings.
    """
    if RHO <= 0:
        raise ValueError("medium density RHO must be positive.")
    if sde_model not in ['energy', 'track_length']:
        raise ValueError(
            "sde_model must be either energy or track_length, depending on the independent variable required."
            )
    if dose_method not in ['SF', 'SK']:
        raise ValueError(
            "dose_method must be either SF or SK, for computation methods shape function or spatial kernel."
            )
    if sampling_type not in ['mc', 'mlmc']:
        raise ValueError(
            "sampling_type must be either mc or mlmc for Monte Carlo and Multilevel Monte Carlo sampling."
            )
    if SPATIAL_DIM not in [2, 3]:
        raise ValueError("spatial dimension must be either 2 or 3.")
    if theta>1 or theta<0:
        raise ValueError("mlmc error weighting parameter theta must be in interval [0,1].")
    if EMIN > 1:
        warnings.warn(f"EMIN of {EMIN} may be too large.", UserWarning)
    if min(Eps) < 0.2 or L_conv_test + MLMC_LEVEL_OFFSET >= 16:
        warnings.warn("mlmc likely to be slow under chosen parameterisation.", RuntimeWarning)
    if E0 <= 0:
        raise ValueError("initial energy E0 must be positive.")
    if not(50 <= E0 <= 250):
        warnings.warn(f"E0 of {E0} is outside therapeutic energy range.", UserWarning)
    if N0 < 20000:
        warnings.warn(f"N0 of {N0} may be too small to achieve suitable variable estimates.")
verify_params()

#set up title labels
if dose_method=='SF':
    title_seg = f"{'Bilinear' if SPATIAL_DIM==2 else 'Trilinear'} Basis Function"
elif dose_method=='SK':
    title_seg = "Spatial Kernel"


