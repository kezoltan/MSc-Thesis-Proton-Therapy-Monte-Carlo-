
import os
import numpy as np
from math import ceil, prod

#Units:
#length = cm 
#mass = g
#Beam axis: x

NUM_CPUS = os.cpu_count() - 2 #Leave one free 
E0 = 62 #MeV
EMIN = 1.0 #The stopping power model diverges here
T=E0-EMIN
EPS_0 = 0.005 
ALPHA =  2.633e-3
p = 1.735

def stopping_power(E, a=ALPHA, p=p):
    return 1/(a*p) * E**(1-p)

RHO = 1 #water is 1gcm^{-3} 
if RHO <= 0:
    raise ValueError("density must be positive.")
l=0.05 #0.02 - this is what V uses for 100k sims
l_reciprocal=1/l
SIGMA = l*1/2 #tried making this l/4, definitely too small, images were very rough

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
file_path = r"C:\Users\kathe\OneDrive - Zolution Technologies\Oxford\Dissertation\Code\Dose Map Code\dose map results"
                #r"/home/zoltan/Documents/dose map code repo

#------------KEY PARAMETERS---------------------

sampling_type="mc" #"mc" or "mlmc" or "anti_mlmc"
SPATIAL_DIM = 2 #making this 3 is a nightmare for plotting method 1
method = "KZ"
dose_method="SK" #"SF" or "SK"

#-----------MC PARAMETERS----------------------

SIMS_PER_CPU = 50 #7146 #1064

#-----------MLMC PARAMETERS---------------------

MLMC_LEVEL_OFFSET = 6 #don't make this exceed 9 please (mc level)
                      #i.e. this is mlmc level 0 but stepsize level 7 
base = 2 #refinement base

#mlmc levels,  
Lmin = 2 #min level
Lmax = 5 #10 #max level
N0 = 20 #num samples level 0
final_time = E0 - EMIN #stepping in eta 

#for the mlmc tests, mlmc level!! (don't include the offset)
N_conv_test = 10000 
L_conv_test = 4 #if the step size level exceeds 12 it gets noticably slower 
Eps = [0.2] #0.01, 0.02, 0.05, 0.1, 

if L_conv_test + MLMC_LEVEL_OFFSET >= 13:
    print(f"Caution: max stepsize level: {L_conv_test + MLMC_LEVEL_OFFSET} is likely too fine for mlmc test.")

#-----------------------------------------------

#ds = 0.005*2 #positive!
#dE = -0.09

sim_num = NUM_CPUS * SIMS_PER_CPU
straggling_severity = "Moderate" #None, Light, Moderate, Strong
range_allowance = 1.3
y_scaling = 0.75
origin=np.array([0.0, 0.0, 0.0])
width_sdev_factor = 3 #how many times l should the width be, also depends on E0 ideally
energy_spread_percent = 1/100 #initial energy spread in % from Chronholm and Pryer 2026 
                                #we will interpret this as the % of the mean = standard dev 
energy_spread=True
energy_sdev = 0#E0 * energy_spread_percent #convert to the standard dev 
if energy_sdev == 0:
    energy_spread=False

#if sampling_type=='mlmc':
#    if energy_sdev != 0:
#        print("Setting energy_sdev to zero.")
#        def modify_e_sdev():
#            global energy_sdev
#            energy_sdev = 0
#        modify_e_sdev()
#    if method != 'KZ':
#        print("Setting method to KZ.")
#        def modify_method():
#            global method
#            method = 'KZ'
#        modify_method()

#For Gauss quad in dose method 1
t1 = 1/np.sqrt(3)
quad_nodes = [-t1, t1]

KAPPA = choose_kappa()
if dose_method not in ["SK", "SF"]:
    raise ValueError("dose_method must be either SK (spatial kernel) or SF (sbape function).")
if method not in ["V", "KZ"]:
    raise ValueError("method must be 'KZ' or 'V'.")

def choose_l():
    """
    Calculates l given the number of simulations N for the given dimension, for the FEM method.
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

def nodes(l):
    """
    Defines the length dimensions of the cuboid D and gives the positions of all nodes in D 
    The origin is placed at the corner of the face of domain D where the beam originates, and is such that the coords within D are all positive 
    ALL nodes coordinates must be exactly divisible by the side length l

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
    
    #Make sure the lengths are divisible by l
    beam_axis_depth = ceil(range_upper / l)*l 
    height = ceil(range_upper*y_scaling / l)*l #height from the origin lower corner   
    
    #Calculate the total number of nodes in each direction 
    nx = int(round(beam_axis_depth / l)) + 1
    ny = int(round(height / l)) + 1 
    
    #Turn these into node positions with l
    xs = np.arange(nx) * l
    ys = np.arange(ny) * l  
    
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
    X_meshgrid, Y_meshgrid, Z_meshgrid = nodes(l)
    meshgrid = X_meshgrid, Y_meshgrid, Z_meshgrid
    nodes_array = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel(), Z_meshgrid.ravel()], axis=-1)
if SPATIAL_DIM==2:
    X_meshgrid, Y_meshgrid= nodes(l)
    meshgrid = X_meshgrid, Y_meshgrid
    nodes_array = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel()], axis=-1)

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
        lower_Y = Y_meshgrid[0,0,0] #This file relies on this being 0
        lower_Z = Z_meshgrid[0,0,0]
        upper_Z = Z_meshgrid[-1,-1,-1] #hi

        if lower_Y != 0 or lower_Z != 0:
            raise ValueError("lower domain bounds should be 0.")
        domain_upper_bounds = np.array([upper_X, upper_Y, upper_Z])
        lower_X, lower_Y, lower_Z = domain_lower_bounds
        upper_X, upper_Y, upper_Z = domain_upper_bounds

        y0 = (lower_Y + upper_Y)*1/2
        z0 = (lower_Z + upper_Z)*1/2
        X0 = np.array([0.0,y0,z0])

    elif SPATIAL_DIM ==2:
        X_meshgrid, Y_meshgrid = meshgrid
        upper_X = X_meshgrid[-1,-1]
        upper_Y = Y_meshgrid[-1,-1]
        lower_Y = Y_meshgrid[0,0] #This file relies on this being 0

        if lower_Y != 0:
            raise ValueError("lower domain bounds should be 0.")
        domain_upper_bounds = np.array([upper_X, upper_Y])
        lower_X, lower_Y= domain_lower_bounds
        upper_X, upper_Y= domain_upper_bounds

        y0 = (lower_Y + upper_Y)*1/2
        X0 = np.array([0.0,y0])

    return X0, domain_lower_bounds, domain_upper_bounds

_, domain_lower_bounds, domain_upper_bounds = choose_X0(meshgrid)

dose_shape = prod(X_meshgrid.shape) #More accurately this is a length but it works as a 1D shape
max_steps = sum(X_meshgrid.shape) #All boundaries, just needs to be large enough but relatively small





