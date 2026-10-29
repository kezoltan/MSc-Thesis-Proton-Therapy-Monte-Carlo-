# July 2026 (organised October 2026)
# Parameter settings that explicitly depend on other functions

from ._fixed_params import *
from .core_params import *
from src.setup_functions.proton_setup_Bragg_Kleeman import *
from src.setup_functions.nodal_mesh_setup import *
from math import prod

# Straggling parameter
KAPPA = choose_kappa()

#Initial direction (must be on the unit sphere)
Omega0 = choose_Omega_0()

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

X0_mean, domain_lower_bounds, domain_upper_bounds = choose_X0(meshgrid)
dose_shape = prod(X_meshgrid.shape) #More accurately this is a length but it works as a 1D shape
max_steps = sum(X_meshgrid.shape) #All boundaries, just needs to be large enough but relatively small

# Standardised title element for plots 
if dose_method=='SF':
    title_seg = f"{'Bilinear' if SPATIAL_DIM==2 else 'Trilinear'} Basis Function"
elif dose_method=='SK':
    title_seg = "Spatial Kernel"

# Theta (MLMC only): proportion of error allocated to discretisation (vs. sampling)
if dose_method=='SF':
    theta = 0.25 
if dose_method=='SK':
    theta=0.5

# def choose_l():
#     """
#     Calculates side_len given the number of simulations N for the given dimension, for the FEM method.
#     """
#     N = SIMS_PER_CPU * NUM_CPUS
#     factor = 0.8 #can play with this, have learnt 0.5 is too small
#     return N**(-1/(2*SPATIAL_DIM)) * factor