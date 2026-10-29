
import numpy as np

# Choose parent folder to save all folders/files created during this program
file_path = "/Users/katiez/Documents"

# "/tmp/CHANGE_THIS" #### SET AS USER

# Choose the number of spatial dimensions in which to compute
SPATIAL_DIM = 2 # 2 or 3

# Choose the independent variable for the proton transport SDE system
sde_model = "energy" # "energy" or "track_length"

# Choose the dose computation technique 
dose_method = "SF" # "SF" or "SK" for (shape function or spatial kernel)

# Choose the Monte Carlo sampling method
sampling_type= "mc" # "mc" or "mlmc"

# Standard MC only: number of proton simulations per CPU
SIMS_PER_CPU = 200 #hardware dependent

# MLMC only: (List of) the root mean square error values on which to perform MLMC
Eps = np.array([0.2, 0.1]) #np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2])
