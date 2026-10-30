
import numpy as np
import os

from pbt_mlmc.config import CONFIG_FILE_ENV, read_config

# The CLI sets this before importing parameterisation, so dependent defaults
# are calculated from the user's settings during module initialization.
_config_file = os.environ.get(CONFIG_FILE_ENV)
CONFIG = read_config(_config_file) if _config_file else {}

# Choose parent folder to save all folders/files created during this program
file_path = CONFIG.get("file_path", "/tmp/CHANGE_THIS")

# Choose the number of spatial dimensions in which to compute
SPATIAL_DIM = CONFIG.get("SPATIAL_DIM", 2) # 2 or 3

# Choose the independent variable for the proton transport SDE system
sde_model = CONFIG.get("sde_model", "energy") # "energy" or "track_length"

# Choose the dose computation technique
dose_method = CONFIG.get("dose_method", "SF") # "SF" or "SK" for (shape function or spatial kernel)

# Choose the Monte Carlo sampling method
sampling_type= CONFIG.get("sampling_type", "mc") # "mc" or "mlmc"

# Standard MC only: number of proton simulations per CPU
SIMS_PER_CPU = CONFIG.get("SIMS_PER_CPU", 200) #hardware dependent

# MLMC only: (List of) the root mean square error values on which to perform MLMC
Eps = np.asarray(CONFIG.get("Eps", [0.2, 0.1])) #np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2])
