from src.parameterisation.core_params import *
from src.parameterisation._fixed_params import *
from src.parameterisation._conditional_params import dose_shape, X_meshgrid, Y_meshgrid
from src.setup_functions.nodal_mesh_setup import linear_lumped_mass_matrix

# added defaults so tests are easier to run
def load_mass_matrix():
    """Load a cached lumped mass matrix, or calculate and save it.

    Returns
    -------
    M_diag : numpy.ndarray
        Entries of the linear lumped mass matrix diagonal.
    """
    M_path = os.path.join(file_path, f"{SPATIAL_DIM}D_{dose_shape}_lumped_mass_matrix.npz")
    if os.path.exists(M_path):
        mass_data = np.load(M_path)
        M_diag = mass_data["M_diag"]
        mass_shape = M_diag.shape[0]
        if mass_shape != dose_shape:
            raise ValueError(f"Lumped mass matrix shape must equal {dose_shape}, is {mass_shape}.")
        print("Mass matrix loaded successfully.")
    else:
        M_diag = linear_lumped_mass_matrix(X_meshgrid)
        np.savez(M_path, M_diag=M_diag, side_len=side_len, X=X_meshgrid, Y=Y_meshgrid)
    return M_diag
