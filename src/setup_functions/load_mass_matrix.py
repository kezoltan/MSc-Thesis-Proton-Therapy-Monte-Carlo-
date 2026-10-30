from src.parameterisation import *

# added defaults so tests are easier to run
def load_mass_matrix(
    file_path=file_path,
    spatial_dim=SPATIAL_DIM,
    dose_shape=dose_shape,
    X_meshgrid=X_meshgrid,
    Y_meshgrid=Y_meshgrid,
    side_len=side_len,
):
    """Load a cached lumped mass matrix or calculate and save it.

    Optional arguments allow callers to provide a mesh and cache location.
    When omitted, the corresponding values from ``doseparams`` are used.
    """
    M_path = os.path.join(file_path, f"{spatial_dim}D_{dose_shape}_lumped_mass_matrix.npz")
    if os.path.exists(M_path):
        mass_data = np.load(M_path)
        M_diag = mass_data["M_diag"]
        mass_shape = M_diag.shape[0]
        if mass_shape != dose_shape:
            raise ValueError(f"Lumped mass matrix shape must equal {dose_shape}, is {mass_shape}.")
        print("Mass matrix loaded successfully.")
    else:
        M_diag = linear_lumped_mass_matrix(X_meshgrid, side_len, spatial_dim=spatial_dim)
        np.savez(M_path, M_diag=M_diag, side_len=side_len, X=X_meshgrid, Y=Y_meshgrid)
    return M_diag
