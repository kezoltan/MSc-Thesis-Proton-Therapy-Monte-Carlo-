# October 2026
# Run test to verify that params can be accepted

from pbt_mlmc.parameterisation import *
import warnings

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
            "dose_method must be either SF or SK for computation methods shape function or spatial kernel."
            )
    if sampling_type not in ['mc', 'mlmc']:
        raise ValueError(
            "sampling_type must be either mc or mlmc for Monte Carlo and Multilevel Monte Carlo sampling."
            )
    if SPATIAL_DIM not in [2, 3]:
        raise ValueError("spatial dimension must be either 2 or 3.")
    if theta>1 or theta<0 and sampling_type!='mc':
        raise ValueError("mlmc error weighting parameter theta must be in interval [0,1].")
    if EMIN > 1:
        warnings.warn(f"EMIN of {EMIN} may be too large.", UserWarning, stacklevel=2)
    if (min(Eps) < 0.2 or L_conv_test + MLMC_LEVEL_OFFSET >= 16) and sampling_type!='mc':
        warnings.warn("mlmc likely to be slow under chosen parameterisation.", RuntimeWarning, stacklevel=2)
    if E0 <= 0:
        raise ValueError("initial energy E0 must be positive.")
    if not(50 <= E0 <= 250):
        warnings.warn(f"E0 of {E0} is outside therapeutic energy range.", UserWarning, stacklevel=2)
    if N0 < 20000 and sampling_type!='mc':
        warnings.warn(f"N0 of {N0} may be too small to achieve suitable variable estimates.", stacklevel=2)

if __name__ == '__main__':
    verify_params()
