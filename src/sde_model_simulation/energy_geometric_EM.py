# July 2026
# Supporting function to run the geometric EM scheme for the track length model

from doseparams import *
from dosesetup import *
from math import log


def track_length_diff_coeff(E, a=ALPHA, p=p):
    return a * p * np.sqrt(KAPPA) * E ** (p - 0.5)


# def n_KZ_angular_diffusion(E, Omega, EPS_0=EPS_0, a=ALPHA, p=p):
#     return np.sqrt(2*EPS_0 * a*p)*E**(p/2 - 0.5) * (np.eye(SPATIAL_DIM) - np.outer(Omega, Omega))

# Run one step of the scheme

def energy_one_step_geoEM(h, dB1_onestep, dB2_spatial_dim_onestep, E_n, Omega_n, s_n, X_n):
    """ Simulate one step of the geometric Euler scheme for the energy model.

    Parameters
    ----------
    h : float
        The step-size of the scheme, parameterised in energy (loss). 
    dB1_onestep : float
         
    dB2_spatial_dim_onestep : numpy.ndarray, shape (d,)
         
    E_n : float
        The energy of the proton at the start of the step, must be nonegative.
    Omega_n : numpy.ndarray, shape (d,)
        The direction of the proton at the start of the step.  
    s_n : float
        The accumulated track length of the proton track at the start of the step. 
    X_n : numpy.ndarray, shape (d,)
        The spatial position of the proton at the start of the step. 

    Returns
    -------
    tuple
        The energetic, directional and positional state of the proton at the end of the step. 
    """
    #Set up
    S_inv = reciprocal_stopping_power(E_n) 
    omega_scheme_coeff = calculate_coeff(E_n, sde_model='energy') # coeff_prefactor*E_n**(p/2 - 0.5)
    other_noise = dB2_spatial_dim_onestep - Omega_n * np.dot(Omega_n, dB2_spatial_dim_onestep)

    #Geometric Scheme
    y = omega_scheme_coeff * other_noise
    Omega_n1 = exponential_map_sphere(Omega_n, y)
    s_n1 = s_n - S_inv * h + track_length_diff_coeff(E_n) * dB1_onestep
    path_travelled = s_n1 - s_n
    X_n1 = X_n + Omega_n * (path_travelled)  # Omega n or n1?
    E_n1 = E_n + h

    # The outputs are now at step n+1
    return E_n1, Omega_n1, s_n1, X_n1


# -----------------GAUSSIAN BEAM SETUP-----------------


def sample_gauss_beam(rng, s0=0):
    s = float(s0)
    X = sample_initial_position(meshgrid, rng, side_len)
    Omega = Omega0
    E = initial_energy_spread()
    Y = log(E)
    return s, X, Omega, E, Y


def choose_dE_ds(level: int, T=E0 - EMIN, mu=1, base=base):
    """
    This is chosen such that dE aligns with a mlmc stepsize, purely for convenience.
    And dE and ds align wrt the stopping power.
    And so that the values align relatively closely to those used in Chronholm+Pryer 2026

    mu is in [0,1], chosen freely.
    level is the geometric stepsize level, 9 for MC standard (chosen in main text)
    """
    dE = -T * base ** (-level)
    ds = -dE / (mu * stopping_power(E0))
    return dE, ds


def energy_model_initialise(dE, E_start, a=ALPHA, p=p):
    """
    l is the mlmc level with base=base
    """
    # Stepsize calibration (important if Gaussian energy spread is on, E_start might not be E0)
    h = dE
    if energy_spread:
        base_h = dE
        n = ceil((E_start - EMIN) / abs(base_h))
        Es = np.linspace(E_start, EMIN, n + 1)
        h = Es[1] - Es[0]
        num_steps = len(Es) - 1  # Minus 1 because we don't need to repeat the IC
    else:
        if E_start != E0:
            raise ValueError("energy spread is zero but E_start != E0, energy model.")
        num_steps = int(T // (-h))
        if abs(T % (-h)) > 1e-6:
            raise ValueError(
              f"energy step size is not geometric, h: {h}, base: {base}, T: {E0-EMIN}, num_steps: {num_steps}"
            ) 
    return h, num_steps


def full_energy_model_setup(level, rng):
    s, X, Omega, E, Y = sample_gauss_beam(rng)
    dE, _ = choose_dE_ds(level)
    h, num_steps = energy_model_initialise(dE, E)
    setup_tuple = s, X, Omega, E, Y, h, num_steps
    dZ1 = rng.standard_normal(num_steps) 
    dZ2_spatial_dim = rng.standard_normal((num_steps, SPATIAL_DIM))
    return setup_tuple, (dZ1, dZ2_spatial_dim)
