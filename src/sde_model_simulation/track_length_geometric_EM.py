
#July 2026
#Supporting function to run the geometric EM scheme for the track length model

from math import exp
from src.parameterisation import *
from src.setup_functions import *

def V_energy_loss_var_pul(E, kappa = KAPPA):
    """Calculates the energy loss variance per unit track length (pul).

    Parameters
    ----------
    E : float
        Energy, must be nonegative.
    kappa : float, optional
        Energy straggling strength. Defaults to ``KAPPA`` from src/parameterisation.

    Returns
    -------
    float
        Energy straggling strength evaluated at energy ``E``.
    """
    return kappa*E*stopping_power(E)

def log_energy_drift_coeff(L):
    """ Drift coefficient for the log transformed energy equation, using Ito's lemma.

    Parameters
    ----------
    L : float
        Logarithm of nonegative energy.

    Returns
    -------
    float
        The value of the drift coefficient.
    """
    return -stopping_power(L)*L**(-1) - 0.5*V_energy_loss_var_pul(L)*L**(-2)

def log_energy_diff_coeff(L):
    """ Diffusion coefficient for the log transformed energy equation, using Ito's lemma.

    Parameters
    ----------
    L : float
        Logarithm of nonegative energy.

    Returns
    -------
    float
        The value of the diffusion coefficient.
    """
    return np.sqrt(V_energy_loss_var_pul(L))*L**(-1)

# def n_V_angular_drift(Omega, EPS_0=EPS_0):
#     return -2*EPS_0*Omega #Make sure that you correct this

# def n_V_angular_diffusion(Omega, EPS_0=EPS_0):
#     return np.sqrt(2*EPS_0)*(np.eye(SPATIAL_DIM) - np.outer(Omega, Omega))

# Run one step of the scheme

def track_length_one_step_geoEM(h, dB1_onestep, dB2_spatial_dim_onestep, E_n, Omega_n, s_n, X_n, Y_n):
    """ Simulate one step of the geometric Euler scheme for the track length model.

    Parameters
    ----------
    h : float
        The step-size of the scheme, parameterised in track length.
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
    Y_n : float
        The logarithm of the E_n.

    Returns
    -------
    tuple
        The energetic, directional and positional state of the proton at the end of the step.
    """
    coeff = calculate_coeff(E_n, sde_model='track_length')
    other_noise = dB2_spatial_dim_onestep - Omega_n * np.dot(Omega_n, dB2_spatial_dim_onestep)

    #Geometric Scheme
    y = coeff * other_noise
    Omega_n1 = exponential_map_sphere(Omega_n, y)
    X_n1 = X_n + Omega_n * h #Omega n or n1?
    Y_n1 = Y_n + h * log_energy_drift_coeff(E_n) + log_energy_diff_coeff(E_n) * dB1_onestep
    E_n1 = exp(Y_n1)
    s_n1 = s_n + h

    #The outputs are now at step n+1
    return E_n1, Omega_n1, s_n1, X_n1, Y_n1
