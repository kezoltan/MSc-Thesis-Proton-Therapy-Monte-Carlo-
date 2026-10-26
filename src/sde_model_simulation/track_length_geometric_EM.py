
#July 2026
#Supporting function to run the geometric EM scheme for the track length model

from doseparams import *
from dosesetup import *

def n_V_log_energy_drift(L):
    """
    L = exp(Yn) must be calculated in the main code
    """
    return -stopping_power(L)*L**(-1) - 0.5*T_energy_stragg(L)*L**(-2)

def n_V_log_energy_diffusion(L):
    return np.sqrt(T_energy_stragg(L))*L**(-1)

def n_V_angular_drift(Omega, EPS_0=EPS_0):
    return -2*EPS_0*Omega #Make sure that you correct this

def n_V_angular_diffusion(Omega, EPS_0=EPS_0):
    return np.sqrt(2*EPS_0)*(np.eye(SPATIAL_DIM) - np.outer(Omega, Omega))

# Run one step of the scheme

def one_step_V_g_euler(h, dB1_onestep, dB2_3D_onestep, coeff, E_n, Omega_n, s_n, X_n, Y_n):
    """
    Runs the n+1th step of the geometric Euler scheme for the energy model 
    """
    other_noise = dB2_3D_onestep - Omega_n * np.dot(Omega_n, dB2_3D_onestep)

    #Geometric Scheme
    y = coeff * other_noise
    Omega_n1 = exponential_map_sphere(Omega_n, y)
    X_n1 = X_n + Omega_n * h #Omega n or n1?
    Y_n1 = Y_n + h * n_V_log_energy_drift(E_n) + n_V_log_energy_diffusion(E_n) * dB1_onestep
    E_n1 = exp(Y_n1)
    s_n1 = s_n + h

    #The outputs are now at step n+1
    return E_n1, Omega_n1, s_n1, X_n1, Y_n1 