import numpy as np
from doseparams import dose_shape, MLMC_LEVEL_OFFSET

def mlmcv(mlmc_parallel_l, N0, eps, Lmin, Lmax, alpha0, beta0, gamma0, *args):
    """
    Multi-level Monte Carlo estimation.
    Dynamically vectorized to handle the Primal value and an arbitrary 
    number of sensitivities simultaneously.
    """
    # Check input parameters
    if Lmin < 2:
        raise ValueError("error: needs Lmin >= 2")
    if Lmax < Lmin:
        raise ValueError("error: needs Lmax >= Lmin")
    if N0 <= 0 or eps <= 0:
        raise ValueError("error: needs N0 > 0, eps > 0")

    # Initialization
    alpha = max(0.0, alpha0)
    beta  = max(0.0, beta0)
    gamma = max(0.0, gamma0)

    theta = 0.25
    L = Lmin

    # Arrays directly mapping to levels l = 0, 1, ..., L
    Nl = np.zeros(L + 1, dtype=int)
    costl = np.zeros(L + 1, dtype=float)
    dNl = np.full(L + 1, int(N0), dtype=int)
    
    suml = None
    num_q = dose_shape

    while np.sum(dNl) > 0:
        # Update sample sums
        for l in range(L + 1):
            if dNl[l] > 0:

                print(f"START level {l}: running {dNl[l]} simulations", flush=True)
                sums, cost = mlmc_parallel_l(l, dNl[l], *args)
                if l == 0:
                    print("level 0 sums shape:", sums.shape)
                    print("level 0 max sum(diff):", np.max(np.abs(sums[:, 0])))
                    print("level 0 max sum(diff^2):", np.max(sums[:, 1]))
                print(f"DONE level {l}: completed {dNl[l]} simulations", flush=True)

                #check the dose shape 
                if sums.shape != (dose_shape, 6):
                    raise ValueError(f"mlmc sums vs. dose shape mismatch: expected {(dose_shape, 6)}, got {sums.shape}.")

                Nl[l] += dNl[l]
                costl[l] += cost
                
                # Late initialization: detect number of tracked quantities on the first pass
                if suml is None:
                    suml = np.zeros((2 * num_q, L + 1), dtype=float)
                
                # Dynamically accumulate first and second moments for all parameters
                for k in range(num_q):
                    suml[2 * k, l]     += sums[k,0]      # diff accumulation
                    suml[2 * k + 1, l] += sums[k,1]      # diff**2 accumulation

        # Reshape suml to compute moments across all quantities simultaneously
        suml_reshaped = suml.reshape(num_q, 2, L + 1)
        ml = np.abs(suml_reshaped[:, 0, :] / Nl)
        Vl = np.maximum(0.0, suml_reshaped[:, 1, :] / Nl - ml**2)
        
        # MAXIMIZE across Primal and all Greeks to guarantee precision criteria
        ml_max = np.max(ml, axis=0)
        Vl_max = np.max(Vl, axis=0)
        
        Cl = costl / Nl

        # Fix to cope with possible zero values for extrapolated means/variances
        for l in range(2, L + 1):
            ml_max[l] = max(ml_max[l], 0.5 * ml_max[l-1] / (2.0**alpha))
            Vl_max[l] = max(Vl_max[l], 0.5 * Vl_max[l-1] / (2.0**beta))

        # Use linear regression on the maximum bounds to estimate parameters
        A = np.column_stack((np.arange(1, L + 1), np.ones(L)))

        if alpha0 <= 0:
            y = np.log2(ml_max[1:])
            x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            alpha = max(0.5, -x[0])

        if beta0 <= 0:
            y = np.log2(Vl_max[1:])
            x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            beta = max(0.5, -x[0])

        if gamma0 <= 0:
            y = np.log2(Cl[1:])
            x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            gamma = max(0.5, x[0])

        # Set optimal number of additional samples using worst-case variance bounds
        Ns = np.ceil(np.sqrt(Vl_max / Cl) * np.sum(np.sqrt(Vl_max * Cl)) / ((1 - theta) * eps**2))

        print("\n--- NEW SAMPLE ALLOCATION ---", flush=True)
        print("Nl     =", Nl, flush=True)
        print("ml_max =", ml_max, flush=True)
        print("Vl_max =", Vl_max, flush=True)
        print("Cl     =", Cl, flush=True)
        print("Ns     =", Ns, flush=True)


        dNl = np.maximum(0, Ns - Nl).astype(int)

        # Weak convergence verification
        if np.sum(dNl > 0.01 * Nl) == 0:
            rng = np.arange(0, min(2, L - 1) + 1)
            rem = np.max(ml_max[L - rng] / (2.0**(rng * alpha))) / (2.0**alpha - 1.0)

            if rem > np.sqrt(theta) * eps:
                if L == Lmax:
                    print("*** failed to achieve weak convergence ***")
                else:
                    L += 1
                    
                    # Expand arrays dynamically for the new level
                    Vl_max = np.append(Vl_max, Vl_max[-1] / (2.0**beta))
                    Cl = np.append(Cl, Cl[-1] * (2.0**gamma))
                    Nl = np.append(Nl, 0)
                    suml = np.column_stack((suml, np.zeros(2 * num_q)))
                    costl = np.append(costl, 0.0)

                    # Recompute targets
                    Ns = np.ceil(np.sqrt(Vl_max / Cl) * np.sum(np.sqrt(Vl_max * Cl)) / ((1 - theta) * eps**2))
                    dNl = np.maximum(0, Ns - Nl).astype(int)

                    #Added a check! Because it was taking too long 
                    print("\nMLMC allocation:")
                    print("eps =", eps)
                    print("current Nl =", Nl)
                    print("target Ns =", Ns.astype(int))
                    print("additional dNl =", dNl)

    # Evaluate final multilevel estimators for all tracked outputs
    P_estimates = np.sum(suml[::2, :] / Nl, axis=1)
    
    # Return estimates as a flattened tuple followed by Nl and Cl for clean unpacking
    return tuple(P_estimates) + (Nl, Cl)