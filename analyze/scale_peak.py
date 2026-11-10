#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy.optimize
from dataclasses import dataclass
from scipy.special import logsumexp

# ----------------------------------------------------------------------------
# Common: dataclass declarations for parameters and models
# ----------------------------------------------------------------------------

@dataclass
class Param: 
    name: str
    p0: float
    bounds: tuple = (None, None)

@dataclass
class ScalingModel:
    """
    func: callable      # func(L, p: dict) -> array
    params: list        # list[Param]
    """
    func: callable      
    params: list        

@dataclass
class CollapseModel:
    """
    x_func(L, rho, p)    -> scaled x      (2-D, shape (L.size, rho.size))
    y_func(L, a, da, p)  -> (y, dy)       scaled data and errors
    params               -> list[Param]
    """
    x_func: callable
    y_func: callable
    params: list

# ----------------------------------------------------------------------------
# Helper functions, not supposed to be called directly
# ----------------------------------------------------------------------------

def _split(params, fixed): 
    """
    Split the parameters into free and fixed parameters.
    """
    fixed = dict(fixed or {})
    unknown = set(fixed) - {p.name for p in params}
    if unknown:
        raise ValueError(f"Unknown fixed parameters: {unknown}")
    return [p for p in params if p.name not in fixed], fixed

def _assemble(theta, free, fixed):
    """
    Assemble the parameters into a dictionary.
    """
    p = dict(fixed)
    p.update({q.name: v for q, v in zip(free, theta)})
    return p

# ----------------------------------------------------------------------------
# Helper functions for reading data
# ----------------------------------------------------------------------------

def read_temperatures(temp_file):
    """
    Reads temperatures from a file and returns them as a numpy array.
    """
    temps = []
    with open(temp_file, "r") as f:
        next(f) 
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                temps.append(float(parts[1]))
    return np.array(temps)

# ----------------------------------------------------------------------------
# Peak finder using WHAM, reweighting and jackknife resampling
# ----------------------------------------------------------------------------

def _wham_energy(H_E, E, beta, logZ0=None, max_iterations=1000, tol=1e-6):
    """WHAM on the energy marginal only.
 
    The WHAM denominator depends on E alone, so iterating over (E, M) cells is
    redundant: solve for logZ on the n_E energy bins, then recover g(E, M) once.

    Parameters: 
    - H_E: (n_temps, n_E) counts
    - E: (n_E,) bin centers (all bins non-empty)
    - beta: (n_temps,) inverse temperatures
    - logZ0: initial guess for the partition function (if None, use zeros)
    - max_iterations: maximum number of iterations
    - tol: convergence tolerance
    
    Returns:
    - logZ: Logarithm of the partition function for each temperature
    - D(E): Logarithm of the density of states for each energy bin
    - log H_tot(E): Logarithm of the total histogram for each energy bin

    LLM-assisted
    ------------
    Tool: Claude (October 2026)
    Role: Created the _wham_energy function to perform WHAM on the energy 
    marginal only, based on my orginal implementation.
    """
    logn = np.log(H_E.sum(axis=1))
    logH_tot = np.log(H_E.sum(axis=0))
    A = -beta[:, None] * E[None, :]
 
    logZ = np.zeros(len(beta)) if logZ0 is None else np.asarray(logZ0, float).copy()
    for _ in range(max_iterations):
        D = logsumexp(A + (logn - logZ)[:, None], axis=0)
        new = logsumexp(A + (logH_tot - D)[None, :], axis=1)
        new -= new[0]
        done = np.max(np.abs(new - logZ)) < tol
        logZ = new
        if done:
            break
    D = logsumexp(A + (logn - logZ)[:, None], axis=0)
    return logZ, D, logH_tot
 
def calculate_dlnm_peak(H, E_bins, M_bins, temps, N, n_eval=1000, i=1,
                        logZ0=None, return_logZ=False):
    """
    Calculate the peak of the derivative of the logarithm of the magnetization 
    with respect to beta.

    Parameters:
    - H: 3D array of histograms (n_temps, n_E_bins, n_M_bins)
    - E_bins: 1D array of energy bin edges
    - M_bins: 1D array of magnetization bin edges
    - temps: 1D array of temperatures
    - N: System size (number of spins)
    - n_eval: Number of evaluation points for reweighting
    - i: order of the derivative (default is 1 for first derivative)
    - logZ0: initial guess for the partition function (if None, use zeros)
    - return_logZ: if True, also return the logarithm of the partition function

    Returns:
    - T_eval[k]: Temperature at which the peak occurs
    - dlnm[k]: Value of the peak
    - logZ (optional): Logarithm of the partition function for each temperature
    """
    beta = 1.0 / temps
    E_c = 0.5 * (E_bins[:-1] + E_bins[1:])
    m_c = 0.5 * (M_bins[:-1] + M_bins[1:]) / N
 
    H_tot = H.sum(axis=0)
    valid = H_tot.sum(axis=1) > 0
    E = E_c[valid]
    H_tot = H_tot[valid]
 
    logZ, D, _ = _wham_energy(H[:, valid].sum(axis=2), E, beta, logZ0)
 
    logA0 = np.log(H_tot.sum(axis=1)) - D
    logA1 = np.log(H_tot @ m_c**i) - D
 
    T_eval = np.linspace(temps.min(), temps.max(), n_eval)
    be = (1.0 / T_eval)[:, None]
 
    def mean_E(logA):
        lw = logA[None, :] - be * E[None, :]
        lw -= lw.max(axis=1, keepdims=True)
        w = np.exp(lw)
        return (w @ E) / w.sum(axis=1)
 
    dlnm = mean_E(logA0) - mean_E(logA1)
    k = np.argmax(dlnm)
    out = (T_eval[k], dlnm[k])
    return (*out, logZ) if return_logZ else out

def jackknife_multi_histogram(func, E, m, temps, N, num_bins=50, block_size=100,
                              n_eval=1000, i=1, verbose=False):
    """
    Perform a jackknife resampling on a multi-histogram dataset.

    Parameters:
    - func: Function to compute the observable of interest
    - E: 1D array of energy values
    - m: 1D array of magnetization values
    - temps: 1D array of temperatures
    - N: System size (number of spins)
    - num_bins: Number of bins for the histograms
    - block_size: Size of each block for the jackknife resampling
    - n_eval: Number of evaluation points for reweighting
    - i: order of the derivative (default is 1 for first derivative)
    - verbose: If True, print progress information

    Returns: 
    - peak_T: Temperature at which the peak occurs
    - peak_value: Value of the peak
    - T_err: Error in the peak temperature from jackknife resampling
    - peak_err: Error in the peak value from jackknife resampling
    """
    M = np.abs(m * N)
    n_samples, n_temps = E.shape
    n_blocks = n_samples // block_size
    if n_blocks < 2:
        raise ValueError("Not enough samples for jackknife resampling.")
    nb = num_bins
 
    E_bins = np.linspace(E.min(), E.max(), nb + 1)
    M_bins = np.linspace(M.min(), M.max(), nb + 1)

    iE = np.clip(np.digitize(E, E_bins) - 1, 0, nb - 1)
    iM = np.clip(np.digitize(M, M_bins) - 1, 0, nb - 1)
 
    t = np.arange(n_temps)[None, :]
    H = np.bincount(((t * nb + iE) * nb + iM).ravel(),
                    minlength=n_temps * nb * nb).reshape(n_temps, nb, nb)
 
    peak_T, peak_value, logZ_full = func(H, E_bins, M_bins, temps, N,
                                         n_eval=n_eval, i=i, return_logZ=True)
 
    n_used = n_blocks * block_size
    blk = np.arange(n_used) // block_size
    jk_T = np.zeros((n_temps, n_blocks))
    jk_peak = np.zeros((n_temps, n_blocks))
 
    for r in range(n_temps):
        if verbose:
            print(f"{r+1} of {n_temps} temperatures")

        flat = (blk * nb + iE[:n_used, r]) * nb + iM[:n_used, r]
        Hb = np.bincount(flat, minlength=n_blocks * nb * nb).reshape(n_blocks, nb, nb)
        H_r = H[r].copy()
        for b in range(n_blocks):
            H[r] = H_r - Hb[b]
            jk_T[r, b], jk_peak[r, b] = func(H, E_bins, M_bins, temps, N,
                                             n_eval=n_eval, i=i, logZ0=logZ_full)
        H[r] = H_r
 
    f = (n_blocks - 1) / n_blocks
    T_err = np.sqrt(f * np.sum((jk_T - jk_T.mean(axis=1, keepdims=True))**2))
    peak_err = np.sqrt(f * np.sum((jk_peak - jk_peak.mean(axis=1, keepdims=True))**2))
    return peak_T, peak_value, T_err, peak_err

# ----------------------------------------------------------------------------
# Callable functions for density of states and reweighting
# ----------------------------------------------------------------------------

def density_of_states(H, E_bins, M_bins, temps, max_iterations=1000, tol=1e-6,
                      logZ0=None):
    """
    Compute the density of states from the histogram H(E, M) at 
    different temperatures.

    Parameters:
    - H: 3D array of histograms (n_temps, n_E_bins, n_M_bins)
    - E_bins: 1D array of energy bin edges
    - M_bins: 1D array of magnetization bin edges
    - temps: 1D array of temperatures
    - max_iterations: Maximum number of iterations for convergence
    - tol: Tolerance for convergence

    Returns:
    - logZ: Logarithm of the partition function for each temperature
    - logg: Logarithm of the density of states
    - E_valid: Valid energy values
    - M_valid: Valid magnetization values

    LLM-assisted
    ------------
    Tool: Claude (October 2026)
    Role: Optimized the density_of_states function to utilize the wham 
    energy function. 
    """
    beta = 1.0 / temps
    E_c = 0.5 * (E_bins[:-1] + E_bins[1:])
    M_c = 0.5 * (M_bins[:-1] + M_bins[1:])
 
    H_tot = H.sum(axis=0)
    valid_E = H_tot.sum(axis=1) > 0
    logZ, D, _ = _wham_energy(H[:, valid_E].sum(axis=2), E_c[valid_E], beta,
                              logZ0, max_iterations, tol)
 
    D_full = np.full(len(E_c), np.nan)
    D_full[valid_E] = D
    valid = H_tot > 0
    iE, iM = np.nonzero(valid)
    logg = np.log(H_tot[valid]) - D_full[iE]
    return logZ, logg, E_c[iE], M_c[iM]


def compute_reweighted_observables(observable, E_valid, logg, beta_eval):
    """
    Compute reweighted observables using the density of states and partition functions.

    Parameters:
    - observable: Array of observable values
    - E_valid: Array of valid energy bins
    - logg: Logarithm of the density of states
    - beta_eval: Array of beta values for evaluation

    Returns:
    - reweighted: Reweighted observable
    """
    logw = logg[None, :] - beta_eval[:, None] * E_valid[None, :]
    logw -= logsumexp(logw, axis=1, keepdims=True)
    return np.exp(logw) @ observable

# ----------------------------------------------------------------------------
# Peak scaling model and fitting routines
# ----------------------------------------------------------------------------

def peak_model(n_corr=1, x0=1.0, nu=1.0, a0=0.1, w0=1.0):
    """
    X_max(L) = x0 L^(1/nu) * (1 + sum_{i=1..n_corr} a_i L^(-w_i))
    n_corr=0 gives the pure power law.
    """

    params = [Param("x0", x0), Param("nu", nu, (1e-3, None))]
    for i in range(1, n_corr + 1):
        params.append(Param(f"a{i}", float(a0)))
        params.append(Param(f"w{i}", float(w0), (1e-3, None)))

    def func(L, p): 
        L = np.add(L, 0.0)
        corr = 1.0
        for i in range(1, n_corr + 1): 
            corr += p[f"a{i}"] * L**(-p[f"w{i}"])
        return p["x0"] * L**(1/p["nu"]) * corr

    return ScalingModel(func, params)

def guess_power_law(L, X): 
    """ 
    Initial guess for a power law fit of the form X(L) = x0 L^(1/nu)
    We use a linear fit of log(X) vs log(L) to get the initial guess for x0 and nu.
    """
    coeffs = np.polyfit(np.log(L), np.log(np.abs(X)), 1)
    return np.exp(coeffs[1]) * np.sign(X[0]), 1 / coeffs[0]

def fit_scaling(L, X , dX, model, fixed=None, method="least_squares", options=None):
    """
    Weighted fit of X(L) to 'model'
    """
    L, X, dX = np.asanyarray(L), np.asanyarray(X), np.asanyarray(dX)
    free, fixed = _split(model.params, fixed)
    theta0 = np.array([p.p0 for p in free], float)
    lo = np.array([-np.inf if q.bounds[0] is None else q.bounds[0] for q in free])
    hi = np.array([np.inf if q.bounds[1] is None else q.bounds[1] for q in free])

    def resid(theta): 
        p = _assemble(theta, free, fixed)
        return (X - model.func(L, p)) / dX

    if method == "least_squares":
        res = scipy.optimize.least_squares(
            fun=resid,
            x0=theta0,
            bounds=(lo, hi),
            **(options or {})
        )
        theta, chi = res.x, 2*res.cost
        try: 
            cov = np.linalg.inv(res.jac.T @ res.jac)
            err = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            cov, err = None, np.full(len(theta), np.nan)
    else: 
        res = scipy.optimize.minimize(
            fun=lambda theta: np.sum(resid(theta)**2),
            x0=theta0,
            bounds = list(zip([None if np.isinf(v) else v for v in lo], 
                              [None if np.isinf(v) else v for v in hi])),
                              options=options or {})
        theta, chi = res.x, res.fun
        cov, err = None, np.full(len(theta), np.nan)

    dof = len(L) - len(free)
    return dict(
        params=_assemble(theta, free, fixed),
        errors={q.name: e for q, e in zip(free, err)},
        cov=cov, 
        chi=chi,
        dof=dof,
        red_chi2=chi / dof if dof > 0 else np.nan,
        success=res.success,
        message=res.message,
        raw=res,
    )

def block_jackknife_dlnm(m, E, power=1, num_bins=20, sign=-1):
    """
    d ln<|m|^i>/dbeta = sign * ( <|m|^i E>/<|m|^i> - <E> )

    m, E: (n_samples, n_T) arrays (E = total energy)
    sign = -1 for Boltzmann weight exp(-beta*E); use +1 if your E
    is defined with the opposite sign (weight exp(+K*E)).
    Same block-jackknife scheme as block_jackknife.

    LLM-assisted
    ------------
    Tool: Claude (October 2026)
    Role: Generated the block_jackknife_dlnm function for computing
    the derivative of the logarithm of the magnetization with respect to beta.
    """
    n = (m.shape[0] // num_bins) * num_bins
    b = n // num_bins
    mi = np.abs(m[:n]) ** power
    E = E[:n]

    def binsum(a):
        return a.reshape(num_bins, b, -1).sum(axis=1)

    S = {"m": binsum(mi), "mE": binsum(mi * E), "E": binsum(E)}
    full = {k: v.sum(axis=0) / n for k, v in S.items()}
    loo  = {k: (v.sum(axis=0) - v) / (n - b) for k, v in S.items()}

    stat = lambda M: sign * (M["mE"] / M["m"] - M["E"])
    theta, theta_i = stat(full), stat(loo)
    err = np.sqrt((num_bins - 1) * np.var(theta_i, axis=0, ddof=0))
    return theta, err, theta_i

if __name__ == "__main__":
    import pandas as pd
    L = 256
    si=0

    base_path = "/home/sindrekampennesheim/Documents/PhD/IsingModel/Output/Olivia-run1"

    energy = base_path + f"/data/run_1/energy_lb_sigma-{si}_L{L}.csv"
    temp = base_path + f"/data/run_1/temp_si-{si}.txt"
    magnetization = base_path + f"/data/run_1/magnetization_lb_sigma-{si}_L{L}.csv"
    save_path = base_path + f"/figs/run_1/scale_peak/energy_hist_si-{si}_L{L}.pdf"

    E = pd.read_csv(energy, index_col=0).to_numpy()[:, 1:]
    m = pd.read_csv(magnetization, index_col=0).to_numpy()[:, 1:]
    temps = read_temperatures(temp)
    temps = temps[1:]

    # for block_size in [50, 100, 200, 400, 800]:
    #     print(f"Block size: {block_size}")
    #     peak_T, peak_value, T_err, peak_err = jackknife_multi_histogram(
    #         calculate_dlnm_peak, E, m, temps, L, num_bins=50, block_size=block_size, 
    #         n_eval=1000, i=1
    #     )

    #     print(f"Peak temperature: {peak_T:.8f} ± {T_err:.8f}")
    #     print(f"Peak value: {peak_value:.8f} ± {peak_err:.8f}")

    dlnm, dlnm_error, dlnm_replicates = block_jackknife_dlnm(m, E, power=1, num_bins=20)

    M = np.abs(m * L)
    n_samples, n_temps = E.shape
    nb = 50

    E_bins = np.linspace(E.min(), E.max(), nb + 1)
    M_bins = np.linspace(M.min(), M.max(), nb + 1)
    # same binning as np.histogram2d (last edge closed), without per-block calls
    iE = np.clip(np.digitize(E, E_bins) - 1, 0, nb - 1)
    iM = np.clip(np.digitize(M, M_bins) - 1, 0, nb - 1)
 
    t = np.arange(n_temps)[None, :]
    H = np.bincount(((t * nb + iE) * nb + iM).ravel(),
                    minlength=n_temps * nb * nb).reshape(n_temps, nb, nb)

    print(f"H size : {H.shape}")
    print(f"E_bins size : {E_bins.shape}")

    logZ, logg, E_valid, M_valid = density_of_states(H, E_bins, M_bins, temps)

    m_valid = M_valid / L
    T_eval = np.linspace(temps.min(), temps.max(), 1000)
    beta_eval = 1.0 / T_eval

    e_reweighted = compute_reweighted_observables(E_valid, E_valid, logg, beta_eval)
    m_reweighted = compute_reweighted_observables(m_valid, E_valid, logg, beta_eval)
    me_reweighted = compute_reweighted_observables(E_valid * m_valid, E_valid, logg, beta_eval)
    
    dlnm_reweighted = e_reweighted - me_reweighted / m_reweighted

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))

    plt.errorbar(
        temps, 
        dlnm, 
        yerr=dlnm_error, 
        label="Original", 
        fmt="o"
    )

    plt.plot(
        T_eval, 
        dlnm_reweighted,
        '-', 
        label="Reweighted"
    )
    
    plt.errorbar(
        peak_T, 
        peak_value, 
        xerr=T_err, 
        yerr=peak_err, 
        fmt="ro", 
        label=f"Peak at T={peak_T:.3f} ± {T_err:.3f}"
    )

    plt.xlabel("Temperature")
    plt.ylabel("d ln<|m|>/d beta")
    plt.title("Derivative of the logarithm of magnetization vs Temperature")
    plt.legend()
    plt.show()










    # rng = np.random.default_rng(1)
    # L = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256], float)
    # truth = peak_model(2).func(L, dict(x0=0.4, nu=1.0, a1=0.8, w1=1.0, a2=-0.5, w2=2.0))
    # dX = 0.003 * truth
    # X = truth + rng.normal(size=L.size) * dX
 
    # x0, nu = guess_power_law(L, X)
 
    # # 1) the two shapes you asked for
    # m1 = peak_model(n_corr=1, x0=x0, nu=nu)
    # m3 = peak_model(n_corr=3, x0=x0, nu=nu)
 
    # # 2) free fit vs. fixing nu (2D Ising) and omega_1
    # r = fit_scaling(L, X, dX, m1)
    # print(r["params"], r["red_chi2"])
    # r = fit_scaling(L, X, dX, m1, fixed={"nu": 1.0, "w1": 1.0})
    # print(r["params"], r["red_chi2"])
 
    # # 3) swap optimizer
    # r = fit_scaling(L, X, dX, m1, method="Nelder-Mead",
    #                 options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
 
    # # 4) any other function: just write func(L, p) and a Param list
    # custom = ScalingModel(
    #     func=lambda L, p: p["x0"] * L ** (1 / p["nu"]) + p["c"],   # analytic background
    #     params=[Param("x0", x0), Param("nu", nu, (1e-3, None)), Param("c", 0.0)],
    # )
    # print(fit_scaling(L, X, dX, custom)["params"])

    # import matplotlib.pyplot as plt
    # plt.errorbar(L, X, dX, fmt="o")
    # L_fit = np.linspace(L.min(), L.max(), 100)
    # plt.plot(L_fit, m1.func(L_fit, r["params"]), label="fit")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()