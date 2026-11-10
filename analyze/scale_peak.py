#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy.optimize
from dataclasses import dataclass

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
# User-facing functions
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


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    L = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256], float)
    truth = peak_model(2).func(L, dict(x0=0.4, nu=1.0, a1=0.8, w1=1.0, a2=-0.5, w2=2.0))
    dX = 0.003 * truth
    X = truth + rng.normal(size=L.size) * dX
 
    x0, nu = guess_power_law(L, X)
 
    # 1) the two shapes you asked for
    m1 = peak_model(n_corr=1, x0=x0, nu=nu)
    m3 = peak_model(n_corr=3, x0=x0, nu=nu)
 
    # 2) free fit vs. fixing nu (2D Ising) and omega_1
    r = fit_scaling(L, X, dX, m1)
    print(r["params"], r["red_chi2"])
    r = fit_scaling(L, X, dX, m1, fixed={"nu": 1.0, "w1": 1.0})
    print(r["params"], r["red_chi2"])
 
    # 3) swap optimizer
    r = fit_scaling(L, X, dX, m1, method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
 
    # 4) any other function: just write func(L, p) and a Param list
    custom = ScalingModel(
        func=lambda L, p: p["x0"] * L ** (1 / p["nu"]) + p["c"],   # analytic background
        params=[Param("x0", x0), Param("nu", nu, (1e-3, None)), Param("c", 0.0)],
    )
    print(fit_scaling(L, X, dX, custom)["params"])

    import matplotlib.pyplot as plt
    plt.errorbar(L, X, dX, fmt="o")
    L_fit = np.linspace(L.min(), L.max(), 100)
    plt.plot(L_fit, m1.func(L_fit, r["params"]), label="fit")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()






