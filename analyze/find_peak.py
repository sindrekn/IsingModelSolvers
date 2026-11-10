import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def block_jackknife(x, stat, num_bins=20, return_replicates=False):
    """
    Returns the jackknife estimate of the statistic and its error.

    x: data of size (n_samples, n_T) 
    stat: function of moment dict -> (n_T,) array 
    (Ex of stat: stat = lambda M: 1.0 - M[4] / (3.0 * M[2]**2), 
    the Binder cumulant)
    
    LLM-assisted
    ------------
    Tool: Claude (October 2026)
    Role: Generated the block jackknife function 
    for computing statistics and errors from data.
    """
    n = (x.shape[0] // num_bins) * num_bins
    b = n // num_bins
    x = x[:n]
    S = {k: (x**k).reshape(num_bins, b, -1).sum(axis=1) for k in (1, 2, 4)}
    full = {k: S[k].sum(axis=0) / n for k in S}
    loo  = {k: (S[k].sum(axis=0) - S[k]) / (n - b) for k in S}
    theta, theta_i = stat(full), stat(loo)
    err = np.sqrt((num_bins - 1) * np.var(theta_i, axis=0, ddof=0))
    if return_replicates:
        return theta, err, theta_i
    return theta, err

def read_temperatures(temp_file):
    """
    Reads temperatures from a file and returns them as a numpy array.
    """
    temps = []
    with open(temp_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                temps.append(float(parts[1]))
    return np.array(temps)

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

# ----------------------------------------------------------------------------
# Find data
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# Plot histograms of energy for each temperature
# ----------------------------------------------------------------------------

# fig, ax = plt.subplots(figsize=(10, 6))
# for temp in range(0, 19): 
#     ax.hist(e[:, temp], bins=50, alpha=0.8, label=f"T={temp}")

# ax.set_xlabel("Energy", fontsize=12)
# ax.set_ylabel("Frequency", fontsize=12)
# ax.set_title(f"Histogram of Energy for L={L}, si={si}", fontsize=14)
# ax.legend(title="Temperature", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# # plt.savefig(save_path)
# plt.show()
# plt.close()

# ----------------------------------------------------------------------------
# Single histogram reweighting
# ----------------------------------------------------------------------------

# # Select one reference temperature
# ref_idx = 10
# T0 = temps[ref_idx]

# # Select measurements from that temperature only
# E0 = e[:, ref_idx]
# m0 = np.abs(m[:, ref_idx])
# me0 = m0 * E0

# # Calculate the original mean magnetization
# m_mean = np.mean(np.abs(m), axis=0)
# e_mean = np.mean(e, axis=0)
# me_mean = np.mean(np.abs(m) * e, axis=0)

# # Reweight to all temperatures
# m_mean_reweighted = np.zeros(len(temps))
# e_mean_reweighted = np.zeros(len(temps))
# me_mean_reweighted = np.zeros(len(temps))

# for j, T in enumerate(temps):

#     beta0 = 1.0 / T0
#     beta = 1.0 / T

#     # Logarithm of the Boltzmann weights
#     log_weights = -(beta - beta0) * E0

#     # Numerical stabilization to avoid overflow
#     log_weights -= np.max(log_weights)
#     weights = np.exp(log_weights)

#     # Reweighted expectation value
#     m_mean_reweighted[j] = np.sum(m0 * weights) / np.sum(weights)
#     e_mean_reweighted[j] = np.sum(E0 * weights) / np.sum(weights)
#     me_mean_reweighted[j] = np.sum(me0 * weights) / np.sum(weights)

# dlnm = e_mean - me_mean / m_mean
# dlnm_reweighted = e_mean_reweighted - me_mean_reweighted / m_mean_reweighted

# plt.figure(figsize=(10, 6))
# plt.plot(temps, dlnm, label=r"Original dlnm/d$\beta$", marker='o')
# plt.plot(temps, dlnm_reweighted, label=r"Reweighted dlnm/d$\beta$", marker='x')

# plt.xlabel("Temperature")
# plt.ylabel(r"dlnm/d$\beta$")
# plt.title(fr"dlnm/d$\beta$ for L={L}, si={si}")
# plt.legend()
# plt.tight_layout()
# plt.savefig(base_path + f"/figs/run_1/scale_peak/dlnm_vs_temperature_hist-reweighted-simple_si-{si}_L{L}.pdf")
# plt.show()
# plt.close()

# ----------------------------------------------------------------------------
# Multi-histogram reweighting
# ----------------------------------------------------------------------------
import numpy as np
from scipy.special import logsumexp

def multi_histogram_reweighting(E, m, temps, L, num_bins=50, max_iterations=10000, tol=1e-10):
    """
    Perform multi-histogram reweighting to compute the density of states and reweighted observables.

    Parameters:
    - E: 2D array of energies (n_samples, n_temps)
    - m: 2D array of magnetizations (n_samples, n_temps)
    - temps: 1D array of temperatures
    - L: System size (linear dimension)
    - num_bins: Number of bins for the joint histogram
    - max_iterations: Maximum number of iterations for convergence
    - tol: Tolerance for convergence

    Returns:
    - logZ: Logarithm of the partition function for each temperature
    - dlnm_reweighted: Reweighted derivative of the logarithm of the magnetization
    """
    N = L**2
    M = np.abs(m * N)
    beta = 1.0 / temps

    H = np.zeros((len(temps), num_bins, num_bins))

    E_bins = np.linspace(np.min(E), np.max(E), num_bins + 1)
    M_bins = np.linspace(np.min(M), np.max(M), num_bins + 1)

    E_center = 0.5 * (E_bins[:-1] + E_bins[1:])
    M_center = 0.5 * (M_bins[:-1] + M_bins[1:])

    for i in range(len(temps)):
        H[i] = np.histogram2d(
            E[:, i],
            M[:, i],
            bins=[E_bins, M_bins]
        )[0]

    n = H.sum(axis=(1, 2))

    valid = H.sum(axis=0) > 0

    H_valid = H[:, valid]
    E_valid = np.broadcast_to(
        E_center[:, None], (num_bins, num_bins)
    )[valid]

    M_valid = np.broadcast_to(
        M_center[None, :], (num_bins, num_bins)
    )[valid]

    logH_total = np.log(H_valid.sum(axis=0))

    logZ = np.zeros(len(temps))
    logn = np.log(n)

    for iteration in range(max_iterations):

        old_logZ = logZ.copy()

        denominator = logsumexp(
            logn[:, None]
            - beta[:, None] * E_valid[None, :]
            - logZ[:, None],
            axis=0
        )

        logg = logH_total - denominator

        logZ = logsumexp(
            logg[None, :]
            - beta[:, None] * E_valid[None, :],
            axis=1
        )

        if np.max(np.abs(logZ - old_logZ)) < 1e-10:
            break

    print("Iterations:", iteration + 1)

    T_eval = np.linspace(np.min(temps), np.max(temps), 200)
    beta_eval = 1.0 / T_eval

    return logZ, logg, E_valid, M_valid, T_eval, beta_eval

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
    reweighted = np.zeros(len(beta_eval))

    for j, b in enumerate(beta_eval):

        log_weights = logg - b * E_valid

        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)

        reweighted[j] = np.sum(observable * weights)

    return reweighted

def find_peak(reweighted_observable, T_eval):
    """
    Find the peak of the reweighted observable and its corresponding temperature.

    Parameters:
    - reweighted_observable: Array of reweighted observable values
    - T_eval: Array of temperatures for evaluation

    Returns:
    - peak_temperature: Temperature at which the peak occurs
    - peak_value: Value of the observable at the peak
    """
    peak_index = np.argmax(reweighted_observable)
    peak_temperature = T_eval[peak_index]
    peak_value = reweighted_observable[peak_index]

    return peak_temperature, peak_value

logZ, logg, E_valid, M_valid, T_eval, beta_eval = multi_histogram_reweighting(E, m, temps, L)

N = L**2

m_valid = M_valid / N

m_reweighted = compute_reweighted_observables(m_valid, E_valid, logg, beta_eval)
e_reweighted = compute_reweighted_observables(E_valid, E_valid, logg, beta_eval)
me_reweighted = compute_reweighted_observables(m_valid * E_valid, E_valid, logg, beta_eval)

dlnm_reweighted = e_reweighted - me_reweighted / m_reweighted

dlnm, dlnm_error, dlnm_replicates = block_jackknife_dlnm(m, E, power=1, num_bins=20)

# ----------------------------------------------------------------------------
# Find the peak of dlnm/dBeta and its error
# ----------------------------------------------------------------------------

peak_temperature, peak_dlnm = find_peak(dlnm_reweighted, T_eval)

print(f"Peak temperature: {peak_temperature:.5f}")
print(f"Peak dlnm: {peak_dlnm:.5f}")

plt.figure(figsize=(10, 6))

plt.errorbar(
    temps,
    dlnm,
    yerr=dlnm_error,
    fmt='o',
    label="Original magnetization"
)

plt.plot(
    T_eval,
    dlnm_reweighted,
    '-',
    label="Multi-histogram reweighting"
)

plt.plot(
    peak_temperature,
    peak_dlnm,
    'ro',
    label=f"Peak at T={peak_temperature:.5f}"
)

plt.xlabel("Temperature")
plt.ylabel(r"$\langle |m| \rangle$")
plt.legend()
plt.tight_layout()
plt.savefig(base_path + f"/figs/run_1/scale_peak/dlnm_vs_temperature_multi-hist-reweighted_si-{si}_L{L}.pdf")
plt.show()



# ----------------------------------------------------------------------------
# 3D plot of the joint histogram for a specific temperature
# ----------------------------------------------------------------------------

# # 3d plot of the joint histogram for a specific temperature
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the joint histogram for a specific temperature (e.g., the first one)
# i = 0
# x, y = np.meshgrid(E_bins[:-1], M_bins[:-1], indexing='ij')
# ax.plot_surface(x, y, H[i], cmap='viridis', alpha=0.7)

# ax.set_xlabel("Energy")
# ax.set_ylabel("Magnetization")
# ax.set_zlabel("Frequency")
# ax.set_title(f"Joint Histogram for T={temps[i]}")

# plt.savefig(base_path + f"/figs/run_1/scale_peak/joint_histogram_si-{si}_L{L}.pdf")
# plt.show()
# plt.close()


