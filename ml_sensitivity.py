"""Which parameters actually matter?  -- a sensitivity analysis for the DNS fit.

Plain-English idea:
  1. Randomly sample many combinations of the fit parameters (a space-filling
     Latin-hypercube sample, so the whole box is covered evenly).
  2. For each combination, run the population simulation and measure how well it
     matches the observations, via three scores:
         total_error  - the full fit error (fit.error), lower = better
         ecc_ks       - mismatch of the eccentricity distribution (KS, 0=perfect)
         period_ks    - mismatch of the period distribution     (KS, 0=perfect)
  3. Fit a RandomForest from {parameters -> score} and read off its feature
     importances.  An importance of e.g. 60% means that parameter explains ~60%
     of the variation in that score across the sampled box -- i.e. it is the
     dominant knob for that aspect of the fit.

Everything respects fit.DNS_SUBSET, so the scores describe only the period
subset the fit is currently targeting ("all" / "low" / "high").

Run:  python ml_sensitivity.py
"""
import numpy as np
from scipy.stats import qmc, ks_2samp
from sklearn.ensemble import RandomForestRegressor

import fit

# --- the parameters under study come straight from the fit (single source of truth) ---
PARAM_NAMES = [p[0] for p in fit.PARAMS]
BOUND_LO    = np.array([p[1] for p in fit.PARAMS])
BOUND_HI    = np.array([p[2] for p in fit.PARAMS])
N_PARAMS    = len(fit.PARAMS)

N_SAMPLES = 200    # parameter combinations to try (more = smoother importances, slower)
N_BINARIES = 600   # binaries simulated per combination (more = less noise per score, slower)

# observed targets for the chosen subset (low-P / high-P / all)
OBS_ECC      = fit.OBS_DNS_E_FIT
OBS_LOGPERIOD = np.log10(fit.OBS_DNS_P_FIT)


def _readable(name, value):
    """Human-friendly value: KICK1_SIGMA is stored in sim units, show it in km/s."""
    if name == "KICK1_SIGMA":
        return f"{value / 31.55:.0f} km/s"
    return f"{value:.4g}"


# --- 1. sample the parameter box (Latin hypercube: even coverage, no clustering) ---
samples_norm = qmc.LatinHypercube(d=N_PARAMS, seed=1).random(N_SAMPLES)  # in [0,1]
samples      = BOUND_LO + samples_norm * (BOUND_HI - BOUND_LO)           # real values

print(f"DNS subset = {fit.DNS_SUBSET}  ({len(OBS_ECC)} observed targets)")
print(f"sampling {N_SAMPLES} parameter sets x {N_BINARIES} binaries each ...")

# --- 2. score every sample ---
total_error, ecc_ks, period_ks, survival, median_ecc = [], [], [], [], []
for k in range(N_SAMPLES):
    fit.set_params(samples_norm[k])
    dns_P, dns_E, hmxb_P, hmxb_E = fit.simulate(N_BINARIES, seed=3)  # fixed seed -> fair comparison
    total_error.append(fit.error(dns_P, dns_E, hmxb_P, hmxb_E, N_BINARIES))
    survival.append(len(hmxb_P) / N_BINARIES)

    in_subset = fit._in_subset(dns_P)            # only score the chosen period subset
    P, E = dns_P[in_subset], dns_E[in_subset]
    if len(E) > 20:
        ecc_ks.append(ks_2samp(E, OBS_ECC).statistic)
        period_ks.append(ks_2samp(np.log10(P), OBS_LOGPERIOD).statistic)
        median_ecc.append(np.median(E))
    else:                                         # too few survivors to score this sample
        ecc_ks.append(1.0); period_ks.append(1.0); median_ecc.append(np.nan)

    if (k + 1) % 50 == 0:
        print(f"  ...{k + 1}/{N_SAMPLES}")

total_error = np.array(total_error); ecc_ks = np.array(ecc_ks)
period_ks   = np.array(period_ks);   survival = np.array(survival)
median_ecc  = np.array(median_ecc)


# --- 3. learn which parameter drives each score ---
def show_drivers(score, label):
    """Print each parameter's share of the variation in `score` (RandomForest importance),
    plus whether raising the parameter raises or lowers the score."""
    forest = RandomForestRegressor(n_estimators=400, random_state=0).fit(samples_norm, score)
    importance = forest.feature_importances_
    print(f"\n=== what drives {label} ===")
    print("    (importance = share of the score's variation explained by that parameter)")
    for i in np.argsort(importance)[::-1]:
        trend = np.corrcoef(samples[:, i], score)[0, 1]       # +ve: raising it raises the score
        direction = "raises" if trend > 0 else "lowers"
        print(f"  {PARAM_NAMES[i]:<26} {importance[i] * 100:5.1f}%   (raising it {direction} the score)")


print(f"\nsamples={N_SAMPLES}  binaries/sample={N_BINARIES}")
show_drivers(total_error, "TOTAL FIT ERROR (lower = better fit)")
show_drivers(ecc_ks,      "ECCENTRICITY-distribution mismatch (KS)")
show_drivers(period_ks,   "PERIOD-distribution mismatch (KS)")

# --- best eccentricity match found in the random sample ---
best = int(np.argmin(ecc_ks))
print(f"\n=== best eccentricity match in the sample ===")
print(f"  ecc_KS={ecc_ks[best]:.3f}   median e: sim {median_ecc[best]:.3f} vs obs {np.median(OBS_ECC):.3f}"
      f"   survival={survival[best]:.0%}")
for name, value in zip(PARAM_NAMES, samples[best]):
    print(f"  {name:<26} = {_readable(name, value)}")
