"""Automated parameter fitter for the DNS (period, eccentricity) distribution.

Drives main.main() + main.evolve_dns() with trial parameters, scores the resulting
observable-DNS cloud against the 18 observed systems with a KS-based error, and
optimises the parameters to minimise it.

    python fit.py            # local refine from current params (Nelder-Mead)
    python fit.py global     # global search (differential evolution, slower)

Edit PARAMS to choose which knobs to fit and their bounds.
"""
import sys
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

import main as M
from plot import _OBS_DNS, _OBS_HMXB

# Fit the TWO-CHANNEL common envelope (see PERIOD_BRANCHES.md): alpha*lambda is drawn per system
# from a tight + wide mix, which is what lets the model reproduce BOTH the tight (0.1-0.5 d) and the
# wide (4-45 d) period branches at once.  With this on, the single CE_EFFICIENCY is unused, so the
# fit varies CE_EFF_TIGHT / CE_EFF_WIDE / CE_WIDE_FRAC instead (see PARAMS below).
M.CE_TWO_CHANNEL = True

# Fit NORMAL (Gaussian-magnitude) natal kicks for BOTH supernovae: the kick SPEED is drawn from
# N(mean, sigma) truncated at 0, so the fit varies KICK1_MEAN / KICK2_MEAN (the kick MEANS) rather
# than Maxwellian sigmas.  Without these two lines the *_MEAN params in PARAMS would do nothing
# (the default kick law would ignore them).  See KICK1_DIST / KICK2_DIST in main.py.
M.KICK1_DIST = "normal"
M.KICK2_DIST = "normal"

# --- observed targets: final DNS (post-2nd-SN) and HMXB (post-1st-SN) ---
OBS_DNS_P  = np.array([p for p, _ in _OBS_DNS]);  OBS_DNS_E  = np.array([e for _, e in _OBS_DNS])
OBS_HMXB_P = np.array([p for p, _ in _OBS_HMXB]); OBS_HMXB_E = np.array([e for _, e in _OBS_HMXB])

# Per-stage axis scales: standardise log10(period) and eccentricity by each sample's own
# spread, so a step in period weighs the same as a step in e within that stage's distance.
def _scales(P, E):
    return np.log10(P).std(), E.std()
S_LOGP_DNS,  S_E_DNS  = _scales(OBS_DNS_P,  OBS_DNS_E)
S_LOGP_HMXB, S_E_HMXB = _scales(OBS_HMXB_P, OBS_HMXB_E)

# --- parameters to fit: (attribute, low, high) ---
# Kick sigmas, kick angles, and the NS masses (fixed at 1.4) are held constant;
# these are the phenomenological binary-evolution knobs.  Comment out any line to
# hold it fixed at its current main.py value.
PARAMS = [
    # === 1st-SN stage — sets the HMXB period & eccentricity and how many survive ===
    ("PRESN_LOGPERIOD_MEAN_HI",  1.4,  3.5),       # log10 input orbital period [days]: where the cloud sits in P
    #("PRESN_LOGPERIOD_SIGMA_HI", 0.4,  1.4),       # spread of input period: WIDTH of the cloud in P
                                                   # (needed to populate both edges, e.g. the ~1.2 d point)
    ("MASSLOSS_MEAN_HI",         1.5,  8.0),        # mass lost in the 1st SN: HMXB eccentricity + survival
    # === Two-channel CE — sets the DNS period and, crucially, the SPLIT between the two period
    #     branches.  alpha*lambda is the common-envelope efficiency: LOWER = stronger orbital
    #     shrink = TIGHTER DNS.  Fitting two values + their mix is what reproduces both branches. ===
    ("CE_EFF_TIGHT",             0.01, 0.60),        # strong-shrink channel -> the TIGHT branch (0.1-0.5 d)
    ("CE_EFF_WIDE",              0.60, 3),        # weak-shrink channel  -> the WIDE branch (4-45 d)
    ("CE_WIDE_FRAC",             0.10, 0.45),        # fraction taking the wide channel = weight in the wide tail
    ("CE_MIN_PREPERIOD_DAYS",    0.0, 365.0),        # HMXB period at CE onset must exceed this to eject the envelope
    ("HE_CORE_MASS",             2.0,  4.0),         # He-star mass after CE (also affects the CE shrink)
    # === DNS eccentricity ===
    ("CASE_BB_MASS_SCALE",       0.8,  1.4),         # scales SN mass loss -> Blaauw e and its rise with period (P,e slope)
    ("KICK2_SIGMA",              5.0,  50.0),         # 2nd-SN kick spread [km/s] (normal kick): ecc scatter
    # === detectability (snapshot selection) ===
    ("RADIO_LIFETIME_MYR",       100,  10000),       # radio e-folding time tau: SHORT = young sample caught
                                                     # near birth (less GW evolution) = tighter, wider-tail match
    ("KICK2_MEAN",             10.0,  50.0),        # 2nd-SN kick mean [km/s]: adds eccentricity scatter
    ("KICK1_MEAN",             30.0*31.55,  200.0*31.55),  # 1st-SN kick mean [sim units]: HMXB ecc + survival fraction
    # --- held fixed at sensible values (uncomment to fit) ---
    #("COMPANION_MASS",          8.0,  20.0),        # massive "other" star mass [Msun]: 1st-SN binding + CE envelope
    #("KICK2_LOW_FRAC",          0.6,  0.95),        # bimodal: fraction with the small kick (rest -> high-e tail)
    #("KICK_ANGLE",              0,    180),         # 1st-kick cone half-angle (~isotropic is fine)
    ("GW_MAX_AGE_MYR",       4000,  12000),         # Milky Way disk age; usually fixed near 10 Gyr
    #("KICK1_SIGMA",         80*31.55, 220*31.55),   # 1st-SN kick [km/s x 31.55]: HMXB ecc + survival fraction
]

N_EVAL      = 2000     # binaries per objective evaluation (speed vs. noise)
SEED        = 2026062 # fixed per-eval seed -> deterministic, smooth objective
OUTLIER_W   = 9      # weight on the sim->obs (outlier) term; raise to punish outliers harder
COVERAGE_K  = 15        # an obs point must have ~K simulated neighbours to count as 'covered'
                       # (~4 for a sparse subset like high-P so edge points e.g. ~1.2 d still
                       #  count; ~15 for the full 'all' sample)
                       # (coverage uses the K-th nearest sim point, so a lone stray sim point
                       #  no longer hides a gap in the dense cloud)
PERIOD_KS_WEIGHT = 3 # match the period DISTRIBUTION (KS of log10 P).  KS in [0,1], so weight ~3
                       # makes a bad period CDF (KS~0.4) cost ~1.2, comparable to the Chamfer (~2-3).
                       # The model CAN match this marginal, so it gets solid weight.
ECC_KS_WEIGHT    = 3 # match the eccentricity DISTRIBUTION, weighted equally with period.  (The exact
                       # joint (P,e) correlation is NOT achievable -- Tauris sec 5.4 -- so we target
                       # the two marginals, which ARE, rather than over-weighting the 2D Chamfer.)
FIT_STAGE   = "dns"   # which stage the error scores: "hmxb" (post-1st-SN), "dns" (final), or "both"
# Fit only a period subset of the DNS (both obs target and sim cloud are restricted to it):
#   "all"  : every DNS system
#   "low"  : short-period subset, P <  DNS_SUBSET_PSPLIT days  (the close/merging systems)
#   "high" : long-period subset,  P >= DNS_SUBSET_PSPLIT days  (the wide systems)
DNS_SUBSET        = "low"   # "all", "low", or "high".  Use "all" when fitting the two-channel CE,
                            # since its whole purpose is to match BOTH period branches at once.
DNS_SUBSET_PSPLIT = 1.0   # period split [days] between the low and high subsets
HMXB_WEIGHT = 0.3      # weight of HMXB relative to DNS when FIT_STAGE == "both"
MAX_EVALS   = 300      # optimiser budget (objective evaluations); 8 params need a few hundred
                       #   tip: run  `python fit.py global`  (differential evolution) for robustness
# Survival floor penalty: stops the optimiser 'winning' by unbinding the whole
# population (a huge kick -> few survivors -> tiny Chamfer mean).  Only engages below
# MIN_SURVIVAL_FRAC; no penalty for high survival.  YIELD_PENALTY >> a normal fit error.
MIN_SURVIVAL_FRAC = 0.03    # penalty only engages below this survival fraction (none above)
YIELD_PENALTY     = 100.0   # max penalty (>> a normal fit error of ~30)
_n_eval     = 0


def simulate(n=N_EVAL, seed=SEED):
    """Run the full pipeline; return (dns_P, dns_E, hmxb_P, hmxb_E).

    HMXB = raw post-first-SN state (NS + massive star).
    DNS  = final state after the 2nd SN and GW aging (mergers removed).
    """
    if seed is not None:
        np.random.seed(seed)
    dP, dE, hP, hE = [], [], [], []
    for i in range(1, n + 1):
        r = M.main(i)
        # HMXB stage (post first SN, no aging)
        eh = r.get('ecc_hmxb')
        if eh is not None and 0 <= eh < 1:
            ph = r['period_hmxb'] * 365.25
            if 0 < ph < 1e4:
                hP.append(ph); hE.append(eh)
        # DNS stage (final, GW-aged)
        e = r['ecc_imp']
        if e is None or not (0 <= e < 1):
            continue
        period_yr, e_new, merged = M.evolve_dns(e, r['period'])
        if merged:
            continue
        Pd = period_yr * 365.25
        if Pd <= M.MAX_OBSERVABLE_PERIOD_DAYS:    # observational selection: widest observed DNS ~45 d
            dP.append(Pd); dE.append(e_new)
    return np.array(dP), np.array(dE), np.array(hP), np.array(hE)


def _pts(P, E, s_logP, s_e):
    """(period_days, ecc) -> standardised 2D points [log10 P / s, e / s]."""
    return np.column_stack([np.log10(P) / s_logP, E / s_e])


def _in_subset(P):
    """Boolean mask selecting the chosen period subset (low: P<split, high: P>=split)."""
    P = np.asarray(P, dtype=float)
    if DNS_SUBSET == "low":
        return P < DNS_SUBSET_PSPLIT
    if DNS_SUBSET == "high":
        return P >= DNS_SUBSET_PSPLIT
    return np.ones(len(P), dtype=bool)


# Observed DNS restricted to the chosen subset.  When a subset is selected we also recompute
# the axis scales FROM THE SUBSET, so systems outside it (e.g. the wide/high-P group) have
# zero effect on the standardisation — the error sees only the chosen group.
_m_dns = _in_subset(OBS_DNS_P)
OBS_DNS_P_FIT, OBS_DNS_E_FIT = OBS_DNS_P[_m_dns], OBS_DNS_E[_m_dns]
if DNS_SUBSET != "all" and len(OBS_DNS_P_FIT) > 1:
    S_LOGP_DNS, S_E_DNS = _scales(OBS_DNS_P_FIT, OBS_DNS_E_FIT)
_OBS_DNS_PTS  = _pts(OBS_DNS_P_FIT, OBS_DNS_E_FIT, S_LOGP_DNS, S_E_DNS)
_OBS_HMXB_PTS = _pts(OBS_HMXB_P, OBS_HMXB_E, S_LOGP_HMXB, S_E_HMXB)


def _chamfer(P, E, obs_pts, s_logP, s_e):
    """Squared Chamfer distance: sim->obs (outliers) + obs->sim (coverage).

    The coverage term uses each observed point's distance to its K-th nearest
    simulated point (not its single nearest), so an observation only counts as
    'covered' when it lies in the DENSE part of the simulated cloud — a lone stray
    sim point near an observation no longer hides a gap.
    """
    n = len(P)
    thresh = COVERAGE_K + 1                # need K+1 sim points for the K-th-nearest coverage
    if n < thresh:
        # Too few sim points in this (sub)set to score reliably.  Return a penalty that
        # DECREASES as n approaches the threshold, so the optimiser keeps a gradient back
        # toward parameters that actually populate the subset — instead of the flat 10.0
        # plateau it would otherwise get stuck on (esp. when fitting a sparse subset).
        return 10.0 + 10.0 * (thresh - n) / thresh
    D = cdist(_pts(P, E, s_logP, s_e), obs_pts)
    outlier  = (D.min(axis=1) ** 2).mean()                  # each sim point -> nearest obs
    k        = min(COVERAGE_K, n - 1)
    coverage = (np.partition(D, k, axis=0)[k] ** 2).mean()  # each obs -> K-th nearest sim
    return OUTLIER_W * outlier + coverage


def _ks(sim, obs):
    """KS distance between sim and obs 1D samples; 1.0 if too few sim points."""
    if len(sim) < 10:
        return 1.0
    return ks_2samp(sim, obs).statistic


def error(dP, dE, hP, hE, n_sim):
    """DNS Chamfer error, optionally plus the HMXB (post-1st-SN) term, plus explicit
    period/eccentricity marginal-CDF terms, plus a yield penalty so unbinding the whole
    population is never rewarded."""
    total = 0.0
    if FIT_STAGE in ("dns", "both"):
        m = _in_subset(dP)                       # restrict sim to the same period subset as obs
        dP, dE = dP[m], dE[m]
        total += _chamfer(dP, dE, _OBS_DNS_PTS, S_LOGP_DNS, S_E_DNS)
        # explicit marginal matches: period (log10 P) and optionally eccentricity
        total += PERIOD_KS_WEIGHT * _ks(np.log10(dP), np.log10(OBS_DNS_P_FIT)) if len(dP) else PERIOD_KS_WEIGHT
        if ECC_KS_WEIGHT:
            total += ECC_KS_WEIGHT * _ks(dE, OBS_DNS_E_FIT)
    if FIT_STAGE in ("hmxb", "both"):
        w = HMXB_WEIGHT if FIT_STAGE == "both" else 1.0
        total += w * _chamfer(hP, hE, _OBS_HMXB_PTS, S_LOGP_HMXB, S_E_HMXB)
    surv  = len(hP) / max(1, n_sim)                     # fraction surviving the first SN
    if surv < MIN_SURVIVAL_FRAC:                         # floor only; no penalty for high survival
        total += YIELD_PENALTY * (MIN_SURVIVAL_FRAC - surv) / MIN_SURVIVAL_FRAC
    return total


def set_params(x_norm):
    """Map a normalised [0,1] vector onto the module globals; return the real values."""
    vals = {}
    for xi, (attr, lo, hi) in zip(x_norm, PARAMS):
        v = lo + float(np.clip(xi, 0.0, 1.0)) * (hi - lo)
        setattr(M, attr, v)
        vals[attr] = v
    return vals


def objective(x_norm):
    global _n_eval
    _n_eval += 1
    vals = set_params(x_norm)
    dP, dE, hP, hE = simulate()
    err = error(dP, dE, hP, hE, N_EVAL)
    print(f"[{_n_eval:3d}] err={err:.4f}  surv={len(hP)/N_EVAL:.0%}  "
          + "  ".join(f"{k}={v:.4g}" for k, v in vals.items()))
    return err


def _report(x_norm, err):
    vals = set_params(x_norm)
    _N = 25000
    dP, dE, hP, hE = simulate(n=_N, seed=None)   # fresh seed -> honest, un-overfit numbers
    print("\n=== BEST PARAMETERS ===")
    for k, v in vals.items():
        print(f"  {k:<22} = {v:.4g}")
    print(f"\n  fit error = {err:.4f}   (subset = {DNS_SUBSET})")
    if DNS_SUBSET != "all":                       # show stats over the fitted subset only
        m = _in_subset(dP); dP, dE = dP[m], dE[m]
    f = lambda P, E: (len(P), np.median(P), np.median(E), np.mean(E < 0.4), np.mean(E > 0.5))
    hdr = "   n   medP   medE   fe<.4  fe>.5"
    print(f"               {hdr}")
    print("  DNS  sim   " + "  ".join(f"{v:5.3f}" for v in f(dP, dE)))
    print("  DNS  obs   " + "  ".join(f"{v:5.3f}" for v in f(OBS_DNS_P_FIT, OBS_DNS_E_FIT)))
    print("  HMXB sim   " + "  ".join(f"{v:5.3f}" for v in f(hP, hE)))
    print("  HMXB obs   " + "  ".join(f"{v:5.3f}" for v in f(OBS_HMXB_P, OBS_HMXB_E)))

    # --- ready-to-paste main.py lines for the fitted parameters ---
    print("\n=== paste into main.py ===")
    for k, v in vals.items():
        if k == "KICK1_SIGMA":
            print(f"{k:<24}= {v/31.55:.4g} * 31.55")
        else:
            print(f"{k:<24}= {v:.6g}")


if __name__ == "__main__":
    ndim = len(PARAMS)
    x0 = np.array([np.clip((getattr(M, a) - lo) / (hi - lo), 0, 1) for a, lo, hi in PARAMS])
    print(f"Fitting {ndim} parameters, N={N_EVAL}/eval.  Baseline:")
    base = objective(x0)

    if len(sys.argv) > 1 and sys.argv[1] == "global":
        # global search — robust to local minima/noise; popsize*ndim per generation
        popsize = max(4, MAX_EVALS // (2 * ndim))
        res = differential_evolution(objective, bounds=[(0, 1)] * ndim,
                                     maxiter=MAX_EVALS // (popsize * ndim) + 1, popsize=popsize,
                                     tol=0.002, seed=0, polish=False, init='sobol', mutation=(0.4, 1.2))
        best_x, best_f = res.x, res.fun
    else:
        # local refine from current params, capped at MAX_EVALS evaluations.
        # Use a wide initial simplex (step 0.25 in normalised space) so the small
        # eval budget explores a meaningful region instead of tiny perturbations.
        step0   = 0.25
        simplex = np.vstack([x0] + [np.clip(x0 + step0 * row, 0, 1) for row in np.eye(ndim)])
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": MAX_EVALS, "maxfev": MAX_EVALS,
                                "xatol": 0.005, "fatol": 0.0015, "adaptive": True,
                                "initial_simplex": simplex})
        best_x, best_f = res.x, res.fun

    _report(best_x, best_f)
