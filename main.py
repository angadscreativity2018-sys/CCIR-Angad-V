import numpy as np
import step
import plot as plt_orbit
from plot import orbit_label, _OBS_LOW_E, _OBS_HIGH_E
import time
import multiprocessing

# =============================================================================
# UNIT SYSTEM  
# -----------------------------------------------------------------------------
#  Base units
#    mass     : 1 unit = 1 M_sun  = 1.989 × 10³⁰ kg
#    length   : 1 unit = 10^9 m = 0.00667 AU (1/150)
#    time     : 1 unit = 1 year = 3.15 × 10⁷ s
#
#  Derived units
#    velocity : 1 unit = 31.55 m/s
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- View / frame ---
CENTER_OF_MASS_FRAME     = False   # shift to CoM frame so the pair doesn't drift off screen
PROJECT_TO_ORBITAL_PLANE = False   # project onto orbital plane for 2D view  (single run only)

# --- Output ---
SHOW_PLOT                = False   # single-run orbit plot
SHOW_MULTI_PLOT          = False  # 3D CoM trajectory plot across all runs
SHOW_ECC_STATS           = True   # scatter of post-impulse eccentricity vs period across all runs; also shows histogram of eccentricities on the side
SHOW_MERGER_TIME         = True   # scatter of GW merger time vs period, with observed DNS overlaid
VERBOSE                  = False  # print per-run info (slow for large ITERATIONS)

SHOW_INPUT_POPULATION = False
INPUT_POPULATION_SAMPLES = 100000


MULTI_PLOT_WINDOW        = 1000    # steps shown either side of the impulse
#other stuff
SPLIT                    = True   # bimodal distribution of mass loss
SPLIT_RATIO              =  10*len(_OBS_HIGH_E)/len(_OBS_LOW_E+_OBS_HIGH_E)  # fraction of runs that use the high-mass-loss distribution

# --- Multi-run sweep ---
ITERATIONS               = 100000   # number of binaries simulated; raise for a denser cloud
PERIOD_DAYS_MIN          = 1      # paper lower truncation for pre-SN period [days]
PERIOD_DAYS_MAX          = 3000   # paper upper truncation for pre-SN period [days]
PRESN_LOGPERIOD_MEAN_LO = 1.9   
PRESN_LOGPERIOD_SIGMA_LO = .0

PRESN_LOGPERIOD_MEAN_HI = 2.2
PRESN_LOGPERIOD_SIGMA_HI = 1.2

# --- Impulse (supernova kick) ---
APPLY_KICK            = True   # whether to apply the kick at all

MASSLOSS_MEAN_LO           = 1.8   # low mass-loss distribution  [M☉]
MASSLOSS_SIGMA_LO          = np.sqrt(MASSLOSS_MEAN_LO)/4  # truncated below 0
MASSLOSS_MEAN_HI           = 4.1    # high mass-loss distribution  [M☉]
MASSLOSS_SIGMA_HI          = np.sqrt(MASSLOSS_MEAN_HI)/4  # truncated below 0

# First-SN natal kick.  The first-born NS forms from a normally-stripped (not ultra-stripped)
# progenitor, so it gets a full-strength kick.  KICK1_DIST chooses how the SPEED is drawn:
#   "maxwellian" -> speed = |3D Gaussian| with per-axis sigma KICK1_SIGMA  (the standard).
#   "normal"     -> speed straight from a 1D normal N(KICK1_MEAN, KICK1_SIGMA), truncated at 0.
# Both KICK1_SIGMA and KICK1_MEAN are in SIM velocity units (1 km/s = 31.55 units, e.g. 150 km/s
# = 150*31.55).  Direction is set by KICK_ANGLE / BIAS_DIRECTION below.
KICK1_DIST               = "normal"  # "maxwellian" or "normal"
KICK1_SIGMA              = 16 * 31.55   # sigma  [sim velocity units]  (~150 km/s)
KICK1_MEAN               = 100 * 31.55   # mean kick speed [sim velocity units] (only when "normal")

RANDOM_PHASE             = False   # subtract 0–1 orbital period from impulse time each run
NS1_MASS       = 1.4    # first-born NS mass after kick  [M☉]
COMPANION_MASS = 10.0   # companion (the "other" star: massive 2nd star) mass at the 1st SN [M☉].
                        # This is the OB/He star that survives the 1st SN, becomes the HMXB donor,
                        # ejects its envelope in the CE, and explodes in the 2nd SN.  Heavier -> the
                        # 1st-SN binary is tighter/more bound (survives bigger kicks) and the CE has
                        # a more massive envelope to eject.
KICK_ANGLE               = 31.15   # half-cone angle in degrees: 0 = always along bias dir, 180 = fully isotropic
BIAS_DIRECTION           = "orthogonal"  # "velocity" → along v1;  "orthogonal" → perpendicular to orbital plane

# --- Distance bounds ---
CHECK_DIST_PRE           = False  # filter on pre-SN periastron distance
CHECK_DIST_POST          = False   # filter on post-SN periastron distance
DISTANCE_MIN             = 0 #periastron distance must be greater than this to be included in results  [sim length units]
DISTANCE_MAX             = 413 #periastron distance must be less than this to be included in results  [sim length units]

STOP_EARLY = True  # stop if hyperbolic orbits are detected

# =============================================================================
# SECOND SUPERNOVA KICK  (kick applied to the companion star)
# =============================================================================
# When APPLY_KICK2 = True, the companion also undergoes a SN after the first NS
# is formed.  The kick is applied at a random point in the post-first-SN orbit
# (drawn uniformly in time, i.e. mean anomaly M ~ Uniform[0, 2π]).
# Only bound systems that survive the first kick are eligible.

APPLY_KICK2          = True  # enable second SN on companion star

# Companion remnant (young, ultra-stripped NS).
NS2_MASS   = 1.4    # NS2 mass after second SN  [M_sun]

# Second-SN (ultra-stripped) natal kick.  The kick DIRECTION is set by KICK2_ANGLE (below);
# KICK2_DIST chooses how the kick SPEED is drawn:
#   "single"  -> one Maxwellian: speed = |3D Gaussian| with per-axis sigma KICK2_SIGMA.
#   "bimodal" -> most NSs get a SMALL Maxwellian kick (sigma KICK2_SIGMA_LOW), a rare fraction
#                get a LARGE one (sigma KICK2_SIGMA_HIGH).  Tauris+2017 (sec 6.5.1, Table 7):
#                ultra-stripped SNe give w <~ 50 km/s for most DNS, but B1913+16 / B1534+12 need
#                w ~ 200-450 km/s.  A single small Maxwellian cannot make those moderate/high-e
#                systems; the bimodal high tail does, while tight post-CE orbits survive the large
#                kick (their Fig 18) and so reach high e.
#   "normal"  -> speed drawn DIRECTLY from a 1D normal N(KICK2_MEAN, KICK2_SIGMA), truncated at 0
#                (a Gaussian on the kick MAGNITUDE itself, rather than the Maxwellian |3D Gaussian|).
#   "bimodal_normal" -> like "bimodal" but each component is a NORMAL magnitude, not a Maxwellian:
#                low (bulk) = N(KICK2_MEAN, KICK2_SIGMA_LOW) with prob KICK2_LOW_FRAC, high (rare)
#                = N(KICK2_MEAN_HIGH, KICK2_SIGMA_HIGH).  Both truncated at 0.
KICK2_DIST           = "normal"  # "single", "bimodal", "normal", or "bimodal_normal"
KICK2_SIGMA          = 10     # km/s  Maxwellian sigma ("single") OR normal sigma ("normal")
KICK2_MEAN           = 30     # km/s  mean kick speed ("normal"; also the LOW-component mean for "bimodal_normal")
KICK2_SIGMA_LOW      = 20     # km/s  low component sigma  (the dominant, small-kick population)
KICK2_SIGMA_HIGH     = 200    # km/s  high component sigma (rare; makes the high-e systems)
KICK2_MEAN_HIGH      = 200    # km/s  high-component mean (used only when KICK2_DIST == "bimodal_normal")
KICK2_LOW_FRAC       = 0.85   # fraction of 2nd SNe drawn from the low component

# Kick geometry
# Bias axis is always the original orbital plane normal [0,0,1].
# KICK2_ANGLE = 0   → kick always along [0,0,1]
# KICK2_ANGLE = 180 → fully isotropic (bias ignored)
KICK2_ANGLE          = 180   # cone half-angle  [deg]

# --- Pre-second-SN mass transfer (NS1 accretes from companion before it explodes) ---
# When APPLY_MT2 = True, MT2_DELTA_M is stripped from the companion and a fraction
# MT2_EFFICIENCY is accreted onto NS1; the rest is lost from the system.
#   mass_be_eff  = mass_be  − delta_m                   (less CO core → weaker NS2 kick)
#   mass_ns1_eff = mass_ns1 + MT2_EFFICIENCY × delta_m  (heavier NS1 going into second kick)
# delta_m is clamped so the companion cannot be stripped below NS2_MASS + 0.1 M_sun.
#
# Orbital changes during MT2:
#   ecc_mt2  = ecc  * MT2_ECC_FACTOR    (e.g. 0.0 → fully circularised; 1.0 → unchanged)
#   period_mt2 = period * MT2_PERIOD_FACTOR  (e.g. 0.5 → half the period; 1.0 → unchanged)
# These are applied before the second kick.
# Tuned to the observed DNS sample: the companion is ultra-stripped (Case BB RLO in the
# DNS life cycle), the common envelope shrinks the orbit hard (small period factor), and
# the orbit is strongly circularised before the ultra-stripped SN.
APPLY_MT2            = True  # enable pre-SN2 mass transfer
MT2_DELTA_M          = 8.74   # mass stripped from companion  [M_sun]  (ultra-strips 10 → 1.6)
MT2_EFFICIENCY       = 0.001   # fraction of stripped mass accreted by NS1  (0 → lost; 1 → conservative)
MT2_ECC_FACTOR       = .5  # multiply post-MT2 eccentricity by this  (0 = circular, 1 = no change)
MT2_PERIOD_FACTOR    = 0.01487   # multiply post-MT2 period by this  (< 1 = shrinks, > 1 = widens)
MT2_PERIOD_FLOOR_DAYS = 0.0  # hard lower bound on post-MT2 period [days]; was implicitly 1 day.
                              # This was the cause of the sharp left wall in the (P,e) scatter.
                              # Observed DNS reach ~0.08 d, so keep this below that.

# --- Physically-resolved pre-SN2 mass transfer (Common Envelope + Case BB RLO) ---
# MT2_MODE = "physical" replaces the three multiplicative knobs above with the two
# real evolutionary phases that sit between the two neutron stars:
#
#   (1) COMMON ENVELOPE  (Webbink 1984 alpha-lambda energy formalism).
#       The massive companion expands and engulfs NS1.  Drag ejects the donor's
#       H envelope at the expense of orbital energy, shrinking AND circularising
#       the orbit.  Donor mass (M_donor) -> He core (HE_CORE_MASS).
#           a_f/a_i = (M_he * M_ns) / [ M_donor * (M_ns + 2 M_env/(alpha*lambda*r_L)) ]
#           r_L = Eggleton(q),  q = M_donor/M_ns ;  e -> 0
#
#   (2) CASE BB RLO.  The He star expands after core-He exhaustion and transfers
#       its He envelope onto NS1 (non-conservative, Eddington-limited).  He core
#       (HE_CORE_MASS) -> ultra-stripped CO core (CO_CORE_MASS, the pre-SN2 mass).
#       Orbit follows the isotropic-re-emission angular-momentum balance, integrated
#       numerically; CASEBB_ACCRETION is the fraction of transferred mass NS1 keeps (~0).
#
# This yields the tight, circular NS + ultra-stripped CO-core binary that the
# second (ultra-stripped) SN then turns into a DNS — with the orbit set by real
# stellar masses instead of fitted multipliers.  Set MT2_MODE = "factor" to fall
# back to the old multiplicative behaviour.
MT2_MODE          = "physical"  # "physical" (CE + Case BB) or "factor" (old multipliers)
CE_EFFICIENCY   = 0.1  # common-envelope efficiency * structure parameter (alpha_CE * lambda)
HE_CORE_MASS      = 2.6    # donor He-core mass after the H envelope is ejected  [M_sun]
CO_CORE_MASS      = 1.5    # ultra-stripped CO core after Case BB RLO  [M_sun]  (pre-SN2 companion)
CASEBB_ACCRETION           = 0.001  # fraction of Case-BB transferred mass accreted by NS1 (Eddington ~ 0)

# Period-dependent stripping (Tauris+2017 Table 5/6): a WIDER pre-SN2 orbit fills its Roche
# lobe later, strips LESS, and explodes MORE massive -> larger SN mass loss -> higher Blaauw
# eccentricity.  This period->mass relation is what creates the observed (P, e) correlation;
# with a constant CO_CORE_MASS, eccentricity cannot track period.  When True, the exploding
# He-star mass is interpolated from the grid instead of using CO_CORE_MASS.
CASE_BB_PERIOD_DEPENDENT = True   # exploding mass rises with pre-SN2 period (vs constant CO_CORE_MASS)
CASE_BB_MASS_SCALE       = 1.0    # tunable amplitude of the period-dependent stripping

# Common-envelope merger: if the CE shrinks the orbit so far that the bare He core
# overflows its own Roche lobe at the post-CE separation, the NS spirals all the way
# in and merges inside the companion — NO DNS forms (a "failed CE").  This is the
# dominant DNS-formation bottleneck; it removes the unphysical ultra-tight systems
# and sets a real left edge to the (P, e) cloud.  He-core radius from the He-ZAMS
# mass-radius relation (Hurley, Pols & Tout 2000).
CE_MERGE_CHECK    = True   # remove systems whose He core overflows its Roche lobe after CE
R_SUN_SIM         = 0.6957  # solar radius in sim length units  (6.957e8 m / 1e9 m)

# CE survival (the dominant DNS-formation bottleneck; Tauris+2017 sec 3.4.1-3.4.2, Table 4).
# Only WIDE HMXBs survive the common envelope: if the orbit at CE onset is too tight, the
# donor's envelope is too tightly bound to eject, so the NS spirals all the way in and the
# system coalesces into a Thorne-Zytkow object -> NO DNS forms.  Their individual-system
# fits (Figs 24-38) show surviving DNS descend from pre-CE periods of order >~1 yr, which CE
# then shrinks ~100-1000x to the observed pre-SN2 periods of ~0.1-1 day.  Systems entering CE
# below this period merge.  (The exact threshold is genuinely uncertain - see their sec 1.5.1.)
CE_MIN_PREPERIOD_DAYS = 365.0  # HMXB period at CE onset for ~50% envelope-ejection survival
# CE survival is NOT a sharp step (the threshold is uncertain), so the survival probability ramps
# smoothly with period: logistic in log10(period), centred on CE_MIN_PREPERIOD_DAYS, with this
# transition width [dex].  A wider ramp smears the otherwise-hard left wall in the DNS period
# (set 0 for a hard cutoff).
CE_PREPERIOD_WIDTH = 0.25

# Two-channel common envelope (see PERIOD_BRANCHES.md).  The observed DNS orbital-period
# distribution is broad/bimodal: a tight peak at 0.1-0.5 d AND a wide tail to 45 d.  A single
# CE efficiency produces only ONE funnel (it trades the tight branch against the wide branch),
# so it cannot make both at once.  This reflects the two Case BB stripping regimes of Tauris+2017
# (Tables 5/6, Fig 7): deep ultra-stripping at short post-CE period -> tight low-e DNS, vs minimal
# Case-BC stripping at wide post-CE period -> wide higher-e DNS.  When enabled, alpha*lambda is
# drawn PER SYSTEM from a two-component mix: a strong-shrink (tight) value with probability
# 1-CE_WIDE_FRAC, and a weak-shrink (wide) value with probability CE_WIDE_FRAC.  Set
# CE_TWO_CHANNEL = False to fall back to the single CE_EFFICIENCY above.  These three knobs are
# fittable in fit.py.  Reference config (from search): 0.19 / 1.62 / 0.23 -> period KS 0.18 -> 0.14.
CE_TWO_CHANNEL = False     # ON: per-system alpha*lambda from a tight+wide mix (both period branches).
                          #     OFF: single CE_EFFICIENCY above (one period peak only).
CE_EFF_TIGHT   = 0.19     # alpha*lambda of the STRONG-shrink channel.  SMALL -> orbit collapses hard
                          #     -> short post-CE period -> the TIGHT DNS branch (0.1-0.5 d peak).
CE_EFF_WIDE    = 1.62     # alpha*lambda of the WEAK-shrink channel.  LARGE -> orbit barely shrinks
                          #     -> wide post-CE period (minimal Case-BC strip) -> the WIDE 4-45 d tail.
CE_WIDE_FRAC   = 0.23     # probability a system takes the WIDE channel (rest take the tight one);
                          #     directly sets how much weight lands in the wide-period tail.

# =============================================================================
# POST-SN AGING  —  gravitational-wave orbital decay  (Peters 1964)
# =============================================================================
# After the second SN the DNS loses energy and angular momentum to gravitational
# radiation.  The orbit shrinks AND circularises following Peters (1964),
# Phys. Rev. 136, B1224  (https://doi.org/10.1103/PhysRev.136.B1224):
#
#   da/dt = -(64/5) β/a³ · (1 + 73/24 e² + 37/96 e⁴)/(1-e²)^(7/2)
#   de/dt = -(304/15) e β/a⁴ · (1 + 121/304 e²)/(1-e²)^(5/2)
#   β     = G³ m1 m2 (m1+m2) / c⁵
#
# Each surviving DNS is evolved for a random age drawn uniformly in [0, GW_MAX_AGE_MYR].
# 10 Gyr is the Milky Way disk age: DNS form throughout the disk's star-formation history,
# so observed ages span 0-10 Gyr.  Systems that decay to merger before their drawn age are
# removed (no longer observable as DNS).  Only the tightest systems (P ≲ 0.3 d) evolve
# appreciably; wider ones are effectively frozen, because the merger time scales as a⁴.
# Because eccentric orbits have small periastron they merge fastest, so realistic Galactic
# aging naturally thins the high-e tail (rather than letting it pile up).
APPLY_AGING          = True       # apply GW orbital decay after the natal kicks
GW_MAX_AGE_MYR       = 10000.0    # DNS ages drawn uniformly in [0, this]  [Myr]  (Milky Way disk age)
LIGHTSPEED_SIM             = 9.461e6    # speed of light in sim velocity units (c ÷ 31.69 m/s)
MERGE_SEPARATION           = 1.0e-4     # semi-major axis treated as merged  [sim length units, ~100 km]
GW_STEP_FRAC         = .03     # max fractional change in a (or e) per integration step

# Observational selection on orbital period.  The 2nd SN kick occasionally widens an orbit to
# hundreds-thousands of days (bound, but with enormous period).  No such DNS is observed: the
# widest known Galactic DNS is PSR J1930-1852 at Porb = 45.06 d (Tauris+2017 Table 1; observed
# range 0.10-45 d).  These wide systems form physically but fall outside the period-limited
# observed sample, so a system with final period above this is not counted as an observable DNS.
MAX_OBSERVABLE_PERIOD_DAYS = 45.0   # widest observed Galactic DNS (J1930-1852); wider -> not observable

# -----------------------------------------------------------------------------
# OBSERVATIONAL DETECTABILITY  (which DNS we actually see; replaces the flat age cap)
# -----------------------------------------------------------------------------
# DNS form continuously over Galactic history (roughly constant rate for ~10 Gyr), so the Galaxy
# holds DNS of every age.  But a given DNS is only DETECTABLE as a radio-pulsar binary while THREE
# conditions hold at once, and that detectable lifespan differs hugely from system to system.  Two
# consequences follow from a constant formation rate (Little's law / a steady-state snapshot):
#
#   * the CHANCE of catching a system is proportional to the length of its detectable window,
#     tau_obs  -- long-lived systems are over-represented, short-lived ones are rarely caught;
#   * GIVEN it is caught, its age (time since the 2nd SN) is uniform on [0, tau_obs]
#     (constant birth rate => flat age distribution within the window).
#
# The truncations:
#   (1) RADIO-ACTIVE   the recycled pulsar fades SMOOTHLY, not at a hard age: its radio-detectable
#         survival probability is e^(-age / RADIO_LIFETIME_MYR), so RADIO_LIFETIME_MYR is the
#         e-folding time tau (Tauris+2017 sec 3.4.2; recycled-pulsar radio life ~10^8-10^10 yr).
#   (2) STILL BOUND    age < tau_merge(a0, e0)
#         after GW-driven coalescence there is no binary pulsar.  tau_merge is the
#         Peters (1964) inspiral time -- TIGHT and/or ECCENTRIC orbits merge fastest, so the
#         close, high-e systems have the SHORTEST detectable windows (Tauris+2017 sec 6.9, 9.2).
#   (3) IN PERIOD RANGE  P_orb(age) <= MAX_OBSERVABLE_PERIOD_DAYS   (applied separately, below).
#
#   => observable horizon   t_max  = min(tau_merge(a0, e0),  Galactic age)
#      snapshot capture probability   p_snap = 1 - e^(-t_max / tau)   in [0, 1]
#      (the radio-active fraction caught before the binary merges: -> 1 for systems that never
#      merge within ~tau; ~ t_max/tau for fast-merging close systems).  Given a system is caught,
#      its observed age follows the same radio decay truncated at t_max (a truncated exponential,
#      scale tau).  This is the single most important selection effect in Tauris+2017: it is WHY
#      wide non-merging systems are common while the tightest merging systems are caught only
#      fleetingly.  Set APPLY_DETECTABILITY=False to fall back to the old flat-age-window behaviour.
APPLY_DETECTABILITY   = True       # merger-time + radio-decay snapshot weighting (recommended)
APPLY_RADIO_SELECTION = True       # apply the exponential radio fading (else no radio death)
RADIO_LIFETIME_MYR    = 1000       # recycled-pulsar radio e-folding time tau [Myr]  (fittable)

# Acceleration-search bias (Tauris+2017 sec 2.1): a recycled pulsar in a very tight orbit has its
# signal Doppler-smeared by orbital acceleration within a single integration, lowering its
# detectability once P_orb drops to a few hours.  Modelled as a logistic detection probability in
# log10(P_orb): ~0 far below ACCEL_P50_DAYS, 0.5 at it, ~1 well above.  OFF by default (the bulk of
# the period selection is the merger/radio window above; enable to test the tight-orbit suppression).
ACCEL_BIAS     = True             # down-weight very tight orbits (acceleration smearing)
ACCEL_P50_DAYS = 0.07              # P_orb at 50% acceleration-search detectability [days]
ACCEL_WIDTH    = 0.25              # logistic width in log10(P_orb) [dex]

# High P stuff
PRESN_LOGPERIOD_MEAN_HI = 2.52349
MASSLOSS_MEAN_HI        = 3.80086
KICK1_SIGMA             = 88.26 * 31.55
CE_EFF_TIGHT            = 0.139785
CE_EFF_WIDE             = 2.19979
CE_WIDE_FRAC            = 0.302506
CE_MIN_PREPERIOD_DAYS   = 171.857
HE_CORE_MASS            = 2.24732
CASE_BB_MASS_SCALE      = 0.879346
KICK2_SIGMA_LOW         = 32.3884
RADIO_LIFETIME_MYR      = 2124.31
KICK2_MEAN              = 28.044
KICK1_MEAN              = 3019.17

PRESN_LOGPERIOD_MEAN_HI = 2.67712
MASSLOSS_MEAN_HI        = 4.10393
CE_EFF_TIGHT            = 0.0994683
CE_EFF_WIDE             = 2.65509
CE_WIDE_FRAC            = 0.320986
CE_MIN_PREPERIOD_DAYS   = 239.124
HE_CORE_MASS            = 2.10408
CASE_BB_MASS_SCALE      = 1.0709
KICK2_SIGMA             = 9.56882
RADIO_LIFETIME_MYR      = 100
KICK2_MEAN              = 32.122
KICK1_MEAN              = 3819.88
GW_MAX_AGE_MYR          = 10081.2




# =============================================================================
# SIMULATION
# =============================================================================

def _draw_mass_loss_and_period():
    high_pop = SPLIT and np.random.random() < SPLIT_RATIO

    if high_pop:
        m_mean, m_sigma = MASSLOSS_MEAN_HI, MASSLOSS_SIGMA_HI
        p_mean, p_sigma = PRESN_LOGPERIOD_MEAN_HI, PRESN_LOGPERIOD_SIGMA_HI
    else:
        m_mean, m_sigma = MASSLOSS_MEAN_LO, MASSLOSS_SIGMA_LO
        p_mean, p_sigma = PRESN_LOGPERIOD_MEAN_LO, PRESN_LOGPERIOD_SIGMA_LO

    while (mass_loss := np.random.normal(m_mean, m_sigma)) < 0:
        pass

    while True:
        period_days = 10 ** np.random.normal(p_mean, p_sigma)
        if PERIOD_DAYS_MIN <= period_days <= PERIOD_DAYS_MAX:
            break

    return mass_loss, period_days / 365.25, not high_pop

def generate_population_inputs(n_samples):
    masses = []
    periods = []
    low_flags = []

    for _ in range(n_samples):
        mass_loss, period_years, is_low_mass = _draw_mass_loss_and_period()

        masses.append(mass_loss)
        periods.append(period_years * 365.25)
        low_flags.append(is_low_mass)

    return np.array(masses), np.array(periods), np.array(low_flags)

def _draw_kick_speed():
    """Draw a first-SN natal kick speed [sim velocity units].

    "maxwellian": magnitude of a 3D Gaussian velocity (per-axis sigma KICK1_SIGMA).
    "normal":     1D normal N(KICK1_MEAN, KICK1_SIGMA) on the speed, truncated at 0.
    """
    if KICK1_DIST == "normal":
        return max(0.0, np.random.normal(KICK1_MEAN, KICK1_SIGMA))
    return np.linalg.norm(np.random.normal(0, KICK1_SIGMA, size=3))


def _rotate_z(vec, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]])


def _random_in_cone(axis, half_angle_deg):
    """Uniform random unit vector within a cone of half_angle_deg around axis."""
    if half_angle_deg >= 180:
        v = np.random.normal(size=3)
        return v / np.linalg.norm(v)
    cos_theta = np.cos(np.radians(half_angle_deg))
    cos_alpha = np.random.uniform(cos_theta, 1.0)
    sin_alpha = np.sqrt(1.0 - cos_alpha * cos_alpha)
    phi = np.random.uniform(0, 2 * np.pi)
    z = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, ref); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return sin_alpha * np.cos(phi) * x + sin_alpha * np.sin(phi) * y + cos_alpha * z

def _worker_init():
    import signal, ctypes
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # workers ignore Ctrl+C; main process handles it
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)

def _set_high_priority():
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)


def _solve_kepler(M_anom, ecc, n_iter=12):
    """Newton-Raphson solution of Kepler's equation  M = E - e·sin(E)."""
    E = M_anom
    for _ in range(n_iter):
        E -= (E - ecc * np.sin(E) - M_anom) / (1.0 - ecc * np.cos(E))
    return E


def _eggleton_rl(q):
    """Eggleton (1983) Roche-lobe radius as a fraction of the orbital separation.
    q = M_donor / M_companion."""
    q23 = q ** (2.0 / 3.0)
    return 0.49 * q23 / (0.6 * q23 + np.log(1.0 + q ** (1.0 / 3.0)))


def _common_envelope(a_i, m_ns, m_donor, m_core, alpha_lambda):
    """Webbink (1984) alpha-lambda common-envelope energy formalism.

    The donor's envelope (m_donor - m_core) is ejected at the cost of orbital
    energy, shrinking the orbit; the orbit also circularises.  Returns the post-CE
    separation in the same units as a_i.  NS1 accretes negligibly during the CE.
    """
    if m_core >= m_donor:                       # nothing to eject
        return a_i
    m_env  = m_donor - m_core
    q      = m_donor / m_ns                      # donor fills its Roche lobe at onset
    r_l    = _eggleton_rl(q)
    shrink = (m_core * m_ns) / (m_donor * (m_ns + 2.0 * m_env / (alpha_lambda * r_l)))
    return a_i * shrink


def _he_star_radius(m_he):
    """He-star (He-ZAMS) radius [sim length units].  Hurley, Pols & Tout (2000)."""
    r_rsun = 0.2391 * m_he ** 4.6 / (m_he ** 4 + 0.162 * m_he ** 5.231)
    return r_rsun * R_SUN_SIM


def _case_bb_rlo(a_i, m_ns, m_he, m_co, beta, n_steps=200):
    """Non-conservative Case BB RLO: the He star is stripped m_he -> m_co onto NS1.

    Integrates the circular-orbit angular-momentum balance for isotropic
    re-emission (matter NS1 does not accrete leaves with the accretor's specific
    orbital angular momentum).  `beta` is the accreted fraction.  Returns
    (a_f, m_ns_f); the orbit stays circular throughout.
    """
    if m_co >= m_he:                             # nothing transferred
        return a_i, m_ns
    a, m_d, m_a = a_i, m_he, m_ns                # separation, donor, accretor
    dmd = (m_co - m_he) / n_steps                # < 0: donor loses mass each step
    for _ in range(n_steps):
        m_tot = m_d + m_a
        dma   = -beta * dmd                      # accretor gains beta of transferred mass
        dmtot = dmd + dma                        # net mass lost from the system (< 0)
        # isotropic re-emission: lost matter carries the accretor's specific L_orb
        dj_over_j = (m_d / (m_tot * m_a)) * (1.0 - beta) * dmd
        da_over_a = 2.0 * dj_over_j - 2.0 * dmd / m_d - 2.0 * dma / m_a + dmtot / m_tot
        a   *= (1.0 + da_over_a)
        m_d += dmd
        m_a += dma
    return a, m_a


# Final (exploding) He-star mass vs pre-SN2 orbital period, from the Case BB RLO grid of
# Tauris+2017 (Table 5, M_He,i = 3 Msun): a wider orbit strips less, so the star explodes more
# massive.  M_He,f rises ~1.5 -> 2.4 Msun over 0.05 -> 90 d.  Interpolated in log-period.
_BB_LOGP = np.log10(np.array([0.05, 0.10, 0.31, 1.23, 4.61, 9.51, 20.0, 53.2, 87.1]))
_BB_MHEF = np.array([1.49, 1.58, 1.73, 1.80, 1.86, 2.00, 2.17, 2.32, 2.37])


def _case_bb_final_mass(period_days, m_he):
    """Exploding He-star mass after Case BB RLO vs pre-SN2 period (Tauris+2017 Table 5),
    scaled by CASE_BB_MASS_SCALE and clamped to [NS2+0.05, m_he]."""
    m = np.interp(np.log10(max(period_days, 1e-4)), _BB_LOGP, _BB_MHEF) * CASE_BB_MASS_SCALE
    return float(np.clip(m, NS2_MASS + 0.05, m_he))


def _mt2_physical(mass_ns, mass_donor, period_yr):
    """Common Envelope + Case BB RLO turning the wide eccentric HMXB into a tight,
    circular NS + ultra-stripped CO-core binary just before the 2nd SN.

    Returns (period_yr_new, mass_ns_new, mass_co, ecc_new=0.0), or None if the NS
    merges with the He core during the common envelope (failed CE -> no DNS).
    """
    # CE survival: only wide HMXBs eject the envelope; tighter ones spiral in and coalesce
    # (Thorne-Zytkow object) -> no DNS.  Survival probability ramps smoothly with period
    # (logistic in log10 P) so there is no artificial hard wall.  (Tauris+2017 sec 3.4.1-3.4.2.)
    if CE_PREPERIOD_WIDTH > 0:
        z = (np.log10(period_yr * 365.25) - np.log10(CE_MIN_PREPERIOD_DAYS)) / CE_PREPERIOD_WIDTH
        if np.random.random() > 1.0 / (1.0 + np.exp(-z)):
            return None
    elif period_yr * 365.25 < CE_MIN_PREPERIOD_DAYS:
        return None

    M_i = mass_ns + mass_donor
    a_i = (step.G * M_i * period_yr ** 2 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)

    # (1) common envelope: donor -> He core; orbit shrinks and circularises.
    # Two-channel CE (PERIOD_BRANCHES.md): draw alpha*lambda per system from a tight/wide
    # mix so both the tight (ultra-stripped) and wide (Case-BC) period branches form.
    m_he = min(HE_CORE_MASS, mass_donor)
    if CE_TWO_CHANNEL:
        alpha_lambda = CE_EFF_WIDE if np.random.random() < CE_WIDE_FRAC else CE_EFF_TIGHT
    else:
        alpha_lambda = CE_EFFICIENCY
    a_ce = _common_envelope(a_i, mass_ns, mass_donor, m_he, alpha_lambda)

    # CE merger: if the bare He core overflows its Roche lobe at the post-CE
    # separation, the NS spirals all the way in and merges -> no DNS forms.
    if CE_MERGE_CHECK:
        r_lobe_he = a_ce * _eggleton_rl(m_he / mass_ns)
        if _he_star_radius(m_he) >= r_lobe_he:
            return None

    # (2) Case BB RLO: He star stripped to its final pre-SN mass, then explodes.  The final
    # mass depends on orbital width (period-dependent stripping) -> this is what makes the SN
    # mass loss and the Blaauw eccentricity rise with period (the observed (P,e) correlation).
    if CASE_BB_PERIOD_DEPENDENT:
        period_ce = 2.0 * np.pi * np.sqrt(a_ce ** 3 / (step.G * (mass_ns + m_he)))
        m_co = _case_bb_final_mass(period_ce * 365.25, m_he)
    else:
        m_co = min(CO_CORE_MASS, m_he)
    a_bb, mass_ns_new = _case_bb_rlo(a_ce, mass_ns, m_he, m_co, CASEBB_ACCRETION)

    M_f        = mass_ns_new + m_co
    period_new = 2.0 * np.pi * np.sqrt(a_bb ** 3 / (step.G * M_f))
    period_new = max(period_new, MT2_PERIOD_FLOOR_DAYS / 365.25)
    return period_new, mass_ns_new, m_co, 0.0


def _second_kick(mass_ns1, mass_be, ecc, period_yr):
    """
    Apply the second SN kick to the companion star (body 2).

    Parameters
    ----------
    mass_ns1  : float — first neutron star mass [M_sun]
    mass_be   : float — companion (pre-SN2) mass [M_sun]
    ecc       : float — eccentricity after first SN
    period_yr : float — orbital period after first SN [years]

    Returns (ecc_new, period_new, r_peri_new).
    ecc_new >= 1 means the system is unbound after the second kick.

    The kick is applied at a random orbital phase drawn uniformly in time
    (mean anomaly M ~ Uniform[0, 2π]), so eccentric orbits are sampled
    proportionally to time spent at each separation.
    """
    M_tot = mass_ns1 + mass_be
    a     = (step.G * M_tot * period_yr**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)

    # --- Random orbital phase (uniform in time) ---
    M_anom = np.random.uniform(0.0, 2.0 * np.pi)   # mean anomaly
    E      = _solve_kepler(M_anom, ecc)              # eccentric anomaly

    # True anomaly
    f = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )

    # Relative radius and velocity (orbit in xy-plane, periapsis along +x)
    r_mag = a * (1.0 - ecc * np.cos(E))
    p_sl  = a * (1.0 - ecc**2)                      # semi-latus rectum
    h_mag = np.sqrt(step.G * M_tot * p_sl)           # specific angular momentum
    v_r   = step.G * M_tot / h_mag * ecc * np.sin(f) # radial velocity component
    v_t   = h_mag / r_mag                             # tangential velocity component

    cos_f = np.cos(f); sin_f = np.sin(f)
    r_rel = r_mag * np.array([cos_f, sin_f, 0.0])
    v_rel = np.array([v_r * cos_f - v_t * sin_f,
                      v_r * sin_f + v_t * cos_f,
                      0.0])

    # Convert relative → individual (CoM frame)
    pos1 =  r_rel * (mass_be  / M_tot)   # NS1 position
    pos2 = -r_rel * (mass_ns1 / M_tot)   # companion position
    vel1 =  v_rel * (mass_be  / M_tot)   # NS1 velocity
    vel2 = -v_rel * (mass_ns1 / M_tot)   # companion velocity

    # --- Second SN kick applied to NS2 (formerly the companion) ---
    # Ultra-stripped SN natal kick (Tauris et al. 2017).  Kick SPEED in km/s:
    mass_ns2 = NS2_MASS
    if KICK2_DIST == "normal":
        # speed straight from a 1D normal N(mean, sigma), truncated at 0 (no negative speeds)
        vk2_kms = max(0.0, np.random.normal(KICK2_MEAN, KICK2_SIGMA))
    elif KICK2_DIST == "bimodal_normal":
        # two-component GAUSSIAN mixture (normal magnitudes, not Maxwellians):
        #   low (bulk):  N(KICK2_MEAN,      KICK2_SIGMA_LOW)   with prob KICK2_LOW_FRAC
        #   high (rare): N(KICK2_MEAN_HIGH, KICK2_SIGMA_HIGH)  otherwise
        if np.random.random() < KICK2_LOW_FRAC:
            vk2_kms = max(0.0, np.random.normal(KICK2_MEAN,      KICK2_SIGMA_LOW))
        else:
            vk2_kms = max(0.0, np.random.normal(KICK2_MEAN_HIGH, KICK2_SIGMA_HIGH))
    else:
        # Maxwellian: speed = magnitude of a 3D Gaussian velocity with per-axis sigma sig2
        if KICK2_DIST == "bimodal":
            sig2 = KICK2_SIGMA_LOW if np.random.random() < KICK2_LOW_FRAC else KICK2_SIGMA_HIGH
        else:  # "single"
            sig2 = KICK2_SIGMA
        vk2_kms = np.linalg.norm(np.random.normal(0, sig2, size=3))
    vk2_sim = vk2_kms * 1000.0 / 31.55

    if KICK2_ANGLE >= 180:
        kdir2 = np.random.normal(size=3)
        kdir2 /= np.linalg.norm(kdir2)
    else:
        kdir2 = _random_in_cone(np.array([0.0, 0.0, 1.0]), KICK2_ANGLE)

    vel2 += vk2_sim * kdir2

    # --- Post-second-kick orbital elements ---
    M_new   = mass_ns1 + mass_ns2
    r_new   = pos1 - pos2
    v_new   = vel1 - vel2
    h_new   = np.cross(r_new, v_new)
    r_norm  = np.linalg.norm(r_new)
    eps_new = 0.5 * np.linalg.norm(v_new)**2 - step.G * M_new / r_norm
    ecc_new = np.sqrt(max(0.0, 1.0 + 2.0 * eps_new * np.dot(h_new, h_new) / (step.G * M_new)**2))

    if eps_new < 0.0:
        a_new       = -step.G * M_new / (2.0 * eps_new)
        period_new  = 2.0 * np.pi * np.sqrt(a_new**3 / (step.G * M_new))
        r_peri_new  = np.dot(h_new, h_new) / (step.G * M_new * (1.0 + ecc_new))
    else:
        period_new  = float('inf')
        r_peri_new  = 0.0

    return ecc_new, period_new, r_peri_new


def _gw_evolve(a0, e0, m1, m2, t_years):
    """Evolve a binary's (a, e) under gravitational-wave emission for t_years.

    Integrates the Peters (1964) equations with an adaptive forward step.
    Returns (a_new, e_new, merged).  All quantities in simulation units.
    """
    beta = step.G**3 * m1 * m2 * (m1 + m2) / LIGHTSPEED_SIM**5
    a = a0
    e = min(max(e0, 0.0), 0.999)
    t = 0.0
    while t < t_years and a > MERGE_SEPARATION:
        ome2 = max(1.0 - e * e, 1e-9)
        dadt = -(64.0 / 5.0) * beta / a**3 * (1.0 + (73.0/24.0)*e*e + (37.0/96.0)*e**4) / ome2**3.5
        dedt = -(304.0 / 15.0) * beta * e / a**4 * (1.0 + (121.0/304.0)*e*e) / ome2**2.5

        # adaptive step: cap the fractional change in a (and e) per step
        rate = abs(dadt) / a
        if e > 1e-3:
            rate = max(rate, abs(dedt) / e)
        dt = (GW_STEP_FRAC / rate) if rate > 0.0 else (t_years - t)
        dt = min(dt, t_years - t)
        if dt <= 0.0:
            break

        a += dadt * dt
        e += dedt * dt
        t += dt
        if e < 0.0:
            e = 0.0          # circularised; GW keeps shrinking a (da/dt != 0 at e=0)
        if a <= MERGE_SEPARATION:
            break

    merged = a <= MERGE_SEPARATION
    return a, min(max(e, 0.0), 0.999), merged


def _peters_merger_time(a0, e0, m1, m2):
    """Peters (1964) gravitational-wave inspiral (coalescence) time [sim years].

    Time for a binary starting at (a0, e0) to shrink to merger under GW emission.
    Uses the standard closed form: the circular merger time scaled by an
    eccentricity factor.

        T_c(a0) = a0^4 / (4 beta)               (circular-orbit merger time)
        beta    = G^3 m1 m2 (m1+m2) / c^5       (same beta as _gw_evolve)
        T(e0)   = T_c * (1-e0^2)^(7/2) * g(e0)

    The (1-e^2)^(7/2) is the leading Peters factor (eccentric orbits have a small
    periastron and merge much faster); g(e0) is the Mandel (2021) fit to the exact
    Peters integral, accurate to a few percent over 0 <= e < 1.  Consistent by
    construction with the orbit-averaged da/dt, de/dt integrated in _gw_evolve.
    """
    beta = step.G**3 * m1 * m2 * (m1 + m2) / LIGHTSPEED_SIM**5
    Tc   = a0**4 / (4.0 * beta)
    e    = min(max(e0, 0.0), 0.999)
    g    = 1.0 + 0.27 * e**10 + 0.33 * e**20 + 0.2 * e**1000
    return Tc * (1.0 - e * e) ** 3.5 * g


def evolve_dns(ecc0, period0_years):
    """Apply the detectability snapshot + GW orbital decay (Peters 1964) to one DNS.

    Returns (period_years, ecc, dropped).  `dropped` is True when this system is NOT
    part of the observable sample -- either it has merged or it failed the snapshot
    capture / acceleration cuts; callers treat dropped exactly like a merger (remove it).

    Selection (see the DETECTABILITY block in the config above).  The recycled pulsar does
    NOT switch off at a hard age: its radio-active survival probability decays smoothly,
    e^(-t / RADIO_LIFETIME_MYR), so RADIO_LIFETIME_MYR is the e-folding time tau, not a cutoff.
    For a constant DNS birth rate the chance of catching a system before it merges is the
    radio-active integral up to its merger time, normalised by the never-merging maximum:

        t_max  = min(tau_merge,  Galactic age)              # can't observe past merger or the Galaxy
        p_snap = (integral_0^t_max e^(-t/tau) dt) / (integral_0^inf ...) = 1 - e^(-t_max / tau)

    Fast-merging tight systems (small t_max) get p_snap ~ t_max/tau (down-weighted); long-lived
    systems saturate to p_snap -> 1.  Given the system IS caught, its observed age follows the
    same radio-decay law truncated at t_max -- a truncated exponential on [0, t_max], scale tau
    (older = fainter), instead of a flat draw.  GW-evolve (a,e) to that age.
    """
    M_tot = NS1_MASS + NS2_MASS
    a0    = (step.G * M_tot * period0_years**2 / (4 * np.pi**2)) ** (1 / 3)

    if not APPLY_DETECTABILITY:
        # legacy: flat age window [0, radio life], remove if it happens to have merged
        age_window = min(GW_MAX_AGE_MYR, RADIO_LIFETIME_MYR) if APPLY_RADIO_SELECTION else GW_MAX_AGE_MYR
        t_years = np.random.uniform(0.0, age_window) * 1.0e6
        a_new, e_new, merged = _gw_evolve(a0, ecc0, NS1_MASS, NS2_MASS, t_years)
        if merged:
            return None, None, True
        return 2.0 * np.pi * np.sqrt(a_new**3 / (step.G * M_tot)), e_new, False

    # radio e-folding time tau (exponential pulsar death, not a hard cutoff).  If radio selection
    # is disabled, set tau huge so there is effectively no radio fading.
    tau       = RADIO_LIFETIME_MYR if APPLY_RADIO_SELECTION else 1.0e12
    tau_merge = _peters_merger_time(a0, ecc0, NS1_MASS, NS2_MASS) / 1.0e6   # -> Myr
    t_max     = min(tau_merge, GW_MAX_AGE_MYR)               # observable horizon: merger or Galaxy age

    # snapshot capture probability  p = 1 - e^(-t_max / tau)  (the radio-active fraction caught
    # before the binary merges); drop the system if not caught in this Galactic snapshot.
    p_snap = 1.0 - np.exp(-t_max / tau)
    u = np.random.random()
    if u > p_snap:
        return None, None, True

    # observed age ~ truncated exponential on [0, t_max] with scale tau (radio fades with age).
    # inverse-CDF sampling; note (1 - e^(-t_max/tau)) == p_snap, so reuse it.
    t_years = -tau * np.log(1.0 - np.random.random() * p_snap) * 1.0e6
    a_new, e_new, merged = _gw_evolve(a0, ecc0, NS1_MASS, NS2_MASS, t_years)
    if merged:
        return None, None, True                      # safety (age < tau_merge, so rare)
    period_new = 2.0 * np.pi * np.sqrt(a_new**3 / (step.G * M_tot))

    # optional acceleration-search bias: very tight orbits are harder to detect
    if ACCEL_BIAS:
        z = (np.log10(period_new * 365.25) - np.log10(ACCEL_P50_DAYS)) / ACCEL_WIDTH
        if np.random.random() > 1.0 / (1.0 + np.exp(-z)):
            return None, None, True                  # signal smeared away -> not detected

    return period_new, e_new, False


def main(run_idx=1):
    # --- Parameters ---
    dt           = 0.01
    total_time   = .01
    impulse_time = .01
    e = 0.0

    # --- Initial conditions ---
    mass2 = COMPANION_MASS

    mass_loss, period, is_low_mass = _draw_mass_loss_and_period()
    mass1 = NS1_MASS + mass_loss
    seperation = (step.G * (mass1 + mass2) * period**2 / (4 * np.pi**2)) ** (1/3)

    position1 = np.array([0.0, seperation, 0.0])
    position2 = np.array([0.0, 0.0,        0.0])

    v_rel = np.sqrt(step.G * (mass1 + mass2) * (1 + e) / seperation)
    velocity1 = np.array([v_rel * mass2 / (mass1 + mass2), 0.0, 0.0])
    velocity2 = np.array([-v_rel * mass1 / (mass1 + mass2), 0.0, 0.0])
    period0 = 2*np.pi*np.sqrt((seperation/(2 - seperation*v_rel**2/(step.G*(mass1+mass2))))**3 / (step.G*(mass1+mass2)))

    if CENTER_OF_MASS_FRAME:
        avr_V      = (velocity1*mass1 + velocity2*mass2) / (mass1 + mass2)
        velocity1 -= avr_V
        velocity2 -= avr_V

    M   = mass1 + mass2
    r0  = position1 - position2
    v0  = velocity1 - velocity2
    h   = np.cross(r0, v0)
    eps = 0.5*np.linalg.norm(v0)**2 - step.G*M/np.linalg.norm(r0)
    ecc = np.sqrt(max(0.0, 1 + 2*eps*np.linalg.norm(h)**2 / (step.G*M)**2))
    r_peri = np.dot(h, h) / (step.G * M * (1 + ecc))

    if VERBOSE:
        print(f"Run {run_idx:>3}  T={period:.4f} yr  →  sep={seperation:.1f}  pre e={ecc:.4f} ({orbit_label(ecc)})", end='')

    if RANDOM_PHASE:
        impulse_time += period0 * np.random.uniform(-1, 0)
        impulse_time = max(impulse_time, dt)
    impulse_step = int(impulse_time / dt)
    n_steps      = int(total_time / dt)

    if PROJECT_TO_ORBITAL_PLANE:
        e1 = r0 / np.linalg.norm(r0)
        e3 = h  / np.linalg.norm(h)
        e2 = np.cross(e3, e1)

    n_pre  = impulse_step if APPLY_KICK else n_steps
    n_post = n_steps - impulse_step if APPLY_KICK else 0

    if SHOW_MULTI_PLOT:
        m1_pre   = mass1
        com_pre  = np.zeros((n_pre,  3))
        com_post = np.zeros((n_post, 3))

    if SHOW_PLOT:
        if PROJECT_TO_ORBITAL_PLANE:
            proj1_pre  = np.zeros((n_pre,  2))
            proj2_pre  = np.zeros((n_pre,  2))
            proj1_post = np.zeros((n_post, 2))
            proj2_post = np.zeros((n_post, 2))
        else:
            pos1_pre  = np.zeros((n_pre,  3))
            pos2_pre  = np.zeros((n_pre,  3))
            pos1_post = np.zeros((n_post, 3))
            pos2_post = np.zeros((n_post, 3))

    max_z_pre  = 0.0
    max_z_post = 0.0
    impulse_pos1 = impulse_pos2 = None
    ecc_imp = None
    flag = 1

    if CHECK_DIST_PRE and (r_peri < DISTANCE_MIN or r_peri > DISTANCE_MAX):
        flag = 0

    # Figure 1A is an instantaneous-SN calculation. For batch stats, skip the
    # integrator and compute the post-kick orbital elements directly.
    if not SHOW_PLOT and not SHOW_MULTI_PLOT:
        if flag == 0:
            return {'ecc': ecc, 'ecc_imp': 100, 'separation': seperation, 'period': period, 'is_low_mass': is_low_mass, 'fate': 'pre_sn_filter'}

        if RANDOM_PHASE:
            theta = np.random.uniform(0, 2 * np.pi)
            position1 = _rotate_z(position1, theta)
            position2 = _rotate_z(position2, theta)
            velocity1 = _rotate_z(velocity1, theta)
            velocity2 = _rotate_z(velocity2, theta)

        if APPLY_KICK:
            speed = _draw_kick_speed()
            if KICK_ANGLE >= 180:
                kick_dir = np.random.normal(size=3)
                kick_dir /= np.linalg.norm(kick_dir)
            else:
                bias_dir = np.cross(velocity1, position1 - position2) if BIAS_DIRECTION == "orthogonal" else velocity1.copy()
                kick_dir = _random_in_cone(bias_dir, KICK_ANGLE)
            velocity1 -= speed * kick_dir
            mass1 = NS1_MASS

        M = mass1 + mass2
        r_imp = position1 - position2
        v_imp = velocity1 - velocity2
        h_imp = np.cross(r_imp, v_imp)
        eps_imp = 0.5 * np.linalg.norm(v_imp)**2 - step.G * M / np.linalg.norm(r_imp)
        ecc_imp = np.sqrt(max(0.0, 1 + 2 * eps_imp * np.linalg.norm(h_imp)**2 / (step.G * M)**2))

        if eps_imp < 0:
            semi_major_axis = -step.G * M / (2 * eps_imp)
            period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (step.G * M))
        else:
            period = float('inf')

        # First-SN outcome
        fate = 'bound'
        if ecc_imp >= 1.0:
            fate = 'unbound_sn1'        # first SN disrupts the binary
        elif CHECK_DIST_POST:
            r_peri_imp = np.dot(h_imp, h_imp) / (step.G * M * (1 + ecc_imp))
            if r_peri_imp < DISTANCE_MIN or r_peri_imp > DISTANCE_MAX:
                ecc_imp = 100
                fate = 'filter_sn1'     # removed by post-SN1 distance filter

        # HMXB stage = NS + massive companion, just after the first SN (before MT2 / 2nd kick)
        if ecc_imp is not None and 0.0 <= ecc_imp < 1.0:
            ecc_hmxb, period_hmxb = ecc_imp, period
        else:
            ecc_hmxb, period_hmxb = None, None

        # Second SN: apply kick to companion at random orbital phase.
        # Only runs on systems that are bound and pass the first-kick distance filter.
        ecc_presn2, period_presn2 = None, None
        if APPLY_KICK2 and ecc_imp is not None and 0.0 <= ecc_imp < 1.0:
            mass_ns1_eff = mass1
            mass_be_eff  = mass2
            ce_merged    = False
            if APPLY_MT2:
                if MT2_MODE == "physical":
                    # Common Envelope + Case BB RLO, from real stellar masses
                    res_mt2 = _mt2_physical(mass1, mass2, period)
                    if res_mt2 is None:
                        ce_merged = True            # NS merged with He core during CE
                    else:
                        period, mass_ns1_eff, mass_be_eff, ecc_imp = res_mt2
                else:
                    delta_m      = max(0.0, min(MT2_DELTA_M, mass_be_eff - NS2_MASS - 0.1))
                    mass_be_eff  -= delta_m
                    mass_ns1_eff += MT2_EFFICIENCY * delta_m
                    ecc_imp = np.clip(ecc_imp * MT2_ECC_FACTOR, 0.0, 0.999)
                    period  = max(period * MT2_PERIOD_FACTOR, MT2_PERIOD_FLOOR_DAYS / 365.25)
            if ce_merged:
                ecc_imp = 100               # failed CE: no DNS forms
                fate    = 'ce_merger'
            else:
                # pre-SN2 stage = NS + ultra-stripped He star, after MT2, before the 2nd kick
                ecc_presn2, period_presn2 = ecc_imp, period
                ecc_imp, period, r_peri_k2 = _second_kick(mass_ns1_eff, mass_be_eff, ecc_imp, period)
                if ecc_imp >= 1.0:
                    fate = 'unbound_sn2'    # second (ultra-stripped) SN disrupts the binary
                elif CHECK_DIST_POST and (r_peri_k2 < DISTANCE_MIN or r_peri_k2 > DISTANCE_MAX):
                    ecc_imp = 100
                    fate = 'filter_sn2'     # removed by post-SN2 distance filter
                else:
                    fate = 'bound'          # survives both SNe -> DNS (pre-GW)

        return {'ecc': ecc, 'ecc_imp': ecc_imp, 'separation': seperation, 'period': period,
                'is_low_mass': is_low_mass, 'fate': fate,
                'ecc_hmxb': ecc_hmxb, 'period_hmxb': period_hmxb,
                'ecc_presn2': ecc_presn2, 'period_presn2': period_presn2}

    # --- Integration loop (Velocity Verlet in step.py) ---
    for i in range(n_steps):
        if flag == 0:
            break
        if APPLY_KICK and i == impulse_step:
            impulse_pos1 = position1.copy()
            impulse_pos2 = position2.copy()
            speed = _draw_kick_speed()
            if KICK_ANGLE >= 180:
                kick_dir = np.random.normal(size=3)
                kick_dir /= np.linalg.norm(kick_dir)
            else:
                if BIAS_DIRECTION == "orthogonal":
                    bias_dir = np.cross(velocity1, position1 - position2)
                else:
                    bias_dir = velocity1.copy()
                kick_dir = _random_in_cone(bias_dir, KICK_ANGLE)
            delta_v = speed * kick_dir
            velocity1 += delta_v
            mass1 = NS1_MASS

            M = mass1 + mass2
            r_imp = position1 - position2
            v_imp = velocity1 - velocity2
            h_imp = np.cross(r_imp, v_imp)
            eps_imp = 0.5 * np.linalg.norm(v_imp)**2 - step.G * M / np.linalg.norm(r_imp)
            ecc_imp = np.sqrt(max(0.0, 1 + 2 * eps_imp * np.linalg.norm(h_imp)**2 / (step.G * M)**2))

            if STOP_EARLY and ecc_imp >= 1:
                flag = 0
                break
            if CHECK_DIST_POST:
                r_peri_imp = np.dot(h_imp, h_imp) / (step.G * M * (1 + ecc_imp))
                if r_peri_imp < DISTANCE_MIN or r_peri_imp > DISTANCE_MAX:
                    flag = 0
                    break
            if eps_imp < 0:
                semi_major_axis = -step.G * M / (2 * eps_imp)
                period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (step.G * M))
            else:
                semi_major_axis = float('inf')
                period = float('inf')

            if VERBOSE:
                print(f"  →  post e={ecc_imp} ({orbit_label(ecc_imp)})  Δv=[{delta_v[0]:.2f},{delta_v[1]:.2f},{delta_v[2]:.2f}]")

        position1, velocity1, position2, velocity2 = step.step(
            dt, mass1, velocity1, position1, mass2, velocity2, position2)

        if SHOW_MULTI_PLOT:
            if not APPLY_KICK or i < impulse_step:
                com_pre[i] = (position1 * m1_pre + position2 * mass2) / (m1_pre + mass2)
            else:
                com_post[i - impulse_step] = (position1 * mass1 + position2 * mass2) / (mass1 + mass2)

        if SHOW_PLOT:
            if PROJECT_TO_ORBITAL_PLANE:
                r_cm = (position1*mass1 + position2*mass2) / M
                r1   = position1 - r_cm
                r2   = position2 - r_cm
                z    = np.dot(position1 - position2, e3)
                if not APPLY_KICK or i < impulse_step:
                    proj1_pre[i]       = [np.dot(r1, e1), np.dot(r1, e2)]
                    proj2_pre[i]       = [np.dot(r2, e1), np.dot(r2, e2)]
                    max_z_pre = max(max_z_pre, abs(z))
                else:
                    proj1_post[i - impulse_step] = [np.dot(r1, e1), np.dot(r1, e2)]
                    proj2_post[i - impulse_step] = [np.dot(r2, e1), np.dot(r2, e2)]
                    max_z_post = max(max_z_post, abs(z))
            else:
                if not APPLY_KICK or i < impulse_step:
                    pos1_pre[i]  = position1
                    pos2_pre[i]  = position2
                else:
                    pos1_post[i - impulse_step] = position1
                    pos2_post[i - impulse_step] = position2

    if VERBOSE and not APPLY_KICK:
        print()

    # =============================================================================
    # RESULT
    # =============================================================================
    if (CHECK_DIST_PRE or CHECK_DIST_POST) and flag == 0:
        ecc_imp = 100

    if period > 10**10 and period < float('inf'):
        ecc_imp = 100

    result = {
        'ecc':         ecc,
        'ecc_imp':     ecc_imp,
        'separation':  seperation,
        'period':      period,
        'is_low_mass': is_low_mass,
    }
    if SHOW_MULTI_PLOT:
        result['com_pre']  = com_pre
        result['com_post'] = com_post

    if not SHOW_PLOT:
        return result

    # =============================================================================
    # PLOTTING  (see plot.py)
    # =============================================================================
    impulse_xy = None
    if APPLY_KICK and impulse_pos1 is not None and PROJECT_TO_ORBITAL_PLANE:
        r_cm_i     = (impulse_pos1*mass1 + impulse_pos2*mass2) / M
        impulse_xy = (np.dot(impulse_pos1 - r_cm_i, e1),
                      np.dot(impulse_pos1 - r_cm_i, e2))

    sim = {
        'project_2d':    PROJECT_TO_ORBITAL_PLANE,
        'apply_impulse': APPLY_KICK,
        'mass_loss':     mass_loss,
        'mass1':         mass1,
        'mass2':         mass2,
        'ecc':           ecc,
        'ecc_imp':       ecc_imp,
        'impulse_time':  impulse_time,
        'impulse_pos':   impulse_pos1,
        'impulse_xy':    impulse_xy,
        'n_steps':       n_steps,
        'impulse_step':  impulse_step,
        'dt':            dt,
        'total_time':    total_time,
        'p1_pre':  proj1_pre  if PROJECT_TO_ORBITAL_PLANE else pos1_pre,
        'p2_pre':  proj2_pre  if PROJECT_TO_ORBITAL_PLANE else pos2_pre,
        'p1_post': proj1_post if PROJECT_TO_ORBITAL_PLANE else pos1_post,
        'p2_post': proj2_post if PROJECT_TO_ORBITAL_PLANE else pos2_post,
    }

    plt_orbit.show_orbit(sim)
    return result


if __name__ == '__main__':
    _set_high_priority()
    _t0 = time.time()
    with multiprocessing.Pool(initializer=_worker_init) as pool:
        try:
            async_result = pool.map_async(main, range(1, ITERATIONS + 1))
            while not async_result.ready():
                async_result.wait(timeout=0.5)
            results = np.array(async_result.get())
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            print("\nInterrupted.")
            exit()

    flat = list(results.flat)
    total      = len(flat)

    pct = lambda n, d: f"{100*n/d:.1f}%" if d else "n/a"

    print(f"\nTotal: {ITERATIONS} run(s) in {time.time() - _t0:.2f} s")

    # --- Where each system is lost along the DNS formation channel ---
    from collections import Counter
    fates = Counter(r.get('fate', 'unknown') for r in flat)
    _stage_labels = [
        ('pre_sn_filter', 'Lost: pre-SN distance filter'),
        ('unbound_sn1',   'Lost: unbound by 1st SN'),
        ('filter_sn1',    'Lost: post-SN1 distance filter'),
        ('ce_merger',     'Lost: NS merged in common envelope (failed CE)'),
        ('unbound_sn2',   'Lost: unbound by 2nd (ultra-stripped) SN'),
        ('filter_sn2',    'Lost: post-SN2 distance filter'),
        ('bound',         'Survive both SNe -> DNS (pre-GW)'),
    ]
    print("\n--- Fate of each system ---")
    for key, label in _stage_labels:
        if fates.get(key, 0):
            print(f"  {label:<42}: {fates[key]:>6}  ({pct(fates[key], total)})")
    if fates.get('unknown', 0):
        print(f"  {'Lost: other / not classified':<42}: {fates['unknown']:>6}  ({pct(fates['unknown'], total)})")

    bound = [r for r in flat if r['ecc_imp'] is not None and 0 <= r['ecc_imp'] < 1]

    if APPLY_AGING:
        _t_age = time.time()
        aged = []
        n_merged = 0

        # DNS: both bodies are now neutron stars
        M_tot_age = NS1_MASS + NS2_MASS
        for r in bound:
                ecc0    = r['ecc_imp']
                period0 = r['period']                          # years
                a0      = (step.G * M_tot_age * period0**2 / (4 * np.pi**2)) ** (1/3)
                r_peri  = a0 * (1 - ecc0)

                # random Galactic age, then GW orbital decay (Peters 1964)
                period_new, e_new, merged = evolve_dns(ecc0, period0)

                if merged:
                    n_merged += 1
                    r['ecc_imp'] = 100         # merged: no longer observable as a DNS
                    continue

                a_new      = (step.G * M_tot_age * period_new**2 / (4 * np.pi**2)) ** (1/3)
                r_peri_new = a_new * (1 - e_new)

                ar = {
                    'ecc_natal':           ecc0,
                    'ecc_evolved':         e_new,
                    'period_natal_days':   period0    * 365.25,
                    'period_evolved_days': period_new * 365.25,
                    'r_peri_natal':        r_peri,
                    'r_peri_evolved':      r_peri_new,
                }
                aged.append(ar)

                if CHECK_DIST_POST and (r_peri_new < DISTANCE_MIN or r_peri_new > DISTANCE_MAX):
                    r['ecc_imp'] = 100
                elif period_new * 365.25 > MAX_OBSERVABLE_PERIOD_DAYS:
                    r['ecc_imp'] = 100        # wider than any observed DNS -> not observable
                else:
                    r['ecc_imp'] = e_new
                r['period'] = period_new

        n_aged = len(aged)
        if n_aged:
            ecc_n   = np.array([a['ecc_natal']           for a in aged])
            ecc_e   = np.array([a['ecc_evolved']          for a in aged])
            p_n     = np.array([a['period_natal_days']    for a in aged])
            p_e     = np.array([a['period_evolved_days']  for a in aged])
            delta_e = ecc_n - ecc_e

            sig_circ   = np.sum(delta_e > 0.05)
            n_bound    = n_merged + n_aged
            print(f"\n--- Post-SN GW decay  (Peters 1964, ages 0-{GW_MAX_AGE_MYR:.0f} Myr) "
                  f"[{time.time()-_t_age:.2f} s] ---")
            print(f"  DNS entering GW phase    : {n_bound}")
            print(f"  Merged / not detectable  : {n_merged}  ({pct(n_merged, n_bound)})")  # merged OR failed snapshot/accel cut
            print(f"  Surviving DNS (observable): {n_aged}  ({pct(n_aged, n_bound)})")
            print(f"  Mean natal  ecc          : {ecc_n.mean():.4f}")
            print(f"  Mean evolved ecc         : {ecc_e.mean():.4f}")
            print(f"  Mean d_e (natal-evolved) : {delta_e.mean():.4f}  "
                  f"(median {np.median(delta_e):.4f})")
            print(f"  Significantly circularized (d_e>0.05): {sig_circ}  "
                  f"({pct(sig_circ, n_aged)})")
            print(f"  Mean natal  P            : {p_n.mean():.1f} days")
            print(f"  Mean evolved P           : {p_e.mean():.1f} days")

    # --- Final tally: how many systems end up as observable DNS ---
    final_dns = sum(1 for r in flat if r['ecc_imp'] is not None and 0 <= r['ecc_imp'] < 1)
    print(f"\n=== Final observable DNS: {final_dns}  /  {total}  ({pct(final_dns, total)}) ===")

    if SHOW_MULTI_PLOT:
        plt_orbit.show_multi(results, window=MULTI_PLOT_WINDOW)
    if SHOW_ECC_STATS:
        plt_orbit.show_ecc_stats(results)
    if SHOW_MERGER_TIME:
        plt_orbit.show_merger_time(results)
    if SHOW_INPUT_POPULATION:
        masses, periods, low_flags = generate_population_inputs(
            INPUT_POPULATION_SAMPLES
        )

        plt_orbit.show_population_inputs(
            masses,
            periods,
            low_flags
        )
                