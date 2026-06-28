Population Generation

Each simulation generates a binary system consisting of a fixed 15 M☉ Be-star companion and a compact-object progenitor. The progenitor mass is

M_preSN = M_remnant + M_loss

where the mass loss is drawn from a truncated Gaussian. A bimodal population is supported: a branch flag selects between a low-mass-loss and a high-mass-loss distribution, with the branch probability set by SPLIT_RATIO.

Orbital Period Generation

Periods are drawn before the supernova and converted to separations via Kepler's third law.

When LINK_PERIOD_TO_MASS_LOSS = True the mean log-period is linearly interpolated between the two population means as a function of the drawn mass loss. A Gaussian scatter (PERIOD_LINK_SIGMA1/2) is applied around this mean. This imposes a controllable pre-SN period–mass-loss correlation.

Natal Kick Model

Kick magnitudes follow a Mandel-style prescription:

v_kick ∝ (M_CO − M_remnant) / M_remnant × (1 + scatter)

where scatter ~ Normal(0, KICK_SCATTER_SIGMA). The CO-core mass is approximated as a fixed fraction of the pre-SN stellar mass (CORE_MASS_FRAC_LOW / HIGH), with separate fractions for each population branch.

Kick directions are drawn uniformly within a cone of half-angle KICK_ANGLE around a bias axis. KICK_ANGLE = 0 gives perfectly aligned kicks; KICK_ANGLE = 180 gives isotropic kicks. The bias axis is either along the pre-kick orbital velocity ("velocity") or perpendicular to the orbital plane ("orthogonal").

Orbital Calculations

For population-synthesis runs the integrator is skipped. Post-SN orbital elements are computed analytically from the instantaneous state immediately after the kick:

- orbital energy → bound/unbound flag
- specific angular momentum → eccentricity
- periastron distance
- orbital period (bound systems only)

Selection Criteria

Systems are filtered by periastron distance before the SN (CHECK_DIST_PRE) and/or after the SN (CHECK_DIST_POST). Only systems with DISTANCE_MIN ≤ r_peri ≤ DISTANCE_MAX at each checked stage are retained.

Post-SN Aging

When APPLY_AGING = True, every bound system that passes the post-kick distance filter is evolved forward by AGING_TIME_MYR Myr using a periastron-dependent secular model.

A local evolution timescale is computed from the natal periastron distance:

tau_local = AGING_TAU_MYR × (r_peri / AGING_RPERI_REF) ^ AGING_GAMMA

Systems with small periastron distances have short tau_local and evolve strongly; wide systems have tau_local >> AGING_TIME_MYR and are almost unchanged.

The dimensionless evolution strength is:

factor = AGING_TIME_MYR / tau_local

Eccentricity and period then evolve as:

e_new = e × exp(−AGING_ECC_DAMPING × factor)
P_new = P × exp(AGING_PERIOD_GROWTH × factor) × exp(Normal(0, AGING_SCATTER))

AGING_ECC_DAMPING and AGING_PERIOD_GROWTH control the relative strength of eccentricity damping and period growth. AGING_SCATTER adds stochastic spread to the evolved period via a lognormal draw.

After aging the post-SN distance filter is re-applied to the evolved periastron distance. Systems that drift outside [DISTANCE_MIN, DISTANCE_MAX] are excluded from the output.

The scatter plot and CDF always show evolved orbital elements when APPLY_AGING = True.

Outputs

The primary output is a period–eccentricity scatter plot with a CDF panel. Simulated systems are compared against the observed Be/X-ray binary sample. Additional diagnostic plots show the input mass-loss and period distributions.

Console output reports:
- total run count and wall time
- fraction of unbound or filtered systems (total, low-e branch, high-e branch)
- bound systems with natal P < 11 days
- when aging is on: mean natal and evolved eccentricity, Δe statistics, period shift, and count with evolved P < 11 days
