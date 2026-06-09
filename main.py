import numpy as np
import step
import plot as plt_orbit
from plot import orbit_label
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
VERBOSE                  = False  # print per-run info (slow for large ITERATIONS)

MULTI_PLOT_WINDOW        = 1000    # steps shown either side of the impulse


IMPULSE_MEAN             = 15*31.55 
# --- Multi-run sweep ---
ITERATIONS               = 100000   # runs per group; raise for a denser Figure-1A-style cloud
SEPERATION_GROUPS        = 1      # number of groups in the sweep; 1 when using period
USE_PERIOD               = True   # False → sweep separations;  True → draw periods
INITIAL_SEPARATION       = 600    # starting separation  [sim length units]
SEPARATION_SCALE         = 1.5    # geometric scale between groups  (group j = START × SCALE^j)
PERIOD_DRAW              = "paper_classical"  # "paper_classical", "log_uniform", or "log_normal_years"
PERIOD_DAYS_MIN          = 1      # paper lower truncation for pre-SN period [days]
PERIOD_DAYS_MAX          = 3000   # paper upper truncation for pre-SN period [days]
LOG10_PERIOD_DAYS_MEAN   = 1.8    # Table S3 Classical median: log10(P_preSN/day)
LOG10_PERIOD_DAYS_SIGMA  = .5    # approximate spread from Table S3 Classical interval

# --- Impulse (supernova kick) ---
APPLY_IMPULSE            = True   # whether to apply the impulse kick at all
RANDOM_PHASE             = True   # subtract 0–1 orbital period from impulse time each run
IMPULSE_DIST             = "normal"  # "beta" or "normal" or "maxwell"
IMPULSE_ALPHA            = 3.05   # beta: shape α
IMPULSE_BETA             = 14.6   # beta: shape β
IMPULSE_SCALE            = 4.5*31.55  # beta: speed = Beta(α,β) or SCALE as sigma for maxwell
    # normal: mean kick speed  [sim velocity units]
IMPULSE_SIGMA            = 2*31.55   # normal: sigma kick speed
POST_IMPULSE_MASS1       = 1.4    # body 1 remnant mass after kick  [M☉]  (e.g. neutron star)
MASS_LOSS_MEAN           = (IMPULSE_MEAN/31.55)*.12  # mean ejecta mass  [M☉]; initial mass = POST_IMPULSE_MASS1 + drawn loss
MASS_LOSS_SIGMA          =(IMPULSE_MEAN/31.55)*.12/3  # approximate Table S3 Classical spread; truncated below 0
KICK_ANGLE               = 1  # half-cone angle in degrees: 0 = always along bias dir, 180 = fully isotropic
BIAS_DIRECTION           = "velocity"  # "velocity" → along v1;  "orthogonal" → perpendicular to orbital plane

# --- Distance bounds ---
CHECK_DIST_PRE           = False  # filter on pre-SN periastron distance
CHECK_DIST_POST          = True   # filter on post-SN periastron distance
DISTANCE_MIN             = 21 #periastron distance must be greater than this to be included in results  [sim length units]
DISTANCE_MAX             = 313 #periastron distance must be less than this to be included in results  [sim length units]

STOP_EARLY = True  # stop if hyperbolic orbits are detected

# =============================================================================
# SIMULATION
# =============================================================================
def _draw_initial_period_years():
    """Draw the pre-impulse period used to set the initial circular orbit."""
    if PERIOD_DRAW == "paper_classical":
        while True:
            log_days = np.random.normal(LOG10_PERIOD_DAYS_MEAN, LOG10_PERIOD_DAYS_SIGMA)
            period_days = 10 ** log_days
            if PERIOD_DAYS_MIN <= period_days <= PERIOD_DAYS_MAX:
                return period_days / 365.25
    if PERIOD_DRAW == "log_uniform":
        lo = np.log10(PERIOD_DAYS_MIN / 365.25)
        hi = np.log10(PERIOD_DAYS_MAX / 365.25)
        return 10 ** np.random.uniform(lo, hi)
    if PERIOD_DRAW == "log_normal_years":
        return 10 ** np.random.normal(LOG10_PERIOD_DAYS_MEAN - np.log10(365.25), LOG10_PERIOD_DAYS_SIGMA)
    raise ValueError(f"Unknown PERIOD_DRAW: {PERIOD_DRAW}")


def _draw_kick_speed():
    """Draw kick speed in simulation velocity units."""
    if IMPULSE_DIST == "normal":
        return abs(np.random.normal(IMPULSE_MEAN, IMPULSE_SIGMA))
    if IMPULSE_DIST == "beta":
        return IMPULSE_SCALE * np.random.beta(IMPULSE_ALPHA, IMPULSE_BETA)
    if IMPULSE_DIST == "maxwell":
        return np.linalg.norm(np.random.normal(0, IMPULSE_SCALE, size=3))
    raise ValueError(f"Unknown IMPULSE_DIST: {IMPULSE_DIST}")


def _draw_mass_loss():
    """Draw ejecta mass with the paper's below-zero truncation."""
    if MASS_LOSS_SIGMA == 0:
        return MASS_LOSS_MEAN
    while True:
        mass_loss = np.random.normal(MASS_LOSS_MEAN, MASS_LOSS_SIGMA)
        if mass_loss >= 0:
            return mass_loss


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

def main(run_idx=1, seperation=None):
    # --- Parameters ---
    dt           = 0.01
    total_time   = .01
    impulse_time = .01
    e = 0

    # --- Initial conditions --- 
    mass2 = 15
    mass_loss = _draw_mass_loss()
    mass1 = POST_IMPULSE_MASS1 + mass_loss

    if USE_PERIOD:
        period = _draw_initial_period_years()
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
        if USE_PERIOD:
            print(f"Run {run_idx:>3}  T={period:.4f} yr  →  sep={seperation:.1f}  pre e={ecc:.4f} ({orbit_label(ecc)})", end='')
        else:
            print(f"Run {run_idx:>3}  sep={seperation:.1f}  →  T={period0:.4f} yr  pre e={ecc:.4f} ({orbit_label(ecc)})", end='')

    if RANDOM_PHASE:
        impulse_time += period0 * np.random.uniform(-1, 0)
        impulse_time = max(impulse_time, dt)
    impulse_step = int(impulse_time / dt)
    n_steps      = int(total_time / dt)

    if PROJECT_TO_ORBITAL_PLANE:
        e1 = r0 / np.linalg.norm(r0)
        e3 = h  / np.linalg.norm(h)
        e2 = np.cross(e3, e1)

    n_pre  = impulse_step if APPLY_IMPULSE else n_steps
    n_post = n_steps - impulse_step if APPLY_IMPULSE else 0

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
            return {'ecc': ecc, 'ecc_imp': 100, 'separation': seperation, 'period': period}

        if RANDOM_PHASE:
            theta = np.random.uniform(0, 2 * np.pi)
            position1 = _rotate_z(position1, theta)
            position2 = _rotate_z(position2, theta)
            velocity1 = _rotate_z(velocity1, theta)
            velocity2 = _rotate_z(velocity2, theta)

        if APPLY_IMPULSE:
            speed = _draw_kick_speed()
            if KICK_ANGLE >= 180:
                kick_dir = np.random.normal(size=3)
                kick_dir /= np.linalg.norm(kick_dir)
            else:
                bias_dir = np.cross(velocity1, position1 - position2) if BIAS_DIRECTION == "orthogonal" else velocity1.copy()
                kick_dir = _random_in_cone(bias_dir, KICK_ANGLE)
            velocity1 -= speed * kick_dir
            mass1 = POST_IMPULSE_MASS1

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

        if CHECK_DIST_POST and ecc_imp < 1:
            r_peri_imp = np.dot(h_imp, h_imp) / (step.G * M * (1 + ecc_imp))
            if r_peri_imp < DISTANCE_MIN or r_peri_imp > DISTANCE_MAX:
                ecc_imp = 100

        return {'ecc': ecc, 'ecc_imp': ecc_imp, 'separation': seperation, 'period': period}

    # --- Integration loop (Velocity Verlet in step.py) ---
    for i in range(n_steps):
        if flag == 0:
            break
        if APPLY_IMPULSE and i == impulse_step:
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
            mass1 = POST_IMPULSE_MASS1

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
            if not APPLY_IMPULSE or i < impulse_step:
                com_pre[i] = (position1 * m1_pre + position2 * mass2) / (m1_pre + mass2)
            else:
                com_post[i - impulse_step] = (position1 * mass1 + position2 * mass2) / (mass1 + mass2)

        if SHOW_PLOT:
            if PROJECT_TO_ORBITAL_PLANE:
                r_cm = (position1*mass1 + position2*mass2) / M
                r1   = position1 - r_cm
                r2   = position2 - r_cm
                z    = np.dot(position1 - position2, e3)
                if not APPLY_IMPULSE or i < impulse_step:
                    proj1_pre[i]       = [np.dot(r1, e1), np.dot(r1, e2)]
                    proj2_pre[i]       = [np.dot(r2, e1), np.dot(r2, e2)]
                    max_z_pre = max(max_z_pre, abs(z))
                else:
                    proj1_post[i - impulse_step] = [np.dot(r1, e1), np.dot(r1, e2)]
                    proj2_post[i - impulse_step] = [np.dot(r2, e1), np.dot(r2, e2)]
                    max_z_post = max(max_z_post, abs(z))
            else:
                if not APPLY_IMPULSE or i < impulse_step:
                    pos1_pre[i]  = position1
                    pos2_pre[i]  = position2
                else:
                    pos1_post[i - impulse_step] = position1
                    pos2_post[i - impulse_step] = position2

    if VERBOSE and not APPLY_IMPULSE:
        print()

    # =============================================================================
    # RESULT
    # =============================================================================
    if (CHECK_DIST_PRE or CHECK_DIST_POST) and flag == 0:
        ecc_imp = 100

    if period > 10**10 and period < float('inf'):
        ecc_imp = 100

    result = {
        'ecc':        ecc,
        'ecc_imp':    ecc_imp,
        'separation': seperation,
        'period':     period,
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
    if APPLY_IMPULSE and impulse_pos1 is not None and PROJECT_TO_ORBITAL_PLANE:
        r_cm_i     = (impulse_pos1*mass1 + impulse_pos2*mass2) / M
        impulse_xy = (np.dot(impulse_pos1 - r_cm_i, e1),
                      np.dot(impulse_pos1 - r_cm_i, e2))

    sim = {
        'project_2d':    PROJECT_TO_ORBITAL_PLANE,
        'apply_impulse': APPLY_IMPULSE,
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
    if USE_PERIOD:
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
    else:
        _sweep = [INITIAL_SEPARATION * SEPARATION_SCALE**j for j in range(SEPERATION_GROUPS)]
        results = np.array(
            [[main(i + 1, _sweep[j]) for i in range(ITERATIONS)] for j in range(SEPERATION_GROUPS)])

    OUT = sum(1 for r in results.flat if r['ecc_imp'] is not None and r['ecc_imp'] >= 1)
    print(f"\nTotal: {ITERATIONS} run(s) in {time.time() - _t0:.2f} s")
    print("OUT is:", OUT)
    if SHOW_MULTI_PLOT:
        plt_orbit.show_multi(results, window=MULTI_PLOT_WINDOW)
    if SHOW_ECC_STATS:
        plt_orbit.show_ecc_stats(results)
