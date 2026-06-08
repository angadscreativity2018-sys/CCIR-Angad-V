import numpy as np
import step
import plot as plt_orbit
from plot import orbit_label
import time
import multiprocessing

# =============================================================================
# UNIT SYSTEM  (G = 1 throughout)
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
CENTER_OF_MASS_FRAME     = True   # shift to CoM frame so the pair doesn't drift off screen
PROJECT_TO_ORBITAL_PLANE = True   # project onto orbital plane for 2D view  (single run only)

# --- Output ---
SHOW_PLOT                = False  # single-run orbit plot
SHOW_MULTI_PLOT          = False  # 3D CoM trajectory plot across all runs
SHOW_ECC_STATS           = True
VERBOSE                  = False  # print per-run info (slow for large ITERATIONS)

MULTI_PLOT_WINDOW        = 100    # steps shown either side of the impulse

# --- Multi-run sweep ---
ITERATIONS               = 500000  # runs per group
SEPERATION_GROUPS        = 1      # number of groups in the sweep; 1 when using period
USE_PERIOD               = True   # False → sweep separations;  True → draw periods
INITIAL_SEPARATION       = 600    # starting separation  [sim length units]
SEPARATION_SCALE         = 1.5    # geometric scale between groups  (group j = START × SCALE^j)
LOG10_PERIOD_MEAN        = -0.56  # mean of log10(period) distribution  (e.g. 0 → 1 yr)
LOG10_PERIOD_SIGMA       = 1      # sigma of log10(period) distribution

# --- Impulse (supernova kick) ---
APPLY_IMPULSE            = True   # whether to apply the impulse kick at all
RANDOM_PHASE             = True   # subtract 0–1 orbital period from impulse time each run
IMPULSE_DIST             = "normal"  # "beta" or "normal"
IMPULSE_ALPHA            = 3.05   # beta: shape α
IMPULSE_BETA             = 14.6   # beta: shape β
IMPULSE_SCALE            = 17730/3  # beta: speed = Beta(α,β) × SCALE
IMPULSE_MEAN             = 0      # normal: mean kick speed  [sim velocity units]
IMPULSE_SIGMA            = 8636   # normal: sigma kick speed
POST_IMPULSE_MASS1       = 1.4    # body 1 remnant mass after kick  [M☉]  (e.g. neutron star)
MASS_LOSS_MEAN           = 3.6    # mean ejecta mass  [M☉]; initial mass = POST_IMPULSE_MASS1 + drawn loss
MASS_LOSS_SIGMA          = 0.7    # sigma of ejecta draw; set 0 for a fixed loss equal to MASS_LOSS_MEAN
DIRECTIONAL_BIAS         = .9    # 0 to 1: fraction of kicks in the preferred direction (vs isotropic random)
BIAS_DIRECTION           = "velocity"  # "velocity" → along v1;  "orthogonal" → perpendicular to orbital plane

# --- Distance bounds ---
CHECK_DIST               = True
DISTANCE_MIN             = 40 #periastron distance must be greater than this to be included in results  [sim length units]
DISTANCE_MAX             = 600 #periastron distance must be less than this to be included in results  [sim length units]

STOP_EARLY = True  # stop if hyperbolic orbits are detected

# =============================================================================
# SIMULATION
# =============================================================================
def _worker_init():
    import signal, ctypes
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # workers ignore Ctrl+C; main process handles it
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)

def _set_high_priority():
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)

def main(run_idx=1, seperation=4500):
    # --- Parameters ---
    dt           = 0.01
    total_time   = .5
    impulse_time = .5
    e = 0

    # --- Initial conditions --- 
    mass2 = 15
    mass_loss = MASS_LOSS_MEAN if MASS_LOSS_SIGMA == 0 else abs(np.random.normal(MASS_LOSS_MEAN, MASS_LOSS_SIGMA))
    mass1 = POST_IMPULSE_MASS1 + mass_loss

    if USE_PERIOD:
        period = 10 ** np.random.normal(LOG10_PERIOD_MEAN, LOG10_PERIOD_SIGMA)
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

    if CHECK_DIST and (r_peri < DISTANCE_MIN or r_peri > DISTANCE_MAX):
        flag = 0

    # --- Integration loop (Velocity Verlet in step.py) ---
    for i in range(n_steps):
        if flag == 0:
            break
        if APPLY_IMPULSE and i == impulse_step:
            impulse_pos1 = position1.copy()
            impulse_pos2 = position2.copy()
            if IMPULSE_DIST == "normal":
                speed = np.linalg.norm(np.random.normal(0, IMPULSE_SIGMA, size=3))
            else:
                speed = IMPULSE_SCALE * np.random.beta(IMPULSE_ALPHA, IMPULSE_BETA)
            if BIAS_DIRECTION == "orthogonal":
                bias_dir = np.cross(velocity1, position1 - position2)
            else:
                bias_dir = velocity1.copy()
            bias_dir /= np.linalg.norm(bias_dir)
            random_dir = np.random.normal(size=3)
            random_dir /= np.linalg.norm(random_dir)
            kick_dir = (1 - DIRECTIONAL_BIAS) * random_dir + DIRECTIONAL_BIAS * bias_dir
            kick_dir /= np.linalg.norm(kick_dir)
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
            if CHECK_DIST:
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
    if CHECK_DIST and flag == 0:
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
