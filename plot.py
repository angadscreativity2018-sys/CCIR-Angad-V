import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path


ONLY_SHOW_BOUNDED        = True   # filter multi-plot and stats to only runs that remain bound after the impulse
PERIOD_UNIT              = "days"  # "days" or "years" — x-axis unit in ecc stats plot
SHOW_ECC_HIST            = True   # show eccentricity CDF panel on the right of the stats plot
SHOW_PERIOD_CDF          = True   # show a period CDF panel BELOW the scatter (shares the log-P x-axis)
PERIOD_MIN               = .01       # min period shown on scatter x-axis (in PERIOD_UNIT); None = no limit
PERIOD_MAX               = 10000   # max period shown on scatter x-axis (in PERIOD_UNIT); None = no limit
SHOW_DENSITY_BACKGROUND  = False   # show hexbin density background in stats plot; only makes sense with a large number of runs
DENSITY_GRIDSIZE         = 55
SAVE_ECC_STATS           = False
ECC_STATS_FILENAME       = "Figures/figure_1a_maxwellian.png"
SHOW_OBSERVED_DATA       = True   # overlay observed Be/X-ray binary data on ecc stats plot
SHOW_SEPARATE_BRANCH_CDFS = True   # show separate CDFs for low-e and high-e observed systems in the histogram panel
SHOW_HMXB                = False   # overlay observed HMXB (orange) on the ecc stats plot
PLOT_STAGE               = "dns"  # simulated stage shown: "hmxb" (post-1st-SN), "presn2" (after MT2, pre-2nd-SN), "dns" (final)

# Restrict the (P, e) scatter AND the CDF panel to a period subset (sim + observed both filtered):
#   "all"  : every system
#   "low"  : short-period only, P <  PLOT_SUBSET_PSPLIT days
#   "high" : long-period only,  P >= PLOT_SUBSET_PSPLIT days
PLOT_SUBSET              = "all"  # "all", "low", or "high"
PLOT_SUBSET_PSPLIT       = 1.0    # period split [days] between low and high


OBS_DATA_SET = "dns"   # "original", "additional", "both", "dns", "hmxb"


def _in_plot_subset(period_days):
    """True if a system's period falls in the selected PLOT_SUBSET band."""
    if PLOT_SUBSET == "low":
        return period_days < PLOT_SUBSET_PSPLIT
    if PLOT_SUBSET == "high":
        return period_days >= PLOT_SUBSET_PSPLIT
    return True

# High-mass X-ray binaries (NS + massive companion) — the progenitor stage that
# evolves through common envelope into a DNS.  (orbital period [days], eccentricity).
# Mix of supergiant HMXBs (short period, low e) and Be/X-ray binaries (wide, eccentric).
_OBS_HMXB = [
    (1.408,  0.006),   # LMC X-4
    (2.087,  0.001),   # Cen X-3
    (3.412,  0.22),    # 4U 1700-377
    (3.728,  0.18),    # 4U 1538-522
    (3.892,  0.0004),  # SMC X-1
    (8.964,  0.090),   # Vela X-1
    (10.45,  0.107),   # OAO 1657-415
    (11.6,   0.18),    # 2S 0114+650
    (24.3,   0.34),    # 4U 0115+634
    (33.85,  0.37),    # V 0332+53
    (41.5,   0.462),   # GX 301-2
    (42.1,   0.42),    # 2S 1417-624
    (46.0,   0.41),    # EXO 2030+375
    (105.8,  0.14),    # GS 0834-430
    (111.1,  0.47),    # A 0535+262
    (172.7,  0.33),    # XTE J1946+274
    (249.5,  0.68),    # GRO J1008-57
]

_OBS_DNS = [
    (0.078, 0.064),   # J1946+2052
    (0.102, 0.088),   # J0737-3039
    (0.166, 0.085),   # J1906+0746
    #(0.184, 0.606),  # J1757-1854
    (0.206, 0.090),   # J1913+1102
    (0.320, 0.181),   # J1756-2251
    #(0.323, 0.617),  # B1913+16
    #(0.380, 0.586),  # J0509+3801
    (0.421, 0.274),   # B1534+12
    (0.632, 0.348),   # J1208-5936
    (1.176, 0.139),   # J1829+2456
    (1.81, 0.064),    # J1325-6253
    (2.62, 0.169),    # J1411+2551
    (3.73, 0.145),    # J0641+0448
    (4.07, 0.113),    # J0453+1559
    (8.63, 0.249),    # J1518+4904
    (13.6, 0.304),    # J1753-2240
    (14.45, 0.366),   # J1901+0658
    (45.1, 0.399),    # J1930-1852
]

_OBS_LOW_E_ORIGINAL = [
    (28.3,     0.092),
    (250.3,    0.111),
    (59.69,    0.06),
    (105.8,    0.14),
    (75.56,    0.03),
    (31.303,   0.0351),
    (132.89,   0.08),
    (37.97,    0.127),
    (22.5827,  0.0363),
    (172.7,    0.246),
    (40.415,   0.034),
]

_OBS_LOW_E_ADDITIONAL = [
    (17.13,    0.155),   # SXP 5.05
    (18.38,    0.070),   # SXP 2.37
]

_OBS_HIGH_E_ORIGINAL = [
    (24.3174,    0.339),
    (36.5,       0.417),
    (110.3,      0.47),
    (247.8,      0.68),
    (132.189,    0.524),
    (1236.724,   0.86988),
    (42.12,      0.4169),
    (29.806,     0.360),
    (17000,      0.961),
    (46.02217,   0.4102),
    (12.66536,   0.4055),
    (9.558,      0.30),
]

_OBS_HIGH_E_ADDITIONAL = [
    (21.9,       0.26),   # SXP 6.85
    (28.5,       0.41),   # SXP 8.80
    (36.3,       0.28),   # SXP 11.5
    (36.0,       0.30),   # SXP 15.6
    (33.4,       0.40),   # SXP 74.7
    (17.8,       0.43),   # SXP 18.3
    (137.4,      0.41),   # SXP 46.6
    (272.0,      0.57),   # SXP 504
]

if OBS_DATA_SET == "original":
    _OBS_LOW_E  = _OBS_LOW_E_ORIGINAL
    _OBS_HIGH_E = _OBS_HIGH_E_ORIGINAL
elif OBS_DATA_SET == "additional":
    _OBS_LOW_E  = _OBS_LOW_E_ADDITIONAL
    _OBS_HIGH_E = _OBS_HIGH_E_ADDITIONAL
else: 
    _OBS_LOW_E  = _OBS_LOW_E_ORIGINAL  + _OBS_LOW_E_ADDITIONAL
    _OBS_HIGH_E = _OBS_HIGH_E_ORIGINAL + _OBS_HIGH_E_ADDITIONAL



# =============================================================================
# HELPERS
# =============================================================================

def orbit_label(e):
    """Return a human-readable name for an orbit given its eccentricity."""
    if   e < 1e-6:        return "circular"
    elif e < 1.0 - 1e-6:  return "elliptical"
    elif e < 1.0 + 1e-6:  return "parabolic"
    else:                  return "hyperbolic"


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def show_orbit(sim):
    """Show the orbit — 2D orbital-plane or 3D lab-frame depending on sim['project_2d']."""
    if sim['project_2d']:
        _plot_2d(sim)
    else:
        _plot_3d(sim)


# =============================================================================
# 2D ORBITAL-PLANE VIEW
# =============================================================================

def _plot_2d(sim):
    ecc     = sim['ecc']
    ecc_imp = sim.get('ecc_imp', None)
    mass1   = sim['mass1']
    mass2   = sim['mass2']
    m1_orig = mass1 + sim['mass_loss']   # mass before the kick

    fig = plt.figure()
    plt.subplots_adjust(bottom=0.15)
    ax  = fig.add_subplot(111)

    stride = max(1, sim['n_steps'] // 5000)

    # Trails — pre blue/red, post green/orange
    ax.plot(sim['p1_pre'][::stride, 0],  sim['p1_pre'][::stride, 1],  '.', color='blue',   markersize=1, alpha=0.4, label=f'Body 1 pre  (m={m1_orig:.0f})')
    ax.plot(sim['p2_pre'][::stride, 0],  sim['p2_pre'][::stride, 1],  '.', color='red',    markersize=1, alpha=0.4, label=f'Body 2       (m={mass2:.0f})')
    ax.plot(sim['p1_post'][::stride, 0], sim['p1_post'][::stride, 1], '.', color='green',  markersize=1, alpha=0.4, label=f'Body 1 post (m={mass1:.0f})')
    ax.plot(sim['p2_post'][::stride, 0], sim['p2_post'][::stride, 1], '.', color='orange', markersize=1, alpha=0.4, label='Body 2 post')

    # Bridge lines connecting pre→post at the impulse boundary
    if sim['apply_impulse'] and len(sim['p1_post']) > 0:
        b1 = np.vstack([sim['p1_pre'][-1], sim['p1_post'][0]])
        b2 = np.vstack([sim['p2_pre'][-1], sim['p2_post'][0]])
        ax.plot(b1[:, 0], b1[:, 1], '-', color='blue',  linewidth=0.8, alpha=0.5)
        ax.plot(b2[:, 0], b2[:, 1], '-', color='red',   linewidth=0.8, alpha=0.5)

    # Centre-of-mass marker
    ax.plot([0], [0], '+', color='black', markersize=12, markeredgewidth=2, label='CoM')

    # Impulse star
    if sim['apply_impulse'] and sim['impulse_xy'] is not None:
        ix, iy = sim['impulse_xy']
        ax.plot([ix], [iy], '*', color='yellow', markersize=16,
                markeredgecolor='black', markeredgewidth=0.8, zorder=5,
                label=f"Impulse  t={sim['impulse_time']:.0f}s")

    # Axis labels and title
    ax.set_xlabel('e1  (initial radial direction)')
    ax.set_ylabel('e2  (in-plane transverse)')
    post_str = f'    Post: e={ecc_imp:.4f} ({orbit_label(ecc_imp)})' if ecc_imp is not None else ''
    ax.set_title(f'Two-body orbit — orbital plane projection\nPre: e={ecc:.4f} ({orbit_label(ecc)}){post_str}')
    ax.axis('equal')
    #ax.legend(loc='upper right', markerscale=5, fontsize=8)

    # Movable dots (position updated by slider)
    dot1, = ax.plot([], [], 'o', color='blue', markersize=8)
    dot2, = ax.plot([], [], 'o', color='red',  markersize=8)

    _attach_slider(sim, fig, dot1, dot2, mode='2d')
    plt.show()


# =============================================================================
# 3D LAB-FRAME VIEW
# =============================================================================

def _plot_3d(sim):
    ecc     = sim['ecc']
    ecc_imp = sim.get('ecc_imp', None)
    mass1   = sim['mass1']
    mass2   = sim['mass2']
    m1_orig = mass1 + sim['mass_loss']

    fig = plt.figure()
    plt.subplots_adjust(bottom=0.15)
    ax  = fig.add_subplot(111, projection='3d')

    stride = max(1, sim['n_steps'] // 5000)

    # Trails
    ax.plot(sim['p1_pre'][::stride, 0],  sim['p1_pre'][::stride, 1],  sim['p1_pre'][::stride, 2],  '.', color='blue',   markersize=1, alpha=0.4, label=f'Body 1 pre  (m={m1_orig:.0f})')
    ax.plot(sim['p2_pre'][::stride, 0],  sim['p2_pre'][::stride, 1],  sim['p2_pre'][::stride, 2],  '.', color='red',    markersize=1, alpha=0.4, label=f'Body 2       (m={mass2:.0f})')
    ax.plot(sim['p1_post'][::stride, 0], sim['p1_post'][::stride, 1], sim['p1_post'][::stride, 2], '.', color='green',  markersize=1, alpha=0.4, label=f'Body 1 post (m={mass1:.0f})')
    ax.plot(sim['p2_post'][::stride, 0], sim['p2_post'][::stride, 1], sim['p2_post'][::stride, 2], '.', color='orange', markersize=1, alpha=0.4, label='Body 2 post')

    # Bridge lines connecting pre→post at the impulse boundary
    if sim['apply_impulse'] and len(sim['p1_post']) > 0:
        b1 = np.vstack([sim['p1_pre'][-1], sim['p1_post'][0]])
        b2 = np.vstack([sim['p2_pre'][-1], sim['p2_post'][0]])
        ax.plot(b1[:, 0], b1[:, 1], b1[:, 2], '-', color='blue', linewidth=0.8, alpha=0.5)
        ax.plot(b2[:, 0], b2[:, 1], b2[:, 2], '-', color='red',  linewidth=0.8, alpha=0.5)

    # Impulse star
    if sim['apply_impulse'] and sim['impulse_pos'] is not None:
        p = sim['impulse_pos']
        ax.plot([p[0]], [p[1]], [p[2]], '*', color='yellow', markersize=16,
                markeredgecolor='black', markeredgewidth=0.8, zorder=5,
                label=f"Impulse  t={sim['impulse_time']:.0f}s")

    # Axis labels and title
    post_str = f'    Post: e={ecc_imp:.4f}' if ecc_imp is not None else ''
    ax.set_title(f'Two-body orbit — 3D lab frame\nPre: e={ecc:.4f}{post_str}')
    ax.set_xlabel('X');  ax.set_ylabel('Y');  ax.set_zlabel('Z')

    dot1, = ax.plot([], [], [], 'o', color='blue', markersize=8)
    dot2, = ax.plot([], [], [], 'o', color='red',  markersize=8)

    _attach_slider(sim, fig, dot1, dot2, mode='3d')
    plt.show()


# =============================================================================
# TIME SLIDER  — AI-generated, fairly technical
# Converts the slider's float time value into a step index, decides which
# phase (pre/post impulse) that index belongs to, looks up the right position
# array, and updates the body-marker colour and position without redrawing the
# full figure (draw_idle is cheaper than draw).
# =============================================================================

def _attach_slider(sim, fig, dot1, dot2, mode):
    dt           = sim['dt']
    n_steps      = sim['n_steps']
    impulse_step = sim['impulse_step']
    total_time   = sim['total_time']
    apply_imp    = sim['apply_impulse']

    def in_post_phase(idx):
        post = apply_imp and idx >= impulse_step
        j    = (idx - impulse_step) if post else idx
        return post, j

    ax_time = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider  = Slider(ax_time, 'Time (s)', 0, total_time - dt, valinit=0)

    if mode == '2d':
        def update(val):
            idx = min(int(val / dt), n_steps - 1)
            post, j = in_post_phase(idx)
            p1 = sim['p1_post'][j] if post else sim['p1_pre'][j]
            p2 = sim['p2_post'][j] if post else sim['p2_pre'][j]
            dot1.set_color('green' if post else 'blue')
            dot1.set_data([p1[0]], [p1[1]])
            dot2.set_data([p2[0]], [p2[1]])
            fig.canvas.draw_idle()
    else:
        def update(val):
            idx = min(int(val / dt), n_steps - 1)
            post, j = in_post_phase(idx)
            p1 = sim['p1_post'][j] if post else sim['p1_pre'][j]
            p2 = sim['p2_post'][j] if post else sim['p2_pre'][j]
            dot1.set_color('green' if post else 'blue')
            dot1.set_data([p1[0]], [p1[1]]);  dot1.set_3d_properties([p1[2]])
            dot2.set_data([p2[0]], [p2[1]]);  dot2.set_3d_properties([p2[2]])
            fig.canvas.draw_idle()

    update(0)
    slider.on_changed(update)
    # keep slider alive — matplotlib garbage-collects it otherwise
    ax_time._slider = slider


# =============================================================================
# MULTI-RUN CoM PLOT  (SHOW_MULTI_PLOT)
# Draws the centre-of-mass trajectory for every simulation run in one 3D plot.
# Pre-impulse segment is grey; post-impulse is colored by orbit category:
#   circular (e < 0.1) → royalblue
#   elliptical (0.1 ≤ e < 1) → gold
#   hyperbolic / diverging (e ≥ 1) → crimson
# =============================================================================

_CATEGORIES = [
    (0, 'circular',   'black'),
    (1.0, 'elliptical', 'black'),
    (None,'hyperbolic', 'gold'),
]

def _cat(e):
    for threshold, label, color in _CATEGORIES:
        if threshold is None or e < threshold:
            return label, color
    return _CATEGORIES[-1][1], _CATEGORIES[-1][2]


def _flat(results):
    """Flatten a 1D list or 2D numpy array of result dicts into a plain list."""
    arr = np.asarray(results)
    return arr.flatten().tolist()


def show_multi(results, window=500):
    """window: number of steps to show on each side of the impulse event."""
    flat = _flat(results)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')

    counts = {label: 0 for _, label, _ in _CATEGORIES}
    pre_plotted = False

    for r in flat:
        ecc          = r['ecc_imp'] if r['ecc_imp'] is not None else r['ecc']
        label, color = _cat(ecc)
        counts[label] += 1

        pre  = r['com_pre'][-window:]
        post = r['com_post'][:window]

        if not pre_plotted:   # all pre-trails are identical — draw once
            ax.plot(pre[:, 0], pre[:, 1], pre[:, 2],
                    color='grey', alpha=0.6, linewidth=1.2, label='pre-impulse')
            pre_plotted = True

        if len(post) > 0:
            ax.plot(post[:, 0], post[:, 1], post[:, 2],
                    color=color, alpha=0.7, linewidth=1.2)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color='grey', linewidth=1.5, label='pre-impulse')]
    for _, label, color in _CATEGORIES:
        if counts[label]:
            handles.append(Line2D([0], [0], color=color, linewidth=2,
                                  label=f'{label}  (n={counts[label]})'))
    ax.legend(handles=handles, loc='upper left', fontsize=9)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'Centre-of-mass trajectories — {len(flat)} runs\n'
                 f'post-impulse orbit type  |  window = ±{window} steps')
    plt.show()


# =============================================================================
# ORBIT TYPE STATISTICS  (show_ecc_stats)
# =============================================================================

def show_ecc_stats(results):
    flat = _flat(results)

    by_cat = {label: {'x': [], 'y': []} for _, label, _ in _CATEGORIES}
    for r in flat:
        if PLOT_STAGE == "hmxb":                       # post-first-SN (NS + massive star)
            ecc = r.get('ecc_hmxb')
            per = r.get('period_hmxb')
            if ecc is None or per is None:
                continue
        elif PLOT_STAGE == "presn2":                   # after MT2, before the 2nd SN (NS + He star)
            ecc = r.get('ecc_presn2')
            per = r.get('period_presn2')
            if ecc is None or per is None:
                continue
        else:                                          # "dns": final (post-2nd-SN, GW-aged)
            ecc = r['ecc_imp'] if r['ecc_imp'] is not None else r['ecc']
            per = r['period']
        if ONLY_SHOW_BOUNDED and ecc >= 1:
            continue

        scale = 365.25 if PERIOD_UNIT == "days" else 1.0
        period = per * scale
        if not np.isfinite(period) or period <= 0:
            continue
        if not _in_plot_subset(per * 365.25):          # restrict to the chosen period subset
            continue
        label, _ = _cat(ecc)
        by_cat[label]['x'].append(period)
        by_cat[label]['y'].append(ecc)

    ax_hist = None
    ax_pcdf = None
    if SHOW_ECC_HIST or SHOW_PERIOD_CDF:
        nrows = 2 if SHOW_PERIOD_CDF else 1
        ncols = 2 if SHOW_ECC_HIST else 1
        fig = plt.figure(figsize=(11, 6.5 if SHOW_PERIOD_CDF else 5))
        gs = fig.add_gridspec(nrows, ncols,
                              width_ratios=[4, 1][:ncols],
                              height_ratios=[4, 1.4][:nrows],
                              wspace=0.05, hspace=0.06)
        ax = fig.add_subplot(gs[0, 0])
        if SHOW_ECC_HIST:
            ax_hist = fig.add_subplot(gs[0, 1], sharey=ax)   # eccentricity CDF (shares ecc y-axis)
        if SHOW_PERIOD_CDF:
            ax_pcdf = fig.add_subplot(gs[1, 0], sharex=ax)   # period CDF (shares log-P x-axis)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))

    all_x = [x for cat in by_cat.values() for x in cat['x']]
    all_y = [y for cat in by_cat.values() for y in cat['y']]
    if SHOW_DENSITY_BACKGROUND and all_x:
        ax.hexbin(
            all_x, all_y,
            gridsize=DENSITY_GRIDSIZE,
            xscale='log',
            extent=(np.log10(PERIOD_MIN), np.log10(PERIOD_MAX), 0, 1),
            cmap='Greys',
            mincnt=1,
            linewidths=0,
            alpha=0.55,
        )

    total_sim = sum(len(cat['x']) for cat in by_cat.values())
    pt_size = max(1.0, min(40.0, 15000.0 / max(1, total_sim)))

    for _, label, color in _CATEGORIES:
        xs = by_cat[label]['x']
        ys = by_cat[label]['y']
        if xs:
            ax.scatter(xs, ys, c=color, label=f'{label}  (n={len(xs)})',
                       s=pt_size, alpha=0.35, linewidths=0)

    ax.set_xscale('log')
    ax.set_xlabel(f'Period ({PERIOD_UNIT})')
    ax.set_ylabel('Eccentricity')
    ax.legend(loc='upper left', fontsize=9)
    _stage = {'hmxb': 'HMXB (post-1st-SN)', 'presn2': 'Pre-2nd-SN (post-MT2)'}.get(PLOT_STAGE, 'DNS (final)')
    _sub = '' if PLOT_SUBSET == 'all' else f'  [{PLOT_SUBSET}-P, split {PLOT_SUBSET_PSPLIT:g}d]'
    ax.set_title(f'{_stage} eccentricity vs orbital period{_sub}  -  {len(flat)} runs')
    ax.set_ylim(-0.04, 1.05)
    if PERIOD_MIN is not None or PERIOD_MAX is not None:
        ax.set_xlim(left=PERIOD_MIN, right=PERIOD_MAX)

    if ax_hist is not None:
        all_ecc = np.sort([e for cat in by_cat.values() for e in cat['y'] if e < 1])
        if len(all_ecc):
            cdf = np.arange(1, len(all_ecc) + 1) / len(all_ecc)
            ax_hist.plot(cdf, all_ecc, color='black', linewidth=1.2, label='Simulation')

        if SHOW_HMXB and _OBS_HMXB:
            hmxb_ecc = np.sort([e for p, e in _OBS_HMXB if e < 1 and _in_plot_subset(p)])
            if len(hmxb_ecc):
                hmxb_cdf = np.arange(1, len(hmxb_ecc) + 1) / len(hmxb_ecc)
                ax_hist.step(hmxb_cdf, hmxb_ecc, color='darkorange', linewidth=1.5,
                             label=f'HMXB observed (n={len(hmxb_ecc)})')
                ax_hist.legend(fontsize=8)

        if SHOW_OBSERVED_DATA:
            if OBS_DATA_SET == "dns" and _OBS_DNS:
                dns_ecc = np.sort([e for p, e in _OBS_DNS if e < 1 and _in_plot_subset(p)])
                if len(dns_ecc):
                    dns_cdf = np.arange(1, len(dns_ecc) + 1) / len(dns_ecc)
                    ax_hist.step(dns_cdf, dns_ecc, color='limegreen', linewidth=1.5,
                                 label=f'DNS observed (n={len(dns_ecc)})')
                    ax_hist.legend(fontsize=8)
            elif OBS_DATA_SET == "hmxb" and _OBS_HMXB:
                h_ecc = np.sort([e for p, e in _OBS_HMXB if e < 1 and _in_plot_subset(p)])
                if len(h_ecc):
                    h_cdf = np.arange(1, len(h_ecc) + 1) / len(h_ecc)
                    ax_hist.step(h_cdf, h_ecc, color='darkorange', linewidth=1.5,
                                 label=f'HMXB observed (n={len(h_ecc)})')
                    ax_hist.legend(fontsize=8)
            else:
                obs_ecc = np.sort([e for p, e in (_OBS_LOW_E + _OBS_HIGH_E) if e < 1 and _in_plot_subset(p)])
                if SHOW_SEPARATE_BRANCH_CDFS:
                    low_ecc  = np.sort([e for p, e in _OBS_LOW_E  if e < 1 and _in_plot_subset(p)])
                    high_ecc = np.sort([e for p, e in _OBS_HIGH_E if e < 1 and _in_plot_subset(p)])
                    if len(low_ecc):
                        low_cdf  = np.arange(1, len(low_ecc)  + 1) / len(low_ecc)
                        ax_hist.step(low_cdf,  low_ecc,  color='red',  linewidth=1.5, label='Observed low-e')
                    if len(high_ecc):
                        high_cdf = np.arange(1, len(high_ecc) + 1) / len(high_ecc)
                        ax_hist.step(high_cdf, high_ecc, color='blue', linewidth=1.5, label='Observed high-e')
                if len(obs_ecc):
                    obs_cdf = np.arange(1, len(obs_ecc) + 1) / len(obs_ecc)
                    ax_hist.step(obs_cdf, obs_ecc, color='grey', linewidth=1.2, label='Observed')
                ax_hist.legend(fontsize=8)

        ax_hist.set_xlabel('CDF')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(ax.get_ylim())
        ax_hist.tick_params(labelleft=False)

    if SHOW_HMXB and _OBS_HMXB:
        _hmxb = [(p, e) for p, e in _OBS_HMXB if _in_plot_subset(p)]
        if _hmxb:
            hx, hy = zip(*_hmxb)
            ax.scatter(hx, hy, c='darkorange', s=30, zorder=4, marker='^',
                       label=f'HMXB observed (n={len(_hmxb)})', edgecolors='white', linewidths=0.4)

    if SHOW_OBSERVED_DATA:
        if OBS_DATA_SET == "dns" and _OBS_DNS:
            _dns = [(p, e) for p, e in _OBS_DNS if _in_plot_subset(p)]
            if _dns:
                dx, dy = zip(*_dns)
                ax.scatter(dx, dy, c='limegreen', s=30, zorder=5, marker='D',
                           label=f'DNS observed (n={len(_dns)})', edgecolors='white', linewidths=0.4)
        elif OBS_DATA_SET == "hmxb" and _OBS_HMXB:
            _h = [(p, e) for p, e in _OBS_HMXB if _in_plot_subset(p)]
            if _h:
                hx, hy = zip(*_h)
                ax.scatter(hx, hy, c='darkorange', s=30, zorder=5, marker='^',
                           label=f'HMXB observed (n={len(_h)})', edgecolors='white', linewidths=0.4)
        else:
            _lowe  = [(p, e) for p, e in _OBS_LOW_E  if _in_plot_subset(p)]
            _highe = [(p, e) for p, e in _OBS_HIGH_E if _in_plot_subset(p)]
            if _lowe:
                lx, ly = zip(*_lowe)
                ax.scatter(lx, ly, c='red',  s=30, zorder=5, label='observed low-e',  edgecolors='white', linewidths=0.4)
            if _highe:
                hx, hy = zip(*_highe)
                ax.scatter(hx, hy, c='blue', s=30, zorder=5, label='observed high-e', edgecolors='white', linewidths=0.4)
        ax.legend(loc='upper left', fontsize=9)

    # --- period CDF panel (below the scatter, shares the log-P x-axis) ---
    if ax_pcdf is not None:
        sim_p = np.sort([x for x in all_x if x > 0])
        if len(sim_p):
            pcdf = np.arange(1, len(sim_p) + 1) / len(sim_p)
            ax_pcdf.plot(sim_p, pcdf, color='black', lw=1.3, label='Simulation')
        if SHOW_HMXB and _OBS_HMXB:
            hp = np.sort([p for p, e in _OBS_HMXB if _in_plot_subset(p)])
            if len(hp):
                ax_pcdf.step(hp, np.arange(1, len(hp) + 1) / len(hp), where='post',
                             color='darkorange', lw=1.6, label=f'HMXB obs (n={len(hp)})')
        if SHOW_OBSERVED_DATA:
            if OBS_DATA_SET == "dns" and _OBS_DNS:
                dp = np.sort([p for p, e in _OBS_DNS if _in_plot_subset(p)])
                if len(dp):
                    ax_pcdf.step(dp, np.arange(1, len(dp) + 1) / len(dp), where='post',
                                 color='limegreen', lw=1.6, label=f'DNS obs (n={len(dp)})')
            elif OBS_DATA_SET == "hmxb" and _OBS_HMXB:
                hpp = np.sort([p for p, e in _OBS_HMXB if _in_plot_subset(p)])
                if len(hpp):
                    ax_pcdf.step(hpp, np.arange(1, len(hpp) + 1) / len(hpp), where='post',
                                 color='darkorange', lw=1.6, label=f'HMXB obs (n={len(hpp)})')
            else:
                op = np.sort([p for p, e in (_OBS_LOW_E + _OBS_HIGH_E) if _in_plot_subset(p)])
                if len(op):
                    ax_pcdf.step(op, np.arange(1, len(op) + 1) / len(op), where='post',
                                 color='grey', lw=1.4, label=f'Observed (n={len(op)})')
        ax_pcdf.set_xscale('log')
        ax_pcdf.set_xlabel(f'Period ({PERIOD_UNIT})')
        ax_pcdf.set_ylabel('CDF')
        ax_pcdf.set_ylim(0, 1)
        if PERIOD_MIN is not None or PERIOD_MAX is not None:
            ax_pcdf.set_xlim(left=PERIOD_MIN, right=PERIOD_MAX)
        ax_pcdf.legend(fontsize=8, loc='upper left')
        ax.set_xlabel('')                 # the period axis label now lives under the CDF panel
        ax.tick_params(labelbottom=False)

    plt.tight_layout()
    if SAVE_ECC_STATS:
        out = Path(ECC_STATS_FILENAME)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=250, bbox_inches='tight')
    plt.show()


def _merger_time_gyr(period_days, ecc, m1=1.4, m2=1.4):
    """GW merger time (Peters 1964) for a binary given its current orbit, in Gyr.

    T = (5/256) c^5 a^4 / (G^3 m1 m2 M) * (1-e^2)^(7/2)   (circular time x eccentricity factor)
    """
    G = 6.674e-11; c = 2.998e8; Msun = 1.989e30; day = 86400.0; yr = 3.156e7
    M  = (m1 + m2) * Msun
    P  = np.asarray(period_days, dtype=float) * day
    a  = (G * M * P**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
    Tc = 5.0 * c**5 * a**4 / (256.0 * G**3 * (m1 * Msun) * (m2 * Msun) * M)
    return Tc * (1.0 - np.asarray(ecc, dtype=float)**2) ** 3.5 / yr / 1e9


def show_merger_time(results):
    """GW merger time vs orbital period (left) plus a cumulative-frequency panel
    (right), with observed DNS overlaid on both."""
    flat = _flat(results)

    P, T = [], []
    for r in flat:
        ecc = r['ecc_imp'] if r['ecc_imp'] is not None else r['ecc']
        if ecc is None or ecc < 0 or ecc >= 1:
            continue
        pd = r['period'] * 365.25
        if not np.isfinite(pd) or pd <= 0:
            continue
        if not _in_plot_subset(pd):                    # restrict to the chosen period subset
            continue
        P.append(pd)
        T.append(_merger_time_gyr(pd, ecc))
    if not P:
        return
    P = np.array(P); T = np.array(T)

    ot = None
    if SHOW_OBSERVED_DATA and OBS_DATA_SET == "dns" and _OBS_DNS:
        _dns = [(p, e) for p, e in _OBS_DNS if _in_plot_subset(p)]
        if _dns:
            op = np.array([p for p, _ in _dns])
            oe = np.array([e for _, e in _dns])
            ot = _merger_time_gyr(op, oe)

    fig, (ax, ax_cdf) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
    fig.subplots_adjust(wspace=0.05)

    # --- scatter: merger time vs period ---
    pt = max(1.0, min(40.0, 15000.0 / len(P)))
    ax.scatter(P, T, s=pt, c='0.4', alpha=0.35, linewidths=0, label=f'Simulation (n={len(P)})')
    ax.axhline(13.8, color='crimson', lw=1.0, ls='--', label='Hubble time (13.8 Gyr)')
    if ot is not None:
        ax.scatter(op, ot, c='limegreen', s=35, marker='D', zorder=5,
                   edgecolors='white', linewidths=0.4, label=f'DNS observed (n={len(op)})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('GW merger time (Gyr)')
    ax.set_title('GW merger time vs orbital period')
    ax.legend(loc='upper left', fontsize=9)
    if PERIOD_MIN is not None or PERIOD_MAX is not None:
        ax.set_xlim(left=PERIOD_MIN, right=PERIOD_MAX)

    # --- cumulative-frequency panel (shares the log merger-time y-axis) ---
    Ts  = np.sort(T)
    cdf = np.arange(1, len(Ts) + 1) / len(Ts)
    ax_cdf.plot(cdf, Ts, color='black', lw=1.2, label='Simulation')
    if ot is not None:
        ots  = np.sort(ot)
        ocdf = np.arange(1, len(ots) + 1) / len(ots)
        ax_cdf.step(ocdf, ots, color='limegreen', lw=1.5, label=f'DNS observed (n={len(ots)})')
    ax_cdf.axhline(13.8, color='crimson', lw=1.0, ls='--')
    ax_cdf.set_xlabel('CDF')
    ax_cdf.set_xlim(0, 1)
    ax_cdf.tick_params(labelleft=False)
    ax_cdf.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def show_population_inputs(mass_losses, periods_days, low_flags):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Mass-loss distribution

    counts, bins = np.histogram(mass_losses, bins=60)
    centers = 0.5 * (bins[:-1] + bins[1:])

    axes[0,0].plot(centers, counts, linewidth=2)

    axes[0,0].set_title("Mass-loss Distribution")
    axes[0,0].set_xlabel("Mass loss (Msun)")
    axes[0,0].set_ylabel("Count")

    # Period distribution

# Period distribution

    low_periods  = periods_days[low_flags]
    high_periods = periods_days[~low_flags]

    log_low = np.log10(low_periods)
    log_high = np.log10(high_periods)

    log_bins = np.linspace(
        np.log10(periods_days.min()),
        np.log10(periods_days.max()),
        60
    )

    low_counts, edges = np.histogram(log_low, bins=log_bins)
    high_counts, _ = np.histogram(log_high, bins=log_bins)

    # simple smoothing kernel
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()

    low_counts = np.convolve(low_counts, kernel, mode='same')
    high_counts = np.convolve(high_counts, kernel, mode='same')

    centers = 10**(0.5 * (edges[:-1] + edges[1:]))

    axes[0,1].plot(
        centers,
        low_counts,
        color='red',
        linewidth=2,
        label=f'Low-mass-loss (n={len(low_periods)})'
    )

    axes[0,1].plot(
        centers,
        high_counts,
        color='blue',
        linewidth=2,
        label=f'High-mass-loss (n={len(high_periods)})'
    )

    axes[0,1].set_xscale("log")
    axes[0,1].set_title("Period Distribution")
    axes[0,1].set_xlabel("Period (days)")
    axes[0,1].set_ylabel("Density")
    axes[0,1].legend()
    # Leave room for future diagnostics
    axes[1,0].scatter(
    periods_days[low_flags],
    mass_losses[low_flags],
    s=.05,
    alpha=0.1,
    color='red',
    label='Low mass-loss'
)

    axes[1,0].scatter(
        periods_days[~low_flags],
        mass_losses[~low_flags],
        s=.05,
        alpha=0.1,
        color='blue',
        label='High mass-loss'
    )

    axes[1,0].set_xscale('log')
    axes[1,0].set_xlabel('Period (days)')
    axes[1,0].set_ylabel('Mass loss (Msun)')
    axes[1,0].set_title('Mass Loss vs Period')
    axes[1,0].legend()
    axes[1,1].axis("off")

    plt.tight_layout(pad=2)
    plt.show()