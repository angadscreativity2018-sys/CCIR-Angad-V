import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path


ONLY_SHOW_BOUNDED        = True   # filter multi-plot and stats to only runs that remain bound after the impulse
PERIOD_UNIT              = "days"  # "days" or "years" — x-axis unit in ecc stats plot
SHOW_ECC_HIST            = True   # show eccentricity histogram panel on the right of the stats plot
PERIOD_MIN               = 1       # min period shown on scatter x-axis (in PERIOD_UNIT); None = no limit
PERIOD_MAX               = 10000   # max period shown on scatter x-axis (in PERIOD_UNIT); None = no limit
SHOW_DENSITY_BACKGROUND  = False   # show hexbin density background in stats plot; only makes sense with a large number of runs
DENSITY_GRIDSIZE         = 55
SAVE_ECC_STATS           = False
ECC_STATS_FILENAME       = "Figures/figure_1a_maxwellian.png"



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
    (0.1, 'circular',   'red'),
    (1.0, 'elliptical', 'red'),
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
        ecc = r['ecc_imp'] if r['ecc_imp'] is not None else r['ecc']
        if ONLY_SHOW_BOUNDED and ecc >= 1:
            continue

        scale = 365.25 if PERIOD_UNIT == "days" else 1.0
        period = r['period'] * scale
        if not np.isfinite(period) or period <= 0:
            continue
        label, _ = _cat(ecc)
        by_cat[label]['x'].append(period)
        by_cat[label]['y'].append(ecc)

    if SHOW_ECC_HIST:
        fig, (ax, ax_hist) = plt.subplots(
            1, 2, figsize=(11, 5),
            gridspec_kw={'width_ratios': [4, 1]},
            sharey=True
        )
        fig.subplots_adjust(wspace=0.05)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax_hist = None

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

    for _, label, color in _CATEGORIES:
        xs = by_cat[label]['x']
        ys = by_cat[label]['y']
        if xs:
            ax.scatter(xs, ys, c=color, label=f'{label}  (n={len(xs)})',
                       s=.2, alpha=0.35, linewidths=0)

    ax.set_xscale('log')
    ax.set_xlabel(f'Period ({PERIOD_UNIT})')
    ax.set_ylabel('Eccentricity')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'Post-impulse eccentricity vs orbital period  -  {len(flat)} runs')
    ax.set_ylim(-0.04, 1.05)
    if PERIOD_MIN is not None or PERIOD_MAX is not None:
        ax.set_xlim(left=PERIOD_MIN, right=PERIOD_MAX)

    if ax_hist is not None:
        all_ecc = np.sort([e for cat in by_cat.values() for e in cat['y'] if e < 1])
        if len(all_ecc):
            cdf = np.arange(1, len(all_ecc) + 1) / len(all_ecc)
            ax_hist.plot(cdf, all_ecc, color='black', linewidth=1.2)
        ax_hist.set_xlabel('CDF')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylim(ax.get_ylim())
        ax_hist.tick_params(labelleft=False)

    plt.tight_layout()
    if SAVE_ECC_STATS:
        out = Path(ECC_STATS_FILENAME)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=250, bbox_inches='tight')
    plt.show()
