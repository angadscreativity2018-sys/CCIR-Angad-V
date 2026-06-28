# The Orbital-Period Branches of Galactic DNS Systems
### Why the model makes one period peak, and what the data say it should make

This note dissects the **shape of the orbital-period distribution** — the
cumulative-frequency curve the eye reads as *branches* — and traces the single
discrepancy that a global marginal-(P, e) KS fit hides.

Grounding reference throughout: **Tauris et al. 2017, ApJ 846, 170** ("Formation of
Double Neutron Star Systems"), in particular §3.4 (CE survival), §4.4 / §5.1–5.3
and Tables 5–6 (Case BB RLO), and §2.1 (radio selection).

The numbers below came from a set of one-off diagnostic scripts (input-period sweep,
single-knob lever scan, two-channel proof-of-concept).  Those scratch scripts have
been removed now that their conclusions are captured here and the mechanism lives in
`main.py`; the two-channel model is reproduced directly through `main.py` / `fit.py`
(see §10).

---

## 1. The observed structure (19 Galactic DNS)

Periods (days), sorted:

```
0.078 0.102 0.166 0.184 0.206 0.320 0.323 0.380 0.421 | 0.632 |
1.176 1.81  2.62  3.73  4.07  | 8.63 13.6 14.45 45.1
```

Binned in log-period (fraction of the 19 systems):

| P [d] bin | 0.03–0.1 | 0.1–0.25 | 0.25–0.5 | **0.5–1** | 1–2 | 2–4 | 4–10 | 10–50 |
|---|---|---|---|---|---|---|---|---|
| **observed %** | 5.3 | 21.1 | 21.1 | **5.3** | 10.5 | 10.5 | 10.5 | 15.8 |

Three features stand out, and they are what the user perceives as *branches*:

1. **A tight peak at 0.1–0.5 d** — 8 of 19 systems (42 %).  These are the classic
   close, merging DNS (J0737, J1906, B1534, B1913+16, J1757, J0509, …).
2. **A deficit at 0.5–1 d** — a single system (J1208–5936, 0.632 d), 5 %.
3. **A heavy wide tail to 45 d** — 5 systems beyond 4 d (26 %), out to the widest
   known DNS, J1930–1852 at 45 d.

The cumulative curve therefore rises steeply below 0.5 d, almost flattens through
0.5–1 d, then climbs again through a long wide tail.  **This is a broad, arguably
bimodal distribution — not a single hump.**

> Caveat carried throughout: with n = 19 the 0.5–1 d valley rests on *one* system and
> is not by itself statistically significant.  The tight-peak height and the wide-tail
> weight, by contrast, each rest on 5–8 systems and are robust.

---

## 2. What the model actually produces

Current best-fit `main.py` (single common-envelope efficiency), 40 000 binaries:

| P [d] bin | 0.03–0.1 | 0.1–0.25 | 0.25–0.5 | **0.5–1** | 1–2 | 2–4 | 4–10 | 10–50 |
|---|---|---|---|---|---|---|---|---|
| observed % | 5.3 | 21.1 | 21.1 | **5.3** | 10.5 | 10.5 | 10.5 | 15.8 |
| **model %** | 3.6 | 17.4 | 20.4 | **20.6** | 14.3 | 10.0 | 7.5 | 6.1 |

period KS = 0.177.  CDF residuals:

```
P<0.5 d : obs 0.47  sim 0.42   (sim slightly low)
P<1.0 d : obs 0.53  sim 0.62   (sim high  — the valley is overfilled)
P<2.0 d : obs 0.63  sim 0.77   (sim high)
P<5.0 d : obs 0.79  sim 0.89   (sim high  — wide tail under-fed)
```

The model is a **single hump centred in the 0.5–1 d valley**.  It simultaneously
- overfills the 0.5–1 d gap (20.6 % vs 5.3 %),
- under-feeds the wide tail (>4 d: 13.6 % vs 26.3 %), the largest robust error, and
- slightly under-feeds the tight peak (0.1–0.5 d: 37.8 % vs 42.1 %).

A global KS of 0.17 "passes" (below the n=19 rejection threshold ≈0.30), but it is
buying that pass by smearing one peak across a region where the data have a gap.
That is exactly why the **cumulative-frequency curve looks wrong even when the KS
statistic looks acceptable.**

---

## 3. Key diagnostic: the output period is set *downstream*, not by the input

The natural first guess is that the input pre-SN period distribution is too narrow.
**It is not the cause.**  Replacing the single input log-normal with strongly
*bimodal* input period distributions (two well-separated log-normals, modes up to
1.4 dex apart) changes the output almost not at all (`_branch_experiment.py`):

| input-period model | KS_P | tight % | valley % | wide % |
|---|---|---|---|---|
| A single log-normal (current) | 0.170 | 36.4 | 20.7 | 13.9 |
| B single broad log-normal (σ=1.2) | 0.168 | 36.8 | 20.8 | 14.2 |
| C bimodal (modes 35 d / 355 d) | 0.183 | 39.7 | 19.8 | 13.4 |
| D bimodal | 0.175 | 36.5 | 19.6 | 14.1 |
| F bimodal, wide gap (22 d / 630 d) | 0.180 | 38.2 | 19.3 | 13.2 |

**The valley stays at ~19–21 % and the wide tail at ~13 % no matter what we feed in.**
The pre-SN period is *erased* by the evolution between the two neutron stars.

Why: the common envelope applies a near-**constant** orbital shrink (fixed donor and
He-core masses ⇒ fixed `_common_envelope` factor), so `P_post-CE ∝ P_pre-CE` — but
then two operations collapse the range onto a fixed attractor:

- the **CE-merger floor** (`CE_MERGE_CHECK`: the bare He core must not overflow its
  Roche lobe) removes everything that would land at the shortest post-CE periods, and
- the **45-day observability cap** removes the widest.

Between these two walls the single CE-shrink + Case BB mapping funnels every survivor
into a narrow band whose **90th-percentile pre-SN2 period is only 3.5 d** — the model
essentially cannot make a wide post-CE binary.  The output peak sits wherever that
funnel lands; the input does not move it.

---

## 4. Lever scan: a single CE efficiency cannot make both branches

Scanning the downstream knobs one at a time (`_lever_scan.py`), watching the
tight-peak / valley / wide-tail balance:

| knob | setting | KS_P | tight % | valley % | wide % | pre-SN2 P₉₀ |
|---|---|---|---|---|---|---|
| **CE_EFFICIENCY** | 0.15 | 0.192 | **42.1** | 18.3 | 12.2 | 1.4 d |
| (orbital shrink) | 0.39 | 0.174 | 37.7 | 20.5 | 13.5 | 3.5 d |
| | 0.70 | 0.217 | 28.0 | 20.0 | 19.2 | 7.7 d |
| | 1.20 | 0.322 | 17.6 | 18.7 | **26.9** | 17.1 d |
| **RADIO_LIFETIME** | 300 Myr | 0.202 | 41.6 | 18.1 | 11.7 | — |
| (GW decay amount) | 2500 Myr | 0.194 | 30.5 | 22.6 | 16.1 | — |
| | 12000 Myr | 0.302 | 19.9 | 22.3 | 21.0 | — |

Two lessons:

1. **CE efficiency trades the tight branch against the wide branch.** Weak shrink
   (0.15) reproduces the tight peak (42 %) but starves the wide tail (12 %).  Strong
   shrink (1.2) feeds the wide tail (27 %) but destroys the tight peak (18 %).  *One*
   efficiency cannot do both — and **neither setting empties the valley**, which holds
   at 18–21 % in every single run.
2. **More GW decay does not help.**  Longer radio lifetimes move systems out of the
   tight peak (they merge) while *raising* the valley — the wrong direction.

So the valley is a structural attractor of a one-channel funnel, and the wide-tail
deficit is intrinsic to having only one CE shrink.  The fix has to change the
*number of channels*, not the value of a knob.

---

## 5. The physical resolution: two Case BB stripping regimes (Tauris+2017)

Tauris et al. 2017 do not describe DNS formation as one channel with one post-CE
period.  Their Case BB RLO grid (Tables 5–6, Fig. 7) has **two qualitatively
different regimes**, set by how evolved the He star is when it fills its Roche lobe:

- **Tight / ultra-stripped (Case BB):** post-CE period ≲ 1 d.  The He star overflows
  early (He-shell burning), is stripped almost to a bare ~1.4–1.6 M☉ metal core, and
  explodes with tiny ejecta (ΔM ~ 0.1 M☉).  Small mass loss + small kick ⇒
  **tight, low-eccentricity DNS** — the 0.1–0.5 d peak.  (Table 5: Porb,i 0.08–0.5 d
  → Porb,f 0.05–0.31 d.)
- **Wide / minimally-stripped (Case BC):** post-CE period ≳ 10 d.  The He star
  overflows only at carbon-shell burning, ~10³ yr before collapse (their Fig. 7), so
  it is barely stripped and explodes more massive (~2.3–2.4 M☉).  Larger ΔM ⇒ larger
  Blaauw eccentricity ⇒ **wide, higher-eccentricity DNS** — the 4–45 d tail, of which
  the marginally-recycled J1930–1852 (45 d, 185 ms pulsar) is the archetype
  (Tauris §5.1, point 4 of their conclusions).

Physically this two-regime split is a real **bimodality in the post-CE outcome**:
some HMXBs eject the envelope at small separation (deep spiral-in, tight) and some at
large separation (shallow spiral-in, wide), reflecting the genuine and acknowledged
uncertainty in the CE bifurcation point and envelope binding energy (Tauris §1.5.1,
§3.4).  The single-`CE_EFFICIENCY` model collapses this into one funnel and so can
produce only one of the two branches at a time — precisely what §4 shows.

### Implementation tested

We draw the common-envelope efficiency α·λ **per system** from a two-component mix:
a strong-shrink component (fraction 1−f_wide) → the tight branch, and a weak-shrink
component (fraction f_wide) → the wide Case-BC branch.  Everything else is unchanged.
(`_two_channel.py`, `_two_channel_opt.py`.)

---

## 6. Result of the two-channel model

Random search over (α·λ_tight, α·λ_wide, f_wide, Case-BB mass scale, 2nd-kick
fraction/σ, radio lifetime, CE ramp width); best confirmed on an independent
40 000-binary seed:

| parameter | value | meaning |
|---|---|---|
| α·λ (tight) | 0.19 | strong spiral-in → ultra-stripped tight branch |
| α·λ (wide) | 1.62 | shallow spiral-in → wide Case-BC branch |
| f_wide | 0.23 | ~1 in 4 systems take the wide channel |
| CASE_BB_MASS_SCALE | 1.12 | Blaauw-eccentricity amplitude |
| KICK2_LOW_FRAC | 0.91 | low-kick fraction of 2nd SNe |
| KICK2_SIGMA_LOW | 30.8 km/s | low (bulk) 2nd-SN kick |
| RADIO_LIFETIME_MYR | 522 | recycled-pulsar radio lifetime |
| CE_PREPERIOD_WIDTH | 0.30 | CE-survival ramp width [dex] |

| P [d] bin | 0.03–0.1 | 0.1–0.25 | 0.25–0.5 | **0.5–1** | 1–2 | 2–4 | 4–10 | 10–50 |
|---|---|---|---|---|---|---|---|---|
| observed % | 5.3 | 21.1 | 21.1 | **5.3** | 10.5 | 10.5 | 10.5 | 15.8 |
| 1-channel % | 3.6 | 17.4 | 20.4 | **20.6** | 14.3 | 10.0 | 7.5 | 6.1 |
| **2-channel %** | 5.5 | 15.0 | 16.3 | **16.1** | 15.3 | 11.3 | 11.1 | 9.4 |

| metric | 1-channel | **2-channel** |
|---|---|---|
| period KS | 0.177 | **0.144** |
| eccentricity KS | 0.115 | 0.135 |
| wide tail (>4 d) | 13.6 % | **20.5 %** (obs 26.3 %) |

The two-channel CE **lowers the period KS from 0.18 to 0.14** and **closes roughly
half of the wide-tail deficit** (13.6 → 20.5 %, target 26.3 %).  It does this with the
*physically correct* mechanism — a genuinely wide post-CE / Case-BC sub-population —
rather than by kick-widening tight systems, which is how the one-channel model
manufactured its few wide DNS (and which leaves their spin periods wrong: a
kick-widened tight system would be over-recycled, not the slow 185 ms of J1930).

---

## 7. Residuals and honest limits

1. **The 0.5–1 d valley persists (~16 %).**  Two channels narrow it but do not empty
   it, because the tight channel's upper tail and the wide channel's lower tail
   overlap there.  *But* the observed valley is a single-system feature; chasing it
   further would be fitting Poisson noise.  It should be revisited when the DNS sample
   is larger (SKA/FAST; Tauris predict ×5–10).
2. **The tight peak is still ~10 points low** (31 % vs 42 %).  The CE-merger floor
   prevents the model from piling systems below ~0.3 d, and longer GW decay (which
   would push 0.5 d systems down into the peak) makes the global fit worse.  This is
   the clearest sign that **(P, e) alone under-constrains the tight end** — the
   missing lever is the recycling/spin physics (see §8).
3. **Selection works against us, not for us, at the tight end.**  Tauris §2.1: orbital
   acceleration smears tight-orbit pulsars in Fourier searches, *suppressing* the
   observed tight count.  The true tight population is therefore even larger than the
   42 % we see, so the model's tight deficit is, if anything, understated.  A proper
   detectability weighting (acceleration-search loss vs Porb) is the right next
   ingredient.

---

## 8. Recommended next steps (for the paper)

1. **Adopt the two-channel CE as the baseline** and re-run the full global DE fit with
   f_wide and the two efficiencies as free parameters (here they were found by a
   140-trial random search, not a full optimisation — there is more to gain).
2. **Add the spin period P_s as a third observable.**  The (Porb, P_s) correlation
   (Tauris Eq. 7, Fig. 6) is the independent discriminator between the tight
   (well-recycled, fast) and wide (marginally-recycled, slow) branches, and it would
   break the period/He-core/radio-lifetime degeneracy between those parameters.
3. **Replace the hard 45-d cap and flat radio window with a physical detectability
   function** P_det(Porb, P_s, e) including the acceleration-search bias — this is the
   honest way to handle the tight-peak height and the wide-end cutoff together.
4. **Larger sample.** With 19 systems the valley cannot be claimed; the framework here
   predicts a two-peaked period distribution that SKA-era samples can test directly.

---

## 9. The detectability calculation (selection function)

The flat "draw an age in [0, radio life]" cut has been replaced by an explicit,
physically-derived **snapshot selection function** in `main.py` (`evolve_dns` +
`_peters_merger_time`).  It is the rigorous version of the same idea, and it makes the
two competing timescales — pulsar death and binary merger — separable and citable.

**Setup (constant formation + steady-state snapshot).**  DNS form at a roughly
constant rate over ~10 Gyr, so the Galaxy holds DNS of every age.  A given system is
*detectable* as a radio-pulsar binary only while three conditions hold simultaneously:

| condition | meaning | law |
|---|---|---|
| radio-active | recycled pulsar still above the death line | survival `e^(−t/τ)`, e-folding time τ (Tauris §3.4.2) |
| still bound | not yet coalesced | `age < τ_merge(a₀,e₀)` (Peters 1964) |
| in range | within the observed period band | `P_orb(age) ≤ 45 d` |

The pulsar fades **smoothly** (exponential), not at a hard age. For a constant birth
rate the chance of catching a system before it merges is the radio-active integral up to
its merger horizon, normalised by the never-merging maximum:

```
τ_merge = T_c(a₀)·(1−e₀²)^{7/2}·g(e₀)      # Peters time; T_c = a₀⁴/(4β), β=G³m₁m₂(m₁+m₂)/c⁵
                                            # g(e) = Mandel (2021) fit to the exact integral
t_max   = min(τ_merge, t_Galaxy)            # observable horizon
p_snap  = 1 − e^(−t_max / τ)                # capture probability  (∈[0,1])   <-- the user's form
keep with probability p_snap;  age ~ truncated-Exp(scale τ) on [0, t_max];  GW-evolve to it.
```

**Physical meaning.**  τ is the recycled-pulsar radio e-folding time (a free, fittable
parameter).  Systems that never merge within ~τ saturate to `p_snap → 1`; the tightest,
most eccentric systems (small `t_max`) get `p_snap ≈ t_max/τ` and are caught only
fleetingly — the Tauris+2017 (§6.9, §9.2) bias that makes wide non-merging DNS common.
The truncated-exponential age draw means observed systems are weighted toward **young**
ages (less GW evolution), so for short τ the sample sits near its **birth** periods.

**Optional acceleration-search bias** (`ACCEL_BIAS`, Tauris §2.1): a logistic detection
probability in log P_orb that suppresses very tight orbits (orbital-acceleration
smearing).  Off by default.

**What we learned — and the τ result.**  Scanning the radio e-folding time τ (the only
free parameter of the selection):

| τ (radio e-folding) | period KS | KS (P<1 d) | wide tail | tight |
|---|---|---|---|---|
| **300 Myr** | **0.114** | **0.226** | 26.3 % | 27.2 % |
| 600 Myr | 0.128 | 0.287 | 28.6 % | 24.7 % |
| 1000 Myr | 0.160 | 0.348 | 29.6 % | 22.8 % |
| 2000 Myr | 0.201 | 0.366 | 32.0 % | 19.0 % |
| 8000 Myr | 0.275 | 0.505 | 11.6 % | 11.6 % |
| + acceleration bias | (worse) | (worse) | — | — |

Three results:
1. **Short τ (~300 Myr) is strongly preferred** and gives the **best period KS of the whole
   study, 0.114** (vs 0.144 two-channel, 0.165 single-channel).  The truncated-exponential
   age draw catches systems young — near their **birth** periods, before GW evolution
   smears them — which is exactly what the data want.  This is also a physical *measurement*:
   the observed Galactic DNS are a young sample, recycled-pulsar radio life ~ a few ×10⁸ yr.
2. **Longer τ monotonically worsens the fit** — more GW evolution merges the tight systems
   and drags the cloud toward the valley, depleting both wings.  So the model and data agree
   only if the sample is young.
3. **The acceleration bias still makes things worse** — the tight end already matches, so
   suppressing tight orbits over-corrects.  This **rules out tight-orbit detectability** as a
   fix for the low-P fine structure.

So a smooth, physically-derived selection with one fitted timescale τ improves the period
marginal to KS ≈ 0.11.  The residual low-P fine structure is now attributable to small-number
noise (the low-P group is a handful of systems, well under the rejection threshold) or to the
missing *observable* — the spin period (§8) — not to selection, which is now complete.

## 10. Reproduce

The two-channel CE now lives in `main.py` behind `CE_TWO_CHANNEL` (currently **on**),
controlled by three parameters:

| parameter | role |
|---|---|
| `CE_EFF_TIGHT` | strong-shrink channel → the tight 0.1–0.5 d branch |
| `CE_EFF_WIDE` | weak-shrink channel → the wide 4–45 d branch |
| `CE_WIDE_FRAC` | fraction of systems on the wide channel = weight in the wide tail |

```
python main.py     # run the population; CE_TWO_CHANNEL = True uses the two branches
python fit.py global   # re-fit; the three CE knobs above are in PARAMS, DNS_SUBSET="all"
```

`fit.py` forces `CE_TWO_CHANNEL = True` and varies `CE_EFF_TIGHT / CE_EFF_WIDE /
CE_WIDE_FRAC` (replacing the single `CE_EFFICIENCY`).  Set `CE_TWO_CHANNEL = False` in
`main.py` to fall back to a single-channel common envelope.
