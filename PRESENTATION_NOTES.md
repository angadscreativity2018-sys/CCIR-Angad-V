# DNS Population-Synthesis — Talk Notes

Speaking notes for presenting the double-neutron-star (DNS) modelling work.
Each section is what to *say*; figure call-outs are marked **[SHOW: …]**.

---

## 1. The question (1 slide)

- Galactic DNS systems are rare: **~16–19** known with measured orbital period **P** and
  eccentricity **e**. They span P = 0.078–45 days, e = 0.06–0.62.
- **Goal:** build a population-synthesis model of the formation channel that reproduces the
  observed (P, e) distribution *from first principles* (binary physics + supernova kicks),
  not by hand-placing points.
- Why it matters: the (P, e) distribution encodes the **supernova natal kicks** and the
  **common-envelope physics** — things we can't measure directly. Matching it constrains them.

---

## 2. The formation channel (1 slide — draw the diagram)

```
ZAMS binary
  → RLO (mass transfer 1)        primary stripped of its envelope
  → He-star + 1st SN → NS1       FIRST supernova  (natal kick 1)
  → HMXB                         neutron star + massive star, wide & eccentric
  → Common Envelope (CE)         companion engulfs NS; orbit shrinks 100–1000×, circularises
  → Case BB RLO                  He star ultra-stripped onto NS1 → bare CO core
  → ultra-stripped 2nd SN → NS2  SECOND supernova (small mass loss, small kick)
  → DNS                          → gravitational-wave orbital decay over its observable life
```

- **Talking point:** the final (P, e) is set by two levers — (a) how hard the **common envelope**
  shrinks the orbit (sets period), and (b) how much mass + kick the **ultra-stripped 2nd SN**
  delivers (sets eccentricity).
- Each arrow is a physics module in `main.py`: Webbink (1984) α-λ common envelope,
  Tauris (1996) Case BB mass transfer, Peters (1964) GW decay.

---

## 3. The model is physically grounded, not fit-by-eye (1 slide)

Every ingredient comes from a published prescription:

| Module | Physics | Reference |
|---|---|---|
| Common envelope | α-λ energy formalism | Webbink 1984 |
| Case BB stripping | isotropic re-emission, period-dependent | Tauris 1996, 2017 (Tables 5/6) |
| 1st & 2nd SN kicks | Maxwellian / bimodal natal kicks | Müller+2019; Vigna-Gómez+2018 |
| GW decay | Peters orbital-decay equations | Peters 1964 |
| Merger time | analytic Peters fit | Mandel 2021 |

- **Talking point:** the kick magnitudes and stripped masses are taken from 3D supernova
  simulations (Müller, Tauris et al. 2019), so the model reflects the real channel.

---

## 4. Three problems solved (the story of the work — 1–2 slides)

**(a) The artificial period "wall."**
Early plots showed a hard vertical wall at P ≈ 1.2–1.5 d that the observed short-period DNS sat
*behind*. Diagnosed it to a hard-coded 1-day period floor + a hard CE-survival cutoff. Replaced
both with a smooth logistic survival ramp → wall removed, model reaches the observed sub-day systems.

**(b) Two period branches.**
The data show **two** period groups — a tight branch (0.1–0.5 d) and a wide branch (4–45 d).
A single common-envelope efficiency can only make *one*. **Resolution:** a **two-channel CE**
(`CE_TWO_CHANNEL`) — each system draws α-λ from a tight-shrink or weak-shrink channel, motivated
by the two Case BB stripping regimes in Tauris+2017. This reproduces both branches at once.
**[SHOW: the two-branch period CDF — sim vs observed]**

**(c) Selection effects.**
The observed sample is biased: we only see DNS while the recycled pulsar is radio-loud, and only
out to P ≈ 45 d. Added a **detectability weighting**: radio survival ∝ e^(−t/τ), Peters merger
time removes systems that already merged, and a 45-day observational period cap. The fit *prefers
a short τ ≈ 300–900 Myr* — i.e. the observed sample is **young**, caught early in its radio life.
This is itself a physical result, not just a nuisance correction.

---

## 5. The result — the model matches the data (the money slide)

**[SHOW: (P, e) scatter with the 16–19 observed DNS overlaid + eccentricity CDF + period CDF panels]**

| metric | baseline | **fitted** | observed |
|---|---|---|---|
| period KS | 0.45 | **0.11–0.17** | — |
| eccentricity KS | 0.22 | **0.11** | — |
| median period | 3.4 d | **~0.67 d** | 0.63 d |
| median eccentricity | 0.17 | **0.21** | 0.18 |
| coverage | 18/19 | **19/19** | — |

- **Headline claim:** with n ≈ 19 systems the two-sample KS 95% rejection threshold is ≈ 0.30.
  Both fitted KS values are **well below** it → the simulated period and eccentricity distributions
  are **statistically indistinguishable** from the observed sample. The model cannot be rejected.
- All observed systems — including the three high-e short-period ones (e ≈ 0.6 at P ≈ 0.2–0.4 d) —
  lie inside the simulated cloud. Those come from the rare large-kick tail of the 2nd-SN kick.

---

## 6. What's robust vs what's degenerate (1 slide — shows rigour)

Two independent global fits agreed on the **kick/mass-loss physics** but not on the period
normalisation:

- **Well-constrained** (same in both runs): CE efficiency (~0.37–0.39), 1st-SN kick (~85–100 km/s),
  1st-SN mass loss (~3.85 M☉), 2nd-SN kick spread (~35 km/s), high-kick fraction (~0.8),
  Blaauw amplitude (~1.0).
- **Degenerate** (trade off against each other): input period, He-core mass, radio lifetime —
  they all just shift the cloud left/right, so only their *combination* is pinned.

- **Talking point:** this honesty is the point — the data constrain the **supernova kicks**, which
  is the scientifically interesting part; the period normalisation needs an external prior.
- *(Optional)* **[SHOW: parameter-importance bar chart from `ml_sensitivity.py`]** — which knob
  drives which part of the fit.

---

## 7. Honest limitations (1 slide — pre-empt the prof's questions)

- The **exact joint (P, e) correlation** isn't fully reproducible (Tauris §5.4 — kicks smear it);
  we target the two **marginals**, which *are* reproducible.
- Small-N: 19 systems means the high-e tail is noisy; we deliberately **stop fitting** once both
  marginals are below the n=19 rejection threshold — going further fits noise, not physics.
- HMXB stage matches in the bulk but not the tails (no HMXB-selection / tidal circularization yet)
  — a known next step, not claimed as solved.

---

## 8. Recommended figures (priority order)

1. **(P, e) scatter + period CDF + ecc CDF, DNS overlay** — *the* result figure.
   Generate: in `plot.py` set `OBS_DATA_SET = "dns"`, `SHOW_ECC_STATS = True`,
   `SHOW_PERIOD_CDF = True`; run `python main.py`.
2. **Two-branch period CDF** (sim reproduces tight + wide branches) — supports the two-channel-CE story.
3. **Eccentricity CDF, sim vs observed** (KS ≈ 0.11) — the cleanest "we match" panel.
4. **Formation-channel diagram** (section 2) — draw it; orients the audience.
5. *(Optional)* **Parameter-importance bars** from `ml_sensitivity.py` — rigour/sensitivity.

Avoid showing: the old `Figures/figure_1a_*smoke*` renders — those are throwaway test images.

---

## 9. One-line summary to land

> "A physically-grounded population-synthesis model of the Be/X-ray-binary → DNS channel reproduces
> the observed Galactic DNS period and eccentricity distributions to within the statistical
> resolution of the current sample, and in doing so constrains the supernova natal kicks."

---

### Source documents (for deeper Q&A)
- `FIT_CAMPAIGN.md` — full fit results, best-fit parameters, degeneracy analysis.
- `PERIOD_BRANCHES.md` — the two-period-branch investigation + detectability derivation.
- `DNS_WORK.txt` — earlier work log (left-wall fix, kick grounding).
