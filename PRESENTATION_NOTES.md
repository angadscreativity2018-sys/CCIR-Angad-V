# DNS Population Synthesis — Notes to Talk From

## The science

Double neutron star (DNS) systems are binaries in which both stars have ended their lives as
neutron stars. They are rare — only about sixteen to nineteen are known in the Galaxy with a
well-measured orbital period and eccentricity — but they are scientifically valuable, because they
are the progenitors of the neutron-star mergers that LIGO detects in gravitational waves, and
because the shape of their orbits is a fossil record of how they formed. The orbital period and
eccentricity we observe today were set by two violent events in the binary's past — two supernovae —
and by the mass transfer that happened in between. So if we can build a model that reproduces the
observed spread of periods and eccentricities, that model is effectively measuring things we cannot
observe directly: the strength of the kicks neutron stars receive at birth, and the physics of the
common-envelope phase.

The formation channel I am modelling is the standard one. Two massive stars are born in a binary.
The more massive one evolves first, swells up, and transfers its envelope onto its companion before
exploding as the first supernova, leaving a neutron star. That neutron star then orbits its still-massive
companion as a high-mass X-ray binary. Eventually the companion itself expands and swallows the neutron
star in a common envelope — a dramatic phase in which the orbit shrinks by a factor of a hundred to a
thousand and is circularised. What is left is a tight binary of a neutron star and a stripped helium
star. The helium star transfers still more mass onto the neutron star (Case BB mass transfer), is
stripped down to a bare core, and finally explodes as a second, "ultra-stripped" supernova, producing
the second neutron star. The system is now a DNS, and from that moment it slowly spirals inward through
gravitational-wave emission until, eventually, it merges.

Two stages in that story do most of the work in setting the final orbit. The common envelope sets the
period — how hard it shrinks the orbit determines whether we end up with a tight sub-day binary or a
wide one. The second supernova sets the eccentricity — because the exploding star has been ultra-stripped,
it sheds very little mass and delivers only a modest kick, which is why most DNS are only mildly eccentric,
with a thin tail of high-eccentricity systems from the occasional larger kick. Every step in the model is
taken from published physics rather than tuned by hand: the common envelope uses the Webbink (1984)
energy formalism, the mass transfer follows Tauris (1996), the supernova kicks are drawn from the
distributions found in the three-dimensional explosion simulations of Müller, Tauris and collaborators
(2019), and the gravitational-wave decay uses the Peters (1964) equations.

## The problems I had to solve

The first problem was an artificial wall. Early on, the simulated systems refused to reach short
periods — there was a hard vertical edge around 1.2 to 1.5 days, and the observed short-period systems
sat behind it, in a region the model simply could not populate. Tracking the period through each stage
of the simulation showed the cause: a hard-coded one-day floor on the orbital period, combined with a
hard cutoff on which systems were allowed to survive the common envelope. Both were replaced with a
smooth, physically-motivated survival ramp, and the wall disappeared — the model now reaches down into
the observed sub-day systems.

The second, deeper problem was that the data actually show *two* groups of periods, not one. There is a
tight branch of systems clustered between roughly a tenth of a day and half a day, and a separate wide
branch stretching from a few days out to forty-five days. A single common-envelope efficiency can only
ever produce one peak — tune it tight and you lose the wide systems, tune it wide and you lose the close
ones. The resolution was to let the common envelope operate in two channels: each system draws its
shrinkage from either a strong-shrink or a weak-shrink mode. This is not an arbitrary fix — it reflects
the two distinct stripping regimes that Tauris and collaborators (2017) identify in the Case BB phase.
With the two-channel envelope in place, the model reproduces both period branches at once.

The third problem was selection bias, and this turned out to be more interesting than a nuisance. We do
not see DNS systems at random — we see them only while their recycled pulsar is still radio-bright, and
only out to periods of about forty-five days. To handle this honestly, I added a detectability weighting:
each system's radio visibility decays exponentially with age, systems that have already spiralled in and
merged are removed using the Peters merger time, and a hard observational cap is applied at forty-five
days. When the model is fit with this in place, it consistently prefers a *short* radio lifetime, of
order a few hundred million years. In other words, the model is telling us the observed sample is young —
caught early in its radio-active life, before gravitational-wave decay has had time to reshape its orbit.
That is a physical result in its own right, not just a correction.

## The result

Once those three pieces are in, the model matches the data well enough that it cannot be distinguished
from it statistically. The standard test here is the two-sample Kolmogorov–Smirnov statistic, which
measures the largest gap between the simulated and observed distributions; with a sample of nineteen
systems, the value above which we would reject the model is about 0.30. The fitted model sits at roughly
0.11 to 0.17 in period and around 0.11 in eccentricity — comfortably below that threshold for both. The
median simulated period of about 0.67 days matches the observed 0.63, and the median eccentricity of
about 0.21 matches the observed 0.18. Every observed system, including the three awkward high-eccentricity
short-period ones, falls inside the simulated cloud; those particular systems come from the rare
large-kick tail of the second supernova. The honest way to state the headline is that the model
reproduces the observed Galactic DNS period and eccentricity distributions to within the statistical
resolution of the current sample.

It is worth being clear about what is and is not pinned down. I ran the fit twice, independently, and the
two runs agreed closely on the physics that matters — the common-envelope efficiency, both supernova
kick strengths, the amount of mass lost in the first explosion, and the fraction of second supernovae
that deliver a large kick all came out the same. What the two runs disagreed on were the three quantities
that only shift the whole cloud left or right in period — the input period, the helium-core mass, and the
radio lifetime. These trade off against one another, so only their combination is constrained, not each
individually. This is exactly the behaviour you would expect, and it is reassuring: the scientifically
interesting quantities, the supernova kicks, are well determined by the data, while the period
normalisation is the part that would need an external prior to pin down further.

I also want to be honest about the limitations. The exact joint correlation between period and
eccentricity is not perfectly reproducible — the kicks smear it out, as Tauris notes — so I target the
two distributions separately rather than the full two-dimensional shape, which is achievable. With only
nineteen systems the high-eccentricity tail is genuinely noisy, and I deliberately stopped tuning once
both distributions fell below the rejection threshold, because pushing further would mean fitting the
statistical noise of nineteen points rather than any real physics. And while the DNS stage matches well,
the high-mass X-ray binary stage that precedes it matches only in the bulk and not in the tails, because
the model does not yet include the observational selection or the tidal circularisation that shape the
observed X-ray binary sample — that is a known next step, not something I am claiming to have solved.

## Figures to show

The central figure is the period–eccentricity scatter plot with the observed DNS systems overlaid,
flanked by the cumulative-distribution panels for period and eccentricity — that single figure carries
the whole result, showing both that the cloud sits in the right place and that each distribution matches.
It is generated by setting `OBS_DATA_SET = "dns"`, `SHOW_ECC_STATS = True` and `SHOW_PERIOD_CDF = True`
in `plot.py` and running `python main.py`. Beyond that, the period cumulative-distribution on its own is
worth showing to make the two-branch point — that the model reproduces both the tight and the wide
groups — and the eccentricity cumulative-distribution is the cleanest single panel for "the model
matches the data." A hand-drawn version of the formation channel from the first section is the right
thing to open with, to orient the audience, and if there is time the parameter-importance analysis from
`ml_sensitivity.py` makes a nice closing point about which physical knobs the data actually constrain.

The fuller technical records, for questions, are in `FIT_CAMPAIGN.md` (the fit results, best-fit
parameters and the degeneracy analysis), `PERIOD_BRANCHES.md` (the two-branch investigation and the
detectability derivation), and `DNS_WORK.txt` (the earlier work log).
