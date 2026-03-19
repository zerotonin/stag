Pipeline
========

This page describes the algorithmic details of each STAG pipeline stage.
For code examples, see :doc:`usage`.

Data collection
---------------

Thirty-six adult red deer stags (*Cervus elaphus*) were each fitted with
two battery-powered tri-axial accelerometers (±16 g, 50 Hz; TechnoSmart
Europe):

- **Head unit** (axyTrek) — mounted at the base of the antler pedicle;
  includes a 0.5 Hz GPS logger.
- **Ear unit** (axy) — attached to the ear.

Stags were monitored for 48 h in a paddock with *ad libitum* food and
water. Simultaneous video (30 Hz) served as ground truth.

Feature set
-----------

Six accelerometer axes (three head, three ear) are standardised to zero
mean and unit variance. GPS-derived speed and tortuosity are retained only
for coarse ground-truthing due to the 100× sampling-rate mismatch between
accelerometers (50 Hz) and GPS (0.5 Hz).

**Tortuosity** is defined as the arc–chord ratio over three consecutive
GPS fixes:

.. math::

   \\tau = \\frac{\\|\\vec{p}_1 - \\vec{p}_0\\| + \\|\\vec{p}_2 - \\vec{p}_1\\|}{\\|\\vec{p}_2 - \\vec{p}_0\\|}

A straight trajectory yields τ = 1; higher values indicate more sinuous
paths.

Clustering
----------

*k*-means (Lloyd's algorithm, squared Euclidean distance) partitions the
z-scored feature vectors into *k* prototypical movements. The GPU
implementation uses RAPIDS cuML for scalability.

**Model selection** evaluates k = 2–50 (24 settings × 200 independent
runs = 48 000 fits) using two criteria:

1. **Cluster quality** — Calinski–Harabasz index (higher is better).
2. **Cluster stability** — contiguous leave-out: a block of 1 − *r* of the
   time series is removed (r ∈ {0.50, 0.75, 0.90, 1.00}), the block is
   slid in 2 % steps, and centroid drift is measured via Hungarian
   assignment. Low drift = stable solution.

k = 8 was selected as a joint compromise of quality and stability.

Behavioural prototypes
----------------------

The eight prototypical movements (PM₀–PM₇) fall into three categories:

- **Inactive** (PM₀, PM₁, PM₃) — lying / standing, ± rumination /
  panting. 65.5 % of the time budget.
- **Grazing** (PM₆, PM₇) — grazing while walking, and stationary
  grazing.
- **Ear flicks** (PM₂, PM₄, PM₅) — rapid ear movements in response to
  irritants (e.g. flies), each < 1 s.

Hidden Markov Model
-------------------

The prototype label sequence is modelled as a first-order Markov chain.
**Super-prototypes** are frequent triplets of successive labels exceeding
a probability threshold, representing compound behaviours:

- *Grazing cycle*: stationary graze → step → stationary graze (the most
  frequent triplet).
- *Ear-flick bout*: PM₂ → PM₅ → PM₄ (a top-five triplet, condensing a
  flurry of ear movements into a single event).

Circadian analysis
------------------

Classified data are aggregated into hourly bins over a 24 h period
(second day only, after acclimation). Grazing peaks in morning, midday,
and evening; inactive states dominate overnight. Ear flicks show a marked
daytime peak, correlating with insect activity.

On-animal deployment
--------------------

Classification reduces to a nearest-centroid operation. On an Arduino Uno
(16 MHz ATmega328P) the C implementation achieves 4.3 × 10⁸
classifications per second — four orders of magnitude faster than the
50 Hz sensor sampling rate. Memory footprint is a few kilobytes.
