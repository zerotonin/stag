# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — package                                                  ║
# ║  « wearable-accelerometer behaviour pipeline »                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Five-stage pipeline:                                            ║
# ║    1. sync   — align head + ear accelerometers                   ║
# ║    2. database — SQLite + SQLAlchemy ORM                         ║
# ║    3. gps    — speed and tortuosity from NZMG positions          ║
# ║    4. clustering — GPU k-means with contiguous leave-out         ║
# ║    5. analysis — prevalence, bouts, Markov transitions           ║
# ║                                                                  ║
# ║  See Matthews, Matthews & Geurten (2026) for the                 ║
# ║  accompanying publication.                                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""STAG — Sensor-based Tracking and Analysis of Gait."""

__version__ = "0.1.0"
__author__ = "Alexander R. H. Matthews, Lindsay R. Matthews, Bart R. H. Geurten"
