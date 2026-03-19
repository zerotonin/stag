"""
STAG — Sensor-based Tracking and Analysis of Gait
==================================================

An unsupervised machine-learning pipeline for classifying farmed red deer
(*Cervus elaphus*) behaviour from wearable tri-axial accelerometer data.

The pipeline comprises five stages:

1. **Synchronisation** — align head and ear accelerometer streams via
   calibration-drop events.
2. **Database** — ingest synchronised sensor files into a structured
   SQLite database with SQLAlchemy ORM models.
3. **GPS / trajectory** — compute ground speed and path tortuosity from
   GPS fixes projected onto the New Zealand Map Grid.
4. **Clustering** — GPU-accelerated *k*-means (RAPIDS cuML) with
   contiguous leave-out stability analysis and Calinski–Harabasz scoring.
5. **Behavioural analysis** — transition matrices, bout statistics, and
   Hidden Markov Model super-prototypes from the cluster label sequence.

See Also
--------
Matthews, A. R. H., Matthews, L. R. & Geurten, B. R. H. (2026).
Behavioural Phenotyping in Red Deer: Machine Learning Classification of
Accelerometer Data from Micro-Movements to Grazing.
*Computers and Electronics in Agriculture*.
"""

__version__ = "0.1.0"
__author__ = "Alexander R. H. Matthews, Lindsay R. Matthews, Bart R. H. Geurten"
