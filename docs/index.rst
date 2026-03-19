STAG — Sensor-based Tracking and Analysis of Gait
==================================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

An unsupervised machine-learning pipeline for classifying farmed red deer
(*Cervus elaphus*) behaviour from wearable tri-axial accelerometer data.

STAG discovers prototypical movement patterns directly from sensor streams
using *k*-means clustering, chains them into higher-order behavioural
sequences via a Hidden Markov Model, and runs on a 16 MHz microcontroller
at over 4 × 10⁸ classifications per second.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   pipeline
   api
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
