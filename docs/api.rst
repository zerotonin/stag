API Reference
=============

.. contents:: Subpackages
   :local:
   :depth: 1


stag.constants — Project-wide constants, paths and palettes
-----------------------------------------------------------

Central configuration module.  Reads machine-specific paths via
:mod:`stag.local_paths`; defines the manuscript's PM names, colours,
family categories, canonical labels, and helper functions for figure
output.

.. automodule:: stag.constants
   :members:
   :undoc-members:
   :show-inheritance:


stag.local_paths — Machine-specific path resolver
-------------------------------------------------

One source of truth for every absolute path STAG needs.  Resolves
keys via environment variable → ``local_paths.json`` → caller-supplied
default → :class:`LocalPathNotConfiguredError`.

.. automodule:: stag.local_paths
   :members:
   :undoc-members:
   :show-inheritance:


stag.sync — Sensor synchronisation
----------------------------------

Synchronise the head and ear accelerometer channels into a single
combined stream and write it to the deer database.

.. automodule:: stag.sync.data_sync
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.sync.utils
   :members:
   :undoc-members:
   :show-inheritance:


stag.database — Database models and ingestion
---------------------------------------------

SQLAlchemy ORM, schema management, and high-level ingestion / export
helpers.  See :class:`stag.database.handler.DeerDatabaseHandler` for
the entry-point class.

.. automodule:: stag.database.orm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.database.handler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.database.make_cluster_data
   :members:
   :undoc-members:
   :show-inheritance:


stag.gps — GPS trajectory analysis
----------------------------------

Project WGS84 GPS to the New Zealand Map Grid, compute per-sample
tortuosity and ground speed, and render trajectory + speed figures.

.. automodule:: stag.gps.analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.gps.tortuosity
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.gps.plotting
   :members:
   :undoc-members:
   :show-inheritance:


stag.clustering — k-means clustering and evaluation
---------------------------------------------------

GPU-accelerated k-means (cuML), meta-analysis across the leave-out
sweep, internal cluster-quality metrics (Calinski–Harabasz,
silhouette, Kneedle elbow, Hungarian-matched centroid stability),
and centroid-radar dashboards.

.. automodule:: stag.clustering.kmeans
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.clustering.internal_metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.clustering.meta_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.clustering.plotting
   :members:
   :undoc-members:
   :show-inheritance:


stag.analysis — Behavioural sequence analysis
---------------------------------------------

Per-sample label processing, NaN repair, first-order Markov
transition models, super-prototype detection with first-order Markov
shuffles + BH-FDR significance flagging, circadian rate-by-hour, and
per-animal time budgets.

.. automodule:: stag.analysis.label_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.analysis.nan_handler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.analysis.markov
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.analysis.null_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.analysis.super_prototypes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.analysis.circadian
   :members:
   :undoc-members:
   :show-inheritance:


stag.embedded — Bare-metal Q4.12 nearest-centroid classifier
------------------------------------------------------------

Q4.12 fixed-point export of the manuscript's k = 8 centroids into a
single C header consumable by every MCU in the Sprint 4 benchmark
matrix.  The C implementation itself lives in
``stag/embedded/nearest_centroid.{h,c}``.

.. automodule:: stag.embedded.export_centroids
   :members:
   :undoc-members:
   :show-inheritance:


stag.utils — Cross-cutting helpers
----------------------------------

Filename generators, ASCII-banner generators, and other small
utilities used across the package.

.. automodule:: stag.utils.banners
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.utils.csv_formatter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stag.utils.filename_generator
   :members:
   :undoc-members:
   :show-inheritance:
