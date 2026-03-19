Usage
=====

STAG processes wearable accelerometer data through five sequential stages.
Each stage can be run independently once its inputs are available.

Stage 1 — Sensor synchronisation
---------------------------------

The head and ear accelerometers are aligned using calibration-drop events
(three controlled drops from 1.5 m recorded by both sensors simultaneously).

.. code-block:: python

   import pandas as pd
   from stag.sync.data_sync import BetterDataSync

   head_df = pd.read_csv("raw_data/R1_D1_head.csv")
   ear_df  = pd.read_csv("raw_data/R1_D1_ear.csv")

   syncer = BetterDataSync(
       deer_id="R1_D1",
       head_data=head_df,
       ear_data=ear_df,
       window_dict={"start": 0, "end": 50000},
       mkplot=True,
       plot_folder="plots/sync/",
   )
   syncer.run_synchronization()

Or via the command-line script for HPC submission:

.. code-block:: bash

   python scripts/run_sync_aoraki.py --deer_code R1_D1 --path_sys cluster_paths

Stage 2 — Database construction
--------------------------------

Synchronised ``.h5`` files are ingested into a SQLite database using the
SQLAlchemy ORM defined in :mod:`stag.database`.

.. code-block:: bash

   # Insert a single deer's data
   python stag/database/deer_info.py sqlite:///deer_data.db data/synced/R1_D1.h5

   # Batch insert via SLURM
   sbatch slurm/make_deer_db.sh

Stage 3 — GPS trajectory features
-----------------------------------

Ground speed and tortuosity are computed from GPS fixes projected onto the
New Zealand Map Grid (EPSG:27200).

.. code-block:: python

   from stag.gps.analysis import main as process_gps

   deer_df = process_gps("data/synced/R1_D1.h5")
   print(deer_df[["abs_speed_mPs", "tortuosity"]].describe())

Stage 4 — GPU clustering
--------------------------

*k*-means clustering with contiguous leave-out stability analysis. The
script accepts command-line arguments for integration with SLURM array
jobs.

.. code-block:: bash

   python stag/clustering/kmeans.py \
       -t deer8 -nc 8 -ds 0 -dp 0 -rs 0 \
       -df data/clust_data_deer8.npy \
       -sd results/clustering/

To sweep over the full parameter grid (k = 2–50, deletion sizes
0/10/25/50 %, deletion positions in 2 % steps):

.. code-block:: bash

   sbatch slurm/run_slurm_main_clustering.sh

Post-hoc model selection uses :class:`stag.clustering.meta_analysis.ClusterMetaAnalysis`
to evaluate Calinski–Harabasz quality and Hungarian-matched centroid
stability across all runs.

Stage 5 — Behavioural analysis
-------------------------------

The cluster label sequence is analysed for transition probabilities, bout
durations, and super-prototype motifs.

.. code-block:: python

   from stag.analysis.label_analysis import LabelAnalyser

   analyser = LabelAnalyser("results/clustering/labels.npy", fps=50)
   analyser.main(cutoff=2, save_path="results/label_analysis.json")

The output JSON contains per-centroid statistics (percentage, mean bout
duration ± SEM) and the full transition matrix.
