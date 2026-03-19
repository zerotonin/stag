# Data

This directory contains auxiliary data files used by the STAG pipeline.

## `deer_codes/`

Lookup tables mapping animal identifiers to recording sessions. Each CSV
contains deer codes in the format `R{rep}_D{deer}` (e.g. `R1_D1` through
`R4_D9`) across four recording repetitions of nine stags each (36 animals
total).

| File | Description |
|------|-------------|
| `Deer_codes.csv` | Full set of 36 deer codes |
| `Deer_codes_cutdown.csv` | Reduced subset used during development |
| `Deer_codes_problem.csv` | Animals flagged for data quality issues |
| `Deer_codes_without_first_2.csv` | Codes excluding the first two animals |
| `Deer_Codes_fromDB.csv` | Codes extracted from the SQLite database |

## Raw sensor data

Raw accelerometer (`.csv`) and trajectory (`.h5`) files are not included
in this repository due to their size. Contact the corresponding author
(bart.geurten@otago.ac.nz) for data access.

## `correlation_results.txt`

Pairwise correlation statistics from exploratory feature analysis.
