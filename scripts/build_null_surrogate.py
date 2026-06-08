#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — scripts.build_null_surrogate                             ║
# ║  « one uniform MaxAbs-box surrogate per seed »                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Draws a single ``Uniform(-1, 1)^d`` surrogate of the same       ║
# ║  ``(n, d)`` shape as :data:`stag.constants.MAXABS_CLUSTERING_    ║
# ║  INPUT`` and saves it as a ``.npy`` next to the real input.      ║
# ║                                                                  ║
# ║  Idempotent: if the output already exists, the seed is skipped.  ║
# ║  Designed to be driven by ``slurm/build_stability_null_          ║
# ║  surrogates.sh`` as a 10-task CPU array (one seed per task).     ║
# ║                                                                  ║
# ║  Each draw is ~4.6 GB float32; the full 10-seed ensemble lands   ║
# ║  at ~46 GB on the surrogate cache directory.                     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Build one uniform MaxAbs-box surrogate for the R3 stability null."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stag.analysis.stability_null import build_uniform_surrogate
from stag.constants import MAXABS_CLUSTERING_INPUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, required=True,
        help="PRNG seed for the uniform draw (one .npy per seed).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write null_uniform_seed<SS>.npy into.",
    )
    parser.add_argument(
        "--reference", type=Path, default=MAXABS_CLUSTERING_INPUT,
        help="Real feature matrix used to determine the surrogate shape.",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=("float32", "float64"),
        help="Output dtype (default float32, halves disk + I/O).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"null_uniform_seed{args.seed:02d}.npy"

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Already exists, skipping: {out_path}")
        return

    print(f"Inspecting reference: {args.reference}")
    ref = np.load(args.reference, mmap_mode="r")
    n, d = ref.shape
    print(f"  reference shape = ({n}, {d}), dtype = {ref.dtype}")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    print(f"Drawing Uniform(-1, 1)^{d} surrogate, seed={args.seed} ...")
    surrogate = build_uniform_surrogate(n, d, args.seed, dtype=dtype)
    print(
        f"  surrogate shape = {surrogate.shape}, dtype = {surrogate.dtype}, "
        f"size = {surrogate.nbytes / 1e9:.2f} GB",
    )

    np.save(out_path, surrogate)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
