# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — embedded.export_centroids                                ║
# ║  « Python centroids → C header (Q4.12 fixed-point + float) »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Emits a self-contained C header file with two parallel          ║
# ║  representations of the k = 8 centroid set and the per-axis      ║
# ║  MaxAbs divisors:                                               ║
# ║                                                                  ║
# ║    Q4.12 fixed-point  — 16-bit signed, 4 integer + 12 frac bits, ║
# ║      range [-8, +8), step 1/4096 ≈ 2.4 × 10⁻⁴.  Used by the     ║
# ║      8/16-bit MCUs (AVR, MSP430) where hardware float is        ║
# ║      unavailable or expensive.  Arrays are emitted with the     ║
# ║      ``PROGMEM`` qualifier for the AVR variants; the macro is   ║
# ║      a no-op on non-AVR targets via ``#ifdef __AVR__``.         ║
# ║                                                                  ║
# ║    IEEE-754 float    — used by 32-bit MCUs with hardware FPU    ║
# ║      (Cortex-M4F / M7, ESP32 LX6).  No quantisation loss.       ║
# ║                                                                  ║
# ║  The header also carries the per-axis MaxAbs divisors so the    ║
# ║  on-MCU classifier can rescale raw sensor readings (which       ║
# ║  arrive in g-units) into the same [-1, +1] coordinate system    ║
# ║  the centroids live in, before the squared-distance loop runs.  ║
# ║                                                                  ║
# ║  Round-trip verification — :func:`verify_round_trip` confirms   ║
# ║  that the Q4.12-quantised centroids classify a synthetic test   ║
# ║  set identically to the float reference, to within a            ║
# ║  configurable mismatch tolerance.  The defaults catch any       ║
# ║  encoding bug while tolerating the rare boundary case where     ║
# ║  two cluster distances differ by < 1 LSB.                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Python centroids → C header (Q4.12 fixed-point + float)."""

from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stag.constants import (
    CLUSTER_RESULTS_DIR,
    FEATURE_LABELS,
    MAXABS_SCALER_CSV,
)

DEFAULT_Q_FRAC_BITS: int = 12
DEFAULT_K: int = 8
DEFAULT_TEST_VECTORS: int = 100_000


@dataclass(frozen=True)
class QFormat:
    """Fixed-point format descriptor."""

    int_bits: int   # number of integer bits (incl. sign)
    frac_bits: int  # number of fractional bits

    @property
    def total_bits(self) -> int:
        """Sum of integer and fractional bits (excluding sign storage)."""
        return self.int_bits + self.frac_bits

    @property
    def scale(self) -> int:
        """Integer scale factor ``2 ** frac_bits`` mapping float to fixed-point."""
        return 1 << self.frac_bits

    @property
    def max_signed(self) -> int:
        """Largest signed integer representable in ``total_bits`` bits."""
        return (1 << (self.total_bits - 1)) - 1

    @property
    def min_signed(self) -> int:
        """Smallest (most-negative) signed integer in ``total_bits`` bits."""
        return -(1 << (self.total_bits - 1))

    @property
    def range_float(self) -> tuple[float, float]:
        """Representable float range ``(min, max)`` after fixed-point encoding."""
        return (self.min_signed / self.scale, self.max_signed / self.scale)


def encode_q_format(x: np.ndarray, q: QFormat) -> np.ndarray:
    """Quantise float values to a Q-format signed integer representation.

    Values outside the representable range are clipped (the caller is
    expected to have scaled inputs to fit; we clip rather than crash
    because the float centroids might be a hair over ±1 from floating-
    point round-off, and we want a deterministic encoding).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.size == 0:
        return np.zeros(0, dtype=np.int32)
    scaled = np.round(x_arr * q.scale).astype(np.int64)
    clipped = np.clip(scaled, q.min_signed, q.max_signed)
    return clipped.astype(np.int32)


def decode_q_format(x_int: np.ndarray, q: QFormat) -> np.ndarray:
    """Inverse of :func:`encode_q_format` — returns the float value."""
    return np.asarray(x_int, dtype=np.float64) / q.scale


def _nearest_centroid(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Argmin Σ(x_d − c_{k,d})² over k.  Pure numpy reference."""
    diff = features[:, None, :] - centroids[None, :, :]
    sq = (diff * diff).sum(axis=2)
    return np.argmin(sq, axis=1)


def verify_round_trip(
    centroids_float: np.ndarray,
    centroids_q: np.ndarray,
    q: QFormat,
    n_test: int = DEFAULT_TEST_VECTORS,
    tolerance: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> dict:
    """Quantify Q-format quantisation loss for nearest-centroid classification.

    Generates ``n_test`` synthetic feature vectors uniformly in the
    centroid bounding box, runs the nearest-centroid classifier under
    both the float and Q-format representations, and reports the
    mismatch fraction.

    Args:
        centroids_float: Original ``(K, D)`` float centroids.
        centroids_q:     ``(K, D)`` Q-format encoded centroids
                         (int32 already scaled).
        q:               The Q format used.
        n_test:          Test-vector count.
        tolerance:       Acceptable disagreement fraction.
        rng:             Seeded generator.  ``None`` → fresh.

    Returns:
        Dict: ``mismatch_fraction``, ``max_abs_quant_error``,
              ``n_test``, ``passed``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    centroids_dequantised = decode_q_format(centroids_q, q)

    lo = centroids_float.min(axis=0) - 0.1
    hi = centroids_float.max(axis=0) + 0.1
    test = lo + (hi - lo) * rng.random((n_test, centroids_float.shape[1]))

    pred_float = _nearest_centroid(test, centroids_float)
    pred_q     = _nearest_centroid(test, centroids_dequantised)

    mismatch = float((pred_float != pred_q).mean())
    quant_err = float(np.abs(centroids_float - centroids_dequantised).max())
    return {
        "mismatch_fraction":   mismatch,
        "max_abs_quant_error": quant_err,
        "n_test":              int(n_test),
        "passed":              mismatch <= tolerance,
        "tolerance":           tolerance,
    }


def _emit_q_array_2d(
    name: str, arr: np.ndarray, q: QFormat, comment: str | None = None,
) -> str:
    """Render an ``int16_t name[K][D] PROGMEM = {...}`` declaration."""
    k, d = arr.shape
    lines = []
    if comment:
        lines.append(f"/* {comment} */")
    lines.append(f"static const int16_t {name}[{k}][{d}] PROGMEM = {{")
    for row in arr:
        cells = ", ".join(f"{int(v):>7d}" for v in row)
        lines.append(f"    {{ {cells} }},")
    lines.append("};")
    return "\n".join(lines)


def _emit_float_array_2d(
    name: str, arr: np.ndarray, comment: str | None = None,
) -> str:
    k, d = arr.shape
    lines = []
    if comment:
        lines.append(f"/* {comment} */")
    lines.append(f"static const float {name}[{k}][{d}] = {{")
    for row in arr:
        cells = ", ".join(f"{v:>14.10f}f" for v in row)
        lines.append(f"    {{ {cells} }},")
    lines.append("};")
    return "\n".join(lines)


def _emit_q_array_1d(
    name: str, arr: np.ndarray, q: QFormat, comment: str | None = None,
) -> str:
    lines = []
    if comment:
        lines.append(f"/* {comment} */")
    cells = ", ".join(f"{int(v):>7d}" for v in arr)
    lines.append(f"static const int16_t {name}[{arr.size}] PROGMEM = {{ {cells} }};")
    return "\n".join(lines)


def _emit_float_array_1d(
    name: str, arr: np.ndarray, comment: str | None = None,
) -> str:
    lines = []
    if comment:
        lines.append(f"/* {comment} */")
    cells = ", ".join(f"{v:>14.10f}f" for v in arr)
    lines.append(f"static const float {name}[{arr.size}] = {{ {cells} }};")
    return "\n".join(lines)


def centroids_to_c_header(
    centroids: np.ndarray,
    maxabs_divisors: np.ndarray,
    out_path: Path,
    feature_labels: tuple[str, ...] = FEATURE_LABELS,
    q_frac_bits: int = DEFAULT_Q_FRAC_BITS,
    source_files: dict[str, Path] | None = None,
) -> dict:
    """Emit a self-contained C header with Q-format + float centroids.

    The generated header is consumed by ``stag/embedded/nearest_centroid.c``
    (Sprint 4 Phase 2) and by the per-MCU benchmark binaries.

    Args:
        centroids:        ``(K, D)`` float centroids in MaxAbs-scaled
                          coordinates (i.e. each axis already divided
                          by its absolute maximum).
        maxabs_divisors:  ``(D,)`` per-axis MaxAbs divisors used to
                          produce the scaled inputs at training time.
                          The on-MCU classifier multiplies its raw
                          sensor input by the inverse of these divisors
                          before the distance loop.
        out_path:         Where to write ``centroids.h``.
        feature_labels:   Per-axis names (for comments in the header).
        q_frac_bits:      Fractional bits in the Q format.  Default 12
                          → Q4.12 in 16-bit storage.
        source_files:     Optional dict of provenance fields written
                          into the header comment block (e.g.
                          ``{"centroids": Path("…/k_8/centroids/…npy")}``).

    Returns:
        Verification dict from :func:`verify_round_trip`.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    if centroids.ndim != 2:
        raise ValueError(f"centroids must be 2-D, got {centroids.shape}")
    k, d = centroids.shape

    maxabs_divisors = np.asarray(maxabs_divisors, dtype=np.float64)
    if maxabs_divisors.shape != (d,):
        raise ValueError(
            f"maxabs_divisors must have shape ({d},), got {maxabs_divisors.shape}",
        )

    q = QFormat(int_bits=16 - q_frac_bits, frac_bits=q_frac_bits)
    q_lo, q_hi = q.range_float
    if centroids.min() < q_lo or centroids.max() > q_hi:
        raise ValueError(
            f"Centroid range [{centroids.min():.4f}, {centroids.max():.4f}] "
            f"falls outside Q{q.int_bits - 1}.{q.frac_bits} representable "
            f"range [{q_lo:.4f}, {q_hi:.4f}].  Reduce q_frac_bits or rescale.",
        )

    # Per-axis MaxAbs divisors live in g-units (e.g. 7.99).  The on-MCU
    # classifier needs the *inverse* (multiply incoming raw g-reading
    # by this factor → MaxAbs-scaled coordinate).  Inverse fits inside
    # Q1.15 (≈ 0 .. 1, since divisors are ~8 g).
    inv_divisors = 1.0 / maxabs_divisors
    q_inv = QFormat(int_bits=1, frac_bits=15)
    if inv_divisors.max() >= 1.0:
        # Inverse of an 8-g divisor is 0.125, well under 1.0 — but
        # generalise the Q-format choice if a future MaxAbs file ever
        # has divisors < 1 g.
        q_inv = QFormat(int_bits=q.int_bits, frac_bits=q.frac_bits)

    centroids_q       = encode_q_format(centroids,   q)
    inv_divisors_q    = encode_q_format(inv_divisors, q_inv)
    verify = verify_round_trip(centroids, centroids_q, q)

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    provenance_lines = []
    if source_files:
        for tag, p in source_files.items():
            # Force POSIX separators so the generated header is
            # byte-identical regardless of which OS produced it —
            # otherwise Windows emits `\some\path\` and the diff
            # against a Linux-generated reference flips on every CI run.
            p_posix = Path(p).as_posix() if not isinstance(p, str) else p.replace("\\", "/")
            provenance_lines.append(f" *   {tag:>20}: {p_posix}")

    header = f"""/* Auto-generated by stag.embedded.export_centroids on {timestamp}.
 * DO NOT EDIT BY HAND — re-run the script to regenerate.
 *
 * Provenance:
{chr(10).join(provenance_lines) if provenance_lines else " *   (no source files recorded)"}
 *
 * Round-trip verification (n = {verify['n_test']:,} synthetic vectors):
 *   max abs quantisation error      : {verify['max_abs_quant_error']:.6e}
 *   nearest-centroid mismatch frac. : {verify['mismatch_fraction']:.4e}
 *   tolerance                       : {verify['tolerance']}
 *   passed                          : {verify['passed']}
 */

#ifndef STAG_CENTROIDS_H
#define STAG_CENTROIDS_H

#include <stdint.h>

#ifdef __AVR__
#  include <avr/pgmspace.h>
#else
#  define PROGMEM
#  define pgm_read_word_near(addr) (*(addr))
#  define pgm_read_float_near(addr) (*(addr))
#endif

#define STAG_K_CLUSTERS   {k}
#define STAG_N_FEATURES   {d}
#define STAG_Q_FRAC_BITS  {q.frac_bits}
#define STAG_Q_SCALE      {q.scale}

/* Feature axis names (clockwise from head_x):
 *   {', '.join(feature_labels)}
 */

"""
    header += _emit_q_array_2d(
        "stag_centroids_q",
        centroids_q,
        q,
        comment=(f"Q{q.int_bits - 1}.{q.frac_bits} centroids "
                 f"(int16_t, PROGMEM on AVR).  Multiply by 1/STAG_Q_SCALE "
                 f"to recover the float value."),
    ) + "\n\n"
    header += _emit_float_array_2d(
        "stag_centroids_f",
        centroids,
        comment="IEEE-754 float centroids (for MCUs with hardware FPU).",
    ) + "\n\n"
    header += _emit_q_array_1d(
        "stag_inv_maxabs_q",
        inv_divisors_q,
        q_inv,
        comment=(f"Q{q_inv.int_bits - 1}.{q_inv.frac_bits} inverse MaxAbs "
                 f"divisors.  Multiply a raw g-reading by this to get "
                 f"the MaxAbs-scaled coordinate."),
    ) + "\n\n"
    header += _emit_float_array_1d(
        "stag_inv_maxabs_f",
        inv_divisors,
        comment="Inverse MaxAbs divisors as floats (for FPU MCUs).",
    ) + "\n\n"

    header += "\n#endif  /* STAG_CENTROIDS_H */\n"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header)

    return verify


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────


def _resolve_centroids(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = sorted(
        (CLUSTER_RESULTS_DIR / "delSize_0" / "k_8" / "centroids").glob("*.npy"),
    )
    if not candidates:
        raise SystemExit(
            "No k=8 centroid files found.  Pass --centroids explicitly.",
        )
    return candidates[0]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the centroid-export driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--centroids", type=Path, default=None,
        help="k=8 centroids .npy (default: first match in "
             "cluster_results/.../delSize_0/k_8/centroids/*.npy).",
    )
    parser.add_argument("--divisors", type=Path, default=MAXABS_SCALER_CSV,
                        help="MaxAbs divisors CSV (default: "
                             "stag.constants.MAXABS_SCALER_CSV).")
    parser.add_argument("--out", type=Path,
                        default=Path("stag/embedded/centroids.h"),
                        help="Output C header path.")
    parser.add_argument("--q-frac-bits", type=int, default=DEFAULT_Q_FRAC_BITS,
                        help=f"Fractional bits (default {DEFAULT_Q_FRAC_BITS} → Q4.12).")
    return parser.parse_args()


def main() -> None:
    """Export the k=8 centroids as a C header (driver entry point)."""
    args = parse_args()
    centroids_path = _resolve_centroids(args.centroids)
    print(f"centroids : {centroids_path}")
    print(f"divisors  : {args.divisors}")
    print(f"output    : {args.out}")
    print()

    centroids = np.load(centroids_path)
    divisors = pd.read_csv(args.divisors).iloc[0].to_numpy()

    print(f"centroids shape    : {centroids.shape}")
    print(f"centroids range    : [{centroids.min():.4f}, {centroids.max():.4f}]")
    print(f"|centroids|max     : {np.abs(centroids).max():.4f}")
    print(f"divisors           : {divisors.round(4).tolist()}")
    print()

    verify = centroids_to_c_header(
        centroids=centroids,
        maxabs_divisors=divisors,
        out_path=args.out,
        q_frac_bits=args.q_frac_bits,
        source_files={
            "centroids": centroids_path,
            "divisors":  args.divisors,
        },
    )

    print("── verification ──")
    print(f"  max abs quantisation error : {verify['max_abs_quant_error']:.6e}")
    print(f"  nearest-centroid mismatch  : {verify['mismatch_fraction']:.4e}"
          f"  (tolerance {verify['tolerance']})")
    print(f"  passed                     : {verify['passed']}")
    print()
    print(f"wrote {args.out}  ({args.out.stat().st_size:,} bytes)")
    if not verify["passed"]:
        raise SystemExit(
            "Round-trip verification failed.  Inspect the header before "
            "compiling against it.",
        )


if __name__ == "__main__":
    main()
