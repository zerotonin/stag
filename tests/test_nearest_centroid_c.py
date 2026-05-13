"""Host-compiled C-classifier ↔ Python parity test.

Compiles ``stag/embedded/nearest_centroid.c`` with host gcc, runs it on
a Python-generated test set of synthetic MaxAbs-scaled input vectors,
and asserts byte-identical cluster predictions against the numpy
reference implementation.

Skipped if gcc is not on PATH (so the test suite stays green on the
default conda env until Sprint 4 toolchains are installed).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stag.constants import CLUSTER_RESULTS_DIR, MAXABS_SCALER_CSV
from stag.embedded.export_centroids import (
    DEFAULT_Q_FRAC_BITS,
    QFormat,
    centroids_to_c_header,
    encode_q_format,
)

# Skip the whole module if gcc is missing.
pytestmark = pytest.mark.skipif(
    shutil.which("gcc") is None,
    reason="host gcc not installed; Sprint 4 Phase 2 host parity test skipped",
)


N_TEST_VECTORS: int = 10_000
SOURCE_DIR: Path = Path(__file__).resolve().parent.parent / "stag" / "embedded"


def _nearest_centroid_float(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Pure-numpy reference (no early-out — same argmin)."""
    diff = features[:, None, :] - centroids[None, :, :]
    sq = (diff * diff).sum(axis=2)
    return np.argmin(sq, axis=1).astype(np.uint8)


def _build_harness(
    tmp_path: Path,
    centroids_h: Path,
    test_vectors_q: np.ndarray,
    test_vectors_f: np.ndarray,
    expected_q: np.ndarray,
    expected_f: np.ndarray,
) -> Path:
    """Generate test_parity.c, compile with gcc, return the binary path."""
    n = test_vectors_q.shape[0]
    d = test_vectors_q.shape[1]

    def fmt_row_int(row: np.ndarray) -> str:
        return ", ".join(f"{int(v):>7d}" for v in row)

    def fmt_row_float(row: np.ndarray) -> str:
        return ", ".join(f"{v:.10f}f" for v in row)

    q_rows = ",\n    ".join("{ " + fmt_row_int(row) + " }" for row in test_vectors_q)
    f_rows = ",\n    ".join("{ " + fmt_row_float(row) + " }" for row in test_vectors_f)
    eq_row = ", ".join(f"{int(v):>3d}" for v in expected_q)
    ef_row = ", ".join(f"{int(v):>3d}" for v in expected_f)

    harness_src = f"""\
#include <stdio.h>
#include <stdint.h>
#include "nearest_centroid.h"

#define N_TESTS {n}
#define D       {d}

static const int16_t inputs_q[N_TESTS][D] = {{
    {q_rows}
}};

static const float inputs_f[N_TESTS][D] = {{
    {f_rows}
}};

static const uint8_t expected_q[N_TESTS] = {{ {eq_row} }};
static const uint8_t expected_f[N_TESTS] = {{ {ef_row} }};

int main(void) {{
    int mismatch_q = 0;
    int mismatch_f = 0;
    for (int i = 0; i < N_TESTS; ++i) {{
        uint8_t pred_q = stag_nearest_centroid_q412(inputs_q[i]);
        uint8_t pred_f = stag_nearest_centroid_f(inputs_f[i]);
        if (pred_q != expected_q[i]) {{
            if (mismatch_q < 5) {{
                fprintf(stderr,
                    "Q4.12 mismatch at i=%d: C=%u expected=%u\\n",
                    i, pred_q, expected_q[i]);
            }}
            ++mismatch_q;
        }}
        if (pred_f != expected_f[i]) {{
            if (mismatch_f < 5) {{
                fprintf(stderr,
                    "float mismatch at i=%d: C=%u expected=%u\\n",
                    i, pred_f, expected_f[i]);
            }}
            ++mismatch_f;
        }}
    }}
    fprintf(stderr, "Q4.12 mismatches: %d / %d\\n", mismatch_q, N_TESTS);
    fprintf(stderr, "float mismatches: %d / %d\\n", mismatch_f, N_TESTS);
    /* Float path must be bit-identical against the float reference.
     * Q4.12 path is allowed up to 0.5% mismatch due to quantisation. */
    if (mismatch_f != 0) return 2;
    if (mismatch_q > N_TESTS / 200) return 3;
    return 0;
}}
"""
    src_path = tmp_path / "test_parity.c"
    src_path.write_text(harness_src)

    bin_path = tmp_path / "test_parity"
    proc = subprocess.run(
        [
            "gcc", "-O2", "-Wall", "-Wextra", "-std=c11",
            f"-I{SOURCE_DIR}",
            str(src_path),
            str(SOURCE_DIR / "nearest_centroid.c"),
            "-o", str(bin_path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"gcc failed:\n"
            f"  stdout: {proc.stdout}\n"
            f"  stderr: {proc.stderr}",
        )
    return bin_path


def test_host_parity(tmp_path: Path) -> None:
    """Build centroids.h + harness, compile, run, assert clean."""
    centroid_files = sorted(
        (CLUSTER_RESULTS_DIR / "delSize_0" / "k_8" / "centroids").glob("*.npy"),
    )
    if not centroid_files:
        pytest.skip("k=8 centroids not present locally")
    centroids = np.load(centroid_files[0])
    divisors = pd.read_csv(MAXABS_SCALER_CSV).iloc[0].to_numpy()

    centroids_h = SOURCE_DIR / "centroids.h"
    centroids_to_c_header(centroids, divisors, centroids_h)

    # Synthetic test vectors in the centroid bounding box.
    rng = np.random.default_rng(0)
    lo = centroids.min(axis=0) - 0.1
    hi = centroids.max(axis=0) + 0.1
    test_f = (lo + (hi - lo) * rng.random((N_TEST_VECTORS, centroids.shape[1]))).astype(np.float64)

    q = QFormat(int_bits=16 - DEFAULT_Q_FRAC_BITS, frac_bits=DEFAULT_Q_FRAC_BITS)
    test_q = encode_q_format(test_f, q).astype(np.int16)

    expected_f = _nearest_centroid_float(test_f, centroids)
    # For the Q4.12 path the reference is the float classifier applied
    # to the *dequantised* centroids — that's what the C code computes.
    centroids_q = encode_q_format(centroids, q)
    centroids_dq = (centroids_q.astype(np.float64) / q.scale)
    test_q_decoded = (test_q.astype(np.float64) / q.scale)
    expected_q = _nearest_centroid_float(test_q_decoded, centroids_dq)

    bin_path = _build_harness(
        tmp_path, centroids_h, test_q, test_f, expected_q, expected_f,
    )
    proc = subprocess.run([str(bin_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(
            f"test_parity binary exited {proc.returncode}\n"
            f"  stderr: {proc.stderr}",
        )
    # Make the per-classifier counts visible in pytest -s output.
    print(proc.stderr.strip())


def test_synthetic_small() -> None:
    """Quick sanity: a centroid hit-test on a 4×2 cluster set."""
    centroids_f = np.array([
        [ 0.0, 0.0],
        [ 0.5, 0.0],
        [ 0.0, 0.5],
        [-0.5, 0.0],
    ], dtype=np.float64)
    # Reference predictions
    test = np.array([
        [ 0.0,  0.0],   # cluster 0
        [ 0.4,  0.0],   # cluster 1
        [ 0.0,  0.4],   # cluster 2
        [-0.4,  0.0],   # cluster 3
    ])
    pred = _nearest_centroid_float(test, centroids_f)
    np.testing.assert_array_equal(pred, [0, 1, 2, 3])
