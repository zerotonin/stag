/* ╔══════════════════════════════════════════════════════════════════╗
 * ║  STAG — embedded.nearest_centroid                                ║
 * ║  « bare-metal nearest-centroid classifier »                      ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  Two implementations of the same prototype-assignment loop:     ║
 * ║                                                                  ║
 * ║    stag_nearest_centroid_q412   — Q4.12 fixed-point.  Used on   ║
 * ║                                   8/16-bit MCUs without an FPU. ║
 * ║                                   Reads PROGMEM centroid table  ║
 * ║                                   on AVR; flat memory elsewhere.║
 * ║                                                                  ║
 * ║    stag_nearest_centroid_f      — IEEE-754 float.  Used on      ║
 * ║                                   32-bit MCUs with hardware     ║
 * ║                                   FPU (Cortex-M4F / M7, ESP32   ║
 * ║                                   LX6).                          ║
 * ║                                                                  ║
 * ║  Both functions take a six-element input vector in MaxAbs-      ║
 * ║  scaled coordinates (raw g divided by the per-axis sensor       ║
 * ║  saturation limit; see stag.embedded.export_centroids for the   ║
 * ║  inverse divisors).  Returns the cluster index in [0, K).       ║
 * ║                                                                  ║
 * ║  Bit-identical against the Python reference                     ║
 * ║  stag.analysis._nearest_centroid on every test vector in the    ║
 * ║  host parity test (tests/test_nearest_centroid_c.py).           ║
 * ╚══════════════════════════════════════════════════════════════════╝ */

#ifndef STAG_NEAREST_CENTROID_H
#define STAG_NEAREST_CENTROID_H

#include <stdint.h>
#include "centroids.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Q4.12 fixed-point nearest-centroid classifier.
 *
 * Args:
 *   input  pointer to STAG_N_FEATURES int16_t values in Q4.12
 *          (already MaxAbs-scaled).  No bounds checking.
 * Returns:
 *   cluster index in [0, STAG_K_CLUSTERS).
 */
uint8_t stag_nearest_centroid_q412(const int16_t *input);

/* IEEE-754 float nearest-centroid classifier.
 *
 * Args:
 *   input  pointer to STAG_N_FEATURES floats (MaxAbs-scaled).
 * Returns:
 *   cluster index in [0, STAG_K_CLUSTERS).
 */
uint8_t stag_nearest_centroid_f(const float *input);

#ifdef __cplusplus
}
#endif

#endif  /* STAG_NEAREST_CENTROID_H */
