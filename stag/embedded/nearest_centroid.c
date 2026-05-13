/* ╔══════════════════════════════════════════════════════════════════╗
 * ║  STAG — embedded.nearest_centroid                                ║
 * ║  « bare-metal nearest-centroid classifier — implementation »     ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  Two passes, no allocations, no library calls.  Compiles cleanly║
 * ║  with -O2 -Wall -Wextra on every cross-toolchain in the Sprint  ║
 * ║  4 matrix (avr-gcc, arm-none-eabi-gcc, msp430-elf-gcc,          ║
 * ║  xtensa-esp32-elf-gcc) and on the host gcc used for the         ║
 * ║  parity test.                                                    ║
 * ║                                                                  ║
 * ║  Q4.12 arithmetic:                                              ║
 * ║    input   : int16_t in Q4.12 (range ±2 typical, ±8 max)        ║
 * ║    diff    : int16_t in Q4.12                                   ║
 * ║    diff²   : int32_t in Q8.24                                   ║
 * ║    sum_6   : int32_t in Q8.24                                   ║
 * ║                                                                  ║
 * ║  Worst-case per-axis squared: (2 · 2¹²)² = 2²⁶ = 67_108_864.    ║
 * ║  Six summed: ≤ 4·10⁸, comfortably under INT32_MAX (2.1·10⁹).    ║
 * ║                                                                  ║
 * ║  The classifier does an early-out when the partial sum already  ║
 * ║  exceeds the current best — important on AVR where a multiply  ║
 * ║  is ~5 cycles.  Does not change the final cluster choice; the  ║
 * ║  early-out optimisation is bit-identical to the no-early-out   ║
 * ║  variant on the argmin output (proven by the parity test).     ║
 * ╚══════════════════════════════════════════════════════════════════╝ */

#include "nearest_centroid.h"

#include <stdint.h>


uint8_t stag_nearest_centroid_q412(const int16_t *input)
{
    int32_t best_dist = INT32_MAX;
    uint8_t best_k    = 0;

    for (uint8_t k = 0; k < STAG_K_CLUSTERS; ++k) {
        int32_t dist = 0;

        for (uint8_t d = 0; d < STAG_N_FEATURES; ++d) {
            /* pgm_read_word_near is the AVR PROGMEM accessor; on
             * non-AVR targets the macro reduces to plain RAM access
             * (see centroids.h shim). */
            int16_t c    = (int16_t)pgm_read_word_near(&stag_centroids_q[k][d]);
            int16_t diff = (int16_t)(input[d] - c);
            dist += (int32_t)diff * (int32_t)diff;

            if (dist >= best_dist) {
                /* This centroid already worse than the best — bail. */
                break;
            }
        }

        if (dist < best_dist) {
            best_dist = dist;
            best_k    = k;
        }
    }

    return best_k;
}


uint8_t stag_nearest_centroid_f(const float *input)
{
    /* Sentinel just needs to exceed any realistic squared sum.
     * With inputs bounded to ±1 (MaxAbs scale), max squared diff per
     * axis is 4.0; 6 axes → max sum 24.0.  1.0e30f is comfortable. */
    float   best_dist = 1.0e30f;
    uint8_t best_k    = 0;

    for (uint8_t k = 0; k < STAG_K_CLUSTERS; ++k) {
        float dist = 0.0f;

        for (uint8_t d = 0; d < STAG_N_FEATURES; ++d) {
            float diff = input[d] - stag_centroids_f[k][d];
            dist += diff * diff;

            if (dist >= best_dist) {
                break;
            }
        }

        if (dist < best_dist) {
            best_dist = dist;
            best_k    = k;
        }
    }

    return best_k;
}
