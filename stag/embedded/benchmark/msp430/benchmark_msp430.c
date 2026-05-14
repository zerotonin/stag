/* ╔══════════════════════════════════════════════════════════════════╗
 * ║  STAG — embedded.benchmark.msp430.benchmark_msp430                ║
 * ║  « time the nearest-centroid classifier on MSP430G2553 »          ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  Self-timed via Timer_A0 in continuous mode at SMCLK with no      ║
 * ║  prescaler.  Each classification is timed in isolation: read     ║
 * ║  TA0R before, classify, read TA0R after, accumulate the          ║
 * ║  (uint16_t) delta into a uint32_t total.  Wrap inside the        ║
 * ║  16-bit TAR is harmless because no single classification         ║
 * ║  takes anywhere near 65 535 cycles on MSP430.                    ║
 * ║                                                                  ║
 * ║  Why this matters — the classifier has an early-out               ║
 * ║  optimisation (partial sum exceeds current best → skip the rest  ║
 * ║  of this centroid's distance computation), so cycles/call is     ║
 * ║  *data-dependent*.  Other targets average over the full 64-vector║
 * ║  test set (each times N=1000 / 64 ≈ 15 sweeps) and get a stable  ║
 * ║  number.  We sweep the full 64 vectors exactly once → matches    ║
 * ║  the same population mean.                                       ║
 * ║                                                                  ║
 * ║  Why MSP430G2553 and not F5529:                                  ║
 * ║    mspdebug 0.22 (the Ubuntu noble repo version) implements      ║
 * ║    only the original 16-bit MSP430 ISA — not the MSP430X         ║
 * ║    (CPUX) extensions used by the F5xx family (PUSHM, CALLA,      ║
 * ║    MOVA, etc.).  G2553 is pure MSP430 silicon (TI LaunchPad      ║
 * ║    value-line, 16 MHz, 16 KB flash, 512 B SRAM) — same family   ║
 * ║    label, supported by every available simulator.                ║
 * ╚══════════════════════════════════════════════════════════════════╝ */

#include <msp430.h>
#include <stdint.h>

#include "../../nearest_centroid.h"
#include "test_vectors.h"


#define BENCH_N    TEST_VECTORS_N    /* full 64-vector sweep */


/* Globals the runner reads after exit(). */
volatile uint32_t cycles_total      = 0;
volatile uint16_t n_classifications = 0;
volatile uint16_t bench_status      = 0;


int main(void)
{
    WDTCTL = WDTPW | WDTHOLD;

    /* SMCLK source, continuous mode, no prescaler, no overflow IRQ. */
    TA0CTL = TASSEL_2 | MC_2 | ID_0 | TACLR;

    int16_t  input[STAG_N_FEATURES];
    uint16_t accum = 0;
    uint32_t total = 0;

    for (uint16_t i = 0; i < BENCH_N; ++i) {
        /* Copy this iteration's input vector out of the const table. */
        for (uint8_t d = 0; d < STAG_N_FEATURES; ++d) {
            input[d] = test_inputs_q[i][d];
        }

        const uint16_t t0 = TA0R;
        const uint8_t  cls = stag_nearest_centroid_q412(input);
        const uint16_t t1 = TA0R;

        /* uint16_t subtraction handles wrap correctly. */
        total += (uint16_t)(t1 - t0);
        accum = (uint16_t)(accum + cls);
    }

    cycles_total      = total;
    n_classifications = BENCH_N;
    bench_status      = (uint16_t)(1u | (accum & 0u));

    return 0;
}
