/* ╔══════════════════════════════════════════════════════════════════╗
 * ║  STAG — embedded.benchmark.arm.benchmark_arm                     ║
 * ║  « time the nearest-centroid classifier on Cortex-M via DWT »    ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  Works on every Cortex-M variant with a DWT cycle counter        ║
 * ║  (M3, M4, M4F, M7).  For M0+ targets (which lack DWT) a          ║
 * ║  SysTick-based variant is provided via -DSTAG_NO_DWT.            ║
 * ║                                                                  ║
 * ║  DWT->CYCCNT is a free-running 32-bit cycle counter — at         ║
 * ║  64 MHz (nRF52840 default) it wraps every ~67 s; we measure      ║
 * ║  N = 1000 classifications which is well within the wrap window  ║
 * ║  on every target.  No ISR overhead, no peripheral setup beyond  ║
 * ║  TRCENA + CYCCNTENA.                                             ║
 * ║                                                                  ║
 * ║  Result publication is identical to the AVR firmware: globals   ║
 * ║  cycles_total + n_classifications are filled in, then control   ║
 * ║  parks at the bench_done() no-inline anchor for the gdb-based    ║
 * ║  Renode runner to read.                                          ║
 * ╚══════════════════════════════════════════════════════════════════╝ */

#include <stdint.h>

#include "../../nearest_centroid.h"
#include "test_vectors.h"


#define BENCH_N    1000


/* Renode's nRF52840 / iMXRT1064 / STM32 platform models do not
 * always emulate the DWT cycle counter at 0xE0001000.  Rather than
 * write target-specific cycle-counter shims, we delegate timing to
 * the host-side runner: it sets breakpoints at the two anchors
 * below and reads `cpu ExecutedInstructions` from the Renode monitor
 * via gdb's `monitor` command.  Cycles per benchmark = delta of the
 * two readings.  Works identically on every Renode-emulated target. */

volatile uint32_t cycles_total      = 0;   /* populated by the runner */
volatile uint32_t n_classifications = 0;
volatile uint8_t  bench_status      = 0;


int main(void)
{
    int16_t  input[STAG_N_FEATURES];
    uint16_t accum = 0;

    /* The benchmark runs forever in an outer loop so the gdb-based
     * runner can attach at any point and still catch bench_start →
     * bench_done on the next iteration.  Without the outer loop the
     * Renode .resc would have to start the CPU paused so gdb arms
     * its breakpoints first — Renode 1.15.3 makes that awkward, and
     * the looping approach is portable across every Cortex-M / MSP430
     * target we will benchmark.
     *
     * The two labels live in inline assembly so the optimiser cannot
     * inline or reorder them.  `.global` makes them visible to nm. */
    for (;;) {
        __asm__ volatile (
            ".global bench_start\n"
            "bench_start:\n"
            "    nop\n"
            ::: "memory"
        );

        for (uint16_t i = 0; i < BENCH_N; ++i) {
            const uint8_t row = (uint8_t)(i & (TEST_VECTORS_N - 1u));
            for (uint8_t d = 0; d < STAG_N_FEATURES; ++d) {
                input[d] = test_inputs_q[row][d];
            }
            accum = (uint16_t)(accum + stag_nearest_centroid_q412(input));
        }

        n_classifications = BENCH_N;
        bench_status = (uint8_t)(1u | (accum & 0u));

        __asm__ volatile (
            ".global bench_done\n"
            "bench_done:\n"
            "    nop\n"
            ::: "memory"
        );
    }
}
