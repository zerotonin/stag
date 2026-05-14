/* ╔══════════════════════════════════════════════════════════════════╗
 * ║  STAG — embedded.benchmark.avr.benchmark_avr                     ║
 * ║  « time the nearest-centroid classifier on AVR via Timer1 »      ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  Compiles to atmega328p / atmega32u4 / atmega2560 via the same  ║
 * ║  source — only the -mmcu flag differs (see Makefile).           ║
 * ║                                                                  ║
 * ║  Measurement strategy                                            ║
 * ║  ────────────────────                                            ║
 * ║  Timer1 runs at the CPU clock (prescaler = 1), so each tick is  ║
 * ║  one instruction-cycle.  The Timer1-overflow ISR increments a   ║
 * ║  32-bit accumulator so we can measure long runs without the     ║
 * ║  hardware counter's 16-bit limit.  Total cycles for the         ║
 * ║  classifier loop are stored in ``cycles_total`` and made        ║
 * ║  visible to the host runner via gdb at the ``bench_done()``     ║
 * ║  no-inline anchor.                                               ║
 * ║                                                                  ║
 * ║  64 deterministic Q4.12 input vectors live in PROGMEM and are   ║
 * ║  cycled through during the benchmark — small enough to fit in   ║
 * ║  even the smallest ATmega's flash, large enough to defeat any  ║
 * ║  trivial caching the compiler might attempt.                    ║
 * ║                                                                  ║
 * ║  The runner reads ``cycles_total`` and ``n_classifications``   ║
 * ║  via gdb on hit of ``bench_done`` and computes the per-call    ║
 * ║  cycle cost.                                                     ║
 * ╚══════════════════════════════════════════════════════════════════╝ */

#ifndef F_CPU
#define F_CPU 16000000UL
#endif

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <stdint.h>

/* Pull in the canonical classifier + centroid header. */
#include "../../nearest_centroid.h"


/* How many classifications to time per run.  Set high enough that
 * init / teardown cost is < 0.1 cycle-per-call in the reported figure;
 * low enough that an honestly-emulated AVR finishes inside a minute. */
#define BENCH_N    1000


/* Deterministic 64-vector Q4.12 input table.  Generated once with a
 * fixed RNG seed (Python: stag/embedded/benchmark/avr/gen_inputs.py)
 * so every MCU benchmarks against the exact same input distribution. */
#include "test_vectors.h"


/* ─── Cycle accumulator ───────────────────────────────────────────── */
static volatile uint32_t timer1_ovf  = 0;
volatile uint32_t        cycles_total      = 0;
volatile uint32_t        n_classifications = 0;
volatile uint8_t         bench_status      = 0;


ISR(TIMER1_OVF_vect)
{
    timer1_ovf++;
}


/* gdb breakpoint anchor.  noinline + used so the symbol survives -Os. */
__attribute__((noinline, used))
static void bench_done(void)
{
    /* The instruction at the start of this function is the gdb
     * breakpoint target.  cycles_total and n_classifications are
     * already written by the time we get here. */
    __asm__ volatile ("nop");
}


int main(void)
{
    /* Disable all peripherals we do not use so they cannot perturb
     * the timing (PRR0 on 328P/2560; PRR on 32U4). */
#if defined(__AVR_ATmega32U4__)
    PRR0 = 0xFF;
    PRR1 = 0xFF;
    /* Re-enable Timer1 — it is the one we need. */
    PRR0 &= (uint8_t)~(1 << PRTIM1);
#elif defined(__AVR_ATmega2560__)
    PRR0 = 0xFF;
    PRR1 = 0xFF;
    PRR0 &= (uint8_t)~(1 << PRTIM1);
#else  /* atmega328p */
    PRR  = 0xFF;
    PRR &= (uint8_t)~(1 << PRTIM1);
#endif

    /* Timer1: normal mode, prescaler = 1 (count every CPU cycle). */
    TCCR1A = 0;
    TCCR1B = (1 << CS10);
    TCNT1  = 0;
    TIMSK1 = (1 << TOIE1);
    sei();

    /* ── Benchmark loop ── */
    int16_t input[STAG_N_FEATURES];
    uint16_t accum = 0;   /* prevents the optimiser from deleting the call */

    for (uint16_t i = 0; i < BENCH_N; ++i) {
        const uint8_t row = (uint8_t)(i & (TEST_VECTORS_N - 1u));
        for (uint8_t d = 0; d < STAG_N_FEATURES; ++d) {
            input[d] = (int16_t)pgm_read_word_near(&test_inputs_q[row][d]);
        }
        accum = (uint16_t)(accum + stag_nearest_centroid_q412(input));
    }

    /* Atomically read Timer1 = (ovf << 16) | TCNT1 with interrupts
     * disabled so we do not race the overflow ISR. */
    cli();
    const uint16_t tcnt_now   = TCNT1;
    const uint32_t ovf_now    = timer1_ovf;
    cycles_total              = (ovf_now << 16) | tcnt_now;
    n_classifications         = BENCH_N;
    /* Force the optimiser to keep `accum` live. */
    bench_status              = (uint8_t)(1u | (accum & 0u));

    /* gdb anchor.  All measurement variables are stable from here. */
    bench_done();

    /* Park forever. */
    for (;;) {
        __asm__ volatile ("nop");
    }
}
