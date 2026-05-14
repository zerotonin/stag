/* ╔══════════════════════════════════════════════════════════════════╗
 * ║  STAG — embedded.benchmark.xtensa.esp32.benchmark_esp32           ║
 * ║  « time the nearest-centroid classifier on ESP32 (Xtensa LX6) »   ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  Self-timed via the Xtensa CCOUNT special register read by       ║
 * ║  `rsr.ccount` — always-on cycle counter, no setup needed.        ║
 * ║                                                                  ║
 * ║  Result reporting: write the cycle count + n_classifications     ║
 * ║  to ESP32 UART0 (TX FIFO at 0x3FF40000), QEMU's `-serial file:`  ║
 * ║  captures the bytes for the host-side runner to grep.  Avoids    ║
 * ║  the need for an Xtensa-aware gdb (the Espressif crosstool       ║
 * ║  toolchain ships gcc + binutils but not gdb).                    ║
 * ║                                                                  ║
 * ║  Per-call timing pattern (same as MSP430):                       ║
 * ║    The classifier has a data-dependent early-out optimisation,   ║
 * ║    so cycles/call varies by input.  Sweeping the full 64-vector  ║
 * ║    test set in one pass and accumulating per-call cycles into a  ║
 * ║    32-bit total gives the same population mean the AVR / ARM     ║
 * ║    runs converge on with N=1000.                                 ║
 * ║                                                                  ║
 * ║  Built with -mabi=call0 -mlongcalls -mtext-section-literals so   ║
 * ║  the firmware avoids the windowed register file entirely.        ║
 * ╚══════════════════════════════════════════════════════════════════╝ */

#include <stdint.h>

#include "../../../nearest_centroid.h"
#include "test_vectors.h"


#define BENCH_N    TEST_VECTORS_N    /* full 64-vector sweep */

#define UART0_FIFO_REG  ((volatile uint32_t *)0x3FF40000u)


/* Always-on 32-bit cycle counter on Xtensa LX6. */
static inline uint32_t read_ccount(void)
{
    uint32_t cc;
    __asm__ volatile ("rsr.ccount %0" : "=r"(cc) : : "memory");
    return cc;
}


static inline void uart_putc(char c)
{
    *UART0_FIFO_REG = (uint32_t)c;
}


static void uart_puts(const char *s)
{
    while (*s) {
        uart_putc(*s++);
    }
}


static void uart_puthex(volatile uint32_t x)
{
    uart_putc('0');
    uart_putc('x');
    for (volatile int i = 7; i >= 0; --i) {
        uint32_t d = (x >> (i * 4)) & 0xFu;
        uart_putc(d < 10 ? (char)('0' + d) : (char)('a' + d - 10));
    }
}


void main(void)
{
    int16_t  input[STAG_N_FEATURES];
    uint16_t accum = 0;
    uint32_t total = 0;

    for (uint16_t i = 0; i < BENCH_N; ++i) {
        for (uint8_t d = 0; d < STAG_N_FEATURES; ++d) {
            input[d] = test_inputs_q[i][d];
        }

        const uint32_t t0  = read_ccount();
        const uint8_t  cls = stag_nearest_centroid_q412(input);
        const uint32_t t1  = read_ccount();

        total += (t1 - t0);
        accum = (uint16_t)(accum + cls);
    }

    /* Emit a single line the runner regexes for:
     *   STAG_RESULT cycles=0xXXXXXXXX n=0xXX
     * Halt afterwards — QEMU will keep emulating until the runner
     * times it out and SIGKILLs the process. */
    uart_puts("STAG_RESULT cycles=");
    uart_puthex(total);
    uart_puts(" n=");
    uart_puthex(BENCH_N);
    uart_putc('\n');

    (void)accum;
    while (1) { /* park forever */ }
}
