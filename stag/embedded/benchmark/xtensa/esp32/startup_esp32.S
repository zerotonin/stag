// ──────────────────────────────────────────────────────────────────
//  STAG — embedded.benchmark.xtensa.esp32.startup_esp32
//  « minimal Xtensa LX6 entry stub for QEMU-emulated ESP32 »
//
//  Sole responsibility: set the stack pointer to the top of DRAM
//  and call main().  Everything else (vector tables, FLEX clocks,
//  cache, etc.) is irrelevant for an instruction-count benchmark
//  running under QEMU's `-machine esp32` model — the boot ROM
//  jumps to our _start via the -kernel mechanism with sane
//  default register state.
//
//  CALL0 ABI throughout (built with -mabi=call0) so we do not need
//  PS.WOE / CALLINC setup for windowed-register ENTRY/RETW.
// ──────────────────────────────────────────────────────────────────

.section .text.start, "ax"
.global _start
_start:
    // Stack pointer to high DRAM (0x3FFE0000 + 0x4000 = 0x3FFE4000).
    // ESP32 DRAM extends 0x3FF80000-0x3FFFFFFF; this lands safely
    // below the BootROM scratch area and above the .bss region the
    // linker script places in 0x3FFB0000.
    movi a1, 0x3FFE0000
    addi a1, a1, 0x4000
    call0 main
1:  j 1b
