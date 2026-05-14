/* ──────────────────────────────────────────────────────────────────
 *  STAG — embedded.benchmark.arm.samd21.linker
 *  « minimal Cortex-M0+ linker script for the SAMD21G18A »
 *
 *  Target silicon: Microchip / Atmel SAMD21G18A as found on the
 *  Adafruit Feather M0, Arduino MKR series, and Trinket M0.
 *
 *      FLASH  0x00000000  256 KB
 *      SRAM   0x20000000   32 KB
 *
 *  Cortex-M0+ has no FPU and only the Thumb-1 subset (no IT blocks,
 *  no UMULL, no UDIV).  The compiler emits more single-cycle
 *  instructions per source statement than on M3+, so the
 *  ExecutedInstructions count is expected to be 2–3× higher than
 *  the 414.7 measured on M3+ Thumb-2 targets.
 *
 *  __stack_top via PROVIDE() — same pattern as the M3+ targets to
 *  avoid the .stack-section / bss-over-stack hazard.
 * ────────────────────────────────────────────────────────────────── */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH =  32K
}

ENTRY(Reset_Handler)

PROVIDE(__stack_top = ORIGIN(SRAM) + LENGTH(SRAM) - 4);

SECTIONS
{
    .isr_vector :
    {
        KEEP(*(.isr_vector))
    } > FLASH

    .text :
    {
        *(.text .text.* .gnu.linkonce.t.*)
        *(.rodata .rodata.* .gnu.linkonce.r.*)
        *(.glue_7 .glue_7t)
        *(.ARM.extab* .gnu.linkonce.armextab.*)
        . = ALIGN(4);
        __data_load_start = .;
    } > FLASH

    .data : AT(__data_load_start)
    {
        __data_start = .;
        *(.data .data.* .gnu.linkonce.d.*)
        . = ALIGN(4);
        __data_end = .;
    } > SRAM

    .bss (NOLOAD) :
    {
        __bss_start = .;
        *(.bss .bss.* .gnu.linkonce.b.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > SRAM

    /DISCARD/ :
    {
        *(.ARM.exidx* .ARM.exidx.*)
        *(.eh_frame)
    }
}
