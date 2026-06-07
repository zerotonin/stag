/* ──────────────────────────────────────────────────────────────────
 *  STAG — embedded.benchmark.arm.rp2040.linker
 *  « minimal Cortex-M0+ linker script for the Raspberry Pi RP2040 »
 *
 *  Target silicon: RP2040 as found on the Raspberry Pi Pico, the
 *  Adafruit Feather RP2040, the SparkFun Pro Micro RP2040, etc.
 *
 *      FLASH  0x00000000  2 MB    (modelled at ROM region for the
 *                                  emulator benchmark — see the
 *                                  comment in rp2040.repl for why
 *                                  we do not model XIP at
 *                                  0x10000000 here)
 *      SRAM   0x20000000  256 KB
 *
 *  Cortex-M0+ Thumb-1 only — same caveats as the SAMD21 linker
 *  script.  Single-core slice; the second M0+ core is unused in
 *  this benchmark.  Instruction count is independent of how many
 *  cores exist — we measure one classifier loop on one core.
 * ────────────────────────────────────────────────────────────────── */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 2048K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH =  256K
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
