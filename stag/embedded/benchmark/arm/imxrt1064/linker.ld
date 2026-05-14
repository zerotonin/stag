/* ──────────────────────────────────────────────────────────────────
 *  STAG — embedded.benchmark.arm.imxrt1064.linker
 *  « minimal Cortex-M7 linker script for the i.MX RT1064 »
 *
 *  The 1064 is the highest-perf Cortex-M in the Sprint 4 matrix
 *  (M7 @ 600 MHz, double-issue, FPv5-D16 with hardware double
 *  precision).  Renode's bundled imxrt1064.repl exposes:
 *
 *      ITCM   0x00000000  512 KB   (code, fastest fetch path)
 *      DTCM   0x20000000  512 KB   (data + stack, single-cycle)
 *      OCRAM2 0x20200000  512 KB   (unused here)
 *
 *  We load the firmware into ITCM (so the M7's tight-coupled
 *  fetch path is exercised — the realistic deployment pattern
 *  for iMXRT firmware) and place the read/write sections in DTCM.
 *  __stack_top is provided four bytes below DTCM end, matching
 *  the nrf52840 pattern that avoided the bss-over-own-stack hazard.
 * ────────────────────────────────────────────────────────────────── */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 512K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 512K
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
