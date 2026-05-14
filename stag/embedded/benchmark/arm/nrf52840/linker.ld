/* ──────────────────────────────────────────────────────────────────
 *  STAG — embedded.benchmark.arm.nrf52840.linker
 *  « minimal Cortex-M4 linker script for the nRF52840 »
 *
 *  The previous version used `. = ORIGIN(SRAM) + LENGTH(SRAM) - 4`
 *  inside a .stack section, which made the toolchain report .bss
 *  as 256 KB (the entire SRAM) and — worse — caused Reset_Handler's
 *  bss-zeroing loop to overwrite its own stack frame at boot.  This
 *  version places __stack_top via a PROVIDE() expression and keeps
 *  .bss tight around the actual zero-initialised data only.
 * ────────────────────────────────────────────────────────────────── */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 1024K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH =  256K
}

ENTRY(Reset_Handler)

/* Initial SP value the vector table reads at offset 0.
 * Word-aligned, four bytes below the top of SRAM. */
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

    /* Discard ARM exception-handling tables we do not need. */
    /DISCARD/ :
    {
        *(.ARM.exidx* .ARM.exidx.*)
        *(.eh_frame)
    }
}
