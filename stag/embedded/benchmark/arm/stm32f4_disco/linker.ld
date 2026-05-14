/* ──────────────────────────────────────────────────────────────────
 *  STAG — embedded.benchmark.arm.stm32f4_disco.linker
 *  « minimal Cortex-M4F linker script for the STM32F4-Discovery »
 *
 *  Target silicon: STM32F407VG on the original STM32F4-Discovery
 *  evaluation board.  168 MHz Cortex-M4F with FPv4-SP-D16.
 *
 *      FLASH  0x08000000  1 MB
 *      SRAM   0x20000000  128 KB  (SRAM1, the contiguous block)
 *
 *  Renode's platforms/cpus/stm32f4.repl actually exposes a 2 MB
 *  flash region and a 256 KB SRAM region.  We declare the silicon-
 *  accurate 1 MB / 128 KB so this same linker script ships clean
 *  for a real-hardware port — Renode tolerates an undersized MEMORY
 *  spec without complaint.
 *
 *  __stack_top is provided four bytes below the top of SRAM via a
 *  PROVIDE() expression (same pattern as nrf52840 / imxrt1064; the
 *  alternative .stack-section approach made .bss span the entire
 *  SRAM and Reset_Handler zeroed its own stack frame at boot).
 * ────────────────────────────────────────────────────────────────── */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH =  128K
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
