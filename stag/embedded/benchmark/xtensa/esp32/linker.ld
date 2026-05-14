/* ──────────────────────────────────────────────────────────────────
 *  STAG — embedded.benchmark.xtensa.esp32.linker
 *  « minimal Xtensa LX6 linker script for QEMU-emulated ESP32 »
 *
 *      IRAM   0x40080000  128 KB   (instruction RAM, where code lives
 *                                   when loaded via QEMU `-kernel`)
 *      DRAM   0x3FFB0000  320 KB   (data RAM, bss/data/stack)
 *
 *  These are the ESP32-V3 silicon addresses, which QEMU's
 *  `-machine esp32` emulates verbatim.  No FLASH section because
 *  we sidestep the boot-ROM + flash partition flow by passing the
 *  ELF directly via -kernel.
 *
 *  Stack top is set imperatively in startup_esp32.S
 *  (movi a1, 0x3FFE4000), well above the .bss high-water mark.
 * ────────────────────────────────────────────────────────────────── */

ENTRY(_start)

MEMORY
{
    iram (rx)  : ORIGIN = 0x40080000, LENGTH = 128K
    dram (rwx) : ORIGIN = 0x3FFB0000, LENGTH = 320K
}

SECTIONS
{
    .text 0x40080000 :
    {
        *(.text.start)
        *(.text .text.*)
        *(.literal .literal.*)
    } > iram

    .rodata :
    {
        *(.rodata .rodata.*)
    } > iram

    .data :
    {
        *(.data .data.*)
    } > dram

    .bss (NOLOAD) :
    {
        *(.bss .bss.*)
        *(COMMON)
    } > dram

    /DISCARD/ :
    {
        *(.xt.lit)
        *(.xt.prop)
        *(.note.*)
        *(.comment)
    }
}
