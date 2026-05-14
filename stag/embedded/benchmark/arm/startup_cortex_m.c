/* ╔══════════════════════════════════════════════════════════════════╗
 * ║  STAG — embedded.benchmark.arm.startup_cortex_m                  ║
 * ║  « minimal Cortex-M reset handler + vector table »               ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  One source file covers every Cortex-M target in the Sprint 4   ║
 * ║  benchmark.  Avoids the per-vendor CMSIS / startup boilerplate  ║
 * ║  by using only what the architecture itself defines (the first ║
 * ║  16 exception vectors are core, identical on every Cortex-M).   ║
 * ║                                                                  ║
 * ║  The benchmark does not enable any interrupts, so the table is  ║
 * ║  terminated at IRQ entry 0 — we never need vendor-specific      ║
 * ║  IRQ handlers.                                                   ║
 * ╚══════════════════════════════════════════════════════════════════╝ */

#include <stdint.h>

extern int main(void);

/* Linker-provided symbols (see linker script). */
extern uint32_t __stack_top;
extern uint32_t __data_load_start;
extern uint32_t __data_start;
extern uint32_t __data_end;
extern uint32_t __bss_start;
extern uint32_t __bss_end;


__attribute__((noreturn))
void Reset_Handler(void)
{
    /* Copy initialised data from flash to SRAM. */
    {
        uint32_t *src = &__data_load_start;
        uint32_t *dst = &__data_start;
        while (dst < &__data_end) {
            *dst++ = *src++;
        }
    }
    /* Zero .bss. */
    {
        uint32_t *p = &__bss_start;
        while (p < &__bss_end) {
            *p++ = 0;
        }
    }
    (void)main();
    for (;;) { __asm__ volatile ("wfi"); }
}


__attribute__((noreturn, weak))
void Default_Handler(void)
{
    for (;;) {
        __asm__ volatile ("bkpt #0");
    }
}


/* Aliases for the 14 fault / system exceptions.  All point at
 * Default_Handler unless overridden by the firmware; we do not
 * override any in the benchmark. */
#define ALIAS(x) __attribute__((weak, alias(#x), noreturn))

void NMI_Handler          (void) ALIAS(Default_Handler);
void HardFault_Handler    (void) ALIAS(Default_Handler);
void MemManage_Handler    (void) ALIAS(Default_Handler);
void BusFault_Handler     (void) ALIAS(Default_Handler);
void UsageFault_Handler   (void) ALIAS(Default_Handler);
void SVC_Handler          (void) ALIAS(Default_Handler);
void DebugMon_Handler     (void) ALIAS(Default_Handler);
void PendSV_Handler       (void) ALIAS(Default_Handler);
void SysTick_Handler      (void) ALIAS(Default_Handler);


/* Architectural vector table (first 16 entries). */
__attribute__((section(".isr_vector"), used))
void (* const _vectors[])(void) = {
    (void (*)(void))(&__stack_top),  /* 0  initial SP */
    Reset_Handler,                   /* 1  reset      */
    NMI_Handler,                     /* 2  NMI        */
    HardFault_Handler,               /* 3  hard fault */
    MemManage_Handler,               /* 4  MPU fault  */
    BusFault_Handler,                /* 5  bus fault  */
    UsageFault_Handler,              /* 6  usage      */
    0, 0, 0, 0,                      /* 7-10 reserved */
    SVC_Handler,                     /* 11 SVCall     */
    DebugMon_Handler,                /* 12 debug      */
    0,                               /* 13 reserved   */
    PendSV_Handler,                  /* 14 PendSV     */
    SysTick_Handler,                 /* 15 SysTick    */
};
