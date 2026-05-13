# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — embedded                                                 ║
# ║  « bare-metal nearest-centroid classifier + MCU benchmarks »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Sprint 4 / R3 Q3 item 6 + R1 #8/9: the manuscript's claim that ║
# ║  the classifier runs in real time on essentially every MCU a   ║
# ║  wildlife biologger could be built with.                        ║
# ║                                                                  ║
# ║  Subpackages:                                                   ║
# ║    export_centroids.py — Python → C header (Q4.12 + float)      ║
# ║    nearest_centroid.{c,h} — canonical bare-metal classifier     ║
# ║    benchmark/avr/   — simavr runners (Uno, Micro, Mega)         ║
# ║    benchmark/arm/   — Renode runners (STM32L476, nRF52840,      ║
# ║                       SAMD21, RP2040, iMXRT1062)                ║
# ║    benchmark/xtensa/esp32/    — Renode ESP32                    ║
# ║    benchmark/msp430/          — Renode MSP430F5529              ║
# ║    benchmark/compose_table.py — merges per-MCU CSVs             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""STAG embedded deployment + ten-MCU emulator benchmark."""
