# ──────────────────────────────────────────────────────────────────
#  STAG — embedded.benchmark.xtensa.esp32.Makefile
#  « build the ESP32 (Xtensa LX6) benchmark firmware »
# ──────────────────────────────────────────────────────────────────

TOOLCHAIN_BIN := $(HOME)/opt/xtensa-esp-elf/bin
CC            := $(TOOLCHAIN_BIN)/xtensa-esp32-elf-gcc
OBJDUMP       := $(TOOLCHAIN_BIN)/xtensa-esp32-elf-objdump
SIZE          := $(TOOLCHAIN_BIN)/xtensa-esp32-elf-size

SRC_DIR       := .
# The Makefile lives at stag/embedded/benchmark/xtensa/esp32/.
# Shared embedded code (nearest_centroid.c + the central headers)
# sits three levels up at stag/embedded/.
EMBED_DIR     := ../../..
BUILD_DIR     := build

# CALL0 ABI sidesteps the windowed register file entirely — main()
# uses a normal stack-frame prologue, no ENTRY/RETW pair.
# -mlongcalls keeps far call targets reachable.
# -mtext-section-literals keeps L32R literal pools in the same
# section as the code that loads them; with our linker placing
# .text before .literal anyway, this prevents "dangerous relocation"
# errors during linking.
ESP32_ARCH    := -mabi=call0 -mlongcalls -mtext-section-literals

COMMON_CFLAGS := -Wall -Wextra -std=c11 -Os -g \
                 -fdata-sections -ffunction-sections \
                 -ffreestanding -nostartfiles -nostdlib \
                 -I$(EMBED_DIR) -I$(SRC_DIR)

LDFLAGS       := -Wl,--gc-sections

SOURCES       := $(SRC_DIR)/startup_esp32.S \
                 $(SRC_DIR)/benchmark_esp32.c \
                 $(EMBED_DIR)/nearest_centroid.c

.PHONY: all clean esp32

all: esp32

esp32: $(BUILD_DIR)/esp32/benchmark.elf $(BUILD_DIR)/esp32/benchmark.lss

$(BUILD_DIR)/esp32:
	@mkdir -p $@

$(BUILD_DIR)/esp32/benchmark.elf: $(SOURCES) linker.ld test_vectors.h | $(BUILD_DIR)/esp32
	$(CC) $(ESP32_ARCH) $(COMMON_CFLAGS) $(LDFLAGS) \
		-T $(SRC_DIR)/linker.ld -o $@ $(SOURCES)
	@$(SIZE) $@

$(BUILD_DIR)/esp32/benchmark.lss: $(BUILD_DIR)/esp32/benchmark.elf
	$(OBJDUMP) -h -S $< > $@

# ── Inputs (regenerable) ──────────────────────────────────────────
# The Makefile lives at stag/embedded/benchmark/xtensa/esp32/Makefile;
# the repo root (containing the stag/ package) is five levels up.
REPO_ROOT := ../../../../..

test_vectors.h:
	@cd $(REPO_ROOT) && PYTHONPATH=. python -m stag.embedded.benchmark.avr.gen_inputs \
		--arch xtensa \
		--out stag/embedded/benchmark/xtensa/esp32/test_vectors.h

clean:
	rm -rf $(BUILD_DIR) test_vectors.h
