# ──────────────────────────────────────────────────────────────────
#  STAG — embedded.benchmark.msp430.Makefile
#  « build the MSP430F5529 benchmark firmware »
# ──────────────────────────────────────────────────────────────────

TI_DIR    := $(HOME)/ti/msp430-gcc
CC        := $(TI_DIR)/bin/msp430-elf-gcc
OBJDUMP   := $(TI_DIR)/bin/msp430-elf-objdump
SIZE      := $(TI_DIR)/bin/msp430-elf-size

SRC_DIR   := .
EMBED_DIR := ../..
BUILD_DIR := build

# F5529-specific include + linker search paths.  msp430-elf-gcc needs
# both -I (for msp430g2553.h) and -L (for msp430g2553.ld) — the former
# is auto-found via `-mmcu=msp430g2553` only if the TI include dir is
# the default search path, which it is not for non-system installs.
TI_INCLUDE := $(TI_DIR)/include

CFLAGS  := -mmcu=msp430g2553 \
           -Wall -Wextra -std=c11 -Os -g \
           -fdata-sections -ffunction-sections \
           -I$(TI_INCLUDE) -I$(EMBED_DIR) -I$(SRC_DIR)

# Linker-only flags — do NOT repeat -mmcu= here, the driver passes
# the linker script once already and duplicates trigger
# "linker script file 'msp430g2553.ld' appears multiple times".
LDFLAGS := -L$(TI_INCLUDE) -Wl,--gc-sections

SOURCES := $(SRC_DIR)/benchmark_msp430.c \
           $(EMBED_DIR)/nearest_centroid.c

.PHONY: all clean msp430g2553

all: msp430g2553

msp430g2553: $(BUILD_DIR)/msp430g2553/benchmark.elf $(BUILD_DIR)/msp430g2553/benchmark.lss

$(BUILD_DIR)/msp430g2553:
	@mkdir -p $@

$(BUILD_DIR)/msp430g2553/benchmark.elf: $(SOURCES) test_vectors.h | $(BUILD_DIR)/msp430g2553
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SOURCES)
	@$(SIZE) --format=berkeley $@

$(BUILD_DIR)/msp430g2553/benchmark.lss: $(BUILD_DIR)/msp430g2553/benchmark.elf
	$(OBJDUMP) -h -S $< > $@

# ── Inputs (regenerable) ──────────────────────────────────────────
# The Makefile lives at stag/embedded/benchmark/msp430/Makefile;
# the repo root (containing the stag/ package) is four levels up.
REPO_ROOT := ../../../..

test_vectors.h:
	@cd $(REPO_ROOT) && PYTHONPATH=. python -m stag.embedded.benchmark.avr.gen_inputs \
		--arch msp430 \
		--out stag/embedded/benchmark/msp430/test_vectors.h

clean:
	rm -rf $(BUILD_DIR) test_vectors.h
