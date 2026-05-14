# ──────────────────────────────────────────────────────────────────
#  STAG — embedded.benchmark.avr.Makefile
#  « build benchmark_avr.elf for atmega328p / atmega32u4 / atmega2560 »
# ──────────────────────────────────────────────────────────────────

CC        := avr-gcc
OBJDUMP   := avr-objdump
SIZE      := avr-size
F_CPU     := 16000000UL

SRC_DIR   := .
EMBED_DIR := ../..
BUILD_DIR := build

CFLAGS    := -Wall -Wextra -std=c11 -Os -g \
             -fdata-sections -ffunction-sections \
             -DF_CPU=$(F_CPU) \
             -I$(EMBED_DIR) -I$(SRC_DIR)

LDFLAGS   := -Wl,--gc-sections

SOURCES   := $(SRC_DIR)/benchmark_avr.c $(EMBED_DIR)/nearest_centroid.c

MCUS      := atmega328p atmega32u4 atmega2560

.PHONY: all clean $(MCUS)

all: $(MCUS)

# One target per MCU.  Each builds its own ELF + LSS in build/<mcu>/.
define build_mcu_template
$(1): $$(BUILD_DIR)/$(1)/benchmark.elf $$(BUILD_DIR)/$(1)/benchmark.lss

$$(BUILD_DIR)/$(1):
	@mkdir -p $$@

$$(BUILD_DIR)/$(1)/benchmark.elf: $$(SOURCES) | $$(BUILD_DIR)/$(1)
	$$(CC) -mmcu=$(1) $$(CFLAGS) $$(LDFLAGS) -o $$@ $$(SOURCES)
	@$$(SIZE) --format=avr --mcu=$(1) $$@

$$(BUILD_DIR)/$(1)/benchmark.lss: $$(BUILD_DIR)/$(1)/benchmark.elf
	$$(OBJDUMP) -h -S $$< > $$@
endef

$(foreach m, $(MCUS), $(eval $(call build_mcu_template,$(m))))

clean:
	rm -rf $(BUILD_DIR)
