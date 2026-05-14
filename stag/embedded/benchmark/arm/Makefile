# ──────────────────────────────────────────────────────────────────
#  STAG — embedded.benchmark.arm.Makefile
#  « one source, multiple Cortex-M targets »
# ──────────────────────────────────────────────────────────────────

CC        := arm-none-eabi-gcc
OBJDUMP   := arm-none-eabi-objdump
SIZE      := arm-none-eabi-size

SRC_DIR   := .
EMBED_DIR := ../..
BUILD_DIR := build

COMMON_CFLAGS := -Wall -Wextra -std=c11 -Os -g \
                 -fdata-sections -ffunction-sections \
                 -ffreestanding -nostartfiles \
                 -I$(EMBED_DIR) -I$(SRC_DIR)

COMMON_LDFLAGS := -nostartfiles --specs=nosys.specs --specs=nano.specs \
                  -Wl,--gc-sections

SOURCES   := $(SRC_DIR)/benchmark_arm.c \
             $(SRC_DIR)/startup_cortex_m.c \
             $(EMBED_DIR)/nearest_centroid.c

# ── Per-target architecture flags ─────────────────────────────────
# Cortex-M4F (nRF52840, STM32F407): hard-float, FPv4-SP-D16
NRF52840_ARCH      := -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
STM32F4_DISCO_ARCH := -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
# Cortex-M7 (iMXRT1064)          : hard-float, FPv5-D16 (double precision)
IMXRT1064_ARCH     := -mcpu=cortex-m7 -mthumb -mfloat-abi=hard -mfpu=fpv5-d16
# Add others (samd21, rp2040, etc.) as Phase 4d+ targets land.

.PHONY: all clean nrf52840 imxrt1064 stm32f4_disco

all: nrf52840 imxrt1064 stm32f4_disco

# ── nRF52840 ──────────────────────────────────────────────────────
nrf52840: $(BUILD_DIR)/nrf52840/benchmark.elf $(BUILD_DIR)/nrf52840/benchmark.lss

$(BUILD_DIR)/nrf52840:
	@mkdir -p $@

$(BUILD_DIR)/nrf52840/benchmark.elf: $(SOURCES) nrf52840/linker.ld nrf52840/test_vectors.h | $(BUILD_DIR)/nrf52840
	$(CC) $(NRF52840_ARCH) $(COMMON_CFLAGS) -I$(SRC_DIR)/nrf52840 \
		$(COMMON_LDFLAGS) -T$(SRC_DIR)/nrf52840/linker.ld \
		-o $@ $(SOURCES)
	@$(SIZE) --format=berkeley $@

$(BUILD_DIR)/nrf52840/benchmark.lss: $(BUILD_DIR)/nrf52840/benchmark.elf
	$(OBJDUMP) -h -S $< > $@

# ── iMXRT1064 ─────────────────────────────────────────────────────
imxrt1064: $(BUILD_DIR)/imxrt1064/benchmark.elf $(BUILD_DIR)/imxrt1064/benchmark.lss

$(BUILD_DIR)/imxrt1064:
	@mkdir -p $@

$(BUILD_DIR)/imxrt1064/benchmark.elf: $(SOURCES) imxrt1064/linker.ld imxrt1064/test_vectors.h | $(BUILD_DIR)/imxrt1064
	$(CC) $(IMXRT1064_ARCH) $(COMMON_CFLAGS) -I$(SRC_DIR)/imxrt1064 \
		$(COMMON_LDFLAGS) -T$(SRC_DIR)/imxrt1064/linker.ld \
		-o $@ $(SOURCES)
	@$(SIZE) --format=berkeley $@

$(BUILD_DIR)/imxrt1064/benchmark.lss: $(BUILD_DIR)/imxrt1064/benchmark.elf
	$(OBJDUMP) -h -S $< > $@

# ── STM32F4-Discovery (STM32F407VG) ───────────────────────────────
stm32f4_disco: $(BUILD_DIR)/stm32f4_disco/benchmark.elf $(BUILD_DIR)/stm32f4_disco/benchmark.lss

$(BUILD_DIR)/stm32f4_disco:
	@mkdir -p $@

$(BUILD_DIR)/stm32f4_disco/benchmark.elf: $(SOURCES) stm32f4_disco/linker.ld stm32f4_disco/test_vectors.h | $(BUILD_DIR)/stm32f4_disco
	$(CC) $(STM32F4_DISCO_ARCH) $(COMMON_CFLAGS) -I$(SRC_DIR)/stm32f4_disco \
		$(COMMON_LDFLAGS) -T$(SRC_DIR)/stm32f4_disco/linker.ld \
		-o $@ $(SOURCES)
	@$(SIZE) --format=berkeley $@

$(BUILD_DIR)/stm32f4_disco/benchmark.lss: $(BUILD_DIR)/stm32f4_disco/benchmark.elf
	$(OBJDUMP) -h -S $< > $@

# ── Inputs (regenerable) ──────────────────────────────────────────
# The Makefile lives at stag/embedded/benchmark/arm/Makefile; the repo
# root (containing the stag/ package) is four levels up.
REPO_ROOT := ../../../..

nrf52840/test_vectors.h:
	@cd $(REPO_ROOT) && PYTHONPATH=. python -m stag.embedded.benchmark.avr.gen_inputs \
		--arch arm \
		--out stag/embedded/benchmark/arm/nrf52840/test_vectors.h

imxrt1064/test_vectors.h:
	@cd $(REPO_ROOT) && PYTHONPATH=. python -m stag.embedded.benchmark.avr.gen_inputs \
		--arch arm \
		--out stag/embedded/benchmark/arm/imxrt1064/test_vectors.h

stm32f4_disco/test_vectors.h:
	@cd $(REPO_ROOT) && PYTHONPATH=. python -m stag.embedded.benchmark.avr.gen_inputs \
		--arch arm \
		--out stag/embedded/benchmark/arm/stm32f4_disco/test_vectors.h

clean:
	rm -rf $(BUILD_DIR) nrf52840/test_vectors.h imxrt1064/test_vectors.h \
	       stm32f4_disco/test_vectors.h
