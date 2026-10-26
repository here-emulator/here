# Makefile for building and running linux kernel on `HERE` (and on QEMU for debugging).

JOBS ?= 14
CROSS_COMPILE ?= riscv64-linux-gnu-
PLATFORM_RISCV_ISA ?= rv64gc
FW_PAYLOAD_FDT_ADDR ?= 0x9F000000
RVEMU_ARGS ?=

EMU_DIR ?= $(CURDIR)
LINUX_DIR ?=
OPENSBI_DIR ?=

DTS_FILE ?= $(EMU_DIR)/dts/virt.dts
DTB_FILE ?= $(EMU_DIR)/dts/virt.dtb
LINUX_IMAGE ?= $(LINUX_DIR)/arch/riscv/boot/Image
FW_BIN ?= $(OPENSBI_DIR)/build/platform/generic/firmware/fw_payload.bin

.PHONY: check build-dtb build-linux build-opensbi linux-qemu linux-qemu-gdb linux linux-debug linux-gdb

check:
	@test -n "$(LINUX_DIR)" || (echo "error: LINUX_DIR is empty. set env LINUX_DIR=... or run make LINUX_DIR=..."; exit 1)
	@test -n "$(OPENSBI_DIR)" || (echo "error: OPENSBI_DIR is empty. set env OPENSBI_DIR=... or run make OPENSBI_DIR=..."; exit 1)
	@test -f "$(DTS_FILE)" || (echo "error: missing $(DTS_FILE)"; exit 1)
	@test -f "$(LINUX_DIR)/Makefile" || (echo "error: missing $(LINUX_DIR)/Makefile"; exit 1)
	@test -f "$(OPENSBI_DIR)/Makefile" || (echo "error: missing $(OPENSBI_DIR)/Makefile"; exit 1)

build-dtb: check
	dtc -I dts -O dtb -o "$(DTB_FILE)" "$(DTS_FILE)"

build-linux: check
	$(MAKE) -C "$(LINUX_DIR)" ARCH=riscv CROSS_COMPILE="$(CROSS_COMPILE)" Image -j"$(JOBS)"

build-opensbi: build-dtb build-linux
	@test -f "$(LINUX_IMAGE)" || (echo "error: missing $(LINUX_IMAGE)"; exit 1)
	$(MAKE) -C "$(OPENSBI_DIR)" PLATFORM=generic -j"$(JOBS)" \
		CROSS_COMPILE="$(CROSS_COMPILE)" \
		PLATFORM_RISCV_ISA="$(PLATFORM_RISCV_ISA)" \
		FW_PAYLOAD_PATH="$(LINUX_IMAGE)" \
		FW_PAYLOAD_FDT_ADDR="$(FW_PAYLOAD_FDT_ADDR)"
	@test -f "$(FW_BIN)" || (echo "error: missing $(FW_BIN)"; exit 1)

linux-qemu: build-opensbi
	qemu-system-riscv64 -M virt -m 8G -nographic -bios "$(FW_BIN)"

linux-qemu-gdb: build-opensbi
	qemu-system-riscv64 -M virt -m 8G -nographic -bios "$(FW_BIN)" -s -S

linux: build-opensbi
	cargo run -p here-bin --release -- "$(FW_BIN)" --dtb "$(DTB_FILE)" --dtb-address "$(FW_PAYLOAD_FDT_ADDR)" $(RVEMU_ARGS)

linux-debug: build-opensbi
	cargo run -p here-bin --release -- "$(FW_BIN)" --dtb "$(DTB_FILE)" --dtb-address "$(FW_PAYLOAD_FDT_ADDR)" -g $(RVEMU_ARGS)

linux-gdb: build-opensbi
	cargo run -p here-bin --release -- "$(FW_BIN)" --dtb "$(DTB_FILE)" --dtb-address "$(FW_PAYLOAD_FDT_ADDR)" -G $(RVEMU_ARGS)