ARCH := riscv64-unknown
SRC  := ./src
INC  := ./include
LIB  := ./lib
OBJ  := ./obj
BIN  := ./bin
DUMP := ./dump

# toolchain
RV_CC ?= $(ARCH)-elf-gcc
RV_CFLAG := -ffreestanding -nostdlib -march=rv64gcv -mabi=lp64d -mcmodel=medany -I$(INC) -g
RV_LD ?= $(ARCH)-elf-ld
RV_LDFLAG := -static -g
RV_DUMP ?= $(ARCH)-elf-objdump
RV_DUMPFLAG := -M no-aliases -S
RV_OBJCOPY ?= $(ARCH)-elf-objcopy

# qemu
QEMU ?= qemu-system-$(ARCH)

# prompt
VERBOSE ?= 0
# ifeq ($(VERBOSE), 0)
# 	Q := @
# else
RV_CC_PROMPT := RV_CC
RV_LD_PROMPT := RV_LD
RV_DUMP_PROMPT := RV_DUMP
RV_OBJCOPY_PROMPT := RV_OBJCOPY

TARGETS += main fib prime matrix_mul vector_kernels io_bench trap_test virtio_blk_test ecall_test interrupt_test float clint uart_interrupt_test
TARGET_ELFS := $(patsubst %,$(BIN)/%.elf,$(TARGETS))
TARGET_DUMPS := $(patsubst %,$(DUMP)/%.dump,$(TARGETS))
TARGET ?= main

LIBS := io log boot trap trap_handler power
LIBS_OBJ := $(patsubst %,$(OBJ)/%.o,$(LIBS))

# proctect intermediate file.
.PRECIOUS: $(OBJ)/%.o

.PHONY: build_lib build run clean debug_makefile
DEFAULT_TARGET: build dump_all
echo_cmd = printf "%-10s %s\n" "$(1)" "$(strip $(2))"

build: $(TARGET_ELFS)

# trap.o
$(OBJ)/trap.o: $(LIB)/trap.S
	@ mkdir -p $(OBJ) $(BIN)
	@ $(call echo_cmd, $(RV_CC_PROMPT), $@)
	@ $(RV_CC) $(RV_CFLAG) -c $< -o $@

# boot.o
$(OBJ)/boot.o: $(SRC)/boot.S
	@ mkdir -p $(OBJ) $(BIN)
	@ $(call echo_cmd, $(RV_CC_PROMPT), $@)
	@ $(RV_CC) $(RV_CFLAG) -c $< -o $@

# default lib file.
$(OBJ)/%.o: $(LIB)/%.c
	@ mkdir $(OBJ) $(BIN) -p
	@ $(call echo_cmd, $(RV_CC_PROMPT), $@)
	@ $(RV_CC) $(RV_CFLAG) -c $^ -o $@

$(BIN)/%.elf: $(OBJ)/%.o $(LIBS_OBJ) linker.ld
	@ mkdir -p $(OBJ) $(BIN)
	@ $(call echo_cmd, $(RV_LD_PROMPT), $@)
	@ $(RV_LD) $(RV_LDFLAG) -T linker.ld -o $@ $(filter %.o,$^)
	
# default .c file
$(OBJ)/%.o: $(SRC)/%.c
	@ mkdir -p $(OBJ) $(BIN)
	@ $(call echo_cmd, $(RV_CC_PROMPT), $@)
	@ $(RV_CC) $(RV_CFLAG) -c $< -o $@


# dump target elf
dump_all: $(TARGET_DUMPS)

$(DUMP)/%.dump: $(BIN)/%.elf
	@ mkdir -p $(DUMP)
	@ $(call echo_cmd, $(RV_DUMP_PROMPT), $@)
	@ $(RV_DUMP) -d $(RV_DUMPFLAG) $< > $@


run: build
	$(QEMU) -machine virt -bios $(BIN)/$(TARGET).elf -serial stdio

clean:
	rm -r $(BIN) $(OBJ)