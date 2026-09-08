# HERE

## About

`HERE` is a HEU Educational Rust-based Emulator for the RISC-V architecture :fire:.

The main features of `HERE` include:

- Supported ISA:
  - RV64GC (RV64IMAFDC, Zicsr, Zifencei)
  - Partial `V` extension support (vector floating-point instructions are still incomplete)
- Supported privilege modes:
  - M, S, and U modes
- A simple debugger monitor called rvdb
- GDB support
- Virtual memory
- Devices:
  - CLINT, PLIC, serial, and VirtIO MMIO (block devices only at present)

An online version with the emulator's core functionality is also available: [rvemu-web](http://here-emulator.github.io/here-web/).

## Build

Install Rust nightly, for example, on Arch Linux:

```sh
sudo pacman -S rustup
rustup default nightly
```

Build:

```sh
cargo build
```

## Testing

We use the [riscv-tests](https://github.com/riscv-software-src/riscv-tests) submodule as our test suite. Initialize the submodule, install [riscv-gnu-toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain), and follow the riscv-tests README to build its ISA binaries:

```sh
git submodule update --init --recursive
```

After building the riscv-tests binaries, run:

```sh
cargo test --features riscv-tests
```

Test support for `riscv-arch-test` also exists, but it is not integrated into CI. Unfortunately, the test suite stabilized at 4.x a few months after we implemented support for 3.x, so the suite we use is not up to date at present.

## Usage

### Quick Start

```sh
# Build the demo; make sure you have a RISC-V compiler
make -C ./test_resources

# Run a simple program
cargo run -- ./test_resources/bin/main.elf

# Run with debugger enabled
cargo run -- ./test_resources/bin/main.elf -g
```

### Useful Command-Line Options

- `<PATH>`: RISC-V ELF executable or raw binary image to run
- `-h, --help`: Print help
- `-f, --format <auto|elf|bin>`: Choose the input format; `auto` uses the `.elf` or `.bin` filename extension
- `-g, --debug`: Start the built-in rvdb debugger (run `help` inside rvdb for its command list)
- `-G, --gdb`: Start a GDB remote stub on `localhost:1234`
- `-S, --script <FILE>`: Run rvdb commands from a file before entering the interactive debugger; requires `--debug`
- `-v, --verbose`: Print additional startup details
- `--loglevel <LEVEL>`: Set the logging level (`trace`, `debug`, `info`, `warn`, or `error`)
- `--device <TYPE:PATH>`: Attach a VirtIO block device; may be repeated
  - Use `virtio-block:/path/to/image`; the image must exist and its size must be a multiple of 512 bytes
- `--isa <ISA>`: Configure the decoder with an ISA string (default: `RV64GC`)
- `--max-cycles <COUNT>`: Stop after the requested number of emulated cycles (`0` disables the limit)
- `--dtb <FILE>`: Load a DTB and pass its guest address to OpenSBI in register `a1`
- `--dtb-address <ADDRESS>`: Set the guest physical address for `--dtb` (default: `0x9f000000`)

During normal emulation, press `Ctrl+A`, release the keys, and then press `x` to exit emulator.

### Example Usage

```sh
mkdir -p ./tmp
dd if=/dev/zero of=./tmp/img_blk bs=512 count=8
cargo run -- ./test_resources/bin/virtio_blk_test.elf --device=virtio-block:./tmp/img_blk --loglevel=debug
```

### Running Linux

At present, the emulator can boot the Linux kernel with BusyBox in an initramfs via OpenSBI. A prebuilt packaged kernel + OpenSBI ELF file (`linux-busybox-rv64gc.bin`) is available for download from [prebuilt-kernels](https://github.com/here-emulator/here/releases/tag/prebuilt-kernels).

1. Download `linux-busybox-rv64gc.bin` from [prebuilt-kernels](https://github.com/here-emulator/here/releases/tag/prebuilt-kernels).

2. Compile the provided Device Tree Source (`dts/virt.dts`) into a Device Tree Blob (`dts/virt.dtb`) using `dtc`:
   ```sh
   dtc -I dts -O dtb -o ./dts/virt.dtb ./dts/virt.dts
   ```

3. Boot Linux with the emulator:
   ```sh
   cargo run --release -- $DOWNLOADS_PATH/linux-busybox-rv64gc.bin --dtb=./dts/virt.dtb
   ```

#### Use a VirtIO Device in Linux

To enable a VirtIO block device in Linux:

1. In `$HERE_ROOT/dts/virt.dts`, uncomment the `virtio_mmio@10001000` node:
   ```dts
   		// virtio_mmio 设备
   		virtio_mmio@10001000 {
   			interrupts = <0x01>;
   			interrupt-parent = <0x11>;
   			reg = <0x00 0x10001000 0x00 0x1000>;
   			compatible = "virtio,mmio";
   		};
   ```
   Then recompile `$HERE_ROOT/dts/virt.dtb`:
   ```sh
   dtc -I dts -O dtb -o ./dts/virt.dtb ./dts/virt.dts
   ```

2. Create a sector-aligned backing image and pass `--device=virtio-block:./tmp/img_blk` to `cargo run`:
   ```sh
   mkdir -p ./tmp
   dd if=/dev/zero of=./tmp/img_blk bs=512 count=2048
   cargo run --release -- $DOWNLOADS_PATH/linux-busybox-rv64gc.bin --dtb=./dts/virt.dtb --device=virtio-block:./tmp/img_blk
   ```

When Linux boots, the kernel log will show the device being recognized (2048 * 512-byte = 10 MiB block):

```
[    2.219776] virtio_blk virtio0: 1/0/0 default/read/poll queues
[    2.222720] virtio_blk virtio0: [vda] 2048 512-byte logical blocks (10.5 MB/10.0 MiB)
```

If devtmpfs has not already created `/dev/vda`, create it from the Linux shell:

1. **Determine the device number** — read the major/minor numbers from sysfs:
   ```sh
   cat /sys/block/vda/dev
   ```
   This typically outputs `254:0`.

2. **Create the device node** — use `mknod` to create the block device file:
   ```sh
   mknod /dev/vda b 254 0
   ```

3. **Verify** — check that the device node appears:
   ```sh
   ls /dev
   ```

Once `/dev/vda` is available, you can perform block-level operations:

- **Read/write raw data** with `dd`:
  ```sh
  dd if=/dev/vda bs=512 count=1 2>/dev/null | hexdump -C
  echo "VirtIO-Blk Write Test Success!" | dd of=/dev/vda bs=512 count=1 conv=notrunc
  ```
- **Create a filesystem and mount it** (this overwrites existing data in the backing image):
  ```sh
  mkfs.ext2 /dev/vda
  mount /dev/vda /mnt
  ```

Additional device metadata is available under `/sys/block/vda/`.

## Virt Board

### MMIO Address Map

|       Device       | Address Base |    Address Length    | PLIC Interrupt ID |
| :----------------: | :----------: | :------------------: | :---------------: |
|  `power-manager`   | 0x0010_0000  |        0x1000        |         -         |
|   `test-device`*   | 0x0010_1000  |         0x10         |       0x3f        |
|      `clint`       | 0x0200_0000  |       0x10000        |         -         |
|       `plic`       | 0x0c00_0000  |      0x0400_0000     |         -         |
|       `uart`       | 0x1000_0000  |         0x08         |       0x0a        |
| `virtio-mmio[0]`** | 0x1000_1000  |        0x1000        |       0x01        |
|       `ram`        | 0x8000_0000  | 0x2000_0000 (512 MiB) |         -         |

\* `test-device` is mapped when the `test-device` Cargo feature is enabled; it is part of the default feature set.

\** Additional VirtIO MMIO transports use consecutive 0x1000-byte regions and interrupt IDs. Keep assigned IDs distinct from the UART interrupt ID 0x0a.

## License

This project is licensed under the MIT License.

---

The "RISC-V" trade name is a registered trademark of RISC-V International. This project is not affiliated with, endorsed by, or sponsored by RISC-V International. For more information about RISC-V, please see [https://riscv.org](https://riscv.org).
