use crate::config::arch_config::WordType;

pub const POWER_MANAGER_NAME: &'static str = "virt-power";
pub const POWER_MANAGER_BASE: WordType = 0x10_0000;
// Cannot be too small - the OpenSBI disallow.
pub const POWER_MANAGER_SIZE: WordType = 0x1000;

#[cfg(feature = "test-device")]
pub const SAMPLE_TIMER_BASE: WordType = 0x10_1000;
#[cfg(feature = "test-device")]
pub const SAMPLE_TIMER_SIZE: WordType = 0x10;

pub const CLINT_NAME: &'static str = "clint";
pub const CLINT_BASE: WordType = 0x200_0000;
pub const CLINT_SIZE: WordType = 0x10000;

pub const PLIC_NAME: &'static str = "plic";
pub const PLIC_BASE: WordType = 0xc00_0000;
pub const PLIC_SIZE: WordType = 0x400_0000;

// The emulator does not model a physical input clock.  Keep the reset
// divisor at the value used by the previous UART implementation; guests that
// need a particular baud rate can program DLL/DLM through DLAB.
pub const UART_DEFAULT_DIV: usize = 1;
pub const UART_NAME: &'static str = "uart";
pub const UART_BASE: WordType = 0x1000_0000;
pub const UART_SIZE: WordType = 8;
/// UART PLIC interrupt source ID, must match DTS `interrupts = <0xa>`
pub const UART_IRQ: u32 = 10;

pub const VIRTIO_MMIO_NAME: &'static str = "virtio-mmio-device";
pub const VIRTIO_MMIO_BASE: WordType = 0x1000_1000;
pub const VIRTIO_MMIO_SIZE: WordType = 0x1000;
/// First VirtIO PLIC source ID. Subsequent MMIO transports use consecutive IDs.
pub const VIRTIO_IRQ_BASE: u32 = 1;

// pub const MMIO_FREQ_DIV: usize = 32;
