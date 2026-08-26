use std::fmt;

use crate::config::arch_config::WordType;

macro_rules! dispatch_read_write {
    ($read_impl: ident, $write_impl: ident) => {
        fn read(
            &mut self,
            addr: crate::config::arch_config::WordType,
            len: u32,
        ) -> Result<u64, MemError> {
            match len {
                1 => self.$read_impl::<u8>(addr).map(|v| v.into()),
                2 => self.$read_impl::<u16>(addr).map(|v| v.into()),
                4 => self.$read_impl::<u32>(addr).map(|v| v.into()),
                8 => self.$read_impl::<u64>(addr),
                _ => unreachable!(),
            }
        }

        fn write(
            &mut self,
            addr: crate::config::arch_config::WordType,
            len: u32,
            data: u64,
        ) -> Result<(), MemError> {
            match len {
                1 => self.$write_impl::<u8>(addr, data as u8),
                2 => self.$write_impl::<u16>(addr, data as u16),
                4 => self.$write_impl::<u32>(addr, data as u32),
                8 => self.$write_impl::<u64>(addr, data),
                _ => unreachable!(),
            }
        }
    };

    () => {
        dispatch_read_write!(read_impl, write_impl);
    };
}

pub(crate) mod aclint;
pub(crate) mod config;
pub mod device_manager;
mod id_allocator;
pub mod uart16550a;
pub(crate) use id_allocator::*;
pub(crate) mod mmio;
pub(crate) mod plic;
pub(crate) mod power_manager;
pub(crate) mod sample_timer;
pub(crate) mod spi;
pub(crate) mod virtio;

#[derive(Debug, PartialEq, Eq)]
pub enum MemError {
    LoadPageFault,
    LoadMisaligned,
    LoadFault,
    StorePageFault,
    StoreMisaligned,
    StoreFault,
}

impl fmt::Display for MemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            MemError::LoadPageFault => "load page fault",
            MemError::LoadMisaligned => "load address is misaligned",
            MemError::LoadFault => "load access fault",
            MemError::StorePageFault => "store page fault",
            MemError::StoreMisaligned => "store address is misaligned",
            MemError::StoreFault => "store access fault",
        };
        f.write_str(message)
    }
}

macro_rules! impl_read_for_type {
    ($unsigned_type: ident) => {
        fn ${concat(read_, $unsigned_type)}(&mut self, addr: WordType) -> Result<$unsigned_type, MemError> {
            self.read(addr, size_of::<$unsigned_type>() as u32)
                .map(|x| x as $unsigned_type)
        }
    };
}

macro_rules! impl_write_for_type {
    ($unsigned_type: ident) => {
        fn ${concat(write_, $unsigned_type)}(
            &mut self,
            addr: WordType,
            data: $unsigned_type,
        ) -> Result<(), MemError> {
            self.write(addr, size_of::<$unsigned_type>() as u32, data as u64)
        }
    };
}

// Check align requirement before device.read/write. Most of align requirement was checked in mmio.
pub trait DeviceTrait: 'static {
    fn read(&mut self, addr: WordType, len: u32) -> Result<u64, MemError>;
    fn write(&mut self, addr: WordType, len: u32, data: u64) -> Result<(), MemError>;

    impl_read_for_type! { u8 }
    impl_read_for_type! { u16 }
    impl_read_for_type! { u32 }
    impl_read_for_type! { u64 }

    impl_write_for_type! { u8 }
    impl_write_for_type! { u16 }
    impl_write_for_type! { u32 }
    impl_write_for_type! { u64 }

    fn sync(&mut self);
}

/// A device whose level-triggered interrupt state is sampled by the PLIC.
pub trait PlicDevice: DeviceTrait {
    /// Return the device's current absolute interrupt-line level.
    ///
    /// This sampled interface is intended for level-triggered sources that
    /// remain asserted until serviced.
    fn irq_level(&mut self) -> bool;
}

// TODO: Remove this
pub trait MemMappedDeviceTrait: DeviceTrait {
    fn base() -> WordType;
    fn size() -> WordType;
}
