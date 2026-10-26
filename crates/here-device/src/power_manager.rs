use crate::DeviceTrait;
use crate::MemError;
use here_types::WordType;
use std::sync::atomic::AtomicU16;

pub const POWER_OFF_CODE: u16 = 0x5555;
pub static POWER_STATUS: AtomicU16 = AtomicU16::new(0);

/// MMIO region size for the power manager device.
// Cannot be too small — OpenSBI disallows small mappings.
pub const POWER_MANAGER_SIZE: WordType = 0x1000;

pub struct PowerManager {
    reg: u16,
}

impl PowerManager {
    fn read_impl<T>(&mut self, addr: here_types::WordType) -> Result<T, MemError>
    where
        T: here_types::UnsignedInteger,
    {
        debug_assert!(addr == 0x00);
        debug_assert!(size_of::<T>() >= 2);
        let mut ret: T = ((self.reg >> 8) as u8).into();
        ret <<= 8;
        ret |= (self.reg as u8).into();
        Ok(ret)
    }

    fn write_impl<T>(&mut self, _addr: here_types::WordType, data: T) -> Result<(), MemError>
    where
        T: here_types::UnsignedInteger,
    {
        debug_assert!(_addr == 0x00);
        let data: u64 = data.into();
        self.reg = data as u16;

        if self.reg == POWER_OFF_CODE {
            POWER_STATUS.store(0x5555, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }
}

impl DeviceTrait for PowerManager {
    dispatch_read_write! { read_impl, write_impl }

    fn sync(&mut self) {}
}

impl PowerManager {
    pub fn new() -> Self {
        POWER_STATUS.store(0, std::sync::atomic::Ordering::Release);
        Self { reg: 0 }
    }
}
