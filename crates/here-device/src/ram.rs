use core::panic;
use std::ops::{Index, IndexMut};

use crate::{
    config::arch_config::WordType,
    device::MemError,
    ram_config,
    utils::{read_raw_ptr, write_raw_ptr},
};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RamImageError {
    #[error("RAM image address overflow: start=0x{start:x}, size=0x{size:x}")]
    AddressOverflow { start: WordType, size: usize },

    #[error(
        "RAM image is out of range: start=0x{start:x}, size=0x{size:x}, RAM size=0x{ram_size:x}"
    )]
    OutOfRange {
        start: WordType,
        size: usize,
        ram_size: usize,
    },
}

const RESERVATION_SET_BIT_WIDTH: WordType = 3; // 64 bit

#[derive(Debug, Clone, Copy)]
struct Reservation {
    reservation_set_id: WordType,
}

impl Reservation {
    fn new(addr: WordType) -> Self {
        Self {
            reservation_set_id: addr >> RESERVATION_SET_BIT_WIDTH,
        }
    }

    fn in_the_same_reservation_set(&self, addr: WordType) -> bool {
        // Note this LR might have had a different effective address and data size,
        // but reserved the SC’s address as part of the reservation set.
        self.reservation_set_id == addr >> RESERVATION_SET_BIT_WIDTH
    }
}

pub struct Ram {
    // TODO: 4KB align the inner box ptr for better performance.
    data: Box<[u8]>,
    reserved: Option<Reservation>,
}

impl Index<usize> for Ram {
    type Output = u8;
    fn index(&self, index: usize) -> &Self::Output {
        &(self.data[index])
    }
}

impl IndexMut<usize> for Ram {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut (self.data[index])
    }
}

impl Ram {
    pub fn new() -> Self {
        Self {
            data: vec![0u8; ram_config::SIZE].into_boxed_slice(),
            reserved: None,
        }
    }

    pub fn with_init(byte: u8) -> Self {
        Self {
            data: vec![byte; ram_config::SIZE].into_boxed_slice(),
            reserved: None,
        }
    }

    pub fn with_data(mut data: Vec<u8>) -> Self {
        if data.len() > ram_config::SIZE {
            log::error!(
                "ram::with_data data size {} exceeds RAM size {}",
                data.len(),
                ram_config::SIZE
            );
            panic!();
        }
        data.resize(ram_config::SIZE, 0);
        Self {
            data: data.into_boxed_slice(),
            reserved: None,
        }
    }

    // TODO: physical address should be `u64` instead of `WordType`.
    pub fn try_insert_section(
        &mut self,
        section_data: &[u8],
        start_addr: WordType,
    ) -> Result<(), RamImageError> {
        let Ok(start_addr_usize) = usize::try_from(start_addr) else {
            return Err(RamImageError::AddressOverflow {
                start: start_addr,
                size: section_data.len(),
            });
        };
        let Some(end_addr) = start_addr_usize.checked_add(section_data.len()) else {
            return Err(RamImageError::AddressOverflow {
                start: start_addr,
                size: section_data.len(),
            });
        };
        if end_addr > ram_config::SIZE {
            return Err(RamImageError::OutOfRange {
                start: start_addr,
                size: section_data.len(),
                ram_size: ram_config::SIZE,
            });
        }

        section_data.iter().enumerate().for_each(|(index, v)| {
            self.data[start_addr_usize + index] = *v;
        });

        Ok(())
    }

    pub fn insert_section(&mut self, section_data: &[u8], start_addr: WordType) {
        if let Err(error) = self.try_insert_section(section_data, start_addr) {
            log::error!("ram::insert_section failed: {error}");
            panic!("{error}");
        }
    }

    pub fn read<T>(&self, addr: WordType) -> Result<T, MemError> {
        if !Self::contains_access::<T>(addr) {
            return Err(MemError::LoadFault);
        }

        let data = unsafe { read_raw_ptr::<T>(self.data.as_ptr().add(addr as usize)) };
        if let Some(data) = data {
            Ok(data)
        } else {
            Err(MemError::LoadMisaligned)
        }
    }

    pub fn load_reserved<T>(&mut self, addr: WordType) -> Result<T, MemError> {
        let data = self.read::<T>(addr)?;
        self.reserved = Some(Reservation::new(addr));
        Ok(data)
    }

    pub fn store_conditional<T>(&mut self, addr: WordType, data: T) -> Result<bool, MemError> {
        #[cfg(debug_assertions)]
        if addr >= ram_config::SIZE as WordType {
            self.reserved = None;
            return Err(MemError::StoreFault);
        }

        if let Some(res) = self.reserved {
            if res.in_the_same_reservation_set(addr) {
                self.write(addr, data)?;
                return Ok(true);
            }
        }

        self.reserved = None;
        Ok(false)
    }

    pub fn write<T>(&mut self, addr: WordType, data: T) -> Result<(), MemError> {
        if !Self::contains_access::<T>(addr) {
            return Err(MemError::StoreFault);
        }

        self.try_invalidate_reservation(addr);

        let ret = unsafe { write_raw_ptr(self.data.as_mut_ptr().add(addr as usize), data) };
        if let Some(()) = ret {
            Ok(())
        } else {
            Err(MemError::StoreMisaligned)
        }
    }

    #[inline]
    pub(crate) fn clear_reservation(&mut self) {
        self.reserved = None;
    }

    /// Invalidate an LR reservation when another agent writes an overlapping RAM range.
    pub fn try_invalidate_reservation(&mut self, addr: WordType) {
        if self
            .reserved
            .is_some_and(|res| res.in_the_same_reservation_set(addr))
        {
            self.reserved = None;
        }
    }

    fn contains_access<T>(addr: WordType) -> bool {
        let Ok(start) = usize::try_from(addr) else {
            return false;
        };
        start
            .checked_add(size_of::<T>())
            .is_some_and(|end| end <= ram_config::SIZE)
    }
    #[cfg(any(test, feature = "test-support"))]
    pub fn get_raw_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
}

// // there is nothing to do.
// impl DeviceTrait for Ram {
//     fn sync(&mut self) {}
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_section_and_read() {
        let mut r = Ram::new();

        // 插入一段数据，地址从 ram_config::BASE_ADDR 开始
        let base = 0x00;
        let section = [0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        r.insert_section(&section, base);

        // 验证内存中数据被正确写入
        for (i, &v) in section.iter().enumerate() {
            assert_eq!(r.data[i], v);
        }

        // 测试read_byte
        let b = r.read::<u8>(base as WordType).unwrap();
        assert_eq!(b, 0x12);

        // 测试read_word (2字节)
        let w = r.read::<u16>(base as WordType).unwrap();
        // 注意你的concat_bits顺序是从高位到低位
        // data[base|1], data[base] => 0x34 0x12 -> 0x3412
        assert_eq!(w, 0x3412);

        // 测试read_dword (4字节)
        let d = r.read::<u32>(base as WordType).unwrap();
        // data[base|3], base|2, base|1, base
        // 0x78 0x56 0x34 0x12 -> 0x78563412
        assert_eq!(d, 0x78563412);

        // 测试read_qword (8字节)
        let q = r.read::<u64>(base as WordType).unwrap();
        // data[base|7]...data[base]
        // 0xF0 0xDE 0xBC 0x9A 0x78 0x56 0x34 0x12 -> 0xF0DEBC9A78563412
        assert_eq!(q, 0xF0DEBC9A78563412);
    }

    #[test]
    #[should_panic(expected = "Result::unwrap()")]
    fn test_read_unaligned_address() {
        let r = Ram::new();
        // 这里用一个不对齐的地址试试，比如 addr & align_ilst[size_of_t] != 0 会断言失败
        r.read::<u64>(1).unwrap(); // 如果1不对齐，应该panic
    }

    #[test]
    fn test_write_byte() {
        let mut ram = Ram::new();
        ram.write::<u8>(0x00, 0xAB).unwrap();
        assert_eq!(ram.data[0], 0xAB);

        ram.write::<u16>(0x00, 0x1234).unwrap();
        assert_eq!(ram.data[0], 0x34); // little endian
        assert_eq!(ram.data[1], 0x12);

        ram.write::<u32>(0x00, 0x12345678).unwrap();
        assert_eq!(ram.data[0], 0x78);
        assert_eq!(ram.data[1], 0x56);
        assert_eq!(ram.data[2], 0x34);
        assert_eq!(ram.data[3], 0x12);

        ram.write::<u64>(0x00, 0x1122334455667788).unwrap();
        assert_eq!(ram.data[0], 0x88);
        assert_eq!(ram.data[1], 0x77);
        assert_eq!(ram.data[2], 0x66);
        assert_eq!(ram.data[3], 0x55);
        assert_eq!(ram.data[4], 0x44);
        assert_eq!(ram.data[5], 0x33);
        assert_eq!(ram.data[6], 0x22);
        assert_eq!(ram.data[7], 0x11);
    }

    #[test]
    fn failed_sc_clears_reservation_and_validates_access() {
        let mut ram = Ram::new();
        ram.load_reserved::<u32>(0).unwrap();

        assert_eq!(
            ram.store_conditional::<u32>(ram_config::SIZE as WordType, 1),
            Err(MemError::StoreFault)
        );
        assert!(!ram.store_conditional::<u32>(0, 1).unwrap());
    }

    #[test]
    fn lr_sc_within_64bit_reservation_set_succeeds() {
        let mut ram = Ram::new();
        ram.write::<u64>(0, 0).unwrap();

        // LR the high 32-bit word of the 64-bit reservation set, then SC it.
        ram.load_reserved::<u32>(4).unwrap();
        assert!(ram.store_conditional::<u32>(4, 0x1234_5678).unwrap());
        assert_eq!(ram.read::<u64>(0).unwrap(), 0x1234_5678_0000_0000);

        // LR the high 32-bit word, then SC the low 32-bit word.
        // Both belong to the same 64-bit reservation set, so SC should succeed.
        ram.load_reserved::<u32>(4).unwrap();
        assert!(ram.store_conditional::<u32>(0, 0xdead_beef).unwrap());
        assert_eq!(ram.read::<u64>(0).unwrap(), 0x1234_5678_dead_beef);
    }

    #[test]
    fn test_read_write_reject_end_and_crossing_accesses() {
        let mut ram = Ram::new();
        let last_word = ram_config::SIZE as WordType - size_of::<u64>() as WordType;

        ram.write::<u64>(last_word, 0x1122_3344_5566_7788).unwrap();
        assert_eq!(ram.read::<u64>(last_word).unwrap(), 0x1122_3344_5566_7788);

        assert_eq!(
            ram.read::<u8>(ram_config::SIZE as WordType),
            Err(MemError::LoadFault)
        );
        assert_eq!(
            ram.read::<u64>(ram_config::SIZE as WordType - 4),
            Err(MemError::LoadFault)
        );
        assert_eq!(
            ram.write::<u8>(ram_config::SIZE as WordType, 0xaa),
            Err(MemError::StoreFault)
        );
        assert_eq!(
            ram.write::<u64>(ram_config::SIZE as WordType - 4, 0),
            Err(MemError::StoreFault)
        );
    }

    #[test]
    #[should_panic]
    fn test_insert_section_rejects_crossing_end() {
        let mut ram = Ram::new();
        ram.insert_section(&[1, 2], ram_config::SIZE as WordType - 1);
    }
}
