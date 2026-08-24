use std::{cell::UnsafeCell, cmp::Ordering, ptr::NonNull, rc::Rc};

use crate::{
    config::arch_config::WordType,
    device::{
        DeviceTrait, MemError,
        device_manager::{DeviceArena, DeviceHandle, ErasedDeviceHandle},
    },
    ram::Ram,
    ram_config,
    utils::{TruncateTo, UnsignedInteger, check_align},
};

pub struct MemoryMapItem {
    pub(crate) start: WordType,
    pub(crate) size: WordType,
    pub(crate) device: ErasedDeviceHandle,
}

impl PartialEq for MemoryMapItem {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start
    }
}
impl Eq for MemoryMapItem {}
impl PartialOrd for MemoryMapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MemoryMapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.start.cmp(&other.start)
    }
}

impl MemoryMapItem {
    pub(crate) fn new<D: DeviceTrait>(
        start: WordType,
        size: WordType,
        device: DeviceHandle<D>,
    ) -> Self {
        Self {
            start,
            size,
            device: device.erase(),
        }
    }
}

/// # mmio
/// ## Usage
/// make sure the address was aligned.
/// ```
/// let mut mmio = MemoryMapIO::new();
/// let a = mmio.read::<WordType>(ram_config::BASE_ADDR + 0x08);
/// let b = mmio.read::<u8>(UART1_ADDR + 0x00);
/// mmio.write::<u8>(UART1_ADDR + 0x06);
/// mmio.write::<u32>(ram_config::BASE_ADDR + 0x03); // ILLIGAL! unaligned accesses
/// ```
pub struct MemoryMapIO {
    map: Vec<MemoryMapItem>,
    ram: Rc<UnsafeCell<Ram>>,
    devices: Option<NonNull<DeviceArena>>,
}

impl MemoryMapIO {
    pub fn from_ram(ram: Rc<UnsafeCell<Ram>>) -> Self {
        Self {
            map: Vec::new(),
            ram,
            devices: None,
        }
    }

    pub fn read_by_type<T>(&mut self, p_addr: WordType) -> Result<T, MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        if p_addr >= ram_config::BASE_ADDR {
            return unsafe {
                self.ram
                    .as_mut_unchecked()
                    .read(p_addr - ram_config::BASE_ADDR)
            };
        }

        match self.map.binary_search_by(|device| {
            if p_addr < device.start {
                Ordering::Greater
            } else if p_addr > device.start {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        }) {
            Ok(i) => self.read_from_device(i, p_addr),
            Err(i) => {
                if i == 0 {
                    Err(MemError::LoadFault)
                    // panic!("physical address: {} is not mapped to the device", p_addr);
                } else {
                    self.read_from_device(i - 1, p_addr)
                }
            }
        }
    }

    pub fn write_by_type<T>(&mut self, p_addr: WordType, data: T) -> Result<(), MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        // let _guard = self.lock();
        if p_addr >= ram_config::BASE_ADDR {
            return unsafe {
                self.ram
                    .as_mut_unchecked()
                    .write(p_addr - ram_config::BASE_ADDR, data)
            };
        }
        match self.map.binary_search_by(|device| {
            if p_addr < device.start {
                Ordering::Greater
            } else if p_addr > device.start {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        }) {
            Ok(i) => self.write_to_device(i, p_addr, data),
            Err(i) => {
                if i == 0 {
                    // panic!("physical address: {} is not mapped to the device", p_addr);
                    Err(MemError::StoreFault)
                } else {
                    self.write_to_device(i - 1, p_addr, data)
                }
            }
        }
    }

    pub fn load_reserved<T>(&mut self, p_addr: WordType) -> Result<T, MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        use ram_config::{BASE_ADDR, SIZE};
        const RAM_END: WordType = BASE_ADDR + SIZE as WordType as WordType;
        match p_addr {
            BASE_ADDR..RAM_END => {
                return unsafe {
                    self.ram
                        .as_mut_unchecked()
                        .load_reserved(p_addr - ram_config::BASE_ADDR)
                };
            }
            _ => {
                // LR/SC is only supported for normal RAM. Device registers are not
                // atomic memory and must not create a reservation.
                Err(MemError::LoadFault)
            }
        }
    }

    pub fn store_conditional<T>(&mut self, p_addr: WordType, data: T) -> Result<bool, MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        use ram_config::{BASE_ADDR, SIZE};
        const RAM_END: WordType = BASE_ADDR + SIZE as WordType as WordType;
        match p_addr {
            BASE_ADDR..RAM_END => {
                return unsafe {
                    self.ram
                        .as_mut_unchecked()
                        .store_conditional(p_addr - ram_config::BASE_ADDR, data)
                };
            }
            _ => Err(MemError::StoreFault),
        }
    }

    #[inline]
    pub fn clear_reservation(&self) {
        unsafe { self.ram.as_mut_unchecked() }.clear_reservation();
    }

    /// # Safety
    ///
    /// `devices` must remain valid for the lifetime of this `MemoryMapIO` and
    /// access to the arena must be serialized with MMIO operations.
    pub(crate) unsafe fn from_mmio_items(
        ram: Rc<UnsafeCell<Ram>>,
        devices: NonNull<DeviceArena>,
        mut map: Vec<MemoryMapItem>,
    ) -> Self {
        map.sort();
        Self {
            map,
            ram,
            devices: Some(devices),
        }
    }

    fn devices_mut(&mut self) -> &mut DeviceArena {
        let mut devices = self
            .devices
            .expect("an MMIO device map requires a DeviceArena");
        unsafe { devices.as_mut() }
    }

    fn read_from_device<T>(&mut self, device_index: usize, p_addr: WordType) -> Result<T, MemError>
    where
        T: UnsignedInteger,
    {
        if !check_align::<T>(p_addr) {
            return Err(MemError::LoadMisaligned);
        }
        if !self.can_access::<T>(device_index, p_addr) {
            return Err(MemError::LoadFault);
        }

        let start = self.map[device_index].start;
        let device = self.map[device_index].device;
        self.devices_mut()
            .erased_device_mut(device)
            .read(p_addr - start, size_of::<T>() as u32)
            .map(|x| x.truncate_to())
    }

    // write data to specific device.
    fn write_to_device<T>(
        &mut self,
        device_index: usize,
        p_addr: WordType,
        data: T,
    ) -> Result<(), MemError>
    where
        T: UnsignedInteger,
    {
        if !check_align::<T>(p_addr) {
            return Err(MemError::StoreMisaligned);
        }
        if !self.can_access::<T>(device_index, p_addr) {
            return Err(MemError::StoreFault);
        }

        let start = self.map[device_index].start;
        let device = self.map[device_index].device;
        self.devices_mut().erased_device_mut(device).write(
            p_addr - start,
            size_of::<T>() as u32,
            data.truncate_to(),
        )
    }

    fn can_access<T>(&self, device_index: usize, p_addr: WordType) -> bool {
        let item = &self.map[device_index];
        let Some(offset) = p_addr.checked_sub(item.start) else {
            return false;
        };
        offset
            .checked_add(size_of::<T>() as WordType)
            .is_some_and(|end| end <= item.size)
    }
}

impl DeviceTrait for MemoryMapIO {
    dispatch_read_write! { read_by_type, write_by_type }

    fn sync(&mut self) {
        // let _guard = self.lock();
        for index in 0..self.map.len() {
            let device = self.map[index].device;
            self.devices_mut().erased_device_mut(device).sync();
        }
    }
}

#[cfg(test)]
mod test {
    use crate::device::{
        config::{POWER_MANAGER_BASE, POWER_MANAGER_SIZE, UART_BASE, UART_SIZE},
        power_manager::PowerManager,
        uart16550a::Uart16550A,
    };

    use super::*;

    #[test]
    fn mmio_mem_test() {
        let ram = Rc::new(UnsafeCell::new(Ram::new()));

        let (uart1, _port) = Uart16550A::new();
        let power_manager = PowerManager::new();
        let mut arena_builder = crate::device::device_manager::DeviceArenaBuilder::new();
        let power_manager = arena_builder.register(Box::new(power_manager));
        let uart1 = arena_builder.register(Box::new(uart1));
        let table = vec![
            MemoryMapItem::new(POWER_MANAGER_BASE, POWER_MANAGER_SIZE, power_manager),
            MemoryMapItem::new(UART_BASE, UART_SIZE, uart1),
        ];
        let mut devices = Box::new(arena_builder.build());
        let devices_ptr = NonNull::from(devices.as_mut());

        let mut mmio = unsafe { MemoryMapIO::from_mmio_items(ram, devices_ptr, table) };
        for i in 0 as WordType..100 {
            mmio.write_by_type(ram_config::BASE_ADDR + i * (1 << size_of::<WordType>()), i)
                .unwrap();
        }

        for i in 0 as WordType..100 {
            assert_eq!(
                i,
                mmio.read_by_type(ram_config::BASE_ADDR + i * (1 << size_of::<WordType>()))
                    .unwrap()
            );
        }
    }

    #[test]
    fn mmio_stdout_test() {
        let ram = Rc::new(UnsafeCell::new(Ram::new()));
        let (uart1, mut port1) = Uart16550A::new();
        let power_manager = PowerManager::new();
        let mut arena_builder = crate::device::device_manager::DeviceArenaBuilder::new();
        let power_manager = arena_builder.register(Box::new(power_manager));
        let uart1 = arena_builder.register(Box::new(uart1));
        let table = vec![
            MemoryMapItem::new(POWER_MANAGER_BASE, POWER_MANAGER_SIZE, power_manager),
            MemoryMapItem::new(UART_BASE, UART_SIZE, uart1),
        ];
        let mut devices = Box::new(arena_builder.build());
        let devices_ptr = NonNull::from(devices.as_mut());

        let mut mmio = unsafe { MemoryMapIO::from_mmio_items(ram, devices_ptr, table) };

        mmio.write_by_type(UART_BASE + 0x00, 'a' as u8).unwrap();
        assert_ne!((mmio.read_by_type::<u8>(UART_BASE + 5).unwrap() & 0x20), 0);
        let data = port1.take_output();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], 'a' as u8);
    }

    struct MockDevice;

    impl DeviceTrait for MockDevice {
        fn read(&mut self, _addr: WordType, _len: u32) -> Result<u64, MemError> {
            Ok(0)
        }

        fn write(&mut self, _addr: WordType, _len: u32, _data: u64) -> Result<(), MemError> {
            Ok(())
        }

        fn sync(&mut self) {}
    }

    #[test]
    fn mmio_rejects_accesses_crossing_device_end() {
        let ram = Rc::new(UnsafeCell::new(Ram::new()));
        let mut arena_builder = crate::device::device_manager::DeviceArenaBuilder::new();
        let mock_device = arena_builder.register(Box::new(MockDevice));
        let table = vec![MemoryMapItem::new(0x1000, 4, mock_device)];
        let mut devices = Box::new(arena_builder.build());
        let devices_ptr = NonNull::from(devices.as_mut());

        let mut mmio = unsafe { MemoryMapIO::from_mmio_items(ram, devices_ptr, table) };

        assert_eq!(mmio.read_by_type::<u32>(0x1000), Ok(0));
        assert_eq!(mmio.write_by_type::<u32>(0x1000, 0), Ok(()));
        assert_eq!(mmio.read_by_type::<u64>(0x1000), Err(MemError::LoadFault));
        assert_eq!(
            mmio.write_by_type::<u64>(0x1000, 0),
            Err(MemError::StoreFault)
        );
    }
}
