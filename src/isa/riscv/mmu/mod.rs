pub mod address;
pub mod config;
mod cpu;
mod page_table;

pub use page_table::PageTableError;

use std::{cell::UnsafeCell, rc::Rc};

use self::config::*;
use self::page_table::*;

use crate::{
    config::arch_config::WordType,
    device::{DeviceTrait, MemError, mmio::MemoryMapIO},
    isa::riscv::{debugger::Address, trap::Exception},
    ram::Ram,
    ram_config,
    utils::UnsignedInteger,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
    // TODO: Consider Remove this. We don't have to handle AMO in this way.
    ReadWrite,
    // TODO: Currently we handle ifetch separately, consider adding ifetch here so that we can unify the handling.
}

enum AccessPolicy {
    Direct,
    Translated {
        check: PermissionCheck,
        effect: AccessEffect,
        fault: MemError,
    },
}

pub(crate) struct VirtAddrManager {
    pub(crate) mmio: MemoryMapIO,
    page_table: PageTableWalker,
    ram: Rc<UnsafeCell<Ram>>,
}

/// Performs address translation and physical memory access using policies prepared by the CPU.
impl VirtAddrManager {
    pub(crate) fn from_ram_and_mmio(ram_ref: Rc<UnsafeCell<Ram>>, mmio: MemoryMapIO) -> Self {
        Self {
            mmio: mmio,
            page_table: PageTableWalker::new(0, config::VirtualMemoryMode::None),
            ram: ram_ref,
        }
    }

    fn translate_with_policy(
        &mut self,
        vaddr: WordType,
        policy: AccessPolicy,
    ) -> Result<u64, MemError> {
        match policy {
            AccessPolicy::Direct => Ok(vaddr),
            AccessPolicy::Translated {
                check,
                effect,
                fault,
            } => self
                .translate_vaddr(vaddr, check, effect)
                .map_err(|_| fault),
        }
    }

    fn read_with_policy<T>(&mut self, addr: WordType, policy: AccessPolicy) -> Result<T, MemError>
    where
        T: UnsignedInteger,
    {
        // Don't check alignment here since some devices may allow unaligned access.
        // Only check alignment in device's implementations.

        let paddr = self.translate_with_policy(addr, policy)?;

        self.mmio.read_by_type(paddr)
    }

    fn write_with_policy<T>(
        &mut self,
        addr: WordType,
        data: T,
        policy: AccessPolicy,
    ) -> Result<(), MemError>
    where
        T: UnsignedInteger,
    {
        let paddr = self.translate_with_policy(addr, policy)?;

        self.mmio.write_by_type(paddr, data)
    }

    fn load_reserved_with_policy<T>(
        &mut self,
        addr: WordType,
        policy: AccessPolicy,
    ) -> Result<T, MemError>
    where
        T: UnsignedInteger,
    {
        if !crate::utils::check_align::<T>(addr) {
            return Err(MemError::LoadMisaligned);
        }

        let paddr = self.translate_with_policy(addr, policy)?;

        self.mmio.load_reserved(paddr)
    }

    fn store_conditional_with_policy<T>(
        &mut self,
        addr: WordType,
        data: T,
        policy: AccessPolicy,
    ) -> Result<bool, MemError>
    where
        T: UnsignedInteger,
    {
        if !crate::utils::check_align::<T>(addr) {
            // Every SC attempt invalidates the hart's previous reservation, even
            // when the target address is not normal RAM.
            self.mmio.clear_reservation();
            return Err(MemError::StoreMisaligned);
        }

        let paddr = self.translate_with_policy(addr, policy)?;

        let res = self.mmio.store_conditional(paddr, data);
        self.mmio.clear_reservation();
        res
    }

    /// Atomic Memory Operation.
    fn fetch_and_op_amo_with_policy<T, F>(
        &mut self,
        addr: WordType,
        rhs_val: T,
        policy: AccessPolicy,
        f: F,
    ) -> Result<T, Exception>
    where
        T: UnsignedInteger,
        F: Fn(&T::AtomicType, T) -> Result<T, Exception>,
    {
        let mut paddr = match self.translate_with_policy(addr, policy) {
            Ok(p) => p,
            Err(_) => return Err(Exception::StorePageFault),
        };

        if !crate::utils::check_align::<T>(paddr) {
            return Err(Exception::StoreMisaligned);
        }

        if !(ram_config::BASE_ADDR..ram_config::BASE_ADDR + ram_config::SIZE as WordType)
            .contains(&paddr)
        {
            // The full name of this exception is Store/AMO access fault
            return Err(Exception::StoreFault);
        }

        paddr -= ram_config::BASE_ADDR;

        let ram = unsafe { &mut *self.ram.get() };
        let ptr = &mut ram[paddr as usize] as *mut u8 as *mut T::AtomicType;
        let lhs = unsafe { &*ptr };
        let result = f(lhs, rhs_val);
        if result.is_ok() {
            ram.try_invalidate_reservation(paddr);
        }
        result
    }

    pub(crate) fn read_by_paddr<T>(&mut self, paddr: WordType) -> Result<T, MemError>
    where
        T: UnsignedInteger,
    {
        self.mmio.read_by_type(paddr.into())
    }

    pub(crate) fn write_by_paddr<T>(&mut self, paddr: WordType, data: T) -> Result<(), MemError>
    where
        T: UnsignedInteger,
    {
        self.mmio.write_by_type(paddr.into(), data)
    }

    #[cfg(test)]
    pub(crate) fn get_raw_ptr(&self) -> *mut u8 {
        unsafe { &mut *self.ram.get() }.get_raw_ptr()
    }

    // TODO: These debug functions (and their ability) are chaotic.
    // Think about them to determine what we really need.
    /// Read operation without side-effect of page table, provided for debugger.
    ///
    /// This function dones't respect the current privilege mode.
    pub(crate) fn debug_read<T>(&mut self, addr: Address) -> Result<T, MemError>
    where
        T: UnsignedInteger,
    {
        match addr {
            Address::Phys(addr) => self.read_by_paddr::<T>(addr),
            Address::Virt(addr) => {
                let check = PermissionCheck {
                    any_of: PTEFlags::empty(),
                    exact_mask: PTEFlags::empty(),
                    exact_flags: PTEFlags::empty(),
                };

                if let Ok(paddr) = self.translate_vaddr(addr, check, AccessEffect::None) {
                    self.mmio.read_by_type(paddr)
                } else {
                    Err(MemError::LoadPageFault)
                }
            }
        }
    }

    /// Write operation without side-effect of page table, provided for debugger.
    pub(crate) fn debug_write<T>(&mut self, addr: Address, data: T) -> Result<(), MemError>
    where
        T: UnsignedInteger,
    {
        match addr {
            Address::Phys(addr) => self.write_by_paddr::<T>(addr, data),
            Address::Virt(addr) => {
                let check = PermissionCheck {
                    any_of: PTEFlags::empty(),
                    exact_mask: PTEFlags::empty(),
                    exact_flags: PTEFlags::empty(),
                };

                if let Ok(paddr) = self.translate_vaddr(addr, check, AccessEffect::None) {
                    self.mmio.write_by_type(paddr, data)
                } else {
                    Err(MemError::StorePageFault)
                }
            }
        }
    }

    fn translate_vaddr(
        &mut self,
        vaddr: WordType,
        check: PermissionCheck,
        effect: AccessEffect,
    ) -> Result<u64, PageTableError> {
        self.page_table
            .translate_vaddr(
                unsafe { self.ram.as_mut_unchecked() },
                vaddr.into(),
                check,
                effect,
            )
            .map(|addr| addr.into())
    }

    /// Translates a virtual address to a physical address without checking any PTE flags or updating any bits, provided for debugger.
    ///
    /// This function doesn't respect the current privilege mode or check PTE flags.
    pub(crate) fn debug_vaddr_to_paddr(&mut self, vaddr: WordType) -> Result<u64, PageTableError> {
        self.translate_vaddr(
            vaddr,
            PermissionCheck {
                any_of: PTEFlags::empty(),
                exact_mask: PTEFlags::empty(),
                exact_flags: PTEFlags::empty(),
            },
            AccessEffect::None,
        )
    }

    /// Translates an address to a physical address as a real instruction would, but without side effects (such as writing A/D bits).
    ///
    /// This means the function will check PTE flags, and consider CPU state like CSR settings and privilege mode.
    fn translate_for_debug_with_policy(
        &mut self,
        addr: u64,
        policy: AccessPolicy,
    ) -> Result<u64, PageTableError> {
        match policy {
            AccessPolicy::Direct => Ok(addr),
            AccessPolicy::Translated {
                check,
                effect,
                fault: _fault,
            } => self.translate_vaddr(addr as WordType, check, effect),
        }
    }

    /// Set the virtual memory mode.
    pub fn set_mode(&mut self, mode: u8) {
        self.page_table.set_mode(mode);
    }

    pub fn set_root_ppn(&mut self, ppn: u64) {
        self.page_table.set_root_addr(ppn << PAGE_SIZE_XLEN);
    }

    pub fn set_ad_update_policy(&mut self, policy: AdUpdatePolicy) {
        self.page_table.set_ad_update_policy(policy);
    }

    /// sync MMIO devices
    pub fn sync(&mut self) {
        self.mmio.sync();
    }

    pub fn flush_tlb(&mut self) {
        self.page_table.flush_tlb();
    }
}

#[cfg(test)]
mod tests {
    use std::{ptr::NonNull, sync::atomic::Ordering};

    use super::*;
    use crate::device::{device_manager::DeviceArenaBuilder, mmio::MemoryMapItem};

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
    fn lr_sc_rejects_mmio_and_sc_clears_ram_reservation() {
        let ram = Rc::new(UnsafeCell::new(Ram::new()));
        let mut arena_builder = DeviceArenaBuilder::new();
        let mock_device = arena_builder.register(Box::new(MockDevice));
        let table = vec![MemoryMapItem::new(0x1000, 8, mock_device)];
        let mut devices = Box::new(arena_builder.build());
        let devices_ptr = NonNull::from(devices.as_mut());
        let mmio = unsafe { MemoryMapIO::from_mmio_items(ram.clone(), devices_ptr, table) };
        let mut memory = VirtAddrManager::from_ram_and_mmio(ram, mmio);
        assert_eq!(
            memory.load_reserved_with_policy::<u32>(0x1000, AccessPolicy::Direct),
            Err(MemError::LoadFault)
        );

        let ram_addr = ram_config::BASE_ADDR;
        memory
            .load_reserved_with_policy::<u32>(ram_addr, AccessPolicy::Direct)
            .unwrap();
        assert_eq!(
            memory.store_conditional_with_policy::<u32>(0x1000, 1, AccessPolicy::Direct),
            Err(MemError::StoreFault)
        );
        assert!(
            !memory
                .store_conditional_with_policy::<u32>(ram_addr, 1, AccessPolicy::Direct)
                .unwrap()
        );
    }

    #[test]
    fn amo_invalidates_lr_reservation() {
        let ram = Rc::new(UnsafeCell::new(Ram::new()));
        let mmio = MemoryMapIO::from_ram(ram.clone());
        let mut memory = VirtAddrManager::from_ram_and_mmio(ram, mmio);
        let addr = ram_config::BASE_ADDR + 8;

        memory
            .write_with_policy::<u32>(addr, 0, AccessPolicy::Direct)
            .unwrap();
        memory
            .load_reserved_with_policy::<u32>(addr, AccessPolicy::Direct)
            .unwrap();
        assert_eq!(
            memory
                .fetch_and_op_amo_with_policy::<u32, _>(
                    addr,
                    1,
                    AccessPolicy::Direct,
                    |lhs, rhs| Ok(lhs.fetch_add(rhs, Ordering::SeqCst)),
                )
                .unwrap(),
            0
        );

        assert!(
            !memory
                .store_conditional_with_policy::<u32>(addr, 2, AccessPolicy::Direct)
                .unwrap()
        );
        assert_eq!(
            memory
                .read_with_policy::<u32>(addr, AccessPolicy::Direct)
                .unwrap(),
            1
        );
    }
}
