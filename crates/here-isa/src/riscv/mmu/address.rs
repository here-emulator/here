use crate::{
    config::arch_config::WordType,
    isa::riscv::mmu::config::{PAGE_SIZE, PAGE_SIZE_XLEN, PHYSICAL_ADDR_WIDTH},
};

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(super) struct PhysicalAddr(pub(super) u64); // rv32 physical addr is 34bit-length.

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(super) struct VirtualAddr(pub(super) WordType);

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(super) struct PhysicalPageNum {
    pub(super) address: u64,
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(super) struct VirtualPageNum {
    pub(super) address: WordType,
}

impl VirtualAddr {
    pub(super) fn vpn(self) -> VirtualPageNum {
        VirtualPageNum::from_vaddr(self.0 & !(PAGE_SIZE - 1))
    }
}

impl PhysicalPageNum {
    pub(super) fn from_ppn(ppn: WordType) -> Self {
        PhysicalPageNum {
            address: ppn << PAGE_SIZE_XLEN,
        }
    }
    pub(super) fn from_paddr(paddr: WordType) -> Self {
        PhysicalPageNum {
            address: paddr & !(PAGE_SIZE - 1),
        }
    }
}

// PhysicalAddr
impl From<u64> for PhysicalAddr {
    fn from(value: u64) -> Self {
        PhysicalAddr(value & ((1 << PHYSICAL_ADDR_WIDTH) - 1))
    }
}
impl From<PhysicalAddr> for u64 {
    fn from(addr: PhysicalAddr) -> Self {
        addr.0
    }
}

// Cast between PPN and physical address.
impl From<PhysicalPageNum> for PhysicalAddr {
    fn from(ppn: PhysicalPageNum) -> Self {
        PhysicalAddr(ppn.address)
    }
}
impl From<PhysicalAddr> for PhysicalPageNum {
    fn from(addr: PhysicalAddr) -> Self {
        PhysicalPageNum::from_paddr(addr.0)
    }
}

// VirtualAddr
impl From<WordType> for VirtualAddr {
    fn from(value: WordType) -> Self {
        VirtualAddr(value)
    }
}
impl From<VirtualAddr> for WordType {
    fn from(addr: VirtualAddr) -> Self {
        addr.0
    }
}

// Cast between VPN and virtual address.
impl From<VirtualPageNum> for VirtualAddr {
    fn from(vpn: VirtualPageNum) -> Self {
        VirtualAddr(vpn.address)
    }
}
impl From<VirtualAddr> for VirtualPageNum {
    fn from(addr: VirtualAddr) -> Self {
        addr.vpn()
    }
}

impl VirtualPageNum {
    pub(super) fn from_vaddr(vaddr: WordType) -> Self {
        VirtualPageNum {
            address: vaddr & !(PAGE_SIZE - 1),
        }
    }
}
