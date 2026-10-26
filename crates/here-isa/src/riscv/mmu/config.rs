use crate::{config::arch_config::WordType, utils::is_sign_extended};

pub const PAGE_SIZE_XLEN: usize = 12;
pub const PAGE_SIZE: WordType = 1 << PAGE_SIZE_XLEN;

pub const PHYSICAL_ADDR_WIDTH: usize = 56; // physical address width.
pub const PPN_WIDTH: usize = PHYSICAL_ADDR_WIDTH - PAGE_SIZE_XLEN; // PPN width size.
pub const SUB_VPN_XLEN: usize = 9;
pub const SUB_VPN_MASK: WordType = (1 << SUB_VPN_XLEN) - 1;

// ============================================
// ======= PTE flags in page table entry ======
// ============================================
pub const PTE_FLAG_XLEN: usize = 10;
pub const PTE_FLAG_MASK: WordType = (1 << PTE_FLAG_XLEN) - 1;
pub const PTE_PPN_MASK: WordType = ((1 << 44) - 1) << 10; // bits 10-53

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum VirtualMemoryMode {
    Page32bit = 0,
    Page39bit,
    Page48bit,
    Page57bit,
    // Page64bit,
    None = 0x3f,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AdUpdatePolicy {
    AutoSet,
    FaultOnClear,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AccessEffect {
    None,
    Accessed,
    AccessedDirty,
}

pub trait SvMode {
    const LEVELS: usize;
    const VA_BITS: usize;
    const PTE_SIZE: usize;

    #[inline]
    fn vpn_index(vpn: WordType, level: usize) -> usize {
        ((vpn >> (PAGE_SIZE_XLEN + level * SUB_VPN_XLEN)) & SUB_VPN_MASK) as usize
    }

    #[inline]
    fn is_canonical_vaddr(vaddr: WordType) -> bool {
        let word_bits = core::mem::size_of::<WordType>() * 8;
        if Self::VA_BITS >= word_bits {
            return true;
        }

        is_sign_extended(vaddr, Self::VA_BITS as u32)
    }
}

pub struct Sv39;
impl SvMode for Sv39 {
    const LEVELS: usize = 3;
    const VA_BITS: usize = 39;
    const PTE_SIZE: usize = 8;
}

pub struct Sv48;
impl SvMode for Sv48 {
    const LEVELS: usize = 4;
    const VA_BITS: usize = 48;
    const PTE_SIZE: usize = 8;
}

pub struct Sv57;
impl SvMode for Sv57 {
    const LEVELS: usize = 5;
    const VA_BITS: usize = 57;
    const PTE_SIZE: usize = 8;
}
