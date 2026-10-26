use bitflags::bitflags;
use core::panic;

use super::*;

use crate::{
    config::arch_config::WordType,
    isa::{
        cache::*,
        riscv::mmu::{
            address::{PhysicalAddr, PhysicalPageNum, VirtualAddr, VirtualPageNum},
            config::*,
        },
    },
    ram::Ram,
    ram_config,
};

bitflags! {
    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    pub struct PTEFlags: u8 {
        const V = 1 << 0; // valid
        const R = 1 << 1; // read
        const W = 1 << 2; // write
        const X = 1 << 3; // execute
        const U = 1 << 4; // U-Mode
        const G = 1 << 5; // global mapping, will not be refreshed in TLB.
        const A = 1 << 6; // accessed, means this leaf-page has a mapping to physical memory.
        const D = 1 << 7; // dirty
    }
}

// impl std::fmt::Debug for PTEFlags {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_tuple("PTEFlags").field(&self.0).finish()
//     }
// }

#[derive(Copy, Clone)]
#[repr(C)]
pub struct PageTableEntry {
    bits: WordType,
}

#[allow(unused)]
impl PageTableEntry {
    pub fn new(bits: WordType) -> Self {
        PageTableEntry { bits }
    }

    pub fn empty() -> Self {
        PageTableEntry { bits: 0 }
    }

    fn ppn(&self) -> PhysicalPageNum {
        PhysicalPageNum::from_ppn((self.bits & PTE_PPN_MASK) >> PTE_FLAG_XLEN)
    }

    fn set_ppn(&mut self, ppn: PhysicalPageNum) {
        self.bits &= PTE_FLAG_MASK; // clear ppn bits.
        self.bits |= ppn.address >> 2;
    }

    fn flags(&self) -> PTEFlags {
        PTEFlags::from_bits((self.bits & PTE_FLAG_MASK) as u8).unwrap()
    }

    fn is_leaf(&self) -> bool {
        self.has_any_flag(PTEFlags::R | PTEFlags::W | PTEFlags::X)
    }

    fn is_invalid_encoding(&self) -> bool {
        (self.flags() & (PTEFlags::R | PTEFlags::W)) == PTEFlags::W
    }

    /// Return true if any requested flag is set.
    fn has_any_flag(&self, flags: PTEFlags) -> bool {
        self.flags().intersects(flags)
    }

    fn is_valid(&self) -> bool {
        !(self.flags() & PTEFlags::V).is_empty()
    }
    fn is_readable(&self) -> bool {
        !(self.flags() & PTEFlags::R).is_empty()
    }
    fn is_writable(&self) -> bool {
        !(self.flags() & PTEFlags::W).is_empty()
    }
    fn is_executable(&self) -> bool {
        !(self.flags() & PTEFlags::X).is_empty()
    }
    fn is_u_mode(&self) -> bool {
        !(self.flags() & PTEFlags::U).is_empty()
    }
    fn is_global(&self) -> bool {
        !(self.flags() & PTEFlags::G).is_empty()
    }
    fn is_accessed(&self) -> bool {
        !(self.flags() & PTEFlags::A).is_empty()
    }
    fn is_dirty(&self) -> bool {
        !(self.flags() & PTEFlags::D).is_empty()
    }

    // set flag for PTE.
    fn set_flag(&mut self, flag: PTEFlags) {
        self.bits |= flag.bits() as WordType;
    }
    fn set_valid(&mut self) {
        self.bits |= PTEFlags::V.bits() as WordType;
    }
    fn set_readable(&mut self) {
        self.bits |= PTEFlags::R.bits() as WordType;
    }
    fn set_writable(&mut self) {
        self.bits |= PTEFlags::W.bits() as WordType;
    }
    fn set_executable(&mut self) {
        self.bits |= PTEFlags::X.bits() as WordType;
    }
    fn set_u_mode(&mut self) {
        self.bits |= PTEFlags::U.bits() as WordType;
    }
    fn set_global(&mut self) {
        self.bits |= PTEFlags::G.bits() as WordType;
    }
    fn set_accessed(&mut self) {
        self.bits |= PTEFlags::A.bits() as WordType;
    }
    fn set_dirty(&mut self) {
        self.bits |= PTEFlags::D.bits() as WordType;
    }
}

#[derive(Copy, Clone)]
struct WalkInfo {
    leaf_level: usize,
    leaf_flags: PTEFlags,
    leaf_pte_addr: u64,
    leaf_ppn: PhysicalPageNum,
}

impl Cacheable for WalkInfo {
    const ADDR_SHIFT_BITS: usize = PAGE_SIZE_XLEN;
}

pub struct PageTableWalker {
    tlb: Cache<DirectCache<WalkInfo, 512>>,
    root_address: WordType,
    mode: VirtualMemoryMode,
    ad_update_policy: AdUpdatePolicy,
}

impl Drop for PageTableWalker {
    fn drop(&mut self) {
        log::info!("tlb hit rate: {}", self.tlb.hit_rate())
    }
}

impl PageTableWalker {
    pub fn new(root_address: WordType, mode: VirtualMemoryMode) -> Self {
        Self {
            tlb: Cache::new(),
            root_address,
            mode,
            ad_update_policy: AdUpdatePolicy::FaultOnClear,
        }
    }

    pub fn flush_tlb(&mut self) {
        self.tlb.clear();
    }

    pub fn set_ad_update_policy(&mut self, ad_update_policy: AdUpdatePolicy) {
        self.ad_update_policy = ad_update_policy;
    }

    pub fn set_mode(&mut self, mode: u8) {
        self.mode = match mode {
            0 => VirtualMemoryMode::None,
            1 => VirtualMemoryMode::Page32bit,
            8 => VirtualMemoryMode::Page39bit,
            9 => VirtualMemoryMode::Page48bit,
            10 => VirtualMemoryMode::Page57bit,
            // 11 => VirtualMemoryMode::Page64bit,
            _ => {
                // This is not allow to happen because satp.mode is WARL.
                log::error!("MMU receive unsupported virtual memory mode: {}.", mode);
                panic!()
            }
        }
    }

    pub fn set_root_addr(&mut self, root_address: WordType) {
        self.root_address = root_address;
    }

    pub fn translate_vaddr(
        &mut self,
        mem: &mut Ram,
        vaddr: VirtualAddr,
        check: PermissionCheck,
        effect: AccessEffect,
    ) -> Result<PhysicalAddr, PageTableError> {
        if self.mode == VirtualMemoryMode::None {
            return Ok(vaddr.0.into());
        }
        if !self.is_canonical_vaddr(vaddr) {
            return Err(PageTableError::PageFault);
        }

        let walk_info = if let Some(info) = self.tlb.get(vaddr.vpn().address) {
            info
        } else {
            let info = self.walk_pte(mem, vaddr.vpn())?;
            self.tlb.put(vaddr.vpn().address, info);
            info
        };

        if (walk_info.leaf_flags & check.exact_mask) != check.exact_flags
            || (check.any_of.is_empty() == false
                && (walk_info.leaf_flags & check.any_of) == PTEFlags::empty())
        {
            // privilege fault

            // When running Linux, many such logs are normal, like copy-on-write:
            // it marks pages read-only initially, and when a write access occurs,
            // the kernel will create a private copy.
            // log::info!(
            //     "Privilege fault when translating vaddr: {:#x}, required flags: ({:?}), {}target flags: ({:?})",
            //     vaddr.0,
            //     check.exact_mask.0,
            //     if check.any_of.is_empty() {
            //         String::new()
            //     } else {
            //         format!("and any of {:?}, ", check.any_of)
            //     },
            //     walk_info.leaf_flags.0
            // );
            return Err(PageTableError::PrivilegeFault);
        }

        self.apply_ad_policy(mem, &walk_info, effect)?;

        let page_shift = PAGE_SIZE_XLEN + walk_info.leaf_level * SUB_VPN_XLEN;
        let page_offset_mask = (1 << page_shift) - 1;
        let paddr = walk_info.leaf_ppn.address | (vaddr.0 & page_offset_mask);
        Ok(paddr.into())
    }

    fn apply_ad_policy(
        &self,
        mem: &mut Ram,
        walk_info: &WalkInfo,
        effect: AccessEffect,
    ) -> Result<(), PageTableError> {
        let (need_accessed, need_dirty) = match effect {
            AccessEffect::None => (false, false),
            AccessEffect::Accessed => (!walk_info.leaf_flags.contains(PTEFlags::A), false),
            AccessEffect::AccessedDirty => (
                !walk_info.leaf_flags.contains(PTEFlags::A),
                !walk_info.leaf_flags.contains(PTEFlags::D),
            ),
        };

        if !need_accessed && !need_dirty {
            return Ok(());
        }

        match self.ad_update_policy {
            AdUpdatePolicy::AutoSet => {
                let pte = Self::pte_at(mem, walk_info.leaf_pte_addr);
                if need_accessed {
                    pte.set_accessed();
                }
                if need_dirty {
                    pte.set_dirty();
                }
                Ok(())
            }
            AdUpdatePolicy::FaultOnClear => Err(PageTableError::PageFault),
        }
    }

    fn pte_at(mem: &mut Ram, pte_addr: u64) -> &mut PageTableEntry {
        let offset = (pte_addr - ram_config::BASE_ADDR) as usize;
        let ptr = &mut mem[offset] as *mut u8 as *mut PageTableEntry;
        unsafe { &mut *ptr }
    }

    fn is_canonical_vaddr(&self, vaddr: VirtualAddr) -> bool {
        match self.mode {
            VirtualMemoryMode::Page39bit => Sv39::is_canonical_vaddr(vaddr.0),
            VirtualMemoryMode::Page48bit => Sv48::is_canonical_vaddr(vaddr.0),
            VirtualMemoryMode::Page57bit => Sv57::is_canonical_vaddr(vaddr.0),
            _ => true,
        }
    }

    fn walk_pte(&self, mem: &mut Ram, vpn: VirtualPageNum) -> Result<WalkInfo, PageTableError> {
        match self.mode {
            VirtualMemoryMode::Page39bit => self.walk_pte_with_mode::<Sv39>(mem, vpn),
            VirtualMemoryMode::Page48bit => self.walk_pte_with_mode::<Sv48>(mem, vpn),
            VirtualMemoryMode::Page57bit => self.walk_pte_with_mode::<Sv57>(mem, vpn),
            _ => panic!("Unsupported virtual memory mode in page walker"),
        }
    }

    fn walk_pte_with_mode<M: SvMode>(
        &self,
        mem: &mut Ram,
        vpn: VirtualPageNum,
    ) -> Result<WalkInfo, PageTableError> {
        let mut entry = PhysicalPageNum::from_paddr(self.root_address);

        for i in (0..M::LEVELS).rev() {
            let sub_vpn = M::vpn_index(vpn.address, i);
            let pte_addr = entry.address + (sub_vpn as u64) * (M::PTE_SIZE as u64);
            let pte = PageTableEntry::new(
                mem.read::<WordType>(pte_addr - ram_config::BASE_ADDR)
                    .or(Err(PageTableError::PageFault))?,
                // TODO: let outer code know it's RAM access fault while translating
                // instead of a common page fault, for better debugging.
            );
            if !pte.is_valid() {
                return Err(PageTableError::PageFault);
            }
            if pte.is_invalid_encoding() {
                return Err(PageTableError::PageFault);
            }

            if pte.is_leaf() {
                // A leaf PTE has been reached. If i>0 and pte.ppn[i-1:0] ≠ 0  this is a misaligned superpage;
                // stop and raise a page-fault exception corresponding to the original access type.
                let mask = ((1 << (i * SUB_VPN_XLEN)) - 1) << PTE_FLAG_XLEN;
                if pte.bits & mask != 0 {
                    return Err(PageTableError::AlignFault);
                } else {
                    return Ok(WalkInfo {
                        leaf_level: i,
                        leaf_flags: pte.flags(),
                        leaf_pte_addr: pte_addr,
                        leaf_ppn: pte.ppn(),
                    });
                }
            }

            if i == 0 {
                return Err(PageTableError::PageFault);
            }

            entry = pte.ppn();
        }

        unreachable!()
    }
}

#[cfg(test)]
mod test_sv39 {
    use crate::ram_config;

    use super::*;

    const PT0: u64 = 0x8000_1000;
    const PT1: u64 = 0x8000_2000;
    const PT2: u64 = 0x8000_3000;
    const DATA_PAGE: u64 = 0x8000_4000;

    fn setup_pte(ram: &mut Ram, entry_addr: u64, target_addr: u64, flags: PTEFlags) {
        let mut pte = PageTableEntry::new(flags.bits() as WordType);
        pte.set_ppn(PhysicalPageNum::from_paddr(target_addr));
        ram.write(entry_addr - ram_config::BASE_ADDR, pte.bits)
            .unwrap();
    }

    fn setup_3level_leaf(ram: &mut Ram, leaf_target_addr: u64, leaf_flags: PTEFlags) {
        setup_pte(ram, PT0, PT1, PTEFlags::V);
        setup_pte(ram, PT1, PT2, PTEFlags::V);
        setup_pte(ram, PT2, leaf_target_addr, leaf_flags);
    }

    #[test]
    fn addr_canonical_test() {
        assert_eq!(Sv39::is_canonical_vaddr(0xFFFF_FF7F_FFFF_FFFF), false);
        assert_eq!(Sv39::is_canonical_vaddr(0xF1FF_FFFF_FFFF_FFFF), false);
        assert_eq!(Sv39::is_canonical_vaddr(0xFFFF_FFFF_FFFF_FFFF), true);
        assert_eq!(Sv39::is_canonical_vaddr(0x0000_0000_0000_0000), true);
    }

    #[test]
    fn page_table_test() {
        let mut ram: Ram = Ram::new();

        let leaf_flags = PTEFlags::A | PTEFlags::V | PTEFlags::R | PTEFlags::W;
        setup_3level_leaf(&mut ram, DATA_PAGE, leaf_flags);

        let mut page_table = PageTableWalker::new(PT0.into(), VirtualMemoryMode::Page39bit);
        let paddr = page_table
            .translate_vaddr(
                &mut ram,
                0x0000_0123.into(),
                PermissionCheck {
                    any_of: PTEFlags::empty(),
                    exact_mask: PTEFlags::R,
                    exact_flags: PTEFlags::R,
                },
                AccessEffect::Accessed,
            )
            .unwrap();
        assert_eq!(paddr.0, DATA_PAGE | 0x123);

        let walk_info = page_table
            .walk_pte(&mut ram, VirtualPageNum::from_vaddr(0x0000_0010))
            .unwrap();
        let leaf_pte = PageTableWalker::pte_at(&mut ram, walk_info.leaf_pte_addr);
        assert_eq!(walk_info.leaf_level, 0);
        assert_eq!(leaf_pte.flags(), leaf_flags | PTEFlags::A);
    }

    #[test]
    fn big_page_test() {
        // 2MB Page.
        let mut ram: Ram = Ram::new();
        let pt0 = 0x8000_1000u64;
        let pt1 = 0x8000_2000u64;
        let data_page = 0x8100_0000u64;
        let leaf_flags = PTEFlags::A | PTEFlags::V | PTEFlags::R | PTEFlags::W;

        setup_pte(&mut ram, pt0, pt1, PTEFlags::V);
        setup_pte(&mut ram, pt1, data_page, leaf_flags);

        let mut page_table = PageTableWalker::new(pt0.into(), VirtualMemoryMode::Page39bit);
        let walk_info = page_table
            .walk_pte(&mut ram, VirtualPageNum::from_vaddr(0x0000_8000))
            .unwrap();
        let leaf_pte = PageTableWalker::pte_at(&mut ram, walk_info.leaf_pte_addr);
        assert_eq!(walk_info.leaf_level, 1);
        assert_eq!(leaf_pte.flags(), leaf_flags);

        let paddr = page_table
            .translate_vaddr(
                &mut ram,
                0x0011_4514.into(),
                PermissionCheck {
                    any_of: PTEFlags::empty(),
                    exact_mask: PTEFlags::W,
                    exact_flags: PTEFlags::W,
                },
                AccessEffect::Accessed,
            )
            .unwrap();
        assert_eq!(paddr.0, 0x8111_4514);
    }

    #[test]
    fn rwx_authority_test() {
        let mut ram: Ram = Ram::new();
        let leaf_flags = PTEFlags::V | PTEFlags::R | PTEFlags::W;

        setup_3level_leaf(&mut ram, DATA_PAGE, leaf_flags);

        let mut page_table = PageTableWalker::new(PT0.into(), VirtualMemoryMode::Page39bit);

        // try get instr, without X authority.
        let err = page_table
            .translate_vaddr(
                &mut ram,
                0x0000_0010.into(),
                PermissionCheck {
                    any_of: PTEFlags::empty(),
                    exact_mask: PTEFlags::X,
                    exact_flags: PTEFlags::X,
                },
                AccessEffect::Accessed,
            )
            .unwrap_err();
        assert_eq!(err, PageTableError::PrivilegeFault);
    }

    #[test]
    fn level0_non_leaf_should_fault() {
        let mut ram: Ram = Ram::new();
        let next_pt = 0x8000_4000u64;

        setup_pte(&mut ram, PT0, PT1, PTEFlags::V);
        setup_pte(&mut ram, PT1, PT2, PTEFlags::V);
        setup_pte(&mut ram, PT2, next_pt, PTEFlags::V);

        let mut page_table = PageTableWalker::new(PT0.into(), VirtualMemoryMode::Page39bit);
        let err = page_table
            .translate_vaddr(
                &mut ram,
                0x0000_0123.into(),
                PermissionCheck {
                    any_of: PTEFlags::empty(),
                    exact_mask: PTEFlags::R,
                    exact_flags: PTEFlags::R,
                },
                AccessEffect::Accessed,
            )
            .unwrap_err();

        assert_eq!(err, PageTableError::PageFault);
    }

    #[test]
    fn access_effect_none_should_not_set_ad_bits() {
        let mut ram: Ram = Ram::new();
        let leaf_flags = PTEFlags::V | PTEFlags::R | PTEFlags::W;

        setup_3level_leaf(&mut ram, DATA_PAGE, leaf_flags);

        let mut page_table = PageTableWalker::new(PT0.into(), VirtualMemoryMode::Page39bit);
        let paddr = page_table
            .translate_vaddr(
                &mut ram,
                0x0000_0123.into(),
                PermissionCheck {
                    any_of: PTEFlags::empty(),
                    exact_mask: PTEFlags::R,
                    exact_flags: PTEFlags::R,
                },
                AccessEffect::None,
            )
            .unwrap();
        assert_eq!(paddr.0, DATA_PAGE | 0x123);

        let walk_info = page_table
            .walk_pte(&mut ram, VirtualPageNum::from_vaddr(0x0000_0010))
            .unwrap();
        let leaf_pte = PageTableWalker::pte_at(&mut ram, walk_info.leaf_pte_addr);
        assert_eq!(leaf_pte.flags(), leaf_flags);
    }

    #[test]
    fn fault_on_clear_test() {
        let mut ram: Ram = Ram::new();
        let leaf_flags = PTEFlags::V | PTEFlags::R | PTEFlags::W;

        setup_3level_leaf(&mut ram, DATA_PAGE, leaf_flags);

        let mut page_table = PageTableWalker::new(PT0.into(), VirtualMemoryMode::Page39bit);
        page_table.set_ad_update_policy(AdUpdatePolicy::FaultOnClear);

        let err = page_table
            .translate_vaddr(
                &mut ram,
                0x0000_0123.into(),
                PermissionCheck {
                    any_of: PTEFlags::empty(),
                    exact_mask: PTEFlags::R,
                    exact_flags: PTEFlags::R,
                },
                AccessEffect::Accessed,
            )
            .unwrap_err();

        assert_eq!(err, PageTableError::PageFault);
    }

    #[test]
    fn fault_on_clear_with_none_effect_should_not_fault() {
        let mut ram: Ram = Ram::new();
        let leaf_flags = PTEFlags::V | PTEFlags::R | PTEFlags::W;

        setup_3level_leaf(&mut ram, DATA_PAGE, leaf_flags);

        let mut page_table = PageTableWalker::new(PT0.into(), VirtualMemoryMode::Page39bit);
        page_table.set_ad_update_policy(AdUpdatePolicy::FaultOnClear);

        let paddr = page_table
            .translate_vaddr(
                &mut ram,
                0x0000_0123.into(),
                PermissionCheck {
                    any_of: PTEFlags::empty(),
                    exact_mask: PTEFlags::R,
                    exact_flags: PTEFlags::R,
                },
                AccessEffect::None,
            )
            .unwrap();

        assert_eq!(paddr.0, DATA_PAGE | 0x123);
    }

    #[test]
    fn fault_on_clear_with_preset_ad_should_pass() {
        let mut ram: Ram = Ram::new();
        let leaf_flags = PTEFlags::V | PTEFlags::R | PTEFlags::W | PTEFlags::A | PTEFlags::D;

        setup_3level_leaf(&mut ram, DATA_PAGE, leaf_flags);

        let mut page_table = PageTableWalker::new(PT0.into(), VirtualMemoryMode::Page39bit);
        page_table.set_ad_update_policy(AdUpdatePolicy::FaultOnClear);

        let paddr = page_table
            .translate_vaddr(
                &mut ram,
                0x0000_0123.into(),
                PermissionCheck {
                    any_of: PTEFlags::empty(),
                    exact_mask: PTEFlags::W,
                    exact_flags: PTEFlags::W,
                },
                AccessEffect::AccessedDirty,
            )
            .unwrap();

        assert_eq!(paddr.0, DATA_PAGE | 0x123);
    }
}
