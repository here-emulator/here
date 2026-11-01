use std::cmp::Ordering;
use std::collections::BTreeMap;

use here_types::{BiMap, WordType, XLEN};

use crate::riscv::{
    csr_reg::{
        CSR_PRIVILEGE_TABLE, PrivilegeLevel,
        extension::{
            basic::*,
            f::RVFExtCsrRegfile,
            s::{RVSExtCsrRegfile, Satp},
            v::RVVExtCsrRegfile,
        },
    },
    isa_builder::{Extension, ISABuilder, IsaParseError},
};

pub type CsrDataType = u64;
pub type CsrMaskType = CsrDataType;

#[derive(Default)]
pub struct CsrItem {
    pub data: CsrDataType,
    pub mask: CsrMaskType,
}

pub(super) type CsrReadFn = fn(&CsrRegfile) -> u64;
pub(super) type CsrWriteFn = fn(&mut CsrRegfile, u64);

pub(super) trait SubExtCsrRegfile {
    fn register(regfile: &mut CsrRegfile);
}

#[derive(Default)]
pub struct CsrContext {
    pub extension: WordType, // Used in `misa`
    pub xlen: u8,            // 32 or 64
}

pub struct CsrRegfile {
    pub(super) basic_csr: RVBasicCsrRegfile,
    pub(super) rvs_csr: RVSExtCsrRegfile,
    pub(super) rvf_csr: RVFExtCsrRegfile,
    pub(super) rvv_csr: RVVExtCsrRegfile,

    pub(crate) context: CsrContext,

    pub(super) map: BTreeMap<u64, (CsrReadFn, CsrWriteFn)>,
    pub(super) name_map: BiMap<String, u64>,
    // pub(self) m_indirect_map: BTreeMap<(u8, u64), (CsrReadFn, CsrWriteFn)>,
    // pub(self) s_indirect_map: BTreeMap<(u8, u64), (CsrReadFn, CsrWriteFn)>,
}

impl Default for CsrRegfile {
    fn default() -> Self {
        Self {
            basic_csr: RVBasicCsrRegfile::default(),
            rvs_csr: RVSExtCsrRegfile::default(),
            rvf_csr: RVFExtCsrRegfile::default(),
            rvv_csr: RVVExtCsrRegfile::default(),
            context: CsrContext::default(),
            map: BTreeMap::new(),
            name_map: BiMap::new(),
        }
    }
}

impl CsrRegfile {
    /// Builds a CSR register file for the ISA described by `isa`,
    /// e.g. `"RV64IMAFDC_Zicsr_Zifencei"`. See [`ISABuilder`].
    pub fn from_isa_str(isa: &str) -> Result<Self, IsaParseError> {
        Ok(Self::from_builder(isa.parse::<ISABuilder>()?))
    }

    pub(crate) fn from_builder(builder: ISABuilder) -> Self {
        let mut regfile = Self::default();
        regfile.context = CsrContext {
            extension: builder.extension_bits(),
            xlen: XLEN as u8,
        };
        RVBasicCsrRegfile::register(&mut regfile);
        RVSExtCsrRegfile::register(&mut regfile);

        if builder.has(Extension::F) || builder.has(Extension::D) {
            RVFExtCsrRegfile::register(&mut regfile);
        }

        if builder.has(Extension::V) {
            RVVExtCsrRegfile::register(&mut regfile);
        }

        regfile
    }

    pub fn is_read_priv_legal(&self, csr_addr: WordType, priv_level: PrivilegeLevel) -> bool {
        if csr_addr == Satp::as_addr()
            && priv_level == PrivilegeLevel::S
            && Mstatus::get_tvm(self) == 1
        {
            return false;
        }

        match CSR_PRIVILEGE_TABLE.binary_search_by(|&(k, _)| {
            if k > csr_addr {
                Ordering::Greater
            } else if k == csr_addr {
                Ordering::Equal
            } else {
                Ordering::Less
            }
        }) {
            Ok(i) => priv_level >= CSR_PRIVILEGE_TABLE[i].1,
            Err(i) => priv_level >= CSR_PRIVILEGE_TABLE[i - 1].1,
        }
    }

    pub fn is_write_priv_legal(&self, csr_addr: WordType, priv_level: PrivilegeLevel) -> bool {
        if csr_addr >= 0xC00 {
            false
        } else {
            self.is_read_priv_legal(csr_addr, priv_level)
        }
    }

    #[must_use]
    pub fn read_checked(&self, addr: WordType, priv_level: PrivilegeLevel) -> Option<WordType> {
        if !self.is_read_priv_legal(addr, priv_level) {
            return None;
        }
        self.read_uncheck_privilege(addr)
    }

    #[must_use]
    pub fn write_checked(
        &mut self,
        addr: WordType,
        data: WordType,
        priv_level: PrivilegeLevel,
    ) -> bool {
        if !self.is_write_priv_legal(addr, priv_level) {
            return false;
        }
        self.write_uncheck_privilege(addr, data)
    }

    /// Write without check for privilege level, but still have validation.
    pub fn write_uncheck_privilege(&mut self, addr: WordType, data: WordType) -> bool {
        if let Some(&(_, write_fn)) = self.map.get(&(addr as u64)) {
            write_fn(self, data as u64);
            true
        } else {
            false
        }
    }

    pub fn read_uncheck_privilege(&self, addr: WordType) -> Option<WordType> {
        let &(read_fn, _) = self.map.get(&(addr as u64))?;
        Some(read_fn(self) as WordType)
    }

    pub fn addr_by_name(&self, name: &str) -> Option<WordType> {
        let key = name.to_lowercase();
        self.name_map
            .get_by_left(&key)
            .map(|&addr| addr as WordType)
    }

    pub fn name_by_addr(&self, addr: WordType) -> Option<&str> {
        let key = addr as u64;
        self.name_map.get_by_right(&key).map(String::as_str)
    }

    pub fn csr_names(&self) -> impl Iterator<Item = (&str, WordType)> {
        self.name_map
            .iter()
            .map(|(name, &addr)| (name.as_str(), addr as WordType))
    }
}

pub enum CsrIndex {
    Direct(u64),
    MachineIndirect(u8, u64),
    SupervisorIndirect(u8, u64),
}

pub trait CsrValidator<NamedCsr> {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType;
}

pub trait NamedCsr: Sized + CsrValidator<Self> {
    const EXTENSION: Extension;
    const INDEX: CsrIndex;

    fn register(regfile: &mut CsrRegfile);
    fn read(regfile: &CsrRegfile) -> u64;
    fn write(regfile: &mut CsrRegfile, new_value: u64);
    fn write_directly(regfile: &mut CsrRegfile, new_value: u64);

    #[inline(always)]
    fn as_addr() -> WordType {
        match Self::INDEX {
            CsrIndex::Direct(addr) => addr as WordType,
            _ => panic!("CSR does not have a direct address"),
        }
    }
}
#[cfg(test)]
mod test {
    use crate::riscv::{
        csr_reg::{
            CsrRegfile, Cycle, Fcsr, Mcause, Mcycle, Mepc, Mie, Mip, Mstatus, Mtvec, NamedCsr,
            PrivilegeLevel, Sie, Sip, Sstatus, Vcsr, Vxrm, Vxsat,
        },
        isa_builder::TEST_ISA,
    };

    fn new_all_ext_regfile() -> CsrRegfile {
        CsrRegfile::from_isa_str(TEST_ISA).expect("TEST_ISA must be valid")
    }

    #[test]
    fn test_rw_by_addr() {
        let mut reg = new_all_ext_regfile();
        assert!(reg.write_checked(Mcause::as_addr(), 3, PrivilegeLevel::M));
        assert!(reg.write_checked(Mepc::as_addr(), 0x1234_5678, PrivilegeLevel::M));

        let mcause = reg
            .read_checked(Mcause::as_addr(), PrivilegeLevel::M)
            .unwrap();
        let mepc = reg
            .read_checked(Mepc::as_addr(), PrivilegeLevel::M)
            .unwrap();

        assert_eq!(mcause, 3);
        assert_eq!(mepc, 0x1234_5678);
    }

    #[test]
    fn test_rw_by_type() {
        let mut reg = new_all_ext_regfile();

        Mstatus::set_mpp(&mut reg, 3);
        Mstatus::set_sie(&mut reg, 1);

        assert_eq!(Mstatus::get_mpp(&reg), 3);
        assert_eq!(Mstatus::get_sie(&reg), 1);
        let mstatus_val = reg
            .read_checked(Mstatus::as_addr(), PrivilegeLevel::M)
            .unwrap();
        assert_eq!(mstatus_val, (1 << 1) | (3 << 11));

        Mstatus::set_mpp(&mut reg, 0b01);
        assert_eq!(Mstatus::get_mpp(&reg), 0b01);

        assert!(reg.write_checked(Mtvec::as_addr(), 0x114514, PrivilegeLevel::M));
        assert_eq!(Mtvec::get_base(&reg), 0x114514 >> 2);
    }

    #[test]
    fn test_read_privilege() {
        let mut reg = new_all_ext_regfile();
        assert!(reg.write_checked(Mcause::as_addr(), 0xFEFE, PrivilegeLevel::M));
        assert_eq!(
            reg.read_uncheck_privilege(Mcause::as_addr()).unwrap(),
            0xFEFE
        );

        assert!(!reg.write_checked(Mcause::as_addr(), 0xFEFE, PrivilegeLevel::S));
        assert_eq!(
            reg.read_uncheck_privilege(Mcause::as_addr()).unwrap(),
            0xFEFE
        );
    }

    #[test]
    fn test_mcycle() {
        let mut reg = new_all_ext_regfile();

        // `mcycle` is a normal writable CSR.

        // Test NamedCsr API.
        Mcycle::set_mcycle(&mut reg, 1);
        assert_eq!(Mcycle::get_mcycle(&reg), 1);

        // Test CsrRegfile API.
        assert_eq!(reg.read_uncheck_privilege(Mcycle::as_addr()).unwrap(), 1);
        reg.write_uncheck_privilege(Mcycle::as_addr(), 2);
        assert_eq!(reg.read_uncheck_privilege(Mcycle::as_addr()).unwrap(), 2);

        // `cycle` is a shadow CSR of `mcycle`.
        reg.write_uncheck_privilege(Cycle::as_addr(), 3);
        assert_eq!(reg.read_uncheck_privilege(Cycle::as_addr()).unwrap(), 2);
    }

    #[test]
    fn test_vector_fixed_point_shadow_csr() {
        let mut reg = new_all_ext_regfile();

        reg.write_uncheck_privilege(Vxrm::as_addr(), 0b10);
        reg.write_uncheck_privilege(Vxsat::as_addr(), 1);

        assert_eq!(reg.read_uncheck_privilege(Vxrm::as_addr()).unwrap(), 0b10);
        assert_eq!(reg.read_uncheck_privilege(Vxsat::as_addr()).unwrap(), 1);
        assert_eq!(Vcsr::get_vxrm(&reg), 0b10);
        assert_eq!(Vcsr::get_vxsat(&reg), 1);
        assert_eq!(reg.read_uncheck_privilege(Fcsr::as_addr()).unwrap(), 0);
    }

    #[test]
    fn test_xstatus() {
        let mut reg = new_all_ext_regfile();

        Mstatus::set_spp(&mut reg, 1);
        assert_eq!(Mstatus::get_spp(&reg), 1);
        assert_eq!(Sstatus::get_spp(&reg), 1);

        reg.write_uncheck_privilege(Mstatus::as_addr(), 0x1802);
        assert_eq!(
            reg.read_uncheck_privilege(Sstatus::as_addr()).unwrap(),
            0x02
        );
        assert_eq!(
            reg.read_uncheck_privilege(Mstatus::as_addr()).unwrap(),
            0x1802
        );
    }

    #[test]
    fn test_s_mode_shadow_csr() {
        let mut csr = new_all_ext_regfile();

        Mip::set_seip(&mut csr, 1);
        assert_eq!(Sip::get_seip(&csr), 1);
        Mip::set_meip(&mut csr, 1);
        assert_eq!(csr.read_uncheck_privilege(Sip::as_addr()).unwrap(), 0x200);
        assert_eq!(csr.read_uncheck_privilege(Mip::as_addr()).unwrap(), 0xa00);

        Mie::set_seie(&mut csr, 1);
        assert_eq!(Sie::get_seie(&csr), 1);
    }

    #[test]
    fn test_mstatus_mpp_validation() {
        use crate::riscv::isa_builder::misa_bit;

        let mut reg = new_all_ext_regfile();
        Mstatus::set_mpp(&mut reg, PrivilegeLevel::M as u64);
        assert_eq!(Mstatus::get_mpp(&reg), PrivilegeLevel::M as u64);

        // Disable U and S extensions in context
        reg.context.extension &= !(misa_bit('U') | misa_bit('S'));

        // Attempting to write U (0) or S (1) should preserve old MPP (M = 3)
        Mstatus::set_mpp(&mut reg, PrivilegeLevel::U as u64);
        assert_eq!(Mstatus::get_mpp(&reg), PrivilegeLevel::M as u64);

        Mstatus::set_mpp(&mut reg, PrivilegeLevel::S as u64);
        assert_eq!(Mstatus::get_mpp(&reg), PrivilegeLevel::M as u64);

        // Re-enable S extension -> writing S succeeds
        reg.context.extension |= misa_bit('S');
        Mstatus::set_mpp(&mut reg, PrivilegeLevel::S as u64);
        assert_eq!(Mstatus::get_mpp(&reg), PrivilegeLevel::S as u64);
    }

    #[test]
    fn test_csr_name_map() {
        let reg = new_all_ext_regfile();
        assert_eq!(reg.addr_by_name("mstatus"), Some(Mstatus::as_addr()));
        assert_eq!(reg.addr_by_name("MSTATUS"), Some(Mstatus::as_addr()));
        assert_eq!(reg.name_by_addr(Mstatus::as_addr()), Some("mstatus"));
        assert_eq!(reg.addr_by_name("fcsr"), Some(Fcsr::as_addr()));
        assert_eq!(reg.name_by_addr(Fcsr::as_addr()), Some("fcsr"));
        assert_eq!(reg.addr_by_name("sstatus"), Some(Sstatus::as_addr()));
        assert_eq!(reg.name_by_addr(Sstatus::as_addr()), Some("sstatus"));
    }
}
