use std::collections::{BTreeMap, HashSet};

use crate::riscv::{
    csr_reg::extension::{basic::*, f::RVFExtCsrRegfile, s::RVSExtCsrRegfile, v::RVVExtCsrRegfile},
    isa_builder::Extension,
};

pub type CsrDataType = u64;
pub type CsrMaskType = CsrDataType;
pub struct CsrItem {
    pub data: CsrDataType,
    pub mask: CsrMaskType,
}

pub(super) type CsrReadFn = fn(&mut CsrRegfile) -> u64;
pub(super) type CsrWriteFn = fn(&mut CsrRegfile, u64);

pub(super) trait SubExtCsrRegfile {
    fn register(regfile: &mut CsrRegfile);
}

pub struct CsrContext {
    pub extension: WordType, // Used in `misa`
    pub xlen: u8,            // 32 or 64
}

#[derive(Default)]
pub struct CsrRegfile {
    pub(super) basic_csr: RVBasicCsrRegfile,
    pub(super) rvs_csr: RVSExtCsrRegfile,
    pub(super) rvf_csr: RVFExtCsrRegfile,
    pub(super) rvv_csr: RVVExtCsrRegfile,

    pub(crate) context: CsrContext,

    pub(super) map: BTreeMap<u64, (CsrReadFn, CsrWriteFn)>,
    // pub(self) m_indirect_map: BTreeMap<(u8, u64), (CsrReadFn, CsrWriteFn)>,
    // pub(self) s_indirect_map: BTreeMap<(u8, u64), (CsrReadFn, CsrWriteFn)>,
}

impl CsrRegfile {
    pub fn new(extensions: HashSet<Extension>) -> Self {
        let mut regfile = Self::default();

        todo!()
    }

    pub fn is_read_priv_legal(&mut self, csr_addr: WordType) -> bool {
        todo!()
    }

    pub fn is_write_priv_legal(&mut self, csr_addr: WordType) -> bool {
        todo!()
    }

    #[must_use]
    fn read_checked(&mut self, addr: WordType) -> Option<WordType> {
        todo!()
    }

    #[must_use]
    fn write_checked(&mut self, addr: WordType, data: WordType) -> bool {
        todo!()
    }

    /// ONLY used in debugger. Read & write without side-effect.
    /// TODO: Consider remove this.
    pub fn debug(&mut self, addr: WordType, new_value: Option<WordType>) -> Option<WordType> {
        todo!()
    }

    /// Write directly without any check or validation and have no other side effects.
    #[must_use]
    pub fn write_directly(&mut self, addr: WordType, data: WordType) -> bool {
        todo!()
    }

    /// Write without check for privilege level, but still have validation.
    /// If you want to write without any check or validation, use [`Self::write_directly`] instead.
    pub fn write_uncheck_privilege(&mut self, addr: WordType, data: WordType) {
        todo!()
    }

    pub fn read_uncheck_privilege(&self, addr: WordType) -> Option<WordType> {
        todo!()
    }

    /// NOTE: Given that this the CSR type is known, so this function won't check privilege level.
    /// If you ensure the CSR exists, use [`Self::get_by_type_existing`] instead for better performance.
    pub fn get_by_type<T>(&mut self) -> Option<T>
    where
        T: NamedCsrReg,
    {
        todo!()
    }

    /// Similar to `get_by_type`, but this function assumes the CSR definitely exists.
    ///
    /// - In debug builds, this function will panic if the CSR does not exist.
    /// - In release builds, this function will skip the runtime check for performance.
    #[must_use]
    pub fn get_by_type_existing<T>(&mut self) -> T
    where
        T: NamedCsrReg,
    {
        todo!()
    }

    pub fn privelege_level(&self) -> PrivilegeLevel {
        todo!()
    }

    pub fn set_current_privileged(&mut self, new_level: PrivilegeLevel) {
        todo!()
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
    fn read(regfile: &mut CsrRegfile) -> u64;
    fn write(regfile: &mut CsrRegfile, new_value: u64);
    fn write_directly(regfile: &mut CsrRegfile, new_value: u64);
}
