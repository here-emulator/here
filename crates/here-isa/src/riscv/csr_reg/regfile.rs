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
        self.name_map.get_by_left(&key).map(|&addr| addr as WordType)
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
