#[macro_use]
mod write_validator;
#[macro_use]
mod read_validator;
mod cpu;

pub mod csr_macro;
pub mod utils;

use self::{
    csr_macro::{CSR_REG_TABLE, Fcsr, Mstatus, Satp, Vcsr, resolve_shadow_addr},
    read_validator::ReadValidator,
    write_validator::WriteValidator,
};
use crate::config::arch_config::WordType;
use std::cmp::Ordering;

/// Constants in this module are not complete. Use `get_index` static method for each CSR type, like [`Mstatus::get_index`].
// TODO: Consider replace all uses of `csr_index` with the corresponding `CSRType::get_index`, 
// then remove the `csr_index` module.
#[rustfmt::skip]
#[allow(non_upper_case_globals, unused)]
pub(crate) mod csr_index {
    use crate::config::arch_config::WordType;
    pub const mstatus   : WordType  = 0x300;    // CPU 状态寄存器，控制中断使能、特权级别
    pub const misa      : WordType  = 0x301;    // ISA 特性寄存器，标明 CPU 支持的指令集扩展
    pub const mie       : WordType  = 0x304;    // 机器中断使能寄存器
    pub const mtvec     : WordType  = 0x305;    // 异常/中断向量基址
    pub const mscratch  : WordType  = 0x340;    // 痕迹寄存器, 一般用于在模式切换时, 保存寄存器信息(如栈信息等)
    pub const mepc      : WordType  = 0x341;    // 异常返回地址
    pub const mcause    : WordType  = 0x342;    // 异常原因
    pub const mtval     : WordType  = 0x343;    // 异常附加信息（例如非法访问地址）
    pub const mip       : WordType  = 0x344;    // 中断挂起寄存器
    // pub const mhartid   : WordType  = 0xF14;    // CPU hart ID（多核情况下）

    // Floating-Point CSR
    pub const fflags    : WordType  = 0x001;
    pub const frm       : WordType  = 0x002;
    pub const fcsr      : WordType  = 0x003;

    // Vector CSR
    pub const vstart    : WordType  = 0x008;
    pub const vxsat     : WordType  = 0x009;
    pub const vxrm      : WordType  = 0x00a;
    pub const vcsr      : WordType  = 0x00f;
    pub const vl        : WordType  = 0xc20;
    pub const vtype     : WordType  = 0xc21;
    pub const vlenb     : WordType  = 0xc22;
}

#[repr(u8)]
#[derive(Debug, PartialEq, PartialOrd, Ord, Eq, Clone, Copy)]
pub enum PrivilegeLevel {
    U = 0,
    S = 1,
    V = 2,
    M = 3,
}

#[rustfmt::skip]
const CSR_PRIVILEGE_TABLE: &[(WordType, PrivilegeLevel)] = &[
    // READ WRITE.
    (0x000, PrivilegeLevel::U), // 0x0FF
    (0x100, PrivilegeLevel::S), // 0x1FF
    (0x200, PrivilegeLevel::V), // 0x2FF
    (0x300, PrivilegeLevel::M), // 0x3FF

    (0x400, PrivilegeLevel::U), // 0x4FF
    (0x500, PrivilegeLevel::S), // 0x57F
    (0x580, PrivilegeLevel::S), // 0x5BF
    (0x5C0, PrivilegeLevel::S), // 0x5FF (Custom)
    (0x600, PrivilegeLevel::V), // 0x67F
    (0x680, PrivilegeLevel::V), // 0x6BF
    (0x6C0, PrivilegeLevel::V), // 0x6FF (Custom)
    (0x700, PrivilegeLevel::M), // 0x77F
    (0x780, PrivilegeLevel::M), // 0x7BF
    (0x7A0, PrivilegeLevel::M), // 0x7FF
    (0x7B0, PrivilegeLevel::M), // 0x7BF (Debug-mode-only)
    (0x7C0, PrivilegeLevel::M), // 0x7FF (Custom)

    (0x800, PrivilegeLevel::U), // 0x8FF (Custom)
    (0x900, PrivilegeLevel::S), // 0x97F
    (0x980, PrivilegeLevel::S), // 0x9BF
    (0x9C0, PrivilegeLevel::S), // 0x9FF (Custom)
    (0xA00, PrivilegeLevel::V), // 0xA7F
    (0xA80, PrivilegeLevel::V), // 0xABF
    (0xAC0, PrivilegeLevel::V), // 0xAFF (Custom)
    (0xB00, PrivilegeLevel::M), // 0xB7F
    (0xB80, PrivilegeLevel::M), // 0xBBF
    (0xBC0, PrivilegeLevel::M), // 0xBFF (Custom)

    // READ ONLY.
    (0xC00, PrivilegeLevel::U), // 0xC7F
    (0xC80, PrivilegeLevel::U), // 0xCBF
    (0xCC0, PrivilegeLevel::U), // 0xCFF (Custom)
    (0xD00, PrivilegeLevel::S), // 0xD7F
    (0xD80, PrivilegeLevel::S), // 0xD8F
    (0xDC0, PrivilegeLevel::S), // 0xDFF (Custom)
    (0xE00, PrivilegeLevel::V), // 0xE7F
    (0xE80, PrivilegeLevel::V), // 0xE8F
    (0xEC0, PrivilegeLevel::V), // 0xEFF (Custom)
    (0xF00, PrivilegeLevel::M), // 0xF7F
    (0xF80, PrivilegeLevel::M), // 0xF8F
    (0xFC0, PrivilegeLevel::M), // 0xFFF (Custom)
];

impl TryFrom<u8> for PrivilegeLevel {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(PrivilegeLevel::U),
            1 => Ok(PrivilegeLevel::S),
            2 => Ok(PrivilegeLevel::V),
            3 => Ok(PrivilegeLevel::M),
            _ => Err("Invalid privilege level"),
        }
    }
}

pub(crate) trait NamedCsrReg {
    fn new(data: *mut CsrReg, ctx: *mut CsrContext) -> Self;
    fn get_index() -> WordType;
    fn data(&self) -> WordType;
    fn set_data(&mut self, val: WordType);
    fn set_data_directly(&mut self, data: WordType);
}

/// Write `value` to the bits specified by `mask`.
pub(crate) struct CsrWriteOp {
    mask: WordType,
}

impl CsrWriteOp {
    #[inline]
    fn new(mask: WordType) -> CsrWriteOp {
        CsrWriteOp { mask }
    }

    #[inline]
    fn new_write_all() -> CsrWriteOp {
        CsrWriteOp { mask: !0 }
    }

    #[inline]
    fn apply(&self, target: &mut WordType, value: WordType) {
        *target = self.get_new_value(*target, value);
    }

    #[inline]
    fn get_new_value(&self, old_value: WordType, value: WordType) -> WordType {
        (old_value & !self.mask) | (value & self.mask)
    }

    /// Merge two write operations into one by `OR`.
    ///
    /// XXX: You'd better not to pass overlapped masks, but there's no check for this.
    #[inline]
    fn merge(&self, rhs: &CsrWriteOp) -> CsrWriteOp {
        CsrWriteOp {
            mask: self.mask | rhs.mask,
        }
    }

    /// Merge two write operations into one by `AND`.
    #[inline]
    fn mask(&self, rhs: &CsrWriteOp) -> CsrWriteOp {
        CsrWriteOp {
            mask: self.mask & rhs.mask,
        }
    }
}

pub(crate) struct CsrContext {
    pub extension: WordType, // Used in `misa`
    pub xlen: u8,            // 32 or 64
}

impl CsrContext {
    fn new() -> CsrContext {
        CsrContext {
            extension: 0,
            xlen: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CsrReg {
    value: WordType,
    write_validator: Option<WriteValidator>,
    read_validator: Option<ReadValidator>,
}

impl CsrReg {
    fn new(
        value: WordType,
        validator: Option<WriteValidator>,
        shadow_view: Option<ReadValidator>,
    ) -> CsrReg {
        CsrReg {
            value,
            write_validator: validator,
            read_validator: shadow_view,
        }
    }

    fn value(&self) -> WordType {
        self.value
    }

    fn validate(&self, new_value: WordType, context: &CsrContext) -> CsrWriteOp {
        if let Some(validator) = self.write_validator {
            return validator(new_value, context);
        }
        return CsrWriteOp::new_write_all();
    }

    fn write(&mut self, new_value: WordType, context: &CsrContext) {
        if let Some(validator) = self.write_validator {
            let op = validator(new_value, context);
            op.apply(&mut self.value, new_value);
        } else {
            self.value = new_value;
        }
    }

    /// Apply the write operation, with the given CsrWriteOp.
    ///
    /// NOTE: This is currently used when writing to a shadow CSR,
    /// which needs to validate and apply the write operation to its base CSR.
    fn write_with_mask(&mut self, new_value: WordType, context: &CsrContext, mask: CsrWriteOp) {
        if let Some(validator) = self.write_validator {
            let op = validator(new_value, context).mask(&mask);
            op.apply(&mut self.value, new_value);
        } else {
            self.value = new_value;
        }
    }

    /// Write directly without any validation.
    fn write_directly(&mut self, new_value: WordType) {
        self.value = new_value;
    }
}

const CSR_SIZE: usize = 1 << 12;

pub(crate) struct CsrRegFile {
    table: Vec<Option<CsrReg>>,
    cpl: PrivilegeLevel, // current privileged level
    pub(super) ctx: CsrContext,
}

impl CsrRegFile {
    pub fn new() -> Self {
        Self::from(CSR_REG_TABLE)
    }

    pub fn from(csr_table: &[(WordType, WordType, WriteValidator)]) -> Self {
        let mut table = vec![None; CSR_SIZE];
        for (addr, default_value, validator) in csr_table.iter() {
            table[*addr as usize] = Some(CsrReg::new(
                *default_value,
                Some(*validator),
                resolve_shadow_addr(*addr),
            ));
        }

        Self {
            table,
            cpl: PrivilegeLevel::M,
            ctx: CsrContext::new(),
        }
    }

    pub fn is_read_priv_legal(&mut self, csr_addr: WordType) -> bool {
        if csr_addr == Satp::get_index()
            && self.privelege_level() == PrivilegeLevel::S
            && self.get_by_type_existing::<Mstatus>().get_tvm() == 1
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
            Ok(i) => self.privelege_level() >= CSR_PRIVILEGE_TABLE[i].1,
            Err(i) => self.privelege_level() >= CSR_PRIVILEGE_TABLE[i - 1].1,
        }
    }

    pub fn is_write_priv_legal(&mut self, csr_addr: WordType) -> bool {
        if csr_addr >= 0xC00 {
            false
        } else {
            self.is_read_priv_legal(csr_addr)
        }
    }

    #[must_use]
    fn read_checked(&mut self, addr: WordType) -> Option<WordType> {
        if !self.is_read_priv_legal(addr) {
            return None;
        }
        self.read_uncheck_privilege(addr)
    }

    #[must_use]
    fn write_checked(&mut self, addr: WordType, data: WordType) -> bool {
        if !self.is_write_priv_legal(addr) {
            return false;
        }
        self.write_uncheck_privilege(addr, data);
        true
    }

    /// ONLY used in debugger. Read & write without side-effect.
    /// TODO: Consider remove this.
    pub fn debug(&mut self, addr: WordType, new_value: Option<WordType>) -> Option<WordType> {
        let value = self.read_uncheck_privilege(addr);

        if let Some(value) = new_value {
            self.write_uncheck_privilege(addr, value);
        }

        value
    }

    /// Write directly without any check or validation and have no other side effects.
    #[must_use]
    pub fn write_directly(&mut self, addr: WordType, data: WordType) -> bool {
        if addr >= CSR_SIZE as WordType {
            return false;
        }
        if let Some(reg) = self.table[addr as usize].as_mut() {
            reg.write_directly(data);
            true
        } else {
            false
        }
    }

    /// Write without check for privilege level, but still have validation.
    /// If you want to write without any check or validation, use [`Self::write_directly`] instead.
    pub fn write_uncheck_privilege(&mut self, addr: WordType, data: WordType) {
        // Special-case fflags and frm, they have their own addr but are subfields of fcsr.
        if addr == csr_index::fflags {
            let fcsr = self.table[Fcsr::get_index() as usize].as_mut().unwrap();
            fcsr.value = (fcsr.value & !0b11111) | (data & 0b11111);
        } else if addr == csr_index::frm {
            let fcsr = self.table[Fcsr::get_index() as usize].as_mut().unwrap();
            fcsr.value = (fcsr.value & !0b11100000) | ((data & 0b111) << 5);
        } else if addr == csr_index::fcsr {
            // Quoted from RISC-V manual:
            // "Bits 31—8 of the fcsr are reserved for other standard extensions. If these extensions are not present,
            // implementations shall ignore writes to these bits and supply a zero value when read."
            let fcsr = self.table[Fcsr::get_index() as usize].as_mut().unwrap();
            fcsr.value = data & 0xFF;
        } else if addr == csr_index::vxsat {
            let vcsr = self.table[Vcsr::get_index() as usize].as_mut().unwrap();
            vcsr.value = (vcsr.value & !0b1) | (data & 0b1);
        } else if addr == csr_index::vxrm {
            let vcsr = self.table[Vcsr::get_index() as usize].as_mut().unwrap();
            vcsr.value = (vcsr.value & !0b110) | ((data & 0b11) << 1);
        } else {
            if let Some(csr) = self.table[addr as usize].as_mut() {
                if let Some(base_addr) = csr.read_validator {
                    // This is a shadow CSR, write to its base CSR instead.

                    // Validate with the shadow CSR's validator.
                    let op = csr.validate(data, &self.ctx);

                    if let Some(base_csr) = self.table[base_addr.target_index as usize].as_mut() {
                        // Apply the write operation to the base CSR, with the base CSR's validator.
                        base_csr.write_with_mask(data, &self.ctx, op);
                    } else {
                        // TODO: Raise error
                    }
                } else {
                    csr.write(data, &self.ctx);
                }
            } else {
                // TODO: Raise error
            }
        }
    }

    pub fn read_uncheck_privilege(&self, addr: WordType) -> Option<WordType> {
        // Special-case fflags and frm; they are subfields of fcsr
        if addr == csr_index::fflags {
            Some(self.table[Fcsr::get_index() as usize].unwrap().value() & 0b11111)
        } else if addr == csr_index::frm {
            Some((self.table[Fcsr::get_index() as usize].unwrap().value() >> 5) & 0b111)
        } else if addr == csr_index::vxsat {
            Some(self.table[Vcsr::get_index() as usize].unwrap().value() & 0b1)
        } else if addr == csr_index::vxrm {
            Some((self.table[Vcsr::get_index() as usize].unwrap().value() >> 1) & 0b11)
        } else {
            if let Some(base_addr) = resolve_shadow_addr(addr) {
                // This is a shadow CSR.
                // The base CSR must exist because it can be resolved by `resolve_shadow_addr`.
                Some(
                    self.table[base_addr.target_index as usize].unwrap().value()
                        & base_addr.view_mask,
                )
            } else {
                Some(self.table[addr as usize]?.value())
            }
        }
    }

    /// NOTE: Given that this the CSR type is known, so this function won't check privilege level.
    /// If you ensure the CSR exists, use [`Self::get_by_type_existing`] instead for better performance.
    pub fn get_by_type<T>(&mut self) -> Option<T>
    where
        T: NamedCsrReg,
    {
        if let Some(base_addr) = resolve_shadow_addr(T::get_index()) {
            // This is a shadow CSR, get its base CSR instead.
            let reg = self.table[base_addr.target_index as usize].as_mut()?;
            return Some(T::new(reg as *mut CsrReg, &mut self.ctx as *mut CsrContext));
        }

        let reg = self.table[T::get_index() as usize].as_mut()?;
        Some(NamedCsrReg::new(
            reg as *mut CsrReg,
            &mut self.ctx as *mut CsrContext,
        ))
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
        if cfg!(debug_assertions) {
            self.get_by_type::<T>().unwrap()
        } else {
            unsafe { self.get_by_type::<T>().unwrap_unchecked() }
        }
    }

    pub fn privelege_level(&self) -> PrivilegeLevel {
        self.cpl
    }

    pub fn set_current_privileged(&mut self, new_level: PrivilegeLevel) {
        self.cpl = new_level
    }
}

#[cfg(test)]
mod test {
    use crate::isa::riscv::csr_reg::{
        CsrRegFile, NamedCsrReg, PrivilegeLevel, csr_index, csr_macro::*,
    };

    #[test]
    fn test_rw_by_addr() {
        let mut reg = CsrRegFile::new();
        assert!(reg.write_checked(csr_index::mcause, 3));
        assert!(reg.write_checked(csr_index::mepc, 0x1234_5678));

        let mcause = reg.read_checked(csr_index::mcause).unwrap();
        let mepc = reg.read_checked(csr_index::mepc).unwrap();

        assert_eq!(mcause, 3);
        assert_eq!(mepc, 0x1234_5678);
    }

    #[test]
    fn test_rw_by_type() {
        let mut reg = CsrRegFile::new();
        let mstatus = reg.get_by_type::<Mstatus>().unwrap();

        mstatus.set_mpp(3);
        mstatus.set_sie(1);

        assert_eq!(mstatus.get_mpp(), 3);
        assert_eq!(mstatus.get_sie(), 1);
        let mstatus_val = reg.read_checked(csr_index::mstatus).unwrap();
        assert_eq!(mstatus_val, (1 << 1 | 3 << 11));

        mstatus.set_mpp(0b10);
        assert_eq!(mstatus.get_mpp(), 0b10);

        let mtvec = reg.get_by_type::<Mtvec>().unwrap();
        assert!(reg.write_checked(csr_index::mtvec, 0x114514));
        assert_eq!(mtvec.get_base(), 0x114514 >> 2);
    }

    #[test]
    fn test_read_privilege() {
        let mut reg = CsrRegFile::new();
        reg.set_current_privileged(PrivilegeLevel::M);
        assert!(reg.write_checked(csr_index::mcause, 0xFEFE));
        assert_eq!(
            reg.read_uncheck_privilege(csr_index::mcause).unwrap(),
            0xFEFE
        );

        reg.set_current_privileged(PrivilegeLevel::S);
        assert!(!reg.write_checked(csr_index::mcause, 0xFEFE));
        assert_eq!(
            reg.read_uncheck_privilege(csr_index::mcause).unwrap(),
            0xFEFE
        );
    }

    #[test]
    fn test_mcycle() {
        let mut reg = CsrRegFile::new();

        // `mcycle` is a normal writable CSR.

        // Test NamedCsr API.
        let mcycle = reg.get_by_type_existing::<Mcycle>();
        mcycle.set_mcycle(1);
        assert_eq!(mcycle.get_mcycle(), 1);

        // Test CsrRegfile API.
        assert_eq!(reg.read_uncheck_privilege(Mcycle::get_index()).unwrap(), 1);
        reg.write_uncheck_privilege(Mcycle::get_index(), 2);
        assert_eq!(reg.read_uncheck_privilege(Mcycle::get_index()).unwrap(), 2);

        // `cycle` is a shadow CSR of `mcycle`.

        // It's known that with NamedCsr API, shadow CSR's validator cannot work properly,
        // so only test CsrRegfile API.
        reg.write_uncheck_privilege(Cycle::get_index(), 3);
        assert_eq!(reg.read_uncheck_privilege(Cycle::get_index()).unwrap(), 2);
    }

    #[test]
    fn test_vector_fixed_point_shadow_csr() {
        let mut reg = CsrRegFile::new();

        reg.write_uncheck_privilege(csr_index::vxrm, 0b10);
        reg.write_uncheck_privilege(csr_index::vxsat, 1);

        assert_eq!(reg.read_uncheck_privilege(csr_index::vxrm).unwrap(), 0b10);
        assert_eq!(reg.read_uncheck_privilege(csr_index::vxsat).unwrap(), 1);
        assert_eq!(reg.get_by_type_existing::<Vcsr>().get_vxrm(), 0b10);
        assert_eq!(reg.get_by_type_existing::<Vcsr>().get_vxsat(), 1);
        assert_eq!(reg.read_uncheck_privilege(Fcsr::get_index()).unwrap(), 0);
    }

    #[test]
    fn test_xstatus() {
        let mut reg = CsrRegFile::new();

        let mstatus = reg.get_by_type_existing::<Mstatus>();
        let sstatus = reg.get_by_type_existing::<Sstatus>();

        mstatus.set_spp(1);
        assert_eq!(mstatus.get_spp(), 1);
        assert_eq!(sstatus.get_spp(), 1);

        reg.write_uncheck_privilege(Mstatus::get_index(), 0x1002);
        assert_eq!(
            reg.read_uncheck_privilege(Sstatus::get_index()).unwrap(),
            0x02
        );
        assert_eq!(
            reg.read_uncheck_privilege(Mstatus::get_index()).unwrap(),
            0x1002
        );
    }

    #[test]
    fn test_s_mode_shadow_csr() {
        let mut csr = CsrRegFile::new();

        let mip = csr.get_by_type_existing::<Mip>();
        let sip = csr.get_by_type_existing::<Sip>();
        mip.set_seip(1);
        assert_eq!(sip.get_seip(), 1);
        mip.set_meip(1);
        assert_eq!(csr.read_uncheck_privilege(Sip::get_index()).unwrap(), 0x200);
        assert_eq!(csr.read_uncheck_privilege(Mip::get_index()).unwrap(), 0xa00);

        let mie = csr.get_by_type_existing::<Mie>();
        let sie = csr.get_by_type_existing::<Sie>();
        mie.set_seie(1);
        assert_eq!(sie.get_seie(), 1);
    }
}
