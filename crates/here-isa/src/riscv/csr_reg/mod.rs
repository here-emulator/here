mod cpu;
pub mod extension;
pub mod regfile;

pub use extension::*;
pub use regfile::{CsrContext, CsrDataType, CsrIndex, CsrItem, CsrMaskType, CsrRegfile, CsrValidator, NamedCsr};

use here_types::WordType;

#[repr(u8)]
#[derive(Debug, PartialEq, PartialOrd, Ord, Eq, Clone, Copy)]
pub enum PrivilegeLevel {
    U = 0,
    S = 1,
    HS = 2,
    M = 3,
}

#[rustfmt::skip]
const CSR_PRIVILEGE_TABLE: &[(WordType, PrivilegeLevel)] = &[
    // READ WRITE.
    (0x000, PrivilegeLevel::U), // 0x0FF
    (0x100, PrivilegeLevel::S), // 0x1FF
    (0x200, PrivilegeLevel::HS), // 0x2FF
    (0x300, PrivilegeLevel::M), // 0x3FF

    (0x400, PrivilegeLevel::U), // 0x4FF
    (0x500, PrivilegeLevel::S), // 0x57F
    (0x580, PrivilegeLevel::S), // 0x5BF
    (0x5C0, PrivilegeLevel::S), // 0x5FF (Custom)
    (0x600, PrivilegeLevel::HS), // 0x67F
    (0x680, PrivilegeLevel::HS), // 0x6BF
    (0x6C0, PrivilegeLevel::HS), // 0x6FF (Custom)
    (0x700, PrivilegeLevel::M), // 0x77F
    (0x780, PrivilegeLevel::M), // 0x7BF
    (0x7A0, PrivilegeLevel::M), // 0x7FF
    (0x7B0, PrivilegeLevel::M), // 0x7BF (Debug-mode-only)
    (0x7C0, PrivilegeLevel::M), // 0x7FF (Custom)

    (0x800, PrivilegeLevel::U), // 0x8FF (Custom)
    (0x900, PrivilegeLevel::S), // 0x97F
    (0x980, PrivilegeLevel::S), // 0x9BF
    (0x9C0, PrivilegeLevel::S), // 0x9FF (Custom)
    (0xA00, PrivilegeLevel::HS), // 0xA7F
    (0xA80, PrivilegeLevel::HS), // 0xABF
    (0xAC0, PrivilegeLevel::HS), // 0xAFF (Custom)
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
    (0xE00, PrivilegeLevel::HS), // 0xE7F
    (0xE80, PrivilegeLevel::HS), // 0xE8F
    (0xEC0, PrivilegeLevel::HS), // 0xEFF (Custom)
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
            2 => Ok(PrivilegeLevel::HS),
            3 => Ok(PrivilegeLevel::M),
            _ => Err("Invalid privilege level"),
        }
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use crate::riscv::{
        csr_reg::{
            CsrRegfile, Cycle, Fcsr, Mcause, Mcycle, Mepc, Mie, Mip, Mstatus, Mtvec, NamedCsr,
            PrivilegeLevel, Sie, Sip, Sstatus, Vcsr, Vxrm, Vxsat,
        },
        isa_builder::Extension,
    };

    fn new_all_ext_regfile() -> CsrRegfile {
        CsrRegfile::new(HashSet::from([
            Extension::I,
            Extension::M,
            Extension::A,
            Extension::F,
            Extension::D,
            Extension::C,
            Extension::V,
            Extension::Zicsr,
            Extension::Zifencei,
        ]))
    }

    #[test]
    fn test_rw_by_addr() {
        let mut reg = new_all_ext_regfile();
        assert!(reg.write_checked(Mcause::as_addr(), 3, PrivilegeLevel::M));
        assert!(reg.write_checked(Mepc::as_addr(), 0x1234_5678, PrivilegeLevel::M));

        let mcause = reg.read_checked(Mcause::as_addr(), PrivilegeLevel::M).unwrap();
        let mepc = reg.read_checked(Mepc::as_addr(), PrivilegeLevel::M).unwrap();

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
        let mstatus_val = reg.read_checked(Mstatus::as_addr(), PrivilegeLevel::M).unwrap();
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
}
