mod cpu;
pub mod regfile;
pub mod extension;

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



// #[cfg(test)]
// mod test {
//     use crate::riscv::csr_reg::CsrRegFile;
//     use crate::riscv::csr_reg::NamedCsrReg;
//     use crate::riscv::csr_reg::PrivilegeLevel;
//     use crate::riscv::csr_reg::csr_index;
//     use crate::riscv::csr_reg::csr_macro::*;

//     #[test]
//     fn test_rw_by_addr() {
//         let mut reg = CsrRegFile::new();
//         assert!(reg.write_checked(csr_index::mcause, 3));
//         assert!(reg.write_checked(csr_index::mepc, 0x1234_5678));

//         let mcause = reg.read_checked(csr_index::mcause).unwrap();
//         let mepc = reg.read_checked(csr_index::mepc).unwrap();

//         assert_eq!(mcause, 3);
//         assert_eq!(mepc, 0x1234_5678);
//     }

//     #[test]
//     fn test_rw_by_type() {
//         let mut reg = CsrRegFile::new();
//         let mstatus = reg.get_by_type::<Mstatus>().unwrap();

//         mstatus.set_mpp(3);
//         mstatus.set_sie(1);

//         assert_eq!(mstatus.get_mpp(), 3);
//         assert_eq!(mstatus.get_sie(), 1);
//         let mstatus_val = reg.read_checked(csr_index::mstatus).unwrap();
//         assert_eq!(mstatus_val, (1 << 1 | 3 << 11));

//         mstatus.set_mpp(0b10);
//         assert_eq!(mstatus.get_mpp(), 0b10);

//         let mtvec = reg.get_by_type::<Mtvec>().unwrap();
//         assert!(reg.write_checked(csr_index::mtvec, 0x114514));
//         assert_eq!(mtvec.get_base(), 0x114514 >> 2);
//     }

//     #[test]
//     fn test_read_privilege() {
//         let mut reg = CsrRegFile::new();
//         reg.set_current_privileged(PrivilegeLevel::M);
//         assert!(reg.write_checked(csr_index::mcause, 0xFEFE));
//         assert_eq!(
//             reg.read_uncheck_privilege(csr_index::mcause).unwrap(),
//             0xFEFE
//         );

//         reg.set_current_privileged(PrivilegeLevel::S);
//         assert!(!reg.write_checked(csr_index::mcause, 0xFEFE));
//         assert_eq!(
//             reg.read_uncheck_privilege(csr_index::mcause).unwrap(),
//             0xFEFE
//         );
//     }

//     #[test]
//     fn test_mcycle() {
//         let mut reg = CsrRegFile::new();

//         // `mcycle` is a normal writable CSR.

//         // Test NamedCsr API.
//         let mcycle = reg.get_by_type_existing::<Mcycle>();
//         mcycle.set_mcycle(1);
//         assert_eq!(mcycle.get_mcycle(), 1);

//         // Test CsrRegfile API.
//         assert_eq!(reg.read_uncheck_privilege(Mcycle::get_index()).unwrap(), 1);
//         reg.write_uncheck_privilege(Mcycle::get_index(), 2);
//         assert_eq!(reg.read_uncheck_privilege(Mcycle::get_index()).unwrap(), 2);

//         // `cycle` is a shadow CSR of `mcycle`.

//         // It's known that with NamedCsr API, shadow CSR's validator cannot work properly,
//         // so only test CsrRegfile API.
//         reg.write_uncheck_privilege(Cycle::get_index(), 3);
//         assert_eq!(reg.read_uncheck_privilege(Cycle::get_index()).unwrap(), 2);
//     }

//     #[test]
//     fn test_vector_fixed_point_shadow_csr() {
//         let mut reg = CsrRegFile::new();

//         reg.write_uncheck_privilege(csr_index::vxrm, 0b10);
//         reg.write_uncheck_privilege(csr_index::vxsat, 1);

//         assert_eq!(reg.read_uncheck_privilege(csr_index::vxrm).unwrap(), 0b10);
//         assert_eq!(reg.read_uncheck_privilege(csr_index::vxsat).unwrap(), 1);
//         assert_eq!(reg.get_by_type_existing::<Vcsr>().get_vxrm(), 0b10);
//         assert_eq!(reg.get_by_type_existing::<Vcsr>().get_vxsat(), 1);
//         assert_eq!(reg.read_uncheck_privilege(Fcsr::get_index()).unwrap(), 0);
//     }

//     #[test]
//     fn test_xstatus() {
//         let mut reg = CsrRegFile::new();

//         let mstatus = reg.get_by_type_existing::<Mstatus>();
//         let sstatus = reg.get_by_type_existing::<Sstatus>();

//         mstatus.set_spp(1);
//         assert_eq!(mstatus.get_spp(), 1);
//         assert_eq!(sstatus.get_spp(), 1);

//         reg.write_uncheck_privilege(Mstatus::get_index(), 0x1002);
//         assert_eq!(
//             reg.read_uncheck_privilege(Sstatus::get_index()).unwrap(),
//             0x02
//         );
//         assert_eq!(
//             reg.read_uncheck_privilege(Mstatus::get_index()).unwrap(),
//             0x1002
//         );
//     }

//     #[test]
//     fn test_s_mode_shadow_csr() {
//         let mut csr = CsrRegFile::new();

//         let mip = csr.get_by_type_existing::<Mip>();
//         let sip = csr.get_by_type_existing::<Sip>();
//         mip.set_seip(1);
//         assert_eq!(sip.get_seip(), 1);
//         mip.set_meip(1);
//         assert_eq!(csr.read_uncheck_privilege(Sip::get_index()).unwrap(), 0x200);
//         assert_eq!(csr.read_uncheck_privilege(Mip::get_index()).unwrap(), 0xa00);

//         let mie = csr.get_by_type_existing::<Mie>();
//         let sie = csr.get_by_type_existing::<Sie>();
//         mie.set_seie(1);
//         assert_eq!(sie.get_seie(), 1);
//     }
// }
