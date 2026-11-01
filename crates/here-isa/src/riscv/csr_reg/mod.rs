mod cpu;
pub mod extension;
pub mod regfile;

pub use extension::*;
pub use regfile::{
    CsrContext, CsrDataType, CsrIndex, CsrItem, CsrMaskType, CsrRegfile, CsrValidator, NamedCsr,
};

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
