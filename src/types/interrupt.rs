use crate::config::arch_config::WordType;

/// Interrupt
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Interrupt {
    UserSoft,
    SupervisorSoft,
    MachineSoft,
    UserTimer,
    SupervisorTimer,
    MachineTimer,
    UserExternal,
    SupervisorExternal,
    MachineExternal,
    Unknown,
}

impl Interrupt {
    pub fn from(nr: usize) -> Self {
        match nr {
            0 => Interrupt::UserSoft,
            1 => Interrupt::SupervisorSoft,
            3 => Interrupt::MachineSoft,
            4 => Interrupt::UserTimer,
            5 => Interrupt::SupervisorTimer,
            7 => Interrupt::MachineTimer,
            8 => Interrupt::UserExternal,
            9 => Interrupt::SupervisorExternal,
            11 => Interrupt::MachineExternal,
            _ => Interrupt::Unknown,
        }
    }
}

impl Into<WordType> for Interrupt {
    fn into(self) -> WordType {
        match self {
            Interrupt::UserSoft => 0,
            Interrupt::SupervisorSoft => 1,
            Interrupt::MachineSoft => 3,
            Interrupt::UserTimer => 4,
            Interrupt::SupervisorTimer => 5,
            Interrupt::MachineTimer => 7,
            Interrupt::UserExternal => 8,
            Interrupt::SupervisorExternal => 9,
            Interrupt::MachineExternal => 11,
            _ => {
                panic!("Unknown Exception.")
            }
        }
    }
}
