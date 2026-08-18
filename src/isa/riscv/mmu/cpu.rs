use crate::{
    config::arch_config::WordType,
    device::MemError,
    isa::riscv::{
        csr_reg::{
            CsrRegFile, PrivilegeLevel,
            csr_macro::{Mstatus, Sstatus},
        },
        executor::RVCPU,
        mmu::{AccessEffect, AccessPolicy, AccessType, PTEFlags, PageTableError, PermissionCheck},
        trap::Exception,
    },
    utils::UnsignedInteger,
};

enum AccessPrivilege {
    UserOnly,
    SupervisorOnly,
    SupervisorAndUser,
    MachineOnly,
}

fn determine_data_access_privilege(csr: &mut CsrRegFile) -> AccessPrivilege {
    match csr.privelege_level() {
        PrivilegeLevel::M => {
            let mstatus = csr.get_by_type_existing::<Mstatus>();

            if mstatus.get_mprv() == 0 {
                AccessPrivilege::MachineOnly
            } else {
                match PrivilegeLevel::try_from(mstatus.get_mpp() as u8)
                    .expect("mstatus.mpp must contain a valid privilege level")
                {
                    PrivilegeLevel::M => AccessPrivilege::MachineOnly,
                    PrivilegeLevel::S => {
                        if mstatus.get_sum() == 0 {
                            AccessPrivilege::SupervisorOnly
                        } else {
                            AccessPrivilege::SupervisorAndUser
                        }
                    }
                    PrivilegeLevel::U => AccessPrivilege::UserOnly,
                    PrivilegeLevel::V => unreachable!(),
                }
            }
        }
        PrivilegeLevel::S => {
            if csr.get_by_type_existing::<Sstatus>().get_sum() == 0 {
                AccessPrivilege::SupervisorOnly
            } else {
                AccessPrivilege::SupervisorAndUser
            }
        }
        PrivilegeLevel::U => AccessPrivilege::UserOnly,
        PrivilegeLevel::V => unreachable!(),
    }
}

fn resolve_data_policy(
    csr: &mut CsrRegFile,
    access: AccessType,
    side_effect: bool,
) -> AccessPolicy {
    let fault = match access {
        AccessType::Read => MemError::LoadPageFault,
        AccessType::Write | AccessType::ReadWrite => MemError::StorePageFault,
    };

    let effect = match (side_effect, access) {
        (false, _) => AccessEffect::None,
        (true, AccessType::Read) => AccessEffect::Accessed,
        (true, AccessType::Write | AccessType::ReadWrite) => AccessEffect::AccessedDirty,
    };

    let (masks, flags) = match determine_data_access_privilege(csr) {
        AccessPrivilege::MachineOnly => return AccessPolicy::Direct,
        AccessPrivilege::SupervisorAndUser => (PTEFlags::empty(), PTEFlags::empty()),
        AccessPrivilege::SupervisorOnly => (PTEFlags::U, PTEFlags::empty()),
        AccessPrivilege::UserOnly => (PTEFlags::U, PTEFlags::U),
    };

    if csr.get_by_type_existing::<Mstatus>().get_mxr() == 1 && access == AccessType::Read {
        return AccessPolicy::Translated {
            check: PermissionCheck {
                any_of: PTEFlags::R | PTEFlags::X,
                exact_mask: masks,
                exact_flags: flags,
            },
            effect,
            fault,
        };
    }

    let rwx_base = match access {
        AccessType::Read => PTEFlags::R,
        AccessType::Write => PTEFlags::W,
        AccessType::ReadWrite => PTEFlags::R | PTEFlags::W,
    };

    AccessPolicy::Translated {
        check: PermissionCheck {
            any_of: PTEFlags::empty(),
            exact_mask: masks | rwx_base,
            exact_flags: flags | rwx_base,
        },
        effect,
        fault,
    }
}

fn resolve_ifetch_policy(csr: &CsrRegFile, side_effect: bool) -> AccessPolicy {
    let effect = if side_effect {
        AccessEffect::Accessed
    } else {
        AccessEffect::None
    };

    match csr.privelege_level() {
        PrivilegeLevel::M => AccessPolicy::Direct,
        PrivilegeLevel::S => AccessPolicy::Translated {
            check: PermissionCheck {
                any_of: PTEFlags::empty(),
                exact_mask: PTEFlags::X | PTEFlags::U,
                exact_flags: PTEFlags::X,
            },
            effect,
            fault: MemError::LoadPageFault,
        },
        PrivilegeLevel::U => AccessPolicy::Translated {
            check: PermissionCheck {
                any_of: PTEFlags::empty(),
                exact_mask: PTEFlags::X | PTEFlags::U,
                exact_flags: PTEFlags::X | PTEFlags::U,
            },
            effect,
            fault: MemError::LoadPageFault,
        },
        PrivilegeLevel::V => unreachable!(),
    }
}

impl RVCPU {
    fn finish_memory_access<T>(
        &mut self,
        addr: WordType,
        result: Result<T, MemError>,
    ) -> Result<T, Exception> {
        result.map_err(|err| {
            self.pending_tval = Some(addr);
            Exception::from_memory_err(err)
        })
    }

    pub(crate) fn read<T>(&mut self, addr: WordType) -> Result<T, Exception>
    where
        T: UnsignedInteger,
    {
        let policy = resolve_data_policy(&mut self.csr, AccessType::Read, true);
        let result = self.memory.read_with_policy(addr, policy);
        self.finish_memory_access(addr, result)
    }

    pub(crate) fn write<T>(&mut self, addr: WordType, data: T) -> Result<(), Exception>
    where
        T: UnsignedInteger,
    {
        let policy = resolve_data_policy(&mut self.csr, AccessType::Write, true);
        let result = self.memory.write_with_policy(addr, data, policy);
        self.finish_memory_access(addr, result)
    }

    pub(crate) fn load_reserved<T>(&mut self, addr: WordType) -> Result<T, Exception>
    where
        T: UnsignedInteger,
    {
        let policy = resolve_data_policy(&mut self.csr, AccessType::Read, true);
        let result = self.memory.load_reserved_with_policy(addr, policy);
        self.finish_memory_access(addr, result)
    }

    pub(crate) fn store_conditional<T>(
        &mut self,
        addr: WordType,
        data: T,
    ) -> Result<bool, Exception>
    where
        T: UnsignedInteger,
    {
        let policy = resolve_data_policy(&mut self.csr, AccessType::Write, true);
        let result = self
            .memory
            .store_conditional_with_policy(addr, data, policy);
        self.finish_memory_access(addr, result)
    }

    pub(crate) fn fetch_and_op_amo<T, F>(
        &mut self,
        addr: WordType,
        rhs_val: T,
        f: F,
    ) -> Result<T, Exception>
    where
        T: UnsignedInteger,
        F: Fn(&T::AtomicType, T) -> Result<T, Exception>,
    {
        let policy = resolve_data_policy(&mut self.csr, AccessType::ReadWrite, true);
        let result = self
            .memory
            .fetch_and_op_amo_with_policy(addr, rhs_val, policy, f);
        result.map_err(|err| {
            self.pending_tval = Some(addr);
            err
        })
    }

    pub(in crate::isa::riscv) fn read_for_ifetch<T>(
        &mut self,
        addr: WordType,
    ) -> Result<T, MemError>
    where
        T: UnsignedInteger,
    {
        let policy = resolve_ifetch_policy(&self.csr, true);
        self.memory.read_with_policy(addr, policy)
    }

    pub(in crate::isa::riscv) fn read_for_debug_ifetch<T>(
        &mut self,
        addr: WordType,
    ) -> Result<T, MemError>
    where
        T: UnsignedInteger,
    {
        let policy = resolve_ifetch_policy(&self.csr, false);
        self.memory.read_with_policy(addr, policy)
    }

    pub(in crate::isa::riscv) fn translate_for_debug(
        &mut self,
        addr: u64,
        access: AccessType,
    ) -> Result<u64, PageTableError> {
        let policy = resolve_data_policy(&mut self.csr, access, false);
        self.memory.translate_for_debug_with_policy(addr, policy)
    }
}
