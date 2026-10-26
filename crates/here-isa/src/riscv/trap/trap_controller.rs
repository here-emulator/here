use crate::DebugTarget;
use crate::riscv::csr_reg::NamedCsrReg;
use crate::riscv::csr_reg::PrivilegeLevel;
use crate::riscv::csr_reg::csr_index;
use crate::riscv::csr_reg::csr_macro::*;
use crate::riscv::executor::RVCPU;
use crate::riscv::trap::Exception;
use crate::riscv::trap::Trap;
use here_types::Interrupt;
use here_types::WordType;

// TODO: Remove static class `TrapController` with sepreate modules for each privilege level.
pub(in crate::riscv) struct TrapController {}

impl TrapController {
    // pub fn new() -> Self {
    //     Self {}
    // }

    // ======================================
    //                M-Mode
    // ======================================
    fn is_exception_delegated_m_mode(cpu: &mut RVCPU, exception: Exception) -> bool {
        let medeleg_val = cpu.csr.get_by_type_existing::<Medeleg>().data();
        (medeleg_val & (1 << exception as u8)) != 0
    }

    fn is_interrupt_delegated_m_mode(cpu: &mut RVCPU, interrupt: Interrupt) -> bool {
        let mideleg = cpu.csr.get_by_type_existing::<Mideleg>();
        match interrupt {
            Interrupt::MachineExternal | Interrupt::MachineSoft | Interrupt::MachineTimer => false,
            Interrupt::SupervisorExternal | Interrupt::UserExternal => mideleg.get_seip() != 0,
            Interrupt::SupervisorSoft | Interrupt::UserSoft => mideleg.get_ssip() != 0,
            Interrupt::SupervisorTimer | Interrupt::UserTimer => mideleg.get_stip() != 0,
            Interrupt::Unknown => {
                unreachable!()
            }
        }
    }

    fn is_delegated_m_mode(cpu: &mut RVCPU, cause: Trap) -> bool {
        match cause {
            Trap::Interrupt(interrupt) => Self::is_interrupt_delegated_m_mode(cpu, interrupt),
            Trap::Exception(exception) => Self::is_exception_delegated_m_mode(cpu, exception),
        }
    }

    fn send_trap_signal_m_mode(cpu: &mut RVCPU, cause: Trap, trap_value: WordType) {
        cpu.csr
            .get_by_type_existing::<Mstatus>()
            .set_mpp(cpu.csr.privelege_level() as u8 as WordType);
        cpu.csr.set_current_privileged(PrivilegeLevel::M);
        cpu.csr
            .write_uncheck_privilege(Mcause::get_index(), cause.into());
        cpu.csr.write_uncheck_privilege(Mepc::get_index(), cpu.pc);

        cpu.csr
            .write_uncheck_privilege(csr_index::mtval, trap_value);

        let mstatus = cpu.csr.get_by_type_existing::<Mstatus>();
        mstatus.set_mpie(mstatus.get_mie());
        mstatus.set_mie(0);

        let mtvec = cpu.csr.get_by_type_existing::<Mtvec>();
        cpu.write_pc(Self::next_pc_by_tvec(
            cause,
            mtvec.get_mode(),
            mtvec.get_base(),
        ));
    }

    pub fn mret(cpu: &mut RVCPU) {
        let mstatus = cpu.csr.get_by_type_existing::<Mstatus>();
        mstatus.set_mie(mstatus.get_mpie());
        mstatus.set_mpie(1);
        cpu.write_pc(cpu.csr.read_uncheck_privilege(Mepc::get_index()).unwrap());

        let new_priv: u8 = mstatus.get_mpp() as u8;

        if new_priv != PrivilegeLevel::M as u8 {
            mstatus.set_mprv(0);
        }

        cpu.csr.set_current_privileged(
            PrivilegeLevel::try_from(new_priv)
                .expect("mstatus.mpp must contain a valid privilege level"),
        );
        mstatus.set_mpp(0);
    }

    // ======================================
    //                S-Mode
    // ======================================
    fn send_trap_signal_s_mode(cpu: &mut RVCPU, cause: Trap, trap_value: WordType) {
        cpu.csr
            .get_by_type_existing::<Sstatus>()
            .set_spp(cpu.csr.privelege_level() as u8 as WordType);
        cpu.csr.set_current_privileged(PrivilegeLevel::S);

        assert!(cpu.csr.write_directly(Scause::get_index(), cause.into()));
        assert!(cpu.csr.write_directly(Sepc::get_index(), cpu.pc));

        assert!(cpu.csr.write_directly(Stval::get_index(), trap_value));

        let sstatus = cpu.csr.get_by_type_existing::<Sstatus>();
        sstatus.set_spie(sstatus.get_sie());
        sstatus.set_sie(0);

        let stvec = cpu.csr.get_by_type_existing::<Stvec>();
        cpu.write_pc(Self::next_pc_by_tvec(
            cause,
            stvec.get_mode(),
            stvec.get_base(),
        ));
    }

    pub fn sret(cpu: &mut RVCPU) {
        let sstatus = cpu.csr.get_by_type_existing::<Sstatus>();
        sstatus.set_sie(sstatus.get_spie());
        sstatus.set_spie(1);
        cpu.write_pc(cpu.csr.read_uncheck_privilege(Sepc::get_index()).unwrap());

        let new_priv: u8 = sstatus.get_spp() as u8;

        // If new_priv != M, then `MPRV` is set to 0 on execution of xRET.
        // `SPP` cannot be M, so no need to check.
        cpu.csr.get_by_type_existing::<Mstatus>().set_mprv(0);

        cpu.csr.set_current_privileged(
            PrivilegeLevel::try_from(new_priv)
                .expect("sstatus.spp must contain a valid privilege level"),
        );
        sstatus.set_spp(PrivilegeLevel::U as u8 as WordType);
    }

    // ======================================
    //                 Common
    // ======================================
    fn try_send_trap_signal(cpu: &mut RVCPU, cause: Trap, trap_value: WordType) -> bool {
        let level = cpu.csr.privelege_level();

        if level == PrivilegeLevel::M || Self::is_delegated_m_mode(cpu, cause) == false {
            match cause {
                Trap::Exception(_) => {
                    Self::send_trap_signal_m_mode(cpu, cause, trap_value);
                    true
                }
                Trap::Interrupt(_) => {
                    // "Interrupts for higher-privilege modes, y>x, are always globally enabled
                    // regardless of the setting of the global yIE bit for the higher-privilege mode."
                    if level < PrivilegeLevel::M
                        || cpu.csr.get_by_type_existing::<Mstatus>().get_mie() == 1
                    {
                        Self::send_trap_signal_m_mode(cpu, cause, trap_value);
                        true
                    } else {
                        false
                    }
                }
            }
        } else {
            match cause {
                Trap::Exception(_) => {
                    Self::send_trap_signal_s_mode(cpu, cause, trap_value);
                    true
                }
                Trap::Interrupt(_) => {
                    // The same as M-Mode interrupt handling.
                    if level < PrivilegeLevel::S
                        || cpu.csr.get_by_type_existing::<Mstatus>().get_sie() == 1
                    {
                        Self::send_trap_signal_s_mode(cpu, cause, trap_value);
                        true
                    } else {
                        false
                    }
                }
            }
        }
    }

    pub fn take_exception(cpu: &mut RVCPU, exception: Exception, trap_value: WordType) {
        let taken = Self::try_send_trap_signal(cpu, Trap::Exception(exception), trap_value);
        debug_assert!(taken, "synchronous exceptions must always be taken");
    }

    pub fn try_take_interrupt(cpu: &mut RVCPU) -> Option<Interrupt> {
        let interrupt = Self::has_interrupt(cpu)?;
        Self::try_send_trap_signal(cpu, Trap::Interrupt(interrupt), 0).then_some(interrupt)
    }

    /// Check if there is any pending interrupt that can be handled now, considering `mip` and `mie`.
    fn has_interrupt(cpu: &mut RVCPU) -> Option<Interrupt> {
        let mip = cpu.csr.get_by_type_existing::<Mip>();
        let mie = cpu.csr.get_by_type_existing::<Mie>();

        let pending_interrupts = mip.data() & mie.data();

        if pending_interrupts == 0 {
            return None;
        }

        for i in 0..=15 {
            if (pending_interrupts & (1 << i)) != 0 {
                let interrupt = Interrupt::from(i as usize);
                return Some(interrupt);
            }
        }

        unreachable!()
    }

    /// Get the next pc value according to the trap vector (like `mtvec` or `stvec`).
    #[must_use]
    fn next_pc_by_tvec(cause: Trap, mode: WordType, base: WordType) -> WordType {
        match (mode, cause) {
            (0, _) | (1, Trap::Exception(_)) => {
                // Direct Mode
                base << 2
            }

            (1, Trap::Interrupt(ir)) => {
                // Vector Mode
                let offset: WordType = ir.into();
                (base << 2) + offset * 4
            }

            _ => {
                unreachable!()
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::riscv::cpu_tester::run_test_cpu_step;
    use crate::riscv::trap::Exception;
    use here_types::ram_config;
    use here_types::ram_config::BASE_ADDR;

    const IRQ_HANDLER_ADDR: WordType = 0x80002000;

    #[test]
    fn test_load_fault() {
        run_test_cpu_step(
            &[0x00003503], // ld a0, 0(zero)
            |builder| builder.csr(Mtvec::get_index(), IRQ_HANDLER_ADDR),
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::get_index(), ram_config::BASE_ADDR)
                    .csr(Mtval::get_index(), 0)
                    .customized(|checker| {
                        let mcause = checker.cpu.csr.get_by_type_existing::<Mcause>();
                        assert_eq!(mcause.get_interrupt_flag(), 0);
                        assert_eq!(mcause.get_cause(), Exception::LoadFault.into());
                        checker
                    })
            },
        );
    }

    #[test]
    fn test_load_misaligned() {
        const BASE_LOAD_MEM: WordType = 0x80001000;
        run_test_cpu_step(
            &[0x0017B503], // ld a0, 1(a5)
            |builder| {
                builder
                    .csr(Mtvec::get_index(), IRQ_HANDLER_ADDR)
                    .reg(15, BASE_LOAD_MEM)
            },
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mcause::get_index(), Exception::LoadMisaligned.into())
                    .csr(Mepc::get_index(), ram_config::BASE_ADDR)
                    .csr(Mtval::get_index(), BASE_LOAD_MEM + 1)
                    .customized(|checker| {
                        let mcause = checker.cpu.csr.get_by_type_existing::<Mcause>();
                        assert_eq!(mcause.get_interrupt_flag(), 0);
                        assert_eq!(mcause.get_cause(), Exception::LoadMisaligned.into());
                        checker
                    })
            },
        );
    }

    #[test]
    fn test_store_fault() {
        run_test_cpu_step(
            &[0x00a7b023], // sd a0, 0(a5)
            |builder| builder.csr(Mtvec::get_index(), IRQ_HANDLER_ADDR | 0b00),
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::get_index(), ram_config::BASE_ADDR)
                    .customized(|checker| {
                        let mcause = checker.cpu.csr.get_by_type_existing::<Mcause>();
                        assert_eq!(mcause.get_interrupt_flag(), 0);
                        assert_eq!(mcause.get_cause(), Exception::StoreFault.into());
                        checker
                    })
            },
        );
    }

    #[test]
    fn test_store_misaligned() {
        const BASE_STORE_MEM: WordType = 0x80001000;
        run_test_cpu_step(
            &[0x00a7b0a3], // sd a0, 1(a5)
            |builder| {
                builder
                    .csr(csr_index::mtvec, IRQ_HANDLER_ADDR)
                    .reg(15, BASE_STORE_MEM)
            },
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::get_index(), ram_config::BASE_ADDR)
                    .csr(Mtval::get_index(), BASE_STORE_MEM + 1)
                    .customized(|checker| {
                        let mcause = checker.cpu.csr.get_by_type_existing::<Mcause>();
                        assert_eq!(mcause.get_interrupt_flag(), 0);
                        assert_eq!(mcause.get_cause(), Exception::StoreMisaligned.into());
                        checker
                    })
            },
        );
    }

    #[test]
    fn test_illegal_instr() {
        const PC_START: WordType = 0x80001000;
        run_test_cpu_step(
            &[0x00a7b023], // Any Instr. Because `PC` do not start as 0x80000000.
            |builder| {
                builder
                    .csr(Mtvec::get_index(), IRQ_HANDLER_ADDR | 0b00)
                    .pc(PC_START)
            },
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::get_index(), PC_START)
                    .customized(|checker| {
                        let mcause = checker.cpu.csr.get_by_type_existing::<Mcause>();
                        assert_eq!(mcause.get_interrupt_flag(), 0);
                        assert_eq!(mcause.get_cause(), Exception::IllegalInstruction.into());
                        checker
                    })
            },
        );
    }

    #[test]
    fn test_instr_fault() {
        const PC_START: WordType = 0x70000000;
        run_test_cpu_step(
            &[0x00a7b023], // Any Instr. Because `PC` do not start as 0x80000000.
            |builder| {
                builder
                    .csr(csr_index::mtvec, IRQ_HANDLER_ADDR | 0b00)
                    .pc(PC_START)
            },
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::get_index(), PC_START)
                    .customized(|checker| {
                        let mcause = checker.cpu.csr.get_by_type_existing::<Mcause>();
                        assert_eq!(mcause.get_interrupt_flag(), 0);
                        assert_eq!(mcause.get_cause(), Exception::InstructionFault.into());
                        checker
                    })
            },
        );
    }

    #[test]
    fn test_instr_misaligned() {
        const PC_START: WordType = 0x80000001;
        run_test_cpu_step(
            &[0x00a7b023], // Any Instr. Because `PC` do not start as 0x80000000.
            |builder| {
                builder
                    .csr(csr_index::mtvec, IRQ_HANDLER_ADDR | 0b00)
                    .pc(PC_START)
            },
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::get_index(), PC_START)
                    .customized(|checker| {
                        let mcause = checker.cpu.csr.get_by_type_existing::<Mcause>();
                        assert_eq!(mcause.get_interrupt_flag(), 0);
                        assert_eq!(mcause.get_cause(), Exception::InstructionMisaligned.into());
                        checker
                    })
            },
        );
    }

    #[test]
    fn test_swap() {
        run_test_cpu_step(
            &[0x34011173], // csrrw sp, mscratch, sp
            |builder| builder.csr(Mscratch::get_index(), 0x114514).reg(2, 0x0721),
            |checker| checker.csr(Mscratch::get_index(), 0x0721).reg(2, 0x114514),
        );
    }

    #[test]
    fn test_next_pc_by_tvec() {
        let base = 0x12345 as WordType;
        assert_eq!(
            TrapController::next_pc_by_tvec(Trap::Exception(Exception::InstructionFault), 1, base),
            base << 2
        );

        assert_eq!(
            TrapController::next_pc_by_tvec(Trap::Interrupt(Interrupt::MachineTimer), 1, base),
            (base << 2) + 0x1c
        );
    }

    #[test]
    fn test_exception() {
        // UNIMP: csrrw x0, cycle, x0 -> illegal instruction,
        // because cycle is readonly CSR.
        run_test_cpu_step(
            &[0xc00c1073],
            |builder| builder.csr(Mtvec::get_index(), IRQ_HANDLER_ADDR),
            |checker| {
                checker
                    .csr(Mcause::get_index(), 2) // illegal instruction
                    .csr(Mtval::get_index(), 0xc00c1073) // instruction itself
                    .csr(Mepc::get_index(), BASE_ADDR) // address of the instruction
                    .pc(IRQ_HANDLER_ADDR)
            },
        );
    }
}
