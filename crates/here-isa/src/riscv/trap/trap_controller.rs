use crate::DebugTarget;
use crate::riscv::cpu::RVCPU;
use crate::riscv::csr_reg::{
    CsrDataType, Mcause, Medeleg, Mepc, Mideleg, Mie, Mip, Mstatus, Mtval, Mtvec, NamedCsr,
    PrivilegeLevel, Scause, Sepc, Sstatus, Stval, Stvec,
};
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
        let medeleg_val = Medeleg::read(&cpu.csr);
        (medeleg_val & (1 << exception as u8)) != 0
    }

    fn is_interrupt_delegated_m_mode(cpu: &mut RVCPU, interrupt: Interrupt) -> bool {
        match interrupt {
            Interrupt::MachineExternal | Interrupt::MachineSoft | Interrupt::MachineTimer => false,
            Interrupt::SupervisorExternal | Interrupt::UserExternal => {
                Mideleg::get_seip(&cpu.csr) != 0
            }
            Interrupt::SupervisorSoft | Interrupt::UserSoft => Mideleg::get_ssip(&cpu.csr) != 0,
            Interrupt::SupervisorTimer | Interrupt::UserTimer => Mideleg::get_stip(&cpu.csr) != 0,
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
        let prev_priv = cpu.privilege_level() as u8 as CsrDataType;
        Mstatus::set_mpp(&mut cpu.csr, prev_priv);
        cpu.set_current_privilege(PrivilegeLevel::M);

        let cause_val: WordType = cause.into();
        Mcause::write(&mut cpu.csr, cause_val as CsrDataType);
        Mepc::write(&mut cpu.csr, cpu.pc as CsrDataType);
        Mtval::write(&mut cpu.csr, trap_value as CsrDataType);

        let mie = Mstatus::get_mie(&cpu.csr);
        Mstatus::set_mpie(&mut cpu.csr, mie);
        Mstatus::set_mie(&mut cpu.csr, 0);

        let mode = Mtvec::get_mode(&cpu.csr) as WordType;
        let base = Mtvec::get_base(&cpu.csr) as WordType;
        cpu.write_pc(Self::next_pc_by_tvec(cause, mode, base));
    }

    pub fn mret(cpu: &mut RVCPU) {
        let mpie = Mstatus::get_mpie(&cpu.csr);
        Mstatus::set_mie(&mut cpu.csr, mpie);
        Mstatus::set_mpie(&mut cpu.csr, 1);
        cpu.write_pc(Mepc::read(&cpu.csr) as WordType);

        let new_priv: u8 = Mstatus::get_mpp(&cpu.csr) as u8;

        if new_priv != PrivilegeLevel::M as u8 {
            Mstatus::set_mprv(&mut cpu.csr, 0);
        }

        cpu.set_current_privilege(
            PrivilegeLevel::try_from(new_priv)
                .expect("mstatus.mpp must contain a valid privilege level"),
        );
        Mstatus::set_mpp(&mut cpu.csr, 0);
    }

    // ======================================
    //                S-Mode
    // ======================================
    fn send_trap_signal_s_mode(cpu: &mut RVCPU, cause: Trap, trap_value: WordType) {
        let prev_priv = cpu.privilege_level() as u8 as CsrDataType;
        Sstatus::set_spp(&mut cpu.csr, prev_priv);
        cpu.set_current_privilege(PrivilegeLevel::S);

        let cause_val: WordType = cause.into();
        Scause::write_directly(&mut cpu.csr, cause_val as CsrDataType);
        Sepc::write_directly(&mut cpu.csr, cpu.pc as CsrDataType);
        Stval::write_directly(&mut cpu.csr, trap_value as CsrDataType);

        let sie = Sstatus::get_sie(&cpu.csr);
        Sstatus::set_spie(&mut cpu.csr, sie);
        Sstatus::set_sie(&mut cpu.csr, 0);

        let mode = Stvec::get_mode(&cpu.csr) as WordType;
        let base = Stvec::get_base(&cpu.csr) as WordType;
        cpu.write_pc(Self::next_pc_by_tvec(cause, mode, base));
    }

    pub fn sret(cpu: &mut RVCPU) {
        let spie = Sstatus::get_spie(&cpu.csr);
        Sstatus::set_sie(&mut cpu.csr, spie);
        Sstatus::set_spie(&mut cpu.csr, 1);
        cpu.write_pc(Sepc::read(&cpu.csr) as WordType);

        let new_priv: u8 = Sstatus::get_spp(&cpu.csr) as u8;

        // If new_priv != M, then `MPRV` is set to 0 on execution of xRET.
        // `SPP` cannot be M, so no need to check.
        Mstatus::set_mprv(&mut cpu.csr, 0);

        cpu.set_current_privilege(
            PrivilegeLevel::try_from(new_priv)
                .expect("sstatus.spp must contain a valid privilege level"),
        );
        Sstatus::set_spp(&mut cpu.csr, PrivilegeLevel::U as u8 as CsrDataType);
    }

    // ======================================
    //                 Common
    // ======================================
    fn try_send_trap_signal(cpu: &mut RVCPU, cause: Trap, trap_value: WordType) -> bool {
        let level = cpu.privilege_level();

        if level == PrivilegeLevel::M || Self::is_delegated_m_mode(cpu, cause) == false {
            match cause {
                Trap::Exception(_) => {
                    Self::send_trap_signal_m_mode(cpu, cause, trap_value);
                    true
                }
                Trap::Interrupt(_) => {
                    // "Interrupts for higher-privilege modes, y>x, are always globally enabled
                    // regardless of the setting of the global yIE bit for the higher-privilege mode."
                    if level < PrivilegeLevel::M || Mstatus::get_mie(&cpu.csr) == 1 {
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
                    if level < PrivilegeLevel::S || Mstatus::get_sie(&cpu.csr) == 1 {
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
        let pending_interrupts = Mip::read(&cpu.csr) & Mie::read(&cpu.csr);

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
    use crate::riscv::csr_reg::Mscratch;
    use crate::riscv::trap::Exception;
    use here_types::ram_config;
    use here_types::ram_config::BASE_ADDR;

    const IRQ_HANDLER_ADDR: WordType = 0x80002000;

    #[test]
    fn test_load_fault() {
        run_test_cpu_step(
            &[0x00003503], // ld a0, 0(zero)
            |builder| builder.csr(Mtvec::as_addr(), IRQ_HANDLER_ADDR),
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::as_addr(), ram_config::BASE_ADDR)
                    .csr(Mtval::as_addr(), 0)
                    .customized(|checker| {
                        assert_eq!(Mcause::get_interrupt_flag(&checker.cpu.csr), 0);
                        assert_eq!(
                            Mcause::get_cause(&checker.cpu.csr) as WordType,
                            Exception::LoadFault.into()
                        );
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
                    .csr(Mtvec::as_addr(), IRQ_HANDLER_ADDR)
                    .reg(15, BASE_LOAD_MEM)
            },
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mcause::as_addr(), Exception::LoadMisaligned.into())
                    .csr(Mepc::as_addr(), ram_config::BASE_ADDR)
                    .csr(Mtval::as_addr(), BASE_LOAD_MEM + 1)
                    .customized(|checker| {
                        assert_eq!(Mcause::get_interrupt_flag(&checker.cpu.csr), 0);
                        assert_eq!(
                            Mcause::get_cause(&checker.cpu.csr) as WordType,
                            Exception::LoadMisaligned.into()
                        );
                        checker
                    })
            },
        );
    }

    #[test]
    fn test_store_fault() {
        run_test_cpu_step(
            &[0x00a7b023], // sd a0, 0(a5)
            |builder| builder.csr(Mtvec::as_addr(), IRQ_HANDLER_ADDR | 0b00),
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::as_addr(), ram_config::BASE_ADDR)
                    .customized(|checker| {
                        assert_eq!(Mcause::get_interrupt_flag(&checker.cpu.csr), 0);
                        assert_eq!(
                            Mcause::get_cause(&checker.cpu.csr) as WordType,
                            Exception::StoreFault.into()
                        );
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
                    .csr(Mtvec::as_addr(), IRQ_HANDLER_ADDR)
                    .reg(15, BASE_STORE_MEM)
            },
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::as_addr(), ram_config::BASE_ADDR)
                    .csr(Mtval::as_addr(), BASE_STORE_MEM + 1)
                    .customized(|checker| {
                        assert_eq!(Mcause::get_interrupt_flag(&checker.cpu.csr), 0);
                        assert_eq!(
                            Mcause::get_cause(&checker.cpu.csr) as WordType,
                            Exception::StoreMisaligned.into()
                        );
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
                    .csr(Mtvec::as_addr(), IRQ_HANDLER_ADDR | 0b00)
                    .pc(PC_START)
            },
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::as_addr(), PC_START)
                    .customized(|checker| {
                        assert_eq!(Mcause::get_interrupt_flag(&checker.cpu.csr), 0);
                        assert_eq!(
                            Mcause::get_cause(&checker.cpu.csr) as WordType,
                            Exception::IllegalInstruction.into()
                        );
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
                    .csr(Mtvec::as_addr(), IRQ_HANDLER_ADDR | 0b00)
                    .pc(PC_START)
            },
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::as_addr(), PC_START)
                    .customized(|checker| {
                        assert_eq!(Mcause::get_interrupt_flag(&checker.cpu.csr), 0);
                        assert_eq!(
                            Mcause::get_cause(&checker.cpu.csr) as WordType,
                            Exception::InstructionFault.into()
                        );
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
                    .csr(Mtvec::as_addr(), IRQ_HANDLER_ADDR | 0b00)
                    .pc(PC_START)
            },
            |checker| {
                checker
                    .pc(IRQ_HANDLER_ADDR)
                    .csr(Mepc::as_addr(), PC_START)
                    .customized(|checker| {
                        assert_eq!(Mcause::get_interrupt_flag(&checker.cpu.csr), 0);
                        assert_eq!(
                            Mcause::get_cause(&checker.cpu.csr) as WordType,
                            Exception::InstructionMisaligned.into()
                        );
                        checker
                    })
            },
        );
    }

    #[test]
    fn test_swap() {
        run_test_cpu_step(
            &[0x34011173], // csrrw sp, mscratch, sp
            |builder| builder.csr(Mscratch::as_addr(), 0x114514).reg(2, 0x0721),
            |checker| checker.csr(Mscratch::as_addr(), 0x0721).reg(2, 0x114514),
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
            |builder| builder.csr(Mtvec::as_addr(), IRQ_HANDLER_ADDR),
            |checker| {
                checker
                    .csr(Mcause::as_addr(), 2) // illegal instruction
                    .csr(Mtval::as_addr(), 0xc00c1073) // instruction itself
                    .csr(Mepc::as_addr(), BASE_ADDR) // address of the instruction
                    .pc(IRQ_HANDLER_ADDR)
            },
        );
    }
}
