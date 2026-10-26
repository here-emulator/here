use super::*;

use gdbstub::common::Signal;
use gdbstub::target::TargetError;
use gdbstub::target::ext;
use gdbstub::target::ext::base::single_register_access::SingleRegisterAccess;
use gdbstub::target::ext::base::singlethread::SingleThreadBase;
use gdbstub_arch::riscv::reg::id::RiscvRegId;

use here_board::Board;
use here_isa::riscv::address::Address;
use here_isa::riscv::csr_reg::PrivilegeLevel;
use here_types::FLOAT_REGFILE_CNT;
use here_types::REGFILE_CNT;
use here_types::WordType;

impl<B: Board> target::Target for GdbDebugger<B> {
    type Arch = desc::Riscv64;
    type Error = &'static str;

    #[inline(always)]
    fn base_ops(&mut self) -> ext::base::BaseOps<'_, Self::Arch, Self::Error> {
        ext::base::BaseOps::SingleThread(self)
    }

    #[inline(always)]
    fn support_breakpoints(&mut self) -> Option<ext::breakpoints::BreakpointsOps<'_, Self>> {
        Some(self)
    }
}

impl<B: Board> SingleThreadBase for GdbDebugger<B> {
    fn read_registers(
        &mut self,
        regs: &mut <Self::Arch as gdbstub::arch::Arch>::Registers,
    ) -> target::TargetResult<(), Self> {
        regs.pc = self.dbg.read_pc();
        for i in 1..REGFILE_CNT {
            regs.x[i] = self.dbg.read_reg(i as u8);
        }

        Ok(())
    }

    fn write_registers(
        &mut self,
        regs: &<Self::Arch as gdbstub::arch::Arch>::Registers,
    ) -> target::TargetResult<(), Self> {
        for i in 1..REGFILE_CNT {
            self.dbg.write_reg(i as u8, regs.x[i]);
        }

        Ok(())
    }

    fn read_addrs(
        &mut self,
        start_addr: <Self::Arch as gdbstub::arch::Arch>::Usize,
        data: &mut [u8],
    ) -> target::TargetResult<usize, Self> {
        for (addr, value_ref) in (start_addr..).zip(data.iter_mut()) {
            match self.dbg.read_memory::<u8>(Address::Virt(addr)) {
                Ok(val) => *value_ref = val,
                Err(_) => return Err(().into()),
            }
        }

        Ok(data.len())
    }

    fn write_addrs(
        &mut self,
        start_addr: <Self::Arch as gdbstub::arch::Arch>::Usize,
        data: &[u8],
    ) -> target::TargetResult<(), Self> {
        for (addr, val) in (start_addr..).zip(data.iter().copied()) {
            if let Err(_) = self.dbg.write_memory::<u8>(Address::Virt(addr), val) {
                return Err(().into());
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn support_single_register_access(
        &mut self,
    ) -> Option<target::ext::base::single_register_access::SingleRegisterAccessOps<'_, (), Self>>
    {
        Some(self)
    }

    #[inline(always)]
    fn support_resume(
        &mut self,
    ) -> Option<target::ext::base::singlethread::SingleThreadResumeOps<'_, Self>> {
        Some(self)
    }
}

impl<B: Board> SingleRegisterAccess<()> for GdbDebugger<B> {
    fn read_register(
        &mut self,
        _tid: (),
        reg_id: <Self::Arch as gdbstub::arch::Arch>::RegId,
        buf: &mut [u8],
    ) -> target::TargetResult<usize, Self> {
        match reg_id {
            RiscvRegId::Gpr(i) => {
                if i >= REGFILE_CNT as u8 {
                    return Err(().into());
                }
                let w = self.dbg.read_reg(i);
                buf.copy_from_slice(&w.to_le_bytes());
            }
            RiscvRegId::Fpr(i) => {
                if i >= FLOAT_REGFILE_CNT as u8 {
                    return Err(().into());
                }
                let w = self.dbg.read_float_reg(i).1;
                buf.copy_from_slice(&w.to_le_bytes());
            }
            RiscvRegId::Pc => {
                let w = self.dbg.read_pc();
                buf.copy_from_slice(&w.to_le_bytes());
            }
            RiscvRegId::Csr(i) => {
                let w = self.dbg.read_csr(i as u64);
                if let Some(w) = w {
                    buf.copy_from_slice(&w.to_le_bytes());
                } else {
                    return Err(().into());
                }
            }
            RiscvRegId::Priv => {
                let w = self.dbg.get_current_privilege() as u8;
                buf.copy_from_slice(&w.to_le_bytes());
            }
            _ => {
                return Err(().into());
            }
        }

        Ok(buf.len())
    }

    fn write_register(
        &mut self,
        _tid: (),
        reg_id: <Self::Arch as gdbstub::arch::Arch>::RegId,
        data: &[u8],
    ) -> target::TargetResult<(), Self> {
        let w = WordType::from_le_bytes(
            data.try_into()
                .map_err(|_| TargetError::Fatal("invalid data"))?,
        );

        match reg_id {
            RiscvRegId::Gpr(i) => {
                if i >= REGFILE_CNT as u8 {
                    return Err(().into());
                }
                self.dbg.write_reg(i, w);
            }
            RiscvRegId::Fpr(i) => {
                if i >= FLOAT_REGFILE_CNT as u8 {
                    return Err(().into());
                }
                let arr: [u8; 8] = data
                    .try_into()
                    .map_err(|_| TargetError::Fatal("invalid data"))?;
                self.dbg.write_float_reg(i, f64::from_le_bytes(arr));
            }
            RiscvRegId::Pc => {
                self.dbg.write_pc(w);
            }
            RiscvRegId::Csr(i) => {
                self.dbg
                    .write_csr(i as u64, w)
                    .map_err(|_| TargetError::NonFatal)?;
            }
            RiscvRegId::Priv => {
                let Ok(privilege) = PrivilegeLevel::try_from(w as u8) else {
                    return Err(().into());
                };
                self.dbg.set_current_privilege(privilege);
            }
            _ => {
                return Err(().into());
            }
        }

        Ok(())
    }
}

impl<B: Board> ext::base::singlethread::SingleThreadResume for GdbDebugger<B> {
    fn resume(&mut self, signal: Option<Signal>) -> Result<(), Self::Error> {
        if signal.is_some() {
            return Err("no support for continuing with signal");
        }

        self.exec_mode = ExecMode::Continue;

        Ok(())
    }

    #[inline(always)]
    fn support_single_step(
        &mut self,
    ) -> Option<target::ext::base::singlethread::SingleThreadSingleStepOps<'_, Self>> {
        Some(self)
    }
}

impl<B: Board> target::ext::base::singlethread::SingleThreadSingleStep for GdbDebugger<B> {
    fn step(&mut self, signal: Option<Signal>) -> Result<(), Self::Error> {
        if signal.is_some() {
            return Err("no support for stepping with signal");
        }

        self.exec_mode = ExecMode::Step;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdbstub::target::ext::base::singlethread::SingleThreadBase;
    use here_board::virt::VirtBoard;

    #[test]
    fn read_addrs_fails_when_any_byte_is_unreadable() {
        let board = VirtBoard::from_binary_with(&[], Default::default()).unwrap();
        let mut debugger = GdbDebugger::new(board);
        let mut data = [0xaa; 4];

        assert!(SingleThreadBase::read_addrs(&mut debugger, 0, &mut data).is_err());
    }
}
