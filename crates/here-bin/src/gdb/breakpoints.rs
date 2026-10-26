use crate::isa::riscv::address::Address;

use super::*;

impl<B: Board> ext::breakpoints::Breakpoints for GdbDebugger<B> {
    #[inline(always)]
    fn support_sw_breakpoint(&mut self) -> Option<ext::breakpoints::SwBreakpointOps<'_, Self>> {
        Some(self)
    }
}

impl<B: Board> ext::breakpoints::SwBreakpoint for GdbDebugger<B> {
    fn add_sw_breakpoint(
        &mut self,
        addr: <Self::Arch as gdbstub::arch::Arch>::Usize,
        _kind: <Self::Arch as gdbstub::arch::Arch>::BreakpointKind,
    ) -> gdbstub::target::TargetResult<bool, Self> {
        match self.dbg.set_breakpoint(Address::Virt(addr)) {
            Ok(is_set) => Ok(is_set),
            Err(_) => Ok(false),
        }
    }

    fn remove_sw_breakpoint(
        &mut self,
        addr: <Self::Arch as gdbstub::arch::Arch>::Usize,
        _kind: <Self::Arch as gdbstub::arch::Arch>::BreakpointKind,
    ) -> gdbstub::target::TargetResult<bool, Self> {
        match self.dbg.clear_breakpoint(Address::Virt(addr)) {
            Ok(is_cleared) => Ok(is_cleared),
            Err(_) => Ok(false),
        }
    }
}
