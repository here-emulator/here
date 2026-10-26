use here_isa::riscv::executor::RVCPU;

use super::JitContext;
use super::JitExit;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchContext {}

#[inline]
pub unsafe fn enter_jit_trampoline(
    _entry: *const u8,
    _cpu: &mut RVCPU,
    _context: &mut JitContext,
) -> JitExit {
    panic!()
}

#[inline]
pub unsafe fn abort_jit_unwind(_arch: &ArchContext) -> ! {
    panic!()
}
