use crate::{
    config::arch_config::WordType,
    isa::riscv::{executor::RVCPU, trap::Exception},
};

cfg_select! {
    target_arch = "x86_64" => {
        mod x64;
        use x64 as arch;
    }
    _ => {
        mod unsupported;
        use unsupported as arch;
    }
}

pub use arch::ArchContext;

/// State shared between runtime helpers and JIT-compiled code.
#[repr(C)]
#[derive(Default)]
pub struct JitContext {
    pub arch: ArchContext,
    pub budget: u64,
    pub guest_pc: WordType,
    pub exception: Option<(Exception, WordType)>,
}

impl JitContext {
    pub unsafe fn raise(
        context: *mut Self,
        guest_pc: WordType,
        consumed_cycles: u64,
        exception: Exception,
        tval: WordType,
    ) -> ! {
        let context_ref = unsafe { &mut *context };
        context_ref.guest_pc = guest_pc;
        context_ref.budget -= consumed_cycles;
        context_ref.exception = Some((exception, tval));
        unsafe { jit_abort(context_ref) }
    }
}

#[inline]
unsafe fn jit_abort(context: &mut JitContext) -> ! {
    unsafe { arch::abort_jit_unwind(&context.arch) }
}

/// Compiled native basic block.
///
/// Execution convention:
/// - Returns `(next_pc, has_exception)` on exit.
/// - Maintains execution budget in [`JitContext::budget`].
#[derive(Clone, Copy)]
pub struct JitFunction {
    entry: *const u8,
}

impl JitFunction {
    pub fn from_ptr(entry: *const u8) -> Self {
        Self { entry }
    }

    #[inline]
    pub fn as_ptr(self) -> *const u8 {
        self.entry
    }

    #[inline]
    pub fn as_mut_ptr(self) -> *mut u8 {
        self.entry as *mut u8
    }

    #[inline]
    pub unsafe fn call(&mut self, cpu: &mut RVCPU, context: &mut JitContext) -> (WordType, bool) {
        unsafe { arch::enter_jit_trampoline(self.entry, cpu, context) }
    }

    #[cfg(test)]
    pub fn address(self) -> usize {
        self.entry as usize
    }
}
