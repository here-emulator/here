// TODO: mirgrate from legacy code, can be modified.

use std::arch::asm;

use libc::{c_int, c_void};

use crate::{
    config::arch_config::WordType,
    isa::riscv::{executor::RVCPU, trap::Exception},
};

unsafe extern "C" {
    fn longjmp(env: *mut c_void, value: c_int) -> !;
}

#[repr(C, align(16))]
#[derive(Default)]
pub(crate) struct JmpBuf {
    storage: [usize; 32],
}

/// State shared between runtime helpers and JIT-compiled code.
#[repr(C)]
#[derive(Default)]
pub struct JitContext {
    jmp_buf: JmpBuf,
    pub budget: u64,
    pub guest_pc: WordType,
    pub exception: Option<(Exception, WordType)>,
}

impl JitContext {
    pub fn jmp_buf_ptr(&mut self) -> *mut c_void {
        self.jmp_buf.storage.as_mut_ptr().cast::<c_void>()
    }

    pub unsafe fn raise(
        context: *mut Self,
        guest_pc: WordType,
        consumed_cycles: u64,
        exception: Exception,
        tval: WordType,
    ) -> ! {
        let context = unsafe { &mut *context };
        context.guest_pc = guest_pc;
        context.budget -= consumed_cycles;
        context.exception = Some((exception, tval));
        let env = context.jmp_buf_ptr();
        unsafe { longjmp(env, 1) }
    }
}

/// Compiled native basic block.
///
/// Execution convention:
/// - Returns `next_pc` on exit.
/// - Maintains execution budget in [`JitContext::budget`].
#[derive(Clone, Copy)]
pub struct JitFunction {
    entry: *const u8,
}

impl JitFunction {
    pub fn from_ptr(entry: *const u8) -> Self {
        Self { entry }
    }

    pub unsafe fn call(self, cpu: *mut RVCPU) -> WordType {
        let next_pc: WordType;
        #[cfg(not(target_family = "windows"))]
        unsafe {
            asm!(
                "push rbx",
                "push rbp",
                "push r12",
                "push r13",
                "push r14",
                "push r15",
                "sub rsp, 8",
                "mov rbp, r11",
                "call r10",
                "add rsp, 8",
                "pop r15",
                "pop r14",
                "pop r13",
                "pop r12",
                "pop rbp",
                "pop rbx",
                in("r10") self.entry,
                in("r11") cpu,
                lateout("rax") next_pc,
                clobber_abi("C"),
            );
        }

        #[cfg(target_family = "windows")]
        unsafe {
            asm!(
                "push rbx",
                "push rbp",
                "push r12",
                "push r13",
                "push r14",
                "push r15",
                // Keep the JIT stack aligned and leave Windows shadow space available.
                "sub rsp, 40",
                "mov rbp, r11",
                "call r10",
                "add rsp, 40",
                "pop r15",
                "pop r14",
                "pop r13",
                "pop r12",
                "pop rbp",
                "pop rbx",
                in("r10") self.entry,
                in("r11") cpu,
                lateout("rax") next_pc,
                clobber_abi("C"),
            );
        }

        next_pc
    }

    #[cfg(test)]
    pub fn address(self) -> usize {
        self.entry as usize
    }
}
