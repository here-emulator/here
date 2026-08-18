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
struct JmpBuf {
    storage: [usize; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct JitInfo {
    pub instr_pc: WordType,
    pub instr_cnt: u64,
    pub instr_len: WordType,
    pub ialign: WordType,
}

#[repr(C)]
#[derive(Default)]
pub(super) struct JitContext {
    jmp_buf: JmpBuf,
    pub guest_pc: WordType,
    pub icount: u64,
    pub remaining_cycles: u64,
    pub exception: Option<(Exception, WordType)>,
}

impl JitContext {
    pub(super) fn jmp_buf_ptr(&mut self) -> *mut c_void {
        self.jmp_buf.storage.as_mut_ptr().cast::<c_void>()
    }

    pub(super) unsafe fn raise(
        context: *mut Self,
        guest_pc: WordType,
        icount: u64,
        exception: Exception,
        tval: WordType,
    ) -> ! {
        let context = unsafe { &mut *context };
        context.guest_pc = guest_pc;
        context.icount = icount;
        context.exception = Some((exception, tval));
        let env = context.jmp_buf_ptr();
        unsafe { longjmp(env, 1) }
    }
}

#[derive(Clone, Copy)]
pub(super) struct JitFunction {
    entry: *const u8,
}

impl JitFunction {
    pub(super) fn from_ptr(entry: *const u8) -> Self {
        Self { entry }
    }

    /// Calls generated code while preserving the host's callee-saved registers.
    /// The translator's `CPU_REG` convention is RBP, so this wrapper loads the
    /// CPU pointer into RBP before entering generated code.
    pub(super) unsafe fn call(self, cpu: *mut RVCPU) -> WordType {
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
    pub(super) fn address(self) -> usize {
        self.entry as usize
    }
}
