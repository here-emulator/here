use here_isa::riscv::executor::RVCPU;

use super::JitContext;
use super::JitExit;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchContext {
    pub saved_rsp: usize,
}

#[allow(improper_ctypes)]
unsafe extern "sysv64" {
    fn enter_jit(
        jit_fn: *const u8,
        saved_rsp_pointer: *mut usize,
        rvcpu: *mut RVCPU,
        context: *mut JitContext,
    ) -> JitExit;

    fn jit_abort_unwind(saved_rsp: usize) -> !;
}

core::arch::global_asm!(
    r#"
    .global enter_jit
    .global jit_abort_unwind

    enter_jit: # (rdi: jit_fn, rsi: saved_rsp_ptr, rdx: cpu, rcx: context)
        # Save callee-saved registers.
        push rbx
        push rbp
        push r12
        push r13
        push r14
        push r15

        # Align the stack, the stack needs to be 16 byte aligned
        # *before* the call instruction is executed.
        sub rsp, 8

        # Save stack pointer for abort unwinding.
        mov [rsi], rsp

        # Pass CPU (RBP) and JitContext (RBX) to JIT code.
        mov rbp, rdx
        mov rbx, rcx

        call rdi

        # Normal return: has_exception = false (edx = 0).
        xor edx, edx

    .Lexit_trampoline:
        # Restore stack and callee-saved registers.
        add rsp, 8
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbp
        pop rbx
        ret

    jit_abort_unwind: # (rdi: saved_rsp)
        # Restore stack, set has_exception = true
        mov rsp, rdi
        xor eax, eax
        mov edx, 1
        jmp .Lexit_trampoline
    "#
);

#[inline]
pub unsafe fn enter_jit_trampoline(
    entry: *const u8,
    cpu: &mut RVCPU,
    context: &mut JitContext,
) -> JitExit {
    unsafe {
        enter_jit(
            entry,
            &mut context.arch.saved_rsp as *mut usize,
            cpu as *mut RVCPU,
            context as *mut JitContext,
        )
    }
}

#[inline]
pub unsafe fn abort_jit_unwind(arch: &ArchContext) -> ! {
    let saved_rsp = arch.saved_rsp;
    unsafe { jit_abort_unwind(saved_rsp) };
}
