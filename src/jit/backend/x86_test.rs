use super::*;
use crate::jit::jit_buffer::JitBuffer;

use X86Reg::*;

fn emitted_bytes(emit: impl FnOnce(&mut X86Assembler)) -> Vec<u8> {
    let mut asm = X86Assembler::new();
    emit(&mut asm);
    asm.build()
}

unsafe fn execute_returning_u64(mut asm: X86Assembler) -> u64 {
    asm.ret();
    let mut buffer = JitBuffer::new();
    let function = unsafe { buffer.emit_code(&asm.build()) };
    unsafe { function.call(std::ptr::null_mut()) }
}

unsafe extern "C" fn mixed_helper_args(
    first: u64,
    second: u64,
    third: u64,
    fourth: u64,
    fifth: u64,
    sixth: u64,
) -> u64 {
    first + second * 10 + third * 100 + fourth * 1_000 + fifth * 10_000 + sixth * 100_000
}

#[test]
fn selects_signed_displacement_width_at_boundaries() {
    assert!(matches!(
        RegMem::from_disp(RDI, -129),
        RegMem::MemDisp32(RDI, -129)
    ));
    assert!(matches!(
        RegMem::from_disp(RDI, -128),
        RegMem::MemDisp8(RDI, -128)
    ));
    assert!(matches!(
        RegMem::from_disp(RDI, -1),
        RegMem::MemDisp8(RDI, -1)
    ));
    assert!(matches!(RegMem::from_disp(RDI, 0), RegMem::Mem(RDI)));
    assert!(matches!(
        RegMem::from_disp(RDI, 127),
        RegMem::MemDisp8(RDI, 127)
    ));
    assert!(matches!(
        RegMem::from_disp(RDI, 128),
        RegMem::MemDisp32(RDI, 128)
    ));
}

#[test]
fn encodes_rex_modrm_alu_and_shift_variants() {
    let alu_cases = [
        (AluOp::Add, 0x01, 0xc0),
        (AluOp::Or, 0x09, 0xc8),
        (AluOp::Adc, 0x11, 0xd0),
        (AluOp::Sbb, 0x19, 0xd8),
        (AluOp::And, 0x21, 0xe0),
        (AluOp::Sub, 0x29, 0xe8),
        (AluOp::Xor, 0x31, 0xf0),
        (AluOp::Cmp, 0x39, 0xf8),
    ];
    for (operation, register_opcode, immediate_modrm) in alu_cases {
        assert_eq!(
            emitted_bytes(|asm| asm.alu_rm32_r32(operation, R8, R9)),
            [0x45, register_opcode, 0xc8]
        );
        assert_eq!(
            emitted_bytes(|asm| asm.alu_rm64_r64(operation, R8, R9)),
            [0x4d, register_opcode, 0xc8]
        );
        assert_eq!(
            emitted_bytes(|asm| asm.alu_rm64_imm32(operation, R8, 0x89ab_cdef)),
            [0x49, 0x81, immediate_modrm, 0xef, 0xcd, 0xab, 0x89]
        );
    }

    let shifts = [
        (ShiftOp::SHL, 0xe0),
        (ShiftOp::SHR, 0xe8),
        (ShiftOp::SAR, 0xf8),
    ];
    for (operation, modrm) in shifts {
        assert_eq!(
            emitted_bytes(|asm| {
                asm.shift_rm_imm8(OperandSize::Dword, operation, R8.into(), 31)
            }),
            [0x41, 0xc1, modrm, 0x1f]
        );
        assert_eq!(
            emitted_bytes(|asm| {
                asm.shift_rm_imm8(OperandSize::Qword, operation, R8.into(), 63)
            }),
            [0x49, 0xc1, modrm, 0x3f]
        );
    }

    assert_eq!(
        emitted_bytes(|asm| asm.mov_r64_rm64(R8, RegMem::from_disp(RDI, -129))),
        [0x4c, 0x8b, 0x87, 0x7f, 0xff, 0xff, 0xff]
    );
    assert_eq!(
        emitted_bytes(|asm| asm.mov_rm64_r64(RegMem::from_disp(RBP, -8), R9)),
        [0x4c, 0x89, 0x4d, 0xf8]
    );
    assert_eq!(
        emitted_bytes(|asm| asm.movsxd_r64_rm32(R9, R8.into())),
        [0x4d, 0x63, 0xc8]
    );
    assert_eq!(
        emitted_bytes(|asm| asm.setcc_r8(ConditionCode::Below, R8)),
        [0x41, 0x0f, 0x92, 0xc0]
    );
    assert_eq!(
        emitted_bytes(|asm| asm.movzx_r32_r8(R8, R8)),
        [0x45, 0x0f, 0xb6, 0xc0]
    );
    assert_eq!(
        emitted_bytes(|asm| asm.cmovcc_r64_rm64(ConditionCode::NotEqual, R8, R9.into())),
        [0x4d, 0x0f, 0x45, 0xc1]
    );
    assert_eq!(emitted_bytes(|asm| asm.call_rm64(R11)), [0x41, 0xff, 0xd3]);
    assert_eq!(
        emitted_bytes(|asm| asm.mov_rsp_disp64_r64(32, R10)),
        [0x4c, 0x89, 0x54, 0x24, 0x20]
    );
    assert_eq!(
        emitted_bytes(|asm| asm.movsx_r64_rm8(RAX, RAX.into())),
        [0x48, 0x0f, 0xbe, 0xc0]
    );
    assert_eq!(
        emitted_bytes(|asm| asm.movsx_r64_rm16(RAX, RAX.into())),
        [0x48, 0x0f, 0xbf, 0xc0]
    );
    assert_eq!(
        emitted_bytes(|asm| asm.movzx_r32_rm16(RAX, RAX.into())),
        [0x40, 0x0f, 0xb7, 0xc0]
    );
}

#[test]
fn executes_register_arithmetic_and_shifts() {
    let mut asm = X86Assembler::new();
    asm.mov_r64_imm64(R8, 1000);
    asm.mov_r64_imm64(R9, 23);
    asm.alu_rm64_r64(AluOp::Sub, R8, R9);
    asm.alu_rm64_imm32(AluOp::Add, R8, 7);
    asm.shift_rm_imm8(OperandSize::Qword, ShiftOp::SHL, R8.into(), 2);
    asm.mov_r64_imm64(RCX, 1);
    asm.shift_rm_cl(OperandSize::Qword, ShiftOp::SHR, R8.into());
    asm.mov_r64_rm64(RAX, R8);

    assert_eq!(unsafe { execute_returning_u64(asm) }, 1968);

    let mut asm = X86Assembler::new();
    asm.mov_r64_imm64(R8, 0x8000_0000_0000_0001);
    asm.shift_rm_imm8(OperandSize::Qword, ShiftOp::SAR, R8.into(), 1);
    asm.mov_r64_rm64(RAX, R8);
    assert_eq!(unsafe { execute_returning_u64(asm) }, 0xc000_0000_0000_0000);

    let mut asm = X86Assembler::new();
    asm.mov_r64_imm64(R8, u64::MAX);
    asm.alu_rm32_imm32(AluOp::Add, R8, 1);
    asm.mov_r64_rm64(RAX, R8);
    assert_eq!(unsafe { execute_returning_u64(asm) }, 0);
}

#[test]
fn jit_function_returns_the_rax_value() {
    let mut asm = X86Assembler::new();
    asm.mov_r64_imm64(RAX, 0x0123_4567_89ab_cdef);

    assert_eq!(unsafe { execute_returning_u64(asm) }, 0x0123_4567_89ab_cdef);
}

#[test]
fn calls_helper_with_mixed_register_and_immediate_arguments() {
    let mut asm = X86Assembler::new();
    asm.mov_r64_imm64(RBX, 2);
    asm.mov_r64_imm64(R12, 4);
    asm.mov_r64_imm64(R13, 6);
    asm.call_helper(
        mixed_helper_args as *const () as u64,
        &[
            CallArg::Imm(1),
            CallArg::Reg(RBX),
            CallArg::Imm(3),
            CallArg::Reg(R12),
            CallArg::Imm(5),
            CallArg::Reg(R13),
        ],
    );

    assert_eq!(unsafe { execute_returning_u64(asm) }, 654_321);
}

#[test]
fn executes_signed_and_unsigned_comparisons() {
    let mut asm = X86Assembler::new();
    asm.mov_r64_imm64(R8, u64::MAX);
    asm.mov_r64_imm64(R9, 1);
    asm.alu_rm64_r64(AluOp::Cmp, R8, R9);
    asm.setcc_r8(ConditionCode::Less, R8);
    asm.movzx_r32_r8(R8, R8);
    asm.mov_r64_rm64(RAX, R8);
    assert_eq!(unsafe { execute_returning_u64(asm) }, 1);

    let mut asm = X86Assembler::new();
    asm.mov_r64_imm64(R8, u64::MAX);
    asm.alu_rm64_imm32(AluOp::Cmp, R8, 1);
    asm.setcc_r8(ConditionCode::Below, R8);
    asm.movzx_r32_r8(R8, R8);
    asm.mov_r64_rm64(RAX, R8);
    assert_eq!(unsafe { execute_returning_u64(asm) }, 0);
}

#[test]
fn executes_disp8_and_disp32_memory_operands() {
    let mut words = [0_u64; 64];
    words[15] = 1000;
    words[31] = 23;
    let base = unsafe { words.as_mut_ptr().add(32) };

    let mut asm = X86Assembler::new();
    asm.mov_r64_rm64(R8, RegMem::from_disp(RBP, -136));
    asm.mov_r64_rm64(R9, RegMem::from_disp(RBP, -8));
    asm.alu_rm64_r64(AluOp::Sub, R8, R9);
    asm.mov_rm64_r64(RegMem::from_disp(RBP, 128), R8);
    asm.ret();

    let mut buffer = JitBuffer::new();
    let function = unsafe { buffer.emit_code(&asm.build()) };
    let _ = unsafe { function.call(base.cast()) };

    assert_eq!(words[48], 977);
}
