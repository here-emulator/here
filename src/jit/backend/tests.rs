use crate::config::arch_config::WordType;
use crate::isa::riscv::cpu_tester::{CPUChecker, TestCPUBuilder};
use crate::isa::riscv::executor::RVCPU;
use crate::jit::backend::NativeCodeGen;
use crate::jit::backend::codegen::CodeGenTrait;
use crate::jit::ir::*;
use crate::jit::jit_buffer::JitBuffer;
use crate::make_ir;

use crate::jit::runtime::JitContext;

fn compile_and_run(block: &IRBlock, cpu: &mut RVCPU) -> WordType {
    let mut ctx = JitContext::default();
    ctx.budget = 100_000;
    compile_and_run_with_ctx(block, cpu, &mut ctx)
}

fn compile_and_run_with_ctx(block: &IRBlock, cpu: &mut RVCPU, ctx: &mut JitContext) -> WordType {
    let mut codegen = NativeCodeGen::try_new().expect("Native CodeGen must be supported");
    let code_block = codegen.compile(block);
    let mut jit_buf = JitBuffer::new();
    let func = unsafe { jit_buf.emit_code(code_block.buf) };
    unsafe { func.call(cpu as *mut RVCPU, ctx as *mut JitContext) }
}

#[test]
fn test_direct_jump() {
    let mut cpu = TestCPUBuilder::new().build();
    let builder = IRBuilder::default();
    let block = builder.build_jmp(0x1234_5678u64.into());

    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x1234_5678);
}

#[test]
fn test_dynamic_jump() {
    let mut cpu = TestCPUBuilder::new().reg(1, 0x8765_4321).build();
    let mut builder = IRBuilder::default();
    let target = make_ir!(builder, [
        target = get_reg 1;
    ], [target]);
    let block = builder.build_jmp(target);

    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x8765_4321);
    CPUChecker::new(&mut cpu).reg(1, 0x8765_4321);
}

#[test]
fn test_branch() {
    let mut cpu = TestCPUBuilder::new().build();

    // Condition taken: 10 < 20
    let mut builder = IRBuilder::default();
    let (v0, v1) = make_ir!(builder, [
        v0 = const 10;
        v1 = const 20;
    ], [v0, v1]);
    let block = builder.build_branch(CmpKind::Lt, v0, v1, 0x1111, 0x2222);
    assert_eq!(compile_and_run(&block, &mut cpu), 0x1111);

    // Condition not taken: 20 < 10
    let mut builder = IRBuilder::default();
    let (v0, v1) = make_ir!(builder, [
        v0 = const 20;
        v1 = const 10;
    ], [v0, v1]);
    let block = builder.build_branch(CmpKind::Lt, v0, v1, 0x1111, 0x2222);
    assert_eq!(compile_and_run(&block, &mut cpu), 0x2222);
}

#[test]
fn test_get_set_reg_and_compute() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v0 = const 15;
        v1 = const 25;
        v_add = compute Add, v0, v1; // 40
        v_sub = compute Sub, v1, v0; // 10
        v_shl = compute Shl, v0, 2;  // 60
        v_slt = compute Slt, v0, v1; // 1
        set_reg 1, v_add;
        set_reg 2, v_sub;
        set_reg 3, v_shl;
        set_reg 4, v_slt;
    ], []);
    let block = builder.build_jmp(0x5000u64.into());

    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x5000);
    CPUChecker::new(&mut cpu)
        .reg(1, 40)
        .reg(2, 10)
        .reg(3, 60)
        .reg(4, 1);
}

#[test]
fn test_x0_semantics() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v0 = const 999;
        set_reg 0, v0;     // Writes to x0 must be discarded
        v1 = get_reg 0;     // Reads from x0 must return 0
        set_reg 5, v1;
    ], []);
    let block = builder.build_jmp(0x6000u64.into());

    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x6000);
    CPUChecker::new(&mut cpu).reg(0, 0).reg(5, 0);
}

#[test]
fn test_call_helper() {
    unsafe extern "C" fn helper_add(a: u64, b: u64) -> u64 {
        a + b
    }

    unsafe extern "C" fn helper_with_cpu(cpu: *mut RVCPU, val: u64) -> u64 {
        unsafe { (*cpu).reg_file.write(7, val) };
        val * 2
    }

    let mut cpu = TestCPUBuilder::new().build();

    // 1. Call without pass_cpu
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v0 = const 100;
        v1 = const 200;
        res = call [] (helper_add as *const () as usize), [v0, v1];
        set_reg 6, res;
    ], []);
    let block = builder.build_jmp(0x7000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x7000);
    CPUChecker::new(&mut cpu).reg(6, 300);

    // 2. Call with pass_cpu
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v0 = const 77;
        res = call [cpu] (helper_with_cpu as *const () as usize), [v0];
        set_reg 8, res;
    ], []);
    let block = builder.build_jmp(0x8000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x8000);
    CPUChecker::new(&mut cpu).reg(7, 77).reg(8, 154);
}

#[test]
fn test_compute_32bit() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v0 = const 0x7fff_ffff; // Max positive i32
        v1 = const 1;
        // 0x7fff_ffff + 1 = 0x8000_0000 (signed i32 is negative, sign-extended to 0xffff_ffff_8000_0000)
        v_addw = compute AddW, v0, v1;
        // 0 - 1 = -1 (sign-extended to 0xffff_ffff_ffff_ffff)
        v_zero = const 0;
        v_subw = compute SubW, v_zero, v1;
        // Shift left 1
        v_shlw = compute ShlW, v0, 1;
        set_reg 10, v_addw;
        set_reg 11, v_subw;
        set_reg 12, v_shlw;
    ], []);
    let block = builder.build_jmp(0x9000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x9000);
    CPUChecker::new(&mut cpu)
        .reg(10, 0xffff_ffff_8000_0000)
        .reg(11, 0xffff_ffff_ffff_ffff)
        .reg(12, 0xffff_ffff_ffff_fffe);
}

#[test]
fn test_branch_unsigned() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    // As unsigned: 1 < 0xffff_ffff_ffff_ffff
    // As signed: 1 > -1
    let (v_pos, v_neg) = make_ir!(builder, [
        v_pos = const 1;
        v_neg = const 0xffff_ffff_ffff_ffff;
    ], [v_pos, v_neg]);

    // Unsigned: 1 < MAX -> Below is true -> then_pc
    let b1 = builder.build_branch(CmpKind::LtUnsigned, v_pos, v_neg, 0xaaaa, 0xbbbb);
    assert_eq!(compile_and_run(&b1, &mut cpu), 0xaaaa);

    // Signed: 1 < -1 is false -> else_pc
    let mut builder2 = IRBuilder::default();
    let (v_pos2, v_neg2) = make_ir!(builder2, [
        v_pos2 = const 1;
        v_neg2 = const 0xffff_ffff_ffff_ffff;
    ], [v_pos2, v_neg2]);
    let b2 = builder2.build_branch(CmpKind::Lt, v_pos2, v_neg2, 0xaaaa, 0xbbbb);
    assert_eq!(compile_and_run(&b2, &mut cpu), 0xbbbb);
}

#[test]
fn test_register_spill() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    // Allocate 20 constants and sum them up sequentially while keeping all alive
    make_ir!(builder, [
        c0 = const 1;
        c1 = const 2;
        c2 = const 3;
        c3 = const 4;
        c4 = const 5;
        c5 = const 6;
        c6 = const 7;
        c7 = const 8;
        c8 = const 9;
        c9 = const 10;
        c10 = const 11;
        c11 = const 12;
        c12 = const 13;
        c13 = const 14;
        c14 = const 15;
        c15 = const 16;
        c16 = const 17;
        c17 = const 18;
        c18 = const 19;
        c19 = const 20;

        s1 = compute Add, c0, c1;
        s2 = compute Add, s1, c2;
        s3 = compute Add, s2, c3;
        s4 = compute Add, s3, c4;
        s5 = compute Add, s4, c5;
        s6 = compute Add, s5, c6;
        s7 = compute Add, s6, c7;
        s8 = compute Add, s7, c8;
        s9 = compute Add, s8, c9;
        s10 = compute Add, s9, c10;
        s11 = compute Add, s10, c11;
        s12 = compute Add, s11, c12;
        s13 = compute Add, s12, c13;
        s14 = compute Add, s13, c14;
        s15 = compute Add, s14, c15;
        s16 = compute Add, s15, c16;
        s17 = compute Add, s16, c17;
        s18 = compute Add, s17, c18;
        s19 = compute Add, s18, c19;

        set_reg 1, s19;
    ], []);
    let block = builder.build_jmp(0xcc00u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0xcc00);
    // Sum 1..=20 = 210
    CPUChecker::new(&mut cpu).reg(1, 210);
}

#[test]
fn test_callee_save_across_call() {
    #[cfg(target_arch = "x86_64")]
    unsafe extern "C" fn clobber_caller_saves() {
        unsafe {
            std::arch::asm!(
                "mov rax, 0x11111111",
                "mov rcx, 0x22222222",
                "mov rdx, 0x33333333",
                "mov rsi, 0x44444444",
                "mov rdi, 0x55555555",
                "mov r8,  0x66666666",
                "mov r9,  0x77777777",
                "mov r10, 0x88888888",
                "mov r11, 0x99999999",
                lateout("rax") _,
                lateout("rcx") _,
                lateout("rdx") _,
                lateout("rsi") _,
                lateout("rdi") _,
                lateout("r8") _,
                lateout("r9") _,
                lateout("r10") _,
                lateout("r11") _,
            );
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe extern "C" fn clobber_caller_saves() {}

    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v_saved = const 0x1234_5678_9abc_def0;
        call [] (clobber_caller_saves as *const () as usize), [];
        v_res = compute Add, v_saved, 1;
        set_reg 1, v_res;
    ], []);
    let block = builder.build_jmp(0xdd00u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0xdd00);
    CPUChecker::new(&mut cpu).reg(1, 0x1234_5678_9abc_def1);
}

#[test]
fn test_read_initial_registers() {
    let mut cpu = TestCPUBuilder::new()
        .reg(1, 100)
        .reg(2, 250)
        .reg(3, 0xdead_beef)
        .build();

    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        r1 = get_reg 1;
        r2 = get_reg 2;
        r3 = get_reg 3;
        sum = compute Add, r1, r2;
        diff = compute Sub, r2, r1;
        xor_res = compute Xor, r1, r3;
        set_reg 4, sum;
        set_reg 5, diff;
        set_reg 6, xor_res;
    ], []);
    let block = builder.build_jmp(0x1000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x1000);
    CPUChecker::new(&mut cpu)
        .reg(1, 100)
        .reg(2, 250)
        .reg(3, 0xdead_beef)
        .reg(4, 350)
        .reg(5, 150)
        .reg(6, 100 ^ 0xdead_beef);
}

#[test]
fn test_bitwise_logic() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v0 = const 0x00ff_00ff_a5a5_5a5a;
        v1 = const 0x0f0f_f0f0_ffff_0000;
        res_and = compute And, v0, v1;
        res_or  = compute Or,  v0, v1;
        res_xor = compute Xor, v0, v1;
        set_reg 1, res_and;
        set_reg 2, res_or;
        set_reg 3, res_xor;
    ], []);
    let block = builder.build_jmp(0x2000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x2000);
    let expected_and = 0x00ff_00ff_a5a5_5a5au64 & 0x0f0f_f0f0_ffff_0000u64;
    let expected_or = 0x00ff_00ff_a5a5_5a5au64 | 0x0f0f_f0f0_ffff_0000u64;
    let expected_xor = 0x00ff_00ff_a5a5_5a5au64 ^ 0x0f0f_f0f0_ffff_0000u64;
    CPUChecker::new(&mut cpu)
        .reg(1, expected_and)
        .reg(2, expected_or)
        .reg(3, expected_xor);
}

#[test]
fn test_shifts_64bit() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v_neg = const 0x8000_0000_0000_0000;
        v_val = const 0x0000_0000_0000_0001;
        shl_res = compute Shl, v_val, 63;
        shr_res = compute Shr, v_neg, 3;  // Logical: 0x1000_0000_0000_0000
        sar_res = compute Sar, v_neg, 3;  // Arithmetic: 0xf000_0000_0000_0000
        set_reg 1, shl_res;
        set_reg 2, shr_res;
        set_reg 3, sar_res;
    ], []);
    let block = builder.build_jmp(0x3000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x3000);
    CPUChecker::new(&mut cpu)
        .reg(1, 0x8000_0000_0000_0000)
        .reg(2, 0x1000_0000_0000_0000)
        .reg(3, 0xf000_0000_0000_0000);
}

#[test]
fn test_shifts_32bit() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v_neg32 = const 0x8000_0000;
        v_one   = const 1;
        // Shift left 31 in 32-bit: 1 << 31 = 0x8000_0000 -> sign-extended to 0xffff_ffff_8000_0000
        sllw_res = compute ShlW, v_one, 31;
        // Shift right logical in 32-bit: 0x8000_0000 >> 4 = 0x0800_0000 -> sign-extended to 0x0000_0000_0800_0000
        srlw_res = compute ShrW, v_neg32, 4;
        // Shift right arithmetic in 32-bit: 0x8000_0000 >> 4 = 0xf800_0000 -> sign-extended to 0xffff_ffff_f800_0000
        sraw_res = compute SarW, v_neg32, 4;
        set_reg 1, sllw_res;
        set_reg 2, srlw_res;
        set_reg 3, sraw_res;
    ], []);
    let block = builder.build_jmp(0x4000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x4000);
    CPUChecker::new(&mut cpu)
        .reg(1, 0xffff_ffff_8000_0000)
        .reg(2, 0x0000_0000_0800_0000)
        .reg(3, 0xffff_ffff_f800_0000);
}

#[test]
fn test_slt_and_sltu() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    // Comparing -1 (0xffff_ffff_ffff_ffff) and 1
    make_ir!(builder, [
        v_minus_one = const 0xffff_ffff_ffff_ffff;
        v_one       = const 1;
        // Signed: -1 < 1 is true (1)
        res_slt  = compute Slt,  v_minus_one, v_one;
        // Unsigned: u64::MAX < 1 is false (0)
        res_sltu = compute Sltu, v_minus_one, v_one;
        // Unsigned: 1 < u64::MAX is true (1)
        res_sltu2 = compute Sltu, v_one, v_minus_one;
        set_reg 1, res_slt;
        set_reg 2, res_sltu;
        set_reg 3, res_sltu2;
    ], []);
    let block = builder.build_jmp(0x5000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x5000);
    CPUChecker::new(&mut cpu).reg(1, 1).reg(2, 0).reg(3, 1);
}

#[test]
fn test_branch_equality() {
    let mut cpu = TestCPUBuilder::new().build();

    // Eq taken: 42 == 42
    let mut b_eq = IRBuilder::default();
    let (v0, v1) = make_ir!(b_eq, [
        v0 = const 42;
        v1 = const 42;
    ], [v0, v1]);
    let blk = b_eq.build_branch(CmpKind::Eq, v0, v1, 0x1111, 0x2222);
    assert_eq!(compile_and_run(&blk, &mut cpu), 0x1111);

    // Eq not taken: 42 == 43
    let mut b_eq2 = IRBuilder::default();
    let (v0, v1) = make_ir!(b_eq2, [
        v0 = const 42;
        v1 = const 43;
    ], [v0, v1]);
    let blk = b_eq2.build_branch(CmpKind::Eq, v0, v1, 0x1111, 0x2222);
    assert_eq!(compile_and_run(&blk, &mut cpu), 0x2222);

    // Ne taken: 42 != 43
    let mut b_ne = IRBuilder::default();
    let (v0, v1) = make_ir!(b_ne, [
        v0 = const 42;
        v1 = const 43;
    ], [v0, v1]);
    let blk = b_ne.build_branch(CmpKind::NotEq, v0, v1, 0x3333, 0x4444);
    assert_eq!(compile_and_run(&blk, &mut cpu), 0x3333);

    // Ne not taken: 42 != 42
    let mut b_ne2 = IRBuilder::default();
    let (v0, v1) = make_ir!(b_ne2, [
        v0 = const 42;
        v1 = const 42;
    ], [v0, v1]);
    let blk = b_ne2.build_branch(CmpKind::NotEq, v0, v1, 0x3333, 0x4444);
    assert_eq!(compile_and_run(&blk, &mut cpu), 0x4444);
}

#[test]
fn test_branch_greater_equal() {
    let mut cpu = TestCPUBuilder::new().build();

    // Signed Ge: -1 >= -2 is true
    let mut b1 = IRBuilder::default();
    let (v0, v1) = make_ir!(b1, [
        v0 = const 0xffff_ffff_ffff_ffff; // -1
        v1 = const 0xffff_ffff_ffff_fffe; // -2
    ], [v0, v1]);
    let blk = b1.build_branch(CmpKind::GtEq, v0, v1, 0x1000, 0x2000);
    assert_eq!(compile_and_run(&blk, &mut cpu), 0x1000);

    // Signed Ge: -2 >= -1 is false
    let mut b2 = IRBuilder::default();
    let (v0, v1) = make_ir!(b2, [
        v0 = const 0xffff_ffff_ffff_fffe; // -2
        v1 = const 0xffff_ffff_ffff_ffff; // -1
    ], [v0, v1]);
    let blk = b2.build_branch(CmpKind::GtEq, v0, v1, 0x1000, 0x2000);
    assert_eq!(compile_and_run(&blk, &mut cpu), 0x2000);

    // Unsigned Ge: 10 >= 10 is true
    let mut b3 = IRBuilder::default();
    let (v0, v1) = make_ir!(b3, [
        v0 = const 10;
        v1 = const 10;
    ], [v0, v1]);
    let blk = b3.build_branch(CmpKind::GtEqUnsigned, v0, v1, 0x3000, 0x4000);
    assert_eq!(compile_and_run(&blk, &mut cpu), 0x3000);

    // Unsigned Ge: 5 >= 10 is false
    let mut b4 = IRBuilder::default();
    let (v0, v1) = make_ir!(b4, [
        v0 = const 5;
        v1 = const 10;
    ], [v0, v1]);
    let blk = b4.build_branch(CmpKind::GtEqUnsigned, v0, v1, 0x3000, 0x4000);
    assert_eq!(compile_and_run(&blk, &mut cpu), 0x4000);
}

#[test]
fn test_multiple_helper_calls_with_args() {
    unsafe extern "C" fn helper_mul(a: u64, b: u64) -> u64 {
        a * b
    }

    unsafe extern "C" fn helper_sub(a: u64, b: u64) -> u64 {
        a - b
    }

    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v10 = const 10;
        v20 = const 20;
        v5  = const 5;
        prod = call [] (helper_mul as *const () as usize), [v10, v20]; // 200
        diff = call [] (helper_sub as *const () as usize), [prod, v5];  // 195
        set_reg 1, prod;
        set_reg 2, diff;
    ], []);
    let block = builder.build_jmp(0x7700u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x7700);
    CPUChecker::new(&mut cpu).reg(1, 200).reg(2, 195);
}

#[test]
fn test_spill_across_call() {
    unsafe extern "C" fn dummy_call(x: u64) -> u64 {
        x + 1
    }

    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    // Keep 15 constants alive across a helper call
    make_ir!(builder, [
        c0 = const 10;
        c1 = const 20;
        c2 = const 30;
        c3 = const 40;
        c4 = const 50;
        c5 = const 60;
        c6 = const 70;
        c7 = const 80;
        c8 = const 90;
        c9 = const 100;
        c10 = const 110;
        c11 = const 120;
        c12 = const 130;
        c13 = const 140;
        c14 = const 150;

        call_arg = const 999;
        call_res = call [] (dummy_call as *const () as usize), [call_arg]; // 1000

        // Use all constants after the call
        s1 = compute Add, c0, c1;
        s2 = compute Add, s1, c2;
        s3 = compute Add, s2, c3;
        s4 = compute Add, s3, c4;
        s5 = compute Add, s4, c5;
        s6 = compute Add, s5, c6;
        s7 = compute Add, s6, c7;
        s8 = compute Add, s7, c8;
        s9 = compute Add, s8, c9;
        s10 = compute Add, s9, c10;
        s11 = compute Add, s10, c11;
        s12 = compute Add, s11, c12;
        s13 = compute Add, s12, c13;
        s14 = compute Add, s13, c14;
        final_sum = compute Add, s14, call_res;

        set_reg 1, final_sum;
    ], []);
    let block = builder.build_jmp(0x8800u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x8800);
    // Sum 10+20+...+150 = 1200, plus 1000 = 2200
    CPUChecker::new(&mut cpu).reg(1, 2200);
}

#[test]
fn test_call_helper_with_cpu() {
    unsafe extern "C" fn helper_inspect_cpu(cpu: *mut RVCPU, addend: u64) -> u64 {
        let cpu = unsafe { &mut *cpu };
        cpu.reg_file[5] + addend
    }

    let mut cpu = TestCPUBuilder::new().reg(5, 100).build();
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        val = const 23;
        res = call [cpu] (helper_inspect_cpu as *const () as usize), [val];
        set_reg 6, res;
    ], []);
    let block = builder.build_jmp(0x9900u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x9900);
    CPUChecker::new(&mut cpu).reg(5, 100).reg(6, 123);
}

#[test]
fn test_dynamic_jump_computed() {
    let mut cpu = TestCPUBuilder::new().reg(1, 0x2000).build();
    let mut builder = IRBuilder::default();
    let target = make_ir!(builder, [
        base = get_reg 1;
        offset = const 0x14;
        sum = compute Add, base, offset;
        set_reg 2, base;
    ], [sum]);
    let block = builder.build_jmp(target);
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x2014);
    CPUChecker::new(&mut cpu).reg(1, 0x2000).reg(2, 0x2000);
}

#[test]
fn test_subw_wrap() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        v0 = const 0;
        v1 = const 1;
        // 0 - 1 = 0xffff_ffff in 32-bit -> sign-extended to -1 (0xffff_ffff_ffff_ffff)
        v2 = compute SubW, v0, v1;
        set_reg 1, v2;
    ], []);
    let block = builder.build_jmp(0x1000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x1000);
    CPUChecker::new(&mut cpu).reg(1, 0xffff_ffff_ffff_ffff);
}

#[test]
fn test_budget_deduction() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::new(0x2000);
    builder.set_instr_cnt(5);
    let block = builder.build_jmp(0x3000u64.into());

    let mut ctx = JitContext::default();
    ctx.budget = 100;
    let next_pc = compile_and_run_with_ctx(&block, &mut cpu, &mut ctx);
    assert_eq!(next_pc, 0x3000);
    assert_eq!(ctx.budget, 95);
}

#[test]
fn test_out_of_budget_exit() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::new(0x2000);
    builder.set_instr_cnt(5);
    let block = builder.build_jmp(0x3000u64.into());

    // budget is 4, which is less than block.instr_cnt (5)
    let mut ctx = JitContext::default();
    ctx.budget = 4;
    let next_pc = compile_and_run_with_ctx(&block, &mut cpu, &mut ctx);
    // Should exit immediately returning start_pc (0x2000), without decrementing budget
    assert_eq!(next_pc, 0x2000);
    assert_eq!(ctx.budget, 4);
}

#[test]
fn test_budget_exact_match() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::new(0x2000);
    builder.set_instr_cnt(5);
    let block = builder.build_jmp(0x3000u64.into());

    let mut ctx = JitContext::default();
    ctx.budget = 5;
    let next_pc = compile_and_run_with_ctx(&block, &mut cpu, &mut ctx);
    assert_eq!(next_pc, 0x3000);
    assert_eq!(ctx.budget, 0);
}

#[test]
fn test_budget_deduction_branch() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::new(0x4000);
    builder.set_instr_cnt(3);
    let (v0, v1) = make_ir!(builder, [
        v0 = const 10;
        v1 = const 20;
    ], [v0, v1]);
    let block = builder.build_branch(CmpKind::Lt, v0, v1, 0x5000, 0x6000);

    let mut ctx = JitContext::default();
    ctx.budget = 10;
    let next_pc = compile_and_run_with_ctx(&block, &mut cpu, &mut ctx);
    assert_eq!(next_pc, 0x5000);
    assert_eq!(ctx.budget, 7);
}

#[test]
fn test_budget_deduction_dynamic_jmp() {
    let mut cpu = TestCPUBuilder::new().reg(1, 0x7777).build();
    let mut builder = IRBuilder::new(0x1000);
    builder.set_instr_cnt(2);
    let target = make_ir!(builder, [
        target = get_reg 1;
    ], [target]);
    let block = builder.build_jmp(target);

    let mut ctx = JitContext::default();
    ctx.budget = 50;
    let next_pc = compile_and_run_with_ctx(&block, &mut cpu, &mut ctx);
    assert_eq!(next_pc, 0x7777);
    assert_eq!(ctx.budget, 48);
}
