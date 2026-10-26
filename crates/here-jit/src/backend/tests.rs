use crate::config::arch_config::WordType;
use crate::isa::riscv::cpu_tester::{CPUChecker, TestCPUBuilder};
use crate::isa::riscv::executor::RVCPU;
use crate::jit::backend::NativeCodeGen;
use crate::jit::backend::codegen::CodeGenTrait;
use crate::jit::ir::*;
use crate::jit::jit_buffer::JitBuffer;
use crate::make_ir;

use crate::jit::runtime::JitContext;

pub(crate) fn compile_and_exec_with_ctx(
    block: &IRBlock,
    meta: &IRBlockMeta,
    cpu: &mut RVCPU,
    ctx: &mut JitContext,
) -> crate::jit::runtime::jit_function::JitExit {
    let mut codegen = NativeCodeGen::try_new().expect("Native CodeGen must be supported");
    let code_block = codegen.compile(block, meta);
    let mut jit_buf = JitBuffer::with_capacity(64 * 1024);
    let mut func = unsafe { jit_buf.emit_code(code_block.buf) };
    unsafe { func.call(cpu, ctx) }
}

pub(crate) fn compile_and_run_with_ctx(
    block: &IRBlock,
    meta: &IRBlockMeta,
    cpu: &mut RVCPU,
    ctx: &mut JitContext,
) -> WordType {
    let exit = compile_and_exec_with_ctx(block, meta, cpu, ctx);
    assert!(!exit.has_exception, "unexpected exception in test block");
    exit.next_pc
}

pub(crate) fn compile_and_run(block: &IRBlock, cpu: &mut RVCPU) -> WordType {
    let mut ctx = JitContext::default();
    ctx.budget = 100_000;
    compile_and_run_with_ctx(block, &IRBlockMeta::default(), cpu, &mut ctx)
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
fn test_branch_conditions() {
    let cases = [
        (CmpKind::Eq, 42, 42, true),
        (CmpKind::Eq, 42, 43, false),
        (CmpKind::NotEq, 42, 43, true),
        (CmpKind::NotEq, 42, 42, false),
        (CmpKind::Lt, 10, 20, true),
        (CmpKind::Lt, 20, 10, false),
        (CmpKind::Lt, 1, 0xffff_ffff_ffff_ffff, false), // 1 < -1 is false
        (CmpKind::LtUnsigned, 1, 0xffff_ffff_ffff_ffff, true), // 1 < MAX is true
        (
            CmpKind::GtEq,
            0xffff_ffff_ffff_ffff,
            0xffff_ffff_ffff_fffe,
            true,
        ), // -1 >= -2 is true
        (
            CmpKind::GtEq,
            0xffff_ffff_ffff_fffe,
            0xffff_ffff_ffff_ffff,
            false,
        ), // -2 >= -1 is false
        (CmpKind::GtEqUnsigned, 10, 10, true),
        (CmpKind::GtEqUnsigned, 5, 10, false),
    ];

    for (cmp, lhs, rhs, expected_taken) in cases {
        let mut cpu = TestCPUBuilder::new().build();
        let mut builder = IRBuilder::default();
        let (v0, v1) = make_ir!(builder, [
            v0 = const lhs;
            v1 = const rhs;
        ], [v0, v1]);
        let block = builder.build_branch(cmp, v0, v1, 0x1111, 0x2222);
        let expected_pc = if expected_taken { 0x1111 } else { 0x2222 };
        assert_eq!(
            compile_and_run(&block, &mut cpu),
            expected_pc,
            "Failed on {cmp:?} with lhs={lhs:#x}, rhs={rhs:#x}"
        );
    }
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
    unsafe extern "sysv64" fn helper_add(a: u64, b: u64) -> u64 {
        a + b
    }

    unsafe extern "sysv64" fn helper_with_cpu(cpu: *mut RVCPU, val: u64) -> u64 {
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

    // 2. Call with cpu_vreg
    let mut builder = IRBuilder::default();
    let cpu_vreg = builder.cpu_vreg();
    make_ir!(builder, [
        v0 = const 77;
        res = call [] (helper_with_cpu as *const () as usize), [cpu_vreg, v0];
        set_reg 8, res;
    ], []);
    let block = builder.build_jmp(0x8000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x8000);
    CPUChecker::new(&mut cpu).reg(7, 77).reg(8, 154);

    // 3. Call with jit_context_vreg
    unsafe extern "sysv64" fn helper_with_ctx(ctx: *mut JitContext, val: u64) -> u64 {
        unsafe { (*ctx).budget += val };
        val * 3
    }

    let mut builder = IRBuilder::default();
    let ctx_vreg = builder.jit_context_vreg();
    make_ir!(builder, [
        v0 = const 50;
        res = call [] (helper_with_ctx as *const () as usize), [ctx_vreg, v0];
        set_reg 9, res;
    ], []);
    let block = builder.build_jmp(0x9000u64.into());
    let mut ctx = JitContext::default();
    ctx.budget = 1000;
    let meta = IRBlockMeta::default();
    let next_pc = compile_and_run_with_ctx(&block, &meta, &mut cpu, &mut ctx);
    assert_eq!(next_pc, 0x9000);
    assert_eq!(ctx.budget, 1050 - meta.guest_cnt as u64);
    CPUChecker::new(&mut cpu).reg(9, 150);
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
fn test_register_spill() {
    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    // Allocate 20 constants and sum them up sequentially while keeping all alive
    let mut consts = Vec::new();
    for i in 1..=20u64 {
        consts.push(builder.const_value(i));
    }
    let mut sum = consts[0];
    for &c in &consts[1..] {
        sum = builder.compute(ComputeKind::Add, sum, c);
    }
    builder.set_reg(1.into(), sum);
    let block = builder.build_jmp(0xcc00u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0xcc00);
    // Sum 1..=20 = 210
    CPUChecker::new(&mut cpu).reg(1, 210);
}

#[test]
fn test_callee_save_across_call() {
    unsafe extern "sysv64" fn clobber_caller_saves() {
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
fn test_multiple_helper_calls_with_args() {
    unsafe extern "sysv64" fn helper_mul(a: u64, b: u64) -> u64 {
        a * b
    }

    unsafe extern "sysv64" fn helper_sub(a: u64, b: u64) -> u64 {
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
    unsafe extern "sysv64" fn dummy_call(x: u64) -> u64 {
        x + 1
    }

    let mut cpu = TestCPUBuilder::new().build();
    let mut builder = IRBuilder::default();
    // Keep 15 constants alive across a helper call
    let mut consts = Vec::new();
    for i in 1..=15u64 {
        consts.push(builder.const_value(i * 10));
    }

    let call_arg = builder.const_value(999);
    let call_res = builder.call_ret(
        dummy_call as *const () as usize,
        vec![call_arg],
        CallOptions::default(),
    );

    let mut sum = consts[0];
    for &c in &consts[1..] {
        sum = builder.compute(ComputeKind::Add, sum, c);
    }
    let final_sum = builder.compute(ComputeKind::Add, sum, call_res);
    builder.set_reg(1.into(), final_sum);

    let block = builder.build_jmp(0x8800u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x8800);
    // Sum 10+20+...+150 = 1200, plus 1000 = 2200
    CPUChecker::new(&mut cpu).reg(1, 2200);
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
fn test_budget_deduction_and_exit() {
    // 1. Direct jump cases: normal deduction, out-of-budget, and exact match
    let direct_cases = [
        // (start_budget, guest_cnt, expected_next_pc, expected_remaining_budget)
        (100, 5, 0x3000, 95),
        (4, 5, 0x2000, 4), // insufficient budget -> exits immediately with start_pc
        (5, 5, 0x3000, 0), // exact match -> exits with 0 remaining
    ];
    for (budget, guest_cnt, expected_pc, expected_budget) in direct_cases {
        let mut cpu = TestCPUBuilder::new().build();
        let block = IRBuilder::new().build_jmp(0x3000u64.into());
        let meta = IRBlockMeta {
            start_pc: 0x2000,
            guest_cnt,
        };
        let mut ctx = JitContext::default();
        ctx.budget = budget;
        let next_pc = compile_and_run_with_ctx(&block, &meta, &mut cpu, &mut ctx);
        assert_eq!(next_pc, expected_pc);
        assert_eq!(ctx.budget, expected_budget);
    }

    // 2. Branch condition deduction
    {
        let mut cpu = TestCPUBuilder::new().build();
        let mut builder = IRBuilder::new();
        let (v0, v1) = make_ir!(builder, [
            v0 = const 10;
            v1 = const 20;
        ], [v0, v1]);
        let block = builder.build_branch(CmpKind::Lt, v0, v1, 0x5000, 0x6000);
        let meta = IRBlockMeta {
            start_pc: 0x4000,
            guest_cnt: 3,
        };
        let mut ctx = JitContext::default();
        ctx.budget = 10;
        let next_pc = compile_and_run_with_ctx(&block, &meta, &mut cpu, &mut ctx);
        assert_eq!(next_pc, 0x5000);
        assert_eq!(ctx.budget, 7);
    }

    // 3. Dynamic jump deduction
    {
        let mut cpu = TestCPUBuilder::new().reg(1, 0x7777).build();
        let mut builder = IRBuilder::new();
        let target = make_ir!(builder, [
            target = get_reg 1;
        ], [target]);
        let block = builder.build_jmp(target);
        let meta = IRBlockMeta {
            start_pc: 0x1000,
            guest_cnt: 2,
        };
        let mut ctx = JitContext::default();
        ctx.budget = 50;
        let next_pc = compile_and_run_with_ctx(&block, &meta, &mut cpu, &mut ctx);
        assert_eq!(next_pc, 0x7777);
        assert_eq!(ctx.budget, 48);
    }
}

core::arch::global_asm!(
    r#"
    .global helper_get_rsp
    helper_get_rsp:
        mov rax, rsp
        ret
    "#
);

unsafe extern "sysv64" {
    fn helper_get_rsp() -> u64;
}

#[test]
fn test_helper_call_stack_alignment() {
    let mut cpu = TestCPUBuilder::new().build();

    // 1. Call with 0 spill slots
    let mut builder = IRBuilder::default();
    make_ir!(builder, [
        rsp_val = call [] (helper_get_rsp as *const () as usize), [];
        set_reg 1, rsp_val;
    ], []);
    let block = builder.build_jmp(0x1000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x1000);
    let rsp_at_entry = cpu.reg_file[1];
    assert_eq!(
        rsp_at_entry % 16,
        8,
        "RSP before call was not 16-byte aligned (0 spill slots)"
    );

    // 2. Call with multiple spill slots (force spills by keeping registers alive)
    let mut builder = IRBuilder::default();
    let mut consts = Vec::new();
    for i in 1..=15u64 {
        consts.push(builder.const_value(i * 10));
    }
    let rsp_val = builder.call_ret(
        helper_get_rsp as *const () as usize,
        vec![],
        CallOptions::default(),
    );
    let mut sum = consts[0];
    for &c in &consts[1..] {
        sum = builder.compute(ComputeKind::Add, sum, c);
    }
    builder.set_reg(1.into(), rsp_val);
    builder.set_reg(2.into(), sum);

    let block = builder.build_jmp(0x2000u64.into());
    let next_pc = compile_and_run(&block, &mut cpu);
    assert_eq!(next_pc, 0x2000);
    let rsp_at_entry = cpu.reg_file[1];
    assert_eq!(
        rsp_at_entry % 16,
        8,
        "RSP before call was not 16-byte aligned (multiple spill slots)"
    );
    assert_eq!(cpu.reg_file[2], 1200);
}
