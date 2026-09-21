use crate::config::arch_config::WordType;

use super::*;

const CPU_REG: X86Reg = X86Reg::RBP;
const STACK_REG: X86Reg = X86Reg::RSP;

impl TargetInst for X64Inst {
    type Reg = X86Reg;

    #[inline]
    fn emit_to(&self, buf: &mut CodeBuf) {
        self.emit(buf);
    }

    #[inline]
    fn put_asm_text(&self, w: &mut impl std::fmt::Write) -> std::fmt::Result {
        self.put_asm_text(w)
    }

    #[inline]
    fn mov(dst: Self::Reg, src: Self::Reg) -> Self {
        X64Inst::Mov {
            size: OperandSize::Qword,
            dst: dst.into(),
            src: src.into(),
        }
    }

    #[inline]
    fn spill(slot: u32, src: Self::Reg) -> Self {
        let offset = (slot as usize * size_of::<WordType>()) as i32;
        X64Inst::Mov {
            size: OperandSize::Qword,
            dst: RegMem::from_disp(STACK_REG, offset),
            src: src.into(),
        }
    }

    #[inline]
    fn reload(dst: Self::Reg, slot: u32) -> Self {
        let offset = (slot as usize * size_of::<WordType>()) as i32;
        X64Inst::Mov {
            size: OperandSize::Qword,
            dst: dst.into(),
            src: RegMem::from_disp(STACK_REG, offset),
        }
    }
}

use crate::jit::backend::cpu_access::hreg_mem_offset;

#[derive(Default)]
pub struct X86CodeGen {
    slots: u32,
}

impl<'a> TargetBackend<'a, X86> for X86CodeGen {
    #[inline]
    fn new() -> Self {
        Self { slots: 0 }
    }

    fn reset(&mut self) {
        self.slots = 0;
    }

    fn config(&mut self) -> CodeGenConfig {
        // Exclude RSP (stack pointer) and RBP (CPU pointer).
        let alloc_mask =
            X86Reg::REG_MASK & !(1 << (X86Reg::RSP as u8)) & !(1 << (X86Reg::RBP as u8));
        CodeGenConfig::new(alloc_mask)
    }

    fn emit_prologue(&mut self, ctx: &mut CodeGenContext<X86>, slots: u32) {
        self.slots = slots;
        if slots > 0 {
            let aligned_slots = (slots + 1) & !1;
            let bytes = (aligned_slots as usize * size_of::<WordType>()) as u32;
            ctx.emit(&X64Inst::AluImm {
                op: AluOp::Sub,
                size: OperandSize::Qword,
                dst: STACK_REG.into(),
                imm: bytes,
            });
        }
    }

    fn emit_epilogue(&mut self, ctx: &mut CodeGenContext<X86>) {
        if self.slots > 0 {
            let aligned_slots = (self.slots + 1) & !1;
            let bytes = (aligned_slots as usize * size_of::<WordType>()) as u32;
            ctx.emit(&X64Inst::AluImm {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: STACK_REG.into(),
                imm: bytes,
            });
        }
        ctx.emit(&X64Inst::Ret);
    }

    fn codegen(&mut self, ctx: &mut CodeGenContext<X86>, insts: &[IRInst]) -> usize {
        match &insts[0] {
            IRInst::Const { dst, imm } => {
                let r_dst = ctx.def_vreg(*dst);
                ctx.emit(&X64Inst::MovImm {
                    size: OperandSize::Qword,
                    dst: r_dst.into(),
                    imm: *imm,
                });
                ctx.end_instruction();
            }
            IRInst::GetReg { dst, src } => {
                let r_dst = ctx.def_vreg(*dst);
                if src.index() == 0 {
                    ctx.emit(&X64Inst::MovImm {
                        size: OperandSize::Qword,
                        dst: r_dst.into(),
                        imm: 0,
                    });
                } else {
                    let offset = hreg_mem_offset(src.index() as u8) as i32;
                    ctx.emit(&X64Inst::Mov {
                        size: OperandSize::Qword,
                        dst: r_dst.into(),
                        src: RegMem::from_disp(CPU_REG, offset),
                    });
                }
                ctx.end_instruction();
            }
            IRInst::SetReg { dst, src } => {
                if dst.index() != 0 {
                    let r_src = ctx.use_vreg(*src);
                    let offset = hreg_mem_offset(dst.index() as u8) as i32;
                    ctx.emit(&X64Inst::Mov {
                        size: OperandSize::Qword,
                        dst: RegMem::from_disp(CPU_REG, offset),
                        src: r_src.into(),
                    });
                }
                ctx.end_instruction();
            }
            IRInst::Compute {
                kind,
                dst,
                src1,
                src2,
            } => {
                self.codegen_compute(ctx, *kind, *dst, *src1, *src2);
            }
            IRInst::Call {
                pass_cpu,
                func_ptr,
                args,
                ret,
                ..
            } => {
                self.codegen_call(ctx, *pass_cpu, *func_ptr, args, *ret);
            }
            IRInst::DirectJmp { target } => {
                let rax = ctx.alloc_scratch_fixed(X86Reg::RAX);
                ctx.emit(&X64Inst::MovImm {
                    size: OperandSize::Qword,
                    dst: rax.into(),
                    imm: *target,
                });
                self.emit_epilogue(ctx);
                ctx.end_instruction();
            }
            IRInst::DynamicJmp { target } => {
                let _ = ctx.use_vreg_fixed(*target, X86Reg::RAX);
                self.emit_epilogue(ctx);
                ctx.end_instruction();
            }
            IRInst::Branch {
                cmp,
                src1,
                src2,
                then_pc,
                else_pc,
            } => {
                self.codegen_branch(ctx, *cmp, *src1, *src2, *then_pc, *else_pc);
            }
        }
        1
    }
}

impl X86CodeGen {
    fn codegen_compute(
        &mut self,
        ctx: &mut CodeGenContext<X86>,
        kind: ComputeKind,
        dst: VReg,
        src1: VReg,
        src2: VReg,
    ) {
        match kind {
            ComputeKind::Add
            | ComputeKind::Sub
            | ComputeKind::And
            | ComputeKind::Or
            | ComputeKind::Xor
            | ComputeKind::AddW
            | ComputeKind::SubW => {
                let alu_op = match kind {
                    ComputeKind::Add | ComputeKind::AddW => AluOp::Add,
                    ComputeKind::Sub | ComputeKind::SubW => AluOp::Sub,
                    ComputeKind::And => AluOp::And,
                    ComputeKind::Or => AluOp::Or,
                    ComputeKind::Xor => AluOp::Xor,
                    _ => unreachable!(),
                };
                let is_32bit = matches!(kind, ComputeKind::AddW | ComputeKind::SubW);
                let size = if is_32bit {
                    OperandSize::Dword
                } else {
                    OperandSize::Qword
                };

                let r_src1 = ctx.use_vreg(src1);
                let r_src2 = ctx.use_vreg(src2);
                let r_dst = ctx.def_vreg(dst);

                if r_dst != r_src1 {
                    ctx.emit_mov(r_dst, r_src1);
                }
                ctx.emit(&X64Inst::Alu {
                    op: alu_op,
                    size,
                    dst: r_dst.into(),
                    src: r_src2,
                });
                if is_32bit {
                    ctx.emit(&X64Inst::Movsxd {
                        dst: r_dst,
                        src: r_dst.into(),
                    });
                }
                ctx.end_instruction();
            }
            ComputeKind::Shl
            | ComputeKind::Shr
            | ComputeKind::Sar
            | ComputeKind::ShlW
            | ComputeKind::ShrW
            | ComputeKind::SarW => {
                let shift_op = match kind {
                    ComputeKind::Shl | ComputeKind::ShlW => ShiftOp::Shl,
                    ComputeKind::Shr | ComputeKind::ShrW => ShiftOp::Shr,
                    ComputeKind::Sar | ComputeKind::SarW => ShiftOp::Sar,
                    _ => unreachable!(),
                };
                let is_32bit = matches!(
                    kind,
                    ComputeKind::ShlW | ComputeKind::ShrW | ComputeKind::SarW
                );
                let size = if is_32bit {
                    OperandSize::Dword
                } else {
                    OperandSize::Qword
                };

                let _ = ctx.use_vreg_fixed(src2, X86Reg::RCX);
                let r_src1 = ctx.use_vreg(src1);

                // If dst == src2 && src1 != src2, dst cannot be in RCX (which holds shift count).
                let r_dst = if dst == src2 && src1 != src2 {
                    let scratch = ctx.alloc_scratch();
                    ctx.def_vreg_fixed(dst, scratch)
                } else {
                    ctx.def_vreg(dst)
                };

                if r_dst != r_src1 {
                    ctx.emit_mov(r_dst, r_src1);
                }
                ctx.emit(&X64Inst::ShiftCl {
                    op: shift_op,
                    size,
                    dst: r_dst.into(),
                });
                if is_32bit {
                    ctx.emit(&X64Inst::Movsxd {
                        dst: r_dst,
                        src: r_dst.into(),
                    });
                }
                ctx.end_instruction();
            }
            ComputeKind::Slt | ComputeKind::Sltu => {
                let cond = if matches!(kind, ComputeKind::Slt) {
                    ConditionCode::Less
                } else {
                    ConditionCode::Below
                };
                let r_src1 = ctx.use_vreg(src1);
                let r_src2 = ctx.use_vreg(src2);

                ctx.emit(&X64Inst::Alu {
                    op: AluOp::Cmp,
                    size: OperandSize::Qword,
                    dst: r_src1.into(),
                    src: r_src2,
                });

                let r_dst = ctx.def_vreg(dst);
                ctx.emit(&X64Inst::Setcc {
                    condition: cond,
                    dst: r_dst,
                });
                ctx.emit(&X64Inst::Movzx8 {
                    dst: r_dst,
                    src: r_dst.into(),
                });
                ctx.end_instruction();
            }
        }
    }

    fn codegen_call(
        &mut self,
        ctx: &mut CodeGenContext<X86>,
        pass_cpu: bool,
        func_ptr: usize,
        args: &[VReg],
        ret: Option<VReg>,
    ) {
        #[cfg(not(target_family = "windows"))]
        const ARG_REGS: [X86Reg; 6] = [
            X86Reg::RDI,
            X86Reg::RSI,
            X86Reg::RDX,
            X86Reg::RCX,
            X86Reg::R8,
            X86Reg::R9,
        ];
        #[cfg(target_family = "windows")]
        const ARG_REGS: [X86Reg; 4] = [X86Reg::RCX, X86Reg::RDX, X86Reg::R8, X86Reg::R9];

        let total_args = args.len() + pass_cpu as usize;
        assert!(
            total_args <= ARG_REGS.len(),
            "too many arguments for C helper call"
        );

        let cpu_arg = pass_cpu.then_some((ARG_REGS[0], CallArg::Reg(CPU_REG)));
        let offset = pass_cpu as usize;
        let vreg_args = args
            .iter()
            .enumerate()
            .map(|(i, &arg)| (ARG_REGS[i + offset], CallArg::VReg(arg)));

        ctx.prepare_call_args(cpu_arg.into_iter().chain(vreg_args));

        let scratch = ctx.alloc_scratch_fixed(X86Reg::R11);
        ctx.emit(&X64Inst::MovImm {
            size: OperandSize::Qword,
            dst: scratch.into(),
            imm: func_ptr as u64,
        });
        ctx.emit(&X64Inst::Call { target: scratch });

        if let Some(ret_vreg) = ret {
            let _ = ctx.def_vreg_fixed(ret_vreg, X86Reg::RAX);
        }
        ctx.end_instruction();
    }

    fn codegen_branch(
        &mut self,
        ctx: &mut CodeGenContext<X86>,
        cmp: CmpKind,
        src1: VReg,
        src2: VReg,
        then_pc: u64,
        else_pc: u64,
    ) {
        if then_pc == else_pc {
            let rax = ctx.alloc_scratch_fixed(X86Reg::RAX);
            ctx.emit(&X64Inst::MovImm {
                size: OperandSize::Qword,
                dst: rax.into(),
                imm: then_pc,
            });
            self.emit_epilogue(ctx);
            ctx.end_instruction();
            return;
        }

        let condition = match cmp {
            CmpKind::Eq => ConditionCode::Equal,
            CmpKind::NotEq => ConditionCode::NotEqual,
            CmpKind::GtEq => ConditionCode::GreaterOrEqual,
            CmpKind::GtEqUnsigned => ConditionCode::AboveOrEqual,
            CmpKind::Lt => ConditionCode::Less,
            CmpKind::LtUnsigned => ConditionCode::Below,
        };

        let rax = ctx.alloc_scratch_fixed(X86Reg::RAX);
        let rcx = ctx.alloc_scratch_fixed(X86Reg::RCX);

        let r_src1 = ctx.use_vreg(src1);
        let r_src2 = ctx.use_vreg(src2);

        ctx.emit(&X64Inst::Alu {
            op: AluOp::Cmp,
            size: OperandSize::Qword,
            dst: r_src1.into(),
            src: r_src2,
        });

        ctx.emit(&X64Inst::MovImm {
            size: OperandSize::Qword,
            dst: rax.into(),
            imm: else_pc,
        });
        ctx.emit(&X64Inst::MovImm {
            size: OperandSize::Qword,
            dst: rcx.into(),
            imm: then_pc,
        });
        ctx.emit(&X64Inst::Cmovcc {
            condition,
            dst: rax,
            src: rcx.into(),
        });
        self.emit_epilogue(ctx);
        ctx.end_instruction();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::arch_config::WordType;
    use crate::device::mmio::MemoryMapIO;
    use crate::isa::riscv::decoder::Decoder;
    use crate::isa::riscv::executor::RVCPU;
    use crate::isa::riscv::mmu::VirtAddrManager;
    use crate::jit::backend::codegen::CodeGenTrait;
    use crate::jit::ir::*;
    use crate::jit::jit_buffer::JitBuffer;
    use crate::make_ir;
    use crate::ram::Ram;
    use std::cell::UnsafeCell;
    use std::rc::Rc;

    fn make_test_cpu() -> RVCPU {
        let ram = Rc::new(UnsafeCell::new(Ram::new()));
        let mmio = MemoryMapIO::from_ram(ram.clone());
        let memory = VirtAddrManager::from_ram_and_mmio(ram, mmio);
        let decoder = Decoder::from_isa_str("RV64GC").expect("test ISA must parse");
        RVCPU::from_decoder(decoder, memory)
    }

    fn compile_and_run(block: &IRBlock, cpu: &mut RVCPU) -> WordType {
        let mut codegen = CodeGen::<X86>::try_new().expect("X86 CodeGen must be supported");
        let code_block = codegen.compile(block);
        let mut jit_buf = JitBuffer::new();
        let func = unsafe { jit_buf.emit_code(code_block.buf) };
        unsafe { func.call(cpu as *mut RVCPU) }
    }

    #[test]
    fn test_x64_direct_jump() {
        let mut cpu = make_test_cpu();
        let builder = IRBuilder::default();
        let block = builder.build_jmp(0x1234_5678u64.into());

        let next_pc = compile_and_run(&block, &mut cpu);
        assert_eq!(next_pc, 0x1234_5678);
    }

    #[test]
    fn test_x64_dynamic_jump() {
        let mut cpu = make_test_cpu();
        let mut builder = IRBuilder::default();
        let v = builder.const_value(0x8765_4321);
        let block = builder.build_jmp(v);

        let next_pc = compile_and_run(&block, &mut cpu);
        assert_eq!(next_pc, 0x8765_4321);
    }

    #[test]
    fn test_x64_branch() {
        let mut cpu = make_test_cpu();

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
    fn test_x64_get_set_reg_and_compute() {
        let mut cpu = make_test_cpu();
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
        assert_eq!(cpu.reg_file[1], 40);
        assert_eq!(cpu.reg_file[2], 10);
        assert_eq!(cpu.reg_file[3], 60);
        assert_eq!(cpu.reg_file[4], 1);
    }

    #[test]
    fn test_x64_x0_semantics() {
        let mut cpu = make_test_cpu();
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
        assert_eq!(cpu.reg_file[0], 0);
        assert_eq!(cpu.reg_file[5], 0);
    }

    #[test]
    fn test_x64_call_helper() {
        unsafe extern "C" fn helper_add(a: u64, b: u64) -> u64 {
            a + b
        }

        unsafe extern "C" fn helper_with_cpu(cpu: *mut RVCPU, val: u64) -> u64 {
            unsafe { (&mut (*cpu).reg_file)[7] = val };
            val * 2
        }

        let mut cpu = make_test_cpu();

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
        assert_eq!(cpu.reg_file[6], 300);

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
        assert_eq!(cpu.reg_file[7], 77);
        assert_eq!(cpu.reg_file[8], 154);
    }

    #[test]
    fn test_x64_compute_32bit() {
        let mut cpu = make_test_cpu();
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
        assert_eq!(cpu.reg_file[10], 0xffff_ffff_8000_0000);
        assert_eq!(cpu.reg_file[11], 0xffff_ffff_ffff_ffff);
        assert_eq!(cpu.reg_file[12], 0xffff_ffff_ffff_fffe);
    }

    #[test]
    fn test_x64_branch_unsigned() {
        let mut cpu = make_test_cpu();
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
    fn test_x64_register_spill() {
        let mut cpu = make_test_cpu();
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
        assert_eq!(cpu.reg_file[1], 210);
    }

    #[test]
    fn test_x64_callee_save_across_call() {
        assert_eq!(X86Reg::CALLEE_SAVE_MASK & X86Reg::CALLER_SAVE_MASK, 0);
        assert_eq!(
            X86Reg::CALLEE_SAVE_MASK | X86Reg::CALLER_SAVE_MASK,
            X86Reg::REG_MASK
        );

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

        let mut cpu = make_test_cpu();
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
        assert_eq!(cpu.reg_file[1], 0x1234_5678_9abc_def1);
    }
}
