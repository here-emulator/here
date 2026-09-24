use crate::config::arch_config::WordType;

use super::*;

const CPU_REG: X86Reg = X86Reg::RBP;
const STACK_REG: X86Reg = X86Reg::RSP;
const CTX_REG: X86Reg = X86Reg::RBX;
const BUDGET_OFFSET: i32 = std::mem::offset_of!(crate::jit::runtime::JitContext, budget) as i32;

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
    fn mov_imm(dst: Self::Reg, imm: u64) -> Self {
        X64Inst::MovImm {
            size: OperandSize::Qword,
            dst: dst.into(),
            imm,
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
pub struct X86CodeGen;

impl CodeGenBackend<X86> for X86CodeGen {
    fn config() -> CodeGenConfig {
        // Exclude RSP (stack pointer), RBP (CPU pointer), and RBX (JitContext pointer).
        let alloc_mask = X86Reg::REG_MASK
            & !(1 << (X86Reg::RSP as u8))
            & !(1 << (X86Reg::RBP as u8))
            & !(1 << (X86Reg::RBX as u8));
        CodeGenConfig::new(alloc_mask)
    }

    fn emit_stack_alloc(ctx: &mut CodeGenContext<X86>) -> PatchLoc<X64PatchKind> {
        let offset = ctx.buf().len();
        ctx.emit(&X64Inst::AluImm {
            op: AluOp::Sub,
            size: OperandSize::Qword,
            dst: STACK_REG.into(),
            imm: 0,
        });
        let emitted_len = ctx.buf().len() - offset;
        if emitted_len < 7 {
            ctx.emit_zeros(7 - emitted_len);
        }
        PatchLoc::new(offset, X64PatchKind::SubImm32)
    }

    fn patch_stack_alloc(
        buf: &mut CodeBuf,
        asm_buf: Option<&mut String>,
        loc: PatchLoc<X64PatchKind>,
        slots: u32,
    ) {
        let aligned_slots = (slots + 1) & !1;
        let bytes = (aligned_slots as usize * size_of::<WordType>()) as u32;
        unsafe {
            loc.patch(buf.as_mut_ptr(), bytes as i64);
        }
        if let Some(asm) = asm_buf {
            if let Some(pos) = asm.find("sub rsp,") {
                let end = asm[pos..].find('\n').map(|p| pos + p).unwrap_or(asm.len());
                asm.replace_range(pos..end, &format!("sub rsp, {bytes}"));
            }
        }
    }

    fn emit_stack_free(ctx: &mut CodeGenContext<X86>) {
        let slots = ctx.current_use_slot();
        if slots > 0 {
            let aligned_slots = (slots + 1) & !1;
            let bytes = (aligned_slots as usize * size_of::<WordType>()) as u32;
            ctx.emit(&X64Inst::AluImm {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: STACK_REG.into(),
                imm: bytes,
            });
        }
    }

    fn emit_block_exit(ctx: &mut CodeGenContext<X86>, next_pc: WordType) -> usize {
        let offset = ctx.buf().len();
        ctx.emit(&X64Inst::MovImm {
            size: OperandSize::Qword,
            dst: X86Reg::RAX.into(),
            imm: next_pc as u64,
        });
        ctx.emit(&X64Inst::Ret);
        offset
    }

    fn emit_dynamic_jmp(ctx: &mut CodeGenContext<X86>, hreg: X86Reg) {
        if hreg != X86Reg::RAX {
            ctx.emit(&X64Inst::Mov {
                size: OperandSize::Qword,
                dst: X86Reg::RAX.into(),
                src: hreg.into(),
            });
        }
        ctx.emit(&X64Inst::Ret);
    }

    fn emit_static_jmp(ctx: &mut CodeGenContext<X86>) -> PatchLoc<X64PatchKind> {
        let offset = ctx.buf().len();
        ctx.emit(&X64Inst::JmpRel { offset: 0 });
        PatchLoc::new(offset, X64PatchKind::JmpRel32)
    }

    fn emit_branch(
        ctx: &mut CodeGenContext<X86>,
        cmp: CmpKind,
        src1: X86Reg,
        src2: X86Reg,
    ) -> (PatchLoc<X64PatchKind>, PatchLoc<X64PatchKind>) {
        let condition = match cmp {
            CmpKind::Eq => ConditionCode::Equal,
            CmpKind::NotEq => ConditionCode::NotEqual,
            CmpKind::GtEq => ConditionCode::GreaterOrEqual,
            CmpKind::GtEqUnsigned => ConditionCode::AboveOrEqual,
            CmpKind::Lt => ConditionCode::Less,
            CmpKind::LtUnsigned => ConditionCode::Below,
        };
        ctx.emit(&X64Inst::Alu {
            op: AluOp::Cmp,
            size: OperandSize::Qword,
            dst: src1.into(),
            src: src2,
        });
        let then_loc = ctx.buf().len();
        ctx.emit(&X64Inst::Jcc {
            condition,
            offset: 0,
        });
        let else_loc = ctx.buf().len();
        ctx.emit(&X64Inst::JmpRel { offset: 0 });
        (
            PatchLoc::new(then_loc, X64PatchKind::JccRel32),
            PatchLoc::new(else_loc, X64PatchKind::JmpRel32),
        )
    }

    unsafe fn patch(buf: *mut u8, loc: PatchLoc<X64PatchKind>, imm: i64) {
        unsafe {
            loc.patch(buf, imm);
        }
    }

    fn emit_check_budget(
        ctx: &mut CodeGenContext<X86>,
        block_insts: u32,
    ) -> PatchLoc<X64PatchKind> {
        ctx.emit(&X64Inst::AluImm {
            op: AluOp::Cmp,
            size: OperandSize::Qword,
            dst: RegMem::from_disp(CTX_REG, BUDGET_OFFSET),
            imm: block_insts,
        });
        let offset = ctx.buf().len();
        ctx.emit(&X64Inst::Jcc {
            condition: ConditionCode::Below,
            offset: 0,
        });
        PatchLoc::new(offset, X64PatchKind::JccRel32)
    }

    fn emit_update_budget(ctx: &mut CodeGenContext<X86>, block_insts: u32) {
        ctx.emit(&X64Inst::AluImm {
            op: AluOp::Sub,
            size: OperandSize::Qword,
            dst: RegMem::from_disp(CTX_REG, BUDGET_OFFSET),
            imm: block_insts,
        });
    }

    fn codegen(ctx: &mut CodeGenContext<X86>, insts: &[IRInst]) -> usize {
        match &insts[0] {
            IRInst::Const { dst, imm } => {
                ctx.def_const(*dst, *imm);
            }
            IRInst::GetReg { dst, src } => {
                if src.index() == 0 {
                    ctx.def_const(*dst, 0);
                } else {
                    let r_dst = ctx.def_vreg(*dst);
                    let offset = hreg_mem_offset(src.index() as u8) as i32;
                    ctx.emit(&X64Inst::Mov {
                        size: OperandSize::Qword,
                        dst: r_dst.into(),
                        src: RegMem::from_disp(CPU_REG, offset),
                    });
                    ctx.end_instruction();
                }
            }
            IRInst::SetReg { dst, src } => {
                if dst.index() != 0 {
                    let offset = hreg_mem_offset(dst.index() as u8) as i32;
                    if let Some(imm) = ctx.try_get_const_i32(*src) {
                        ctx.emit(&X64Inst::MovImm {
                            size: OperandSize::Qword,
                            dst: RegMem::from_disp(CPU_REG, offset),
                            imm: imm as u64,
                        });
                    } else {
                        let r_src = ctx.use_vreg(*src);
                        ctx.emit(&X64Inst::Mov {
                            size: OperandSize::Qword,
                            dst: RegMem::from_disp(CPU_REG, offset),
                            src: r_src.into(),
                        });
                    }
                }
                ctx.end_instruction();
            }
            IRInst::Compute {
                kind,
                dst,
                src1,
                src2,
            } => {
                Self::codegen_compute(ctx, *kind, *dst, *src1, *src2);
            }
            IRInst::Call {
                pass_cpu,
                func_ptr,
                args,
                ret,
                ..
            } => {
                Self::codegen_call(ctx, *pass_cpu, *func_ptr, args, *ret);
            }
            IRInst::Terminator(_) => crate::debug_unreachable!(),
        }
        1
    }
}

impl X86CodeGen {
    fn codegen_compute(
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

                let is_commutative = matches!(
                    kind,
                    ComputeKind::Add
                        | ComputeKind::And
                        | ComputeKind::Or
                        | ComputeKind::Xor
                        | ComputeKind::AddW
                );
                let (src1, src2) = if is_commutative
                    && ctx.try_get_const_i32(src2).is_none()
                    && ctx.try_get_const_i32(src1).is_some()
                {
                    (src2, src1)
                } else {
                    (src1, src2)
                };

                if let Some(imm) = ctx.try_get_const_i32(src2) {
                    let r_src1 = ctx.use_vreg(src1);
                    let r_dst = ctx.def_vreg(dst);

                    if r_dst != r_src1 {
                        ctx.emit_mov(r_dst, r_src1);
                    }
                    ctx.emit(&X64Inst::AluImm {
                        op: alu_op,
                        size,
                        dst: r_dst.into(),
                        imm: imm as u32,
                    });
                    if is_32bit {
                        ctx.emit(&X64Inst::Movsxd {
                            dst: r_dst,
                            src: r_dst.into(),
                        });
                    }
                    ctx.end_instruction();
                } else {
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

                if let Some(imm) = ctx.try_get_const(src2) {
                    let mask = if is_32bit { 0x1f } else { 0x3f };
                    let shift_imm = (imm as u8) & mask;
                    let r_src1 = ctx.use_vreg(src1);
                    let r_dst = ctx.def_vreg(dst);

                    if r_dst != r_src1 {
                        ctx.emit_mov(r_dst, r_src1);
                    }
                    ctx.emit(&X64Inst::ShiftImm {
                        op: shift_op,
                        size,
                        dst: r_dst.into(),
                        imm: shift_imm,
                    });
                    if is_32bit {
                        ctx.emit(&X64Inst::Movsxd {
                            dst: r_dst,
                            src: r_dst.into(),
                        });
                    }
                    ctx.end_instruction();
                } else {
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
            }
            ComputeKind::Slt | ComputeKind::Sltu => {
                let cond = if matches!(kind, ComputeKind::Slt) {
                    ConditionCode::Less
                } else {
                    ConditionCode::Below
                };
                let r_src1 = ctx.use_vreg(src1);
                if let Some(imm) = ctx.try_get_const_i32(src2) {
                    ctx.emit(&X64Inst::AluImm {
                        op: AluOp::Cmp,
                        size: OperandSize::Qword,
                        dst: r_src1.into(),
                        imm: imm as u32,
                    });
                } else {
                    let r_src2 = ctx.use_vreg(src2);
                    ctx.emit(&X64Inst::Alu {
                        op: AluOp::Cmp,
                        size: OperandSize::Qword,
                        dst: r_src1.into(),
                        src: r_src2,
                    });
                }

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
}
