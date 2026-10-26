use here_types::WordType;

use super::runtime::JitContext;
use super::*;

const CPU_REG: X64Reg = X64Reg::RBP;
const STACK_REG: X64Reg = X64Reg::RSP;
const CTX_REG: X64Reg = X64Reg::RBX;

#[inline]
fn spill_slot_mem(slot: u32) -> RegMem {
    let offset = (slot as usize * size_of::<WordType>()) as i32;
    RegMem::from_disp(STACK_REG, offset)
}

#[inline]
fn guest_reg_mem(reg: GuestReg) -> RegMem {
    let offset = abi::guest_reg_offset(reg.index() as u8) as i32;
    RegMem::from_disp(CPU_REG, offset)
}

impl TargetInst for X64Inst {
    type Reg = X64Reg;

    #[inline]
    fn emit_to(&self, buf: &mut CodeBuf) {
        self.emit(buf);
    }

    #[inline]
    fn mov(dst: Self::Reg, src: Self::Reg) -> Self {
        // TODO: we can't omit mov when `dst == src` here, because this function returns exactly one inst.
        X64Inst::mov_rr(OperandSize::Qword, dst, src)
    }

    #[inline]
    fn mov_imm(dst: Self::Reg, imm: u64) -> Self {
        X64Inst::mov_reg_imm(dst, imm)
    }

    #[inline]
    fn spill(slot: u32, src: Self::Reg) -> Self {
        X64Inst::mov_rm_reg(OperandSize::Qword, spill_slot_mem(slot), src)
    }

    #[inline]
    fn reload(dst: Self::Reg, slot: u32) -> Self {
        X64Inst::mov_reg_rm(OperandSize::Qword, dst, spill_slot_mem(slot))
    }
    fn render_asm(&self, out: &mut String, patch: Option<AsmPatch>) {
        super::asm_text::render_inst(self, out, patch);
    }
}

/// Byte size of the spill frame, rounded to an odd number of slots to keep
/// `rsp` 16-byte aligned before helper calls, satisfying the System V AMD64 ABI.
#[inline]
fn stack_frame_bytes(slots: u32) -> u32 {
    let aligned_slots = slots | 1;
    (aligned_slots as usize * size_of::<WordType>()) as u32
}

#[derive(Default)]
pub struct X64CodeGen;

impl CodeGenBackend<X64> for X64CodeGen {
    fn reg_alloc_config() -> RegAllocConfig<X64Reg> {
        const RESERVED: &[X64Reg] = &[STACK_REG];
        const FIXED: &[(VReg, X64Reg)] = &[
            (VRegAllocator::CPU_VREG, CPU_REG),
            (VRegAllocator::JIT_CONTEXT_VREG, CTX_REG),
        ];
        RegAllocConfig {
            reserved: RESERVED,
            fixed: FIXED,
        }
    }

    fn emit_stack_alloc(ctx: &mut CodeGenContext<X64>) -> PatchLoc<X64PatchKind> {
        let offset = ctx.buf().len();
        // Forced imm32 form to patch.
        ctx.emit(&X64Inst::alu_imm32(
            AluOp::Sub,
            OperandSize::Qword,
            STACK_REG,
            0,
        ));
        PatchLoc::new(offset, X64PatchKind::SubImm32)
    }

    fn patch_stack_alloc(ctx: &mut CodeGenContext<X64>, loc: PatchLoc<X64PatchKind>, slots: u32) {
        let bytes = stack_frame_bytes(slots);
        unsafe {
            loc.patch(ctx.buf().as_mut_ptr(), bytes as i64);
        }
        ctx.record_patch(loc.offset, AsmPatch::Imm(bytes as i64));
    }

    fn emit_stack_free(ctx: &mut CodeGenContext<X64>) {
        let bytes = stack_frame_bytes(ctx.current_use_slot());
        if bytes > 0 {
            let imm = i32::try_from(bytes).unwrap();
            ctx.emit(&X64Inst::alu_imm(
                AluOp::Add,
                OperandSize::Qword,
                STACK_REG,
                imm,
            ));
        }
    }

    fn emit_block_exit(ctx: &mut CodeGenContext<X64>, next_pc: WordType) -> usize {
        let offset = ctx.buf().len();
        ctx.emit(&X64Inst::mov_reg_imm(X64Reg::RAX, next_pc as u64));
        ctx.emit(&X64Inst::Ret);
        offset
    }

    fn emit_dynamic_jmp(ctx: &mut CodeGenContext<X64>, hreg: X64Reg) {
        ctx.emit_mov(X64Reg::RAX, hreg);
        ctx.emit(&X64Inst::Ret);
    }

    fn emit_static_jmp(ctx: &mut CodeGenContext<X64>) -> PatchLoc<X64PatchKind> {
        let offset = ctx.buf().len();
        ctx.emit(&X64Inst::JmpRel { offset: 0 });
        PatchLoc::new(offset, X64PatchKind::JmpRel32)
    }

    fn emit_branch(
        ctx: &mut CodeGenContext<X64>,
        cmp: CmpKind,
        src1: X64Reg,
        src2: X64Reg,
    ) -> (PatchLoc<X64PatchKind>, PatchLoc<X64PatchKind>) {
        let condition = cmp.into();
        ctx.emit(&X64Inst::alu_rm_reg(
            AluOp::Cmp,
            OperandSize::Qword,
            src1,
            src2,
        ));
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

    fn emit_check_budget(ctx: &mut CodeGenContext<X64>, guest_cnt: u32) -> PatchLoc<X64PatchKind> {
        let imm = i32::try_from(guest_cnt).unwrap();
        ctx.emit(&X64Inst::alu_imm(
            AluOp::Cmp,
            OperandSize::Qword,
            RegMem::from_disp(CTX_REG, JitContext::BUDGET_OFFSET),
            imm,
        ));
        let offset = ctx.buf().len();
        ctx.emit(&X64Inst::Jcc {
            condition: ConditionCode::Below,
            offset: 0,
        });
        PatchLoc::new(offset, X64PatchKind::JccRel32)
    }

    fn emit_update_budget(ctx: &mut CodeGenContext<X64>, guest_cnt: u32) {
        let imm = i32::try_from(guest_cnt).unwrap();
        ctx.emit(&X64Inst::alu_imm(
            AluOp::Sub,
            OperandSize::Qword,
            RegMem::from_disp(CTX_REG, JitContext::BUDGET_OFFSET),
            imm,
        ));
    }

    fn codegen(ctx: &mut CodeGenContext<X64>, insts: &[IRInst]) -> usize {
        match &insts[0] {
            IRInst::Const { dst, imm } => {
                ctx.def_const(*dst, *imm);
            }
            IRInst::GetReg { dst, src } => {
                let r_dst = ctx.def_vreg(*dst);
                ctx.emit(&X64Inst::mov_reg_rm(
                    OperandSize::Qword,
                    r_dst,
                    guest_reg_mem(*src),
                ));
            }
            IRInst::SetReg { dst, src } => {
                let mem = guest_reg_mem(*dst);
                if let Some(imm) = ctx.try_get_const_i32(*src) {
                    ctx.emit(&X64Inst::mov_rm_imm(OperandSize::Qword, mem, imm));
                } else {
                    let r_src = ctx.use_vreg(*src);
                    ctx.emit(&X64Inst::mov_rm_reg(OperandSize::Qword, mem, r_src));
                }
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
                func_ptr,
                args,
                ret,
                ..
            } => {
                Self::codegen_call(ctx, *func_ptr, args, *ret);
            }
        }
        1
    }
}

impl X64CodeGen {
    fn codegen_compute(
        ctx: &mut CodeGenContext<X64>,
        kind: ComputeKind,
        dst: VReg,
        src1: VReg,
        src2: VReg,
    ) {
        if let Ok(alu_op) = AluOp::try_from(kind) {
            Self::codegen_alu(ctx, alu_op, kind, dst, src1, src2);
        } else if let Ok(shift_op) = ShiftOp::try_from(kind) {
            Self::codegen_shift(ctx, shift_op, kind, dst, src1, src2);
        } else if matches!(kind, ComputeKind::Slt | ComputeKind::Sltu) {
            Self::codegen_slt(ctx, kind, dst, src1, src2);
        } else {
            panic!("unsupported compute kind: {kind:?}");
        }
    }

    fn codegen_alu(
        ctx: &mut CodeGenContext<X64>,
        op: AluOp,
        kind: ComputeKind,
        dst: VReg,
        src1: VReg,
        src2: VReg,
    ) {
        let is_32bit = kind.is_32bit();
        let size = if is_32bit {
            OperandSize::Dword
        } else {
            OperandSize::Qword
        };

        let (src1, src2) = if kind.commutative()
            && ctx.try_get_const_i32(src2).is_none()
            && ctx.try_get_const_i32(src1).is_some()
        {
            (src2, src1)
        } else {
            (src1, src2)
        };

        let r_src1 = ctx.use_vreg(src1);
        let imm = ctx.try_get_const_i32(src2);
        let r_src2 = if imm.is_none() {
            Some(ctx.use_vreg(src2))
        } else {
            None
        };

        // TODO(perf): we can reuse dst when `dst == src1` if `src1` is on its last use,
        // provide that feature on the CodeGenContext.
        let r_dst = ctx.def_vreg(dst);
        ctx.emit_mov(r_dst, r_src1);

        if let Some(imm) = imm {
            ctx.emit(&X64Inst::alu_imm(op, size, r_dst, imm));
        } else {
            ctx.emit(&X64Inst::alu_rm_reg(op, size, r_dst, r_src2.unwrap()));
        }

        if is_32bit {
            ctx.emit(&X64Inst::Movsxd {
                dst: r_dst,
                src: r_dst.into(),
            });
        }
    }

    fn codegen_shift(
        ctx: &mut CodeGenContext<X64>,
        op: ShiftOp,
        kind: ComputeKind,
        dst: VReg,
        src1: VReg,
        src2: VReg,
    ) {
        let is_32bit = kind.is_32bit();
        let size = if is_32bit {
            OperandSize::Dword
        } else {
            OperandSize::Qword
        };

        let imm_shift = ctx
            .try_get_const(src2)
            .map(|imm| ShiftCount::new(size, imm));
        if imm_shift.is_none() {
            ctx.use_vreg_fixed(src2, X64Reg::RCX);
        }
        let r_src1 = ctx.use_vreg(src1);
        let r_dst = ctx.def_vreg(dst);

        ctx.emit_mov(r_dst, r_src1);

        if let Some(shift_count) = imm_shift {
            ctx.emit(&X64Inst::shift_imm(op, size, r_dst, shift_count));
        } else {
            ctx.emit(&X64Inst::shift_cl(op, size, r_dst));
        }

        if is_32bit {
            ctx.emit(&X64Inst::Movsxd {
                dst: r_dst,
                src: r_dst.into(),
            });
        }
    }

    fn codegen_slt(
        ctx: &mut CodeGenContext<X64>,
        kind: ComputeKind,
        dst: VReg,
        src1: VReg,
        src2: VReg,
    ) {
        let cond = match kind {
            ComputeKind::Slt => ConditionCode::Less,
            ComputeKind::Sltu => ConditionCode::Below,
            _ => unreachable!(),
        };
        let r_src1 = ctx.use_vreg(src1);
        if let Some(imm) = ctx.try_get_const_i32(src2) {
            ctx.emit(&X64Inst::alu_imm(
                AluOp::Cmp,
                OperandSize::Qword,
                r_src1,
                imm,
            ));
        } else {
            let r_src2 = ctx.use_vreg(src2);
            ctx.emit(&X64Inst::alu_rm_reg(
                AluOp::Cmp,
                OperandSize::Qword,
                r_src1,
                r_src2,
            ));
        }

        ctx.clear_reg_pins();

        let r_dst = ctx.def_vreg(dst);
        ctx.emit(&X64Inst::Setcc {
            condition: cond,
            dst: r_dst,
        });
        ctx.emit(&X64Inst::Movzx8 {
            dst: r_dst,
            src: r_dst.into(),
        });
    }

    fn codegen_call(
        ctx: &mut CodeGenContext<X64>,
        func_ptr: usize,
        args: &[VReg],
        ret: Option<VReg>,
    ) {
        // Helper calls always use the SysV x86-64 ABI, regardless of host OS
        // (see `jit::abi::helpers`). At most 6 integer/pointer arguments fit in
        // registers; stack arguments are not supported.
        const ARG_REGS: [X64Reg; 6] = [
            X64Reg::RDI,
            X64Reg::RSI,
            X64Reg::RDX,
            X64Reg::RCX,
            X64Reg::R8,
            X64Reg::R9,
        ];

        let total_args = args.len();
        assert!(
            total_args <= ARG_REGS.len(),
            "too many arguments for a JIT helper call"
        );

        let vreg_args = ARG_REGS.into_iter().zip(args.iter().copied());

        ctx.prepare_call_args(vreg_args);
        let scratch = ctx.alloc_scratch_fixed(X64Reg::R11);
        ctx.emit(&X64Inst::mov_reg_imm(scratch, func_ptr as u64));
        ctx.emit(&X64Inst::Call { target: scratch });
        if let Some(ret_vreg) = ret {
            let _ = ctx.def_vreg_fixed(ret_vreg, X64Reg::RAX);
        }
    }
}
