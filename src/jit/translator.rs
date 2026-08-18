use super::{
    backend::x86::*,
    helpers::{LoadKind, StoreKind, check_instruction_alignment_helper},
    jit_function::{JitContext, JitInfo},
};
use crate::{
    config::arch_config::WordType,
    isa::riscv::{
        decoder::DecodeInstr,
        executor::RVCPU,
        instruction::{RVInstrInfo, instr_table::RiscvInstr},
    },
};

mod reg_assign;
mod rvc;
mod rvi;

use reg_assign::*;

use X86Reg::*;

const CPU_REG: X86Reg = RBP;
const GUEST_CALLEE_REGS: &[X86Reg] = &[RBX, R12, R13, R14, R15];
const GUEST_CALLER_REGS: &[X86Reg] = &[RAX, R10, R11];
const GUEST_ASSIGN_REGS: &[X86Reg] = &[RDI, RSI, RDX, RCX, R8, R9, RBP, RSP];

#[inline]
fn reg_on_mem(guest: u8) -> RegMem {
    let offset = std::mem::offset_of!(RVCPU, reg_file) + guest as usize * size_of::<WordType>();
    let offset =
        i32::try_from(offset).expect("RVCPU register offset must fit in an x86 displacement");
    RegMem::from_disp(CPU_REG, offset)
}

#[derive(Clone, Copy)]
enum OperandWidth {
    Xlen,
    Word,
}

/// In our convention:
///
/// - [`CPU_REG`]: `*mut RVCPU`
/// - [`GUEST_CACHE_REGS`]: cached guest registers and [`RegAssign`]-managed scratch values
/// - caller-saved registers: helper arguments, results, and untracked temporary values (be careful!)
pub(super) struct X86CodeGen {
    asm: X86Assembler,
    regs: RegAssign,
    jit_context: *mut JitContext,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TranslateResult {
    Continue,
    ControlFlow,
    Unsupported,
}

impl X86CodeGen {
    pub fn new(jit_context: *mut JitContext) -> Self {
        Self {
            asm: X86Assembler::new(),
            regs: RegAssign::new(),
            jit_context,
        }
    }

    #[must_use]
    pub fn translate(&mut self, decoded: DecodeInstr, context: JitInfo) -> TranslateResult {
        self.translate_instruction(decoded.instr, decoded.info, context)
    }

    pub fn build(mut self, seq_next_pc: Option<WordType>) -> Vec<u8> {
        self.regs.flush(&mut self.asm);
        if let Some(next_pc) = seq_next_pc {
            let next_pc_reg = self.regs.host(RAX, &mut self.asm);
            self.asm.mov_r64_imm64(*next_pc_reg, next_pc as u64);
        }
        self.asm.ret();
        self.asm.build()
    }

    fn translate_alu_r(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        op: AluOp,
        commutative: bool,
        width: OperandWidth,
    ) {
        if rd == 0 {
            return;
        }

        if rd == rs1 {
            let rs2 = self.regs.guest_read(rs2, &mut self.asm);
            let rd = self.regs.guest_read_write(rd, &mut self.asm);
            emit_alu_r(&mut self.asm, op, *rd, *rs2, width);
        } else if commutative && rd == rs2 {
            let rs1 = self.regs.guest_read(rs1, &mut self.asm);
            let rd = self.regs.guest_read_write(rd, &mut self.asm);
            emit_alu_r(&mut self.asm, op, *rd, *rs1, width);
        } else if rd == rs2 {
            let rs1 = self.regs.guest_read(rs1, &mut self.asm);
            let rd = self.regs.guest_read_write(rd, &mut self.asm);
            let scratch = self.regs.scratch(&mut self.asm);
            self.asm.mov_rm64_r64(*scratch, *rs1);
            emit_alu_r(&mut self.asm, op, *scratch, *rd, width);
            self.asm.mov_rm64_r64(*rd, *scratch);
        } else {
            let rs1 = self.regs.guest_read(rs1, &mut self.asm);
            let rs2 = self.regs.guest_read(rs2, &mut self.asm);
            let rd = self.regs.guest_write(rd, &mut self.asm);
            self.asm.mov_rm64_r64(*rd, *rs1);
            emit_alu_r(&mut self.asm, op, *rd, *rs2, width);
        }
    }

    fn translate_alu_i(&mut self, rd: u8, rs1: u8, imm: WordType, op: AluOp, width: OperandWidth) {
        if rd == 0 {
            return;
        }

        let rs1 = self.regs.guest_read(rs1, &mut self.asm);
        let rd = self.regs.guest_write(rd, &mut self.asm);
        if rd.reg() != rs1.reg() {
            self.asm.mov_rm64_r64(*rd, *rs1);
        }
        emit_alu_i(&mut self.asm, op, *rd, imm as u32, width);
    }

    fn translate_shift_r(&mut self, rd: u8, rs1: u8, rs2: u8, op: ShiftOp, width: OperandWidth) {
        if rd == 0 {
            return;
        }

        let rs1 = self.regs.guest_read(rs1, &mut self.asm);
        let rs2 = self.regs.guest_read(rs2, &mut self.asm);
        let shift_count = self.regs.host(RCX, &mut self.asm);
        self.asm.mov_r64_rm64(*shift_count, *rs2);

        let rd = self.regs.guest_write(rd, &mut self.asm);
        if rd.reg() != rs1.reg() {
            self.asm.mov_rm64_r64(*rd, *rs1);
        }
        self.asm.shift_rm_cl(width.operand_size(), op, (*rd).into());
        sext_word_on_need(&mut self.asm, *rd, width);
    }

    fn translate_shift_i(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: WordType,
        op: ShiftOp,
        width: OperandWidth,
    ) {
        if rd == 0 {
            return;
        }

        let rs1 = self.regs.guest_read(rs1, &mut self.asm);
        let rd = self.regs.guest_write(rd, &mut self.asm);
        if rd.reg() != rs1.reg() {
            self.asm.mov_rm64_r64(*rd, *rs1);
        }
        self.asm
            .shift_rm_imm8(width.operand_size(), op, (*rd).into(), imm as u8);
        sext_word_on_need(&mut self.asm, *rd, width);
    }

    fn translate_compare_r(&mut self, rd: u8, rs1: u8, rs2: u8, condition: ConditionCode) {
        if rd == 0 {
            return;
        }

        let rs1 = self.regs.guest_read(rs1, &mut self.asm);
        let rs2 = self.regs.guest_read(rs2, &mut self.asm);
        let rd = self.regs.guest_write(rd, &mut self.asm);
        self.asm.alu_rm64_r64(AluOp::Cmp, *rs1, *rs2);
        self.asm.setcc_r8(condition, *rd);
        self.asm.movzx_r32_r8(*rd, *rd);
    }

    fn translate_compare_i(&mut self, rd: u8, rs1: u8, imm: WordType, condition: ConditionCode) {
        if rd == 0 {
            return;
        }

        let rs1 = self.regs.guest_read(rs1, &mut self.asm);
        let rd = self.regs.guest_write(rd, &mut self.asm);
        self.asm.alu_rm64_imm32(AluOp::Cmp, *rs1, imm as u32);
        self.asm.setcc_r8(condition, *rd);
        self.asm.movzx_r32_r8(*rd, *rd);
    }

    fn translate_move(&mut self, rd: u8, rs: u8) {
        if rd == 0 {
            return;
        }

        let rs = self.regs.guest_read(rs, &mut self.asm);
        let rd = self.regs.guest_write(rd, &mut self.asm);
        if rd.reg() != rs.reg() {
            self.asm.mov_rm64_r64(*rd, *rs);
        }
    }

    fn translate_load_imm(&mut self, rd: u8, value: WordType) {
        if rd == 0 {
            return;
        }

        let rd = self.regs.guest_write(rd, &mut self.asm);
        self.asm.mov_r64_imm64(*rd, value as u64);
    }

    fn translate_load(&mut self, rd: u8, rs1: u8, imm: WordType, kind: LoadKind, context: JitInfo) {
        let addr_reg = {
            // Keep rs1 cached while computing the effective address.
            let addr = self.regs.scratch(&mut self.asm);
            let base = self.regs.guest_read(rs1, &mut self.asm);
            self.asm.mov_r64_rm64(*addr, *base);
            self.asm.alu_rm64_imm32(AluOp::Add, *addr, imm as u32);
            addr.reg()
        };

        self.regs.flush(&mut self.asm);
        self.regs.flush_for_call(&mut self.asm);

        self.asm.call_helper(
            kind.helper(),
            &[
                CallArg::Reg(CPU_REG),
                CallArg::Reg(addr_reg),
                CallArg::Imm(self.jit_context as u64),
                CallArg::Imm(context.instr_pc as u64),
                CallArg::Imm(context.instr_cnt),
            ],
        );

        if rd == 0 {
            return;
        }

        let value_reg = {
            let value = self.regs.host(RAX, &mut self.asm);
            match kind {
                LoadKind::SignedByte => self.asm.movsx_r64_rm8(*value, (*value).into()),
                LoadKind::UnsignedByte => self.asm.movzx_r32_r8(*value, *value),
                LoadKind::SignedHalf => self.asm.movsx_r64_rm16(*value, (*value).into()),
                LoadKind::UnsignedHalf => self.asm.movzx_r32_rm16(*value, (*value).into()),
                LoadKind::SignedWord => self.asm.movsxd_r64_rm32(*value, (*value).into()),
                LoadKind::UnsignedWord => self.asm.mov_r32_rm32(*value, *value),
                LoadKind::DoubleWord => {}
            }
            value.reg()
        };

        let rd = self.regs.guest_write(rd, &mut self.asm);
        if rd.reg() != value_reg {
            self.asm.mov_rm64_r64(*rd, value_reg);
        }
    }

    fn translate_store(
        &mut self,
        rs1: u8,
        rs2: u8,
        imm: WordType,
        kind: StoreKind,
        context: JitInfo,
    ) {
        let (addr_reg, value_reg) = {
            let addr = self.regs.scratch(&mut self.asm);
            let base = self.regs.guest_read(rs1, &mut self.asm);
            self.asm.mov_r64_rm64(*addr, *base);
            self.asm.alu_rm64_imm32(AluOp::Add, *addr, imm as u32);
            let value = self.regs.guest_read(rs2, &mut self.asm);
            (addr.reg(), value.reg())
        };

        self.regs.flush(&mut self.asm);
        self.regs.flush_for_call(&mut self.asm);
        self.asm.call_helper(
            kind.helper(),
            &[
                CallArg::Reg(CPU_REG),
                CallArg::Reg(addr_reg),
                CallArg::Reg(value_reg),
                CallArg::Imm(self.jit_context as u64),
                CallArg::Imm(context.instr_pc as u64),
                CallArg::Imm(context.instr_cnt),
            ],
        );
    }

    fn emit_check_instruction_alignment(&mut self, target: Option<WordType>, context: JitInfo) {
        self.regs.flush(&mut self.asm);
        let target_reg = {
            let target_reg = self.regs.host(RAX, &mut self.asm);
            if let Some(target) = target {
                self.asm.mov_r64_imm64(*target_reg, target as u64);
            }
            target_reg.reg()
        };
        self.regs.flush_for_call(&mut self.asm);
        self.asm.call_helper(
            check_instruction_alignment_helper(),
            &[
                CallArg::Reg(target_reg),
                CallArg::Imm(self.jit_context as u64),
                CallArg::Imm(context.instr_pc as u64),
                CallArg::Imm(context.instr_cnt),
            ],
        );
    }

    fn translate_branch(
        &mut self,
        rs1: u8,
        rs2: Option<u8>,
        imm: WordType,
        condition: ConditionCode,
        context: JitInfo,
    ) {
        {
            let rs1 = self.regs.guest_read(rs1, &mut self.asm);
            if let Some(rs2) = rs2 {
                let rs2 = self.regs.guest_read(rs2, &mut self.asm);
                self.asm.alu_rm64_r64(AluOp::Cmp, *rs1, *rs2);
            } else {
                self.asm.alu_rm64_imm32(AluOp::Cmp, *rs1, 0);
            }
        }

        let target = context.instr_pc.wrapping_add(imm);
        let fallthrough = context.instr_pc.wrapping_add(context.instr_len);
        {
            let fallthrough_reg = self.regs.host(RAX, &mut self.asm);
            let target_reg = self.regs.host(RDX, &mut self.asm);
            self.asm.mov_r64_imm64(*fallthrough_reg, fallthrough as u64);
            self.asm.mov_r64_imm64(*target_reg, target as u64);
            self.asm
                .cmovcc_r64_rm64(condition, *fallthrough_reg, (*target_reg).into());
        }

        if context.ialign == 4 && (target & 0x3) != 0 {
            // Check the selected PC so a not-taken branch still uses its aligned fallthrough.
            self.emit_check_instruction_alignment(None, context);
        }
    }

    fn translate_jump_imm(&mut self, rd: Option<u8>, imm: WordType, context: JitInfo) {
        let target = context.instr_pc.wrapping_add(imm);
        if context.ialign == 4 && (target & 0x3) != 0 {
            self.emit_check_instruction_alignment(Some(target), context);
        }

        let target_reg = self.regs.host(RAX, &mut self.asm);
        if context.ialign != 4 || (target & 0x3) == 0 {
            self.asm.mov_r64_imm64(*target_reg, target as u64);
        }
        if let Some(rd) = rd {
            if rd != 0 {
                let link = self.regs.guest_write(rd, &mut self.asm);
                self.asm.mov_r64_imm64(
                    *link,
                    context.instr_pc.wrapping_add(context.instr_len) as u64,
                );
            }
        }
        drop(target_reg);
    }

    fn translate_jump_reg(&mut self, rd: Option<u8>, rs1: u8, imm: WordType, context: JitInfo) {
        let target_reg = self.regs.host(RAX, &mut self.asm);
        {
            let source = self.regs.guest_read(rs1, &mut self.asm);
            self.asm.mov_r64_rm64(*target_reg, *source);
        }
        self.asm.alu_rm64_imm32(AluOp::Add, *target_reg, imm as u32);
        self.asm
            .alu_rm64_imm32(AluOp::And, *target_reg, (!1_u64) as u32);

        drop(target_reg);

        if context.ialign == 4 {
            self.emit_check_instruction_alignment(None, context);
        }

        if let Some(rd) = rd {
            if rd != 0 {
                let target_reg = self.regs.host(RAX, &mut self.asm);
                let link = self.regs.guest_write(rd, &mut self.asm);
                self.asm.mov_r64_imm64(
                    *link,
                    context.instr_pc.wrapping_add(context.instr_len) as u64,
                );
                drop(target_reg);
            }
        }
    }

    #[must_use]
    pub(super) fn translate_instruction(
        &mut self,
        instr: RiscvInstr,
        info: RVInstrInfo,
        context: JitInfo,
    ) -> TranslateResult {
        let result = self.translate_rvi(instr, info, context);
        if result != TranslateResult::Unsupported {
            return result;
        }

        self.translate_rvc(instr, info, context)
    }
}

impl OperandWidth {
    fn operand_size(self) -> OperandSize {
        match self {
            Self::Xlen => OperandSize::Qword,
            Self::Word => OperandSize::Dword,
        }
    }
}

fn emit_alu_r(asm: &mut X86Assembler, op: AluOp, dst: X86Reg, src: X86Reg, width: OperandWidth) {
    match width {
        OperandWidth::Xlen => asm.alu_rm64_r64(op, dst, src),
        OperandWidth::Word => asm.alu_rm32_r32(op, dst, src),
    }
    sext_word_on_need(asm, dst, width);
}

fn emit_alu_i(asm: &mut X86Assembler, op: AluOp, dst: X86Reg, imm: u32, width: OperandWidth) {
    match width {
        OperandWidth::Xlen => asm.alu_rm64_imm32(op, dst, imm),
        OperandWidth::Word => asm.alu_rm32_imm32(op, dst, imm),
    }
    sext_word_on_need(asm, dst, width);
}

fn sext_word_on_need(asm: &mut X86Assembler, dst: X86Reg, width: OperandWidth) {
    if matches!(width, OperandWidth::Word) {
        asm.sext(dst);
    }
}
