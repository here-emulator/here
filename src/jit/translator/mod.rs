//! It's not urgent to handle instruction fusion,
//! because many of them will disappear after const folding.
//! but some fusions are helpful, include:
//! - branch + mov => cmov

use crate::{
    config::arch_config::WordType,
    debug_unreachable,
    isa::riscv::{
        decoder::DecodeInstr,
        executor::RVCPU,
        instruction::{RVInstrInfo, instr_table::RiscvInstr},
    },
    make_ir,
};

use super::*;

use super::{
    helpers::{LoadKind, StoreKind},
    runtime::JitContext,
    stats::BlockStopReason,
};

pub const MAX_BLOCK_INSTRUCTIONS: u64 = 1024;

pub struct TranslateResult {
    pub ir_block: IRBlock,
    pub instr_cnt: u64,
    /// for stats
    pub guest_bytes: u64,
    pub stop_reason: BlockStopReason,
}

/// Info needed by translator for every instruction.
struct InstrMeta {
    instr_pc: WordType,
    /// Number of instructions translated in this basic block **before** the current one.
    instr_cnt: u64,
}

/// Info that is immutable during translation.
pub struct TranslatorContext {
    pub jit_context: *mut JitContext,
    pub ialign: WordType,
}

pub struct Translator {
    trans_context: TranslatorContext,
    ir_builder: IRBuilder,
}

// TODO: Remove `len` on `DecodeInstr`, and use that instead of `InstAndInfo`.
type InstrAndInfo = (RiscvInstr, RVInstrInfo);

impl Translator {
    pub fn new(trans_context: TranslatorContext) -> Self {
        Self {
            trans_context,
            ir_builder: IRBuilder::default(),
        }
    }

    fn take_builder(&mut self) -> IRBuilder {
        let mut builder = IRBuilder::default();
        std::mem::swap(&mut self.ir_builder, &mut builder);
        builder
    }

    fn translate_jmp(&mut self, inst: &InstrAndInfo, meta: &InstrMeta) -> Option<IRBlock> {
        use RVInstrInfo as Info;
        use RiscvInstr::*;

        let ilen = (*inst).0.len();
        let (rd, offset, base) = match *inst {
            (JAL, Info::J { rd, imm }) => (Some(rd), imm, None),
            (JALR, Info::I { rs1, rd, imm }) => (Some(rd), imm, Some(rs1)),
            (C_J, Info::CJ { target }) => (None, target, None),
            (C_JR, Info::CR { rd_rs1, rs2: _ }) => (None, 0, Some(rd_rs1)),
            (C_JAL, Info::CJ { target }) => (Some(1), target, None),
            (C_JALR, Info::CR { rd_rs1, rs2: _ }) => (Some(1), 0, Some(rd_rs1)),
            _ => {
                return None;
            }
        };

        let base = if base == Some(0) { None } else { base };

        let mut builder = self.take_builder();

        let target_val = if let Some(base) = base {
            let base = builder.get_reg(base.into());
            let v1 = builder.compute(ComputeKind::Add, base, offset.into());
            builder.compute(ComputeKind::And, v1, (!1).into()) // clear LSB, see the manual
        } else {
            let target = meta.instr_pc.wrapping_add(offset);
            if (target & (self.trans_context.ialign - 1)) != 0 {
                self.ir_builder = builder;
                return None;
            }
            target.into()
        };

        if let Some(rd) = rd {
            if rd != 0 {
                builder.set_reg(rd.into(), meta.instr_pc.wrapping_add(ilen).into());
            }
        };

        let block = builder.build_jmp(target_val);
        Some(block)
    }

    fn translate_branch(&mut self, inst: &InstrAndInfo, meta: &InstrMeta) -> Option<IRBlock> {
        use RVInstrInfo as Info;
        use RiscvInstr::*;

        let (inst, info) = *inst;

        let kind = match inst {
            BEQ | C_BEQZ => CmpKind::Eq,
            BNE | C_BNEZ => CmpKind::NotEq,
            BGE => CmpKind::GtEq,
            BGEU => CmpKind::GtEqUnsigned,
            BLT => CmpKind::Lt,
            BLTU => CmpKind::LtUnsigned,
            _ => return None,
        };

        let (rs1, rs2, offset) = match info {
            Info::B { rs1, rs2, imm } => (rs1, rs2, imm),
            Info::CB { rd_rs1, imm } => (rd_rs1, 0, imm),
            _ => debug_unreachable!(),
        };

        let target = meta.instr_pc.wrapping_add(offset);
        let fallthrough = meta.instr_pc.wrapping_add(inst.len());

        if (target & (self.trans_context.ialign - 1)) != 0 {
            return None;
        }

        let mut builder = self.take_builder();

        let rs1 = builder.get_reg(rs1.into());
        let rs2 = builder.get_reg(rs2.into());

        let block = builder.build_branch(kind, rs1, rs2, target, fallthrough);
        Some(block)
    }

    fn translate_compute(&mut self, inst: &InstrAndInfo, _meta: &InstrMeta) -> bool {
        use RVInstrInfo as Info;
        use RiscvInstr::*;

        let (inst, info) = *inst;

        let kind = match inst {
            ADD | ADDI | C_ADD | C_ADDI | C_ADDI4SPN | C_ADDI16SP => ComputeKind::Add,
            SUB | C_SUB => ComputeKind::Sub,
            AND | ANDI | C_AND | C_ANDI => ComputeKind::And,
            OR | ORI | C_OR => ComputeKind::Or,
            XOR | XORI | C_XOR => ComputeKind::Xor,
            SLL | SLLI | C_SLLI => ComputeKind::Shl,
            SRL | SRLI | C_SRLI => ComputeKind::Shr,
            SRA | SRAI | C_SRAI => ComputeKind::Sar,
            SLT | SLTI => ComputeKind::Slt,
            SLTU | SLTIU => ComputeKind::Sltu,
            ADDW | ADDIW | C_ADDW | C_ADDIW => ComputeKind::AddW,
            SUBW | C_SUBW => ComputeKind::SubW,
            SLLW | SLLIW => ComputeKind::ShlW,
            SRLW | SRLIW => ComputeKind::ShrW,
            SRAW | SRAIW => ComputeKind::SarW,
            _ => return false,
        };

        let (rd, v1, v2) = match (info, inst) {
            (Info::R { rs1, rs2, rd }, _) => {
                let v1 = self.ir_builder.get_reg(rs1.into());
                let v2 = self.ir_builder.get_reg(rs2.into());
                (rd, v1, v2)
            }
            (Info::I { rs1, rd, imm }, _) => {
                let v1 = self.ir_builder.get_reg(rs1.into());
                (rd, v1, imm.into())
            }
            (Info::CR { rd_rs1, rs2 }, C_ADD) => {
                let v1 = self.ir_builder.get_reg(rd_rs1.into());
                let v2 = self.ir_builder.get_reg(rs2.into());
                (rd_rs1, v1, v2)
            }
            (Info::CA { rd_rs1, rs2 }, _) => {
                let v1 = self.ir_builder.get_reg(rd_rs1.into());
                let v2 = self.ir_builder.get_reg(rs2.into());
                (rd_rs1, v1, v2)
            }
            (Info::CI { rd_rs1, imm }, C_ADDI | C_ADDI16SP | C_ADDIW | C_SLLI) => {
                let v1 = self.ir_builder.get_reg(rd_rs1.into());
                (rd_rs1, v1, imm.into())
            }
            (Info::CIW { rd, imm }, C_ADDI4SPN) => {
                let v1 = self.ir_builder.get_reg(2.into());
                (rd, v1, imm.into())
            }
            (Info::CB { rd_rs1, imm }, C_ANDI | C_SRLI | C_SRAI) => {
                let v1 = self.ir_builder.get_reg(rd_rs1.into());
                (rd_rs1, v1, imm.into())
            }
            _ => debug_unreachable!(),
        };

        let val = self.ir_builder.compute(kind, v1, v2);
        self.ir_builder.set_reg(rd.into(), val);

        true
    }

    fn translate_load(&mut self, inst: &InstrAndInfo, meta: &InstrMeta) -> bool {
        use RVInstrInfo as Info;
        use RiscvInstr::*;

        let (inst, info) = *inst;
        let kind = match inst {
            LB => LoadKind::SignedByte,
            LBU => LoadKind::UnsignedByte,
            LH => LoadKind::SignedHalf,
            LHU => LoadKind::UnsignedHalf,
            LW => LoadKind::SignedWord,
            LWU => LoadKind::UnsignedWord,
            LD => LoadKind::DoubleWord,
            C_LW => LoadKind::SignedWord,
            C_LD => LoadKind::DoubleWord,
            C_LWSP => LoadKind::SignedWord,
            C_LDSP => LoadKind::DoubleWord,
            _ => return false,
        };

        let (rd, v_addr) = match info {
            Info::I { rs1, rd, imm } | Info::CL { rd, rs1, imm } => {
                let v_rs1 = self.ir_builder.get_reg(rs1.into());
                let v_addr = self.ir_builder.compute(ComputeKind::Add, v_rs1, imm.into());
                (rd, v_addr)
            }
            Info::CI { rd_rs1, imm } => {
                let v_rs1 = self.ir_builder.get_reg(2.into());
                let v_addr = self.ir_builder.compute(ComputeKind::Add, v_rs1, imm.into());
                (rd_rs1, v_addr)
            }
            _ => debug_unreachable!(),
        };

        let jit_ctx = self.trans_context.jit_context as usize;

        let value = make_ir!(self.ir_builder, [
            value = call [cpu, trap] kind.helper(),
                [v_addr, jit_ctx, meta.instr_pc, meta.instr_cnt + 1];
        ], [value]);

        self.ir_builder.set_reg(rd.into(), value);
        true
    }

    fn translate_store(&mut self, inst: &InstrAndInfo, meta: &InstrMeta) -> bool {
        use RVInstrInfo as Info;
        use RiscvInstr::*;

        let (inst, info) = *inst;
        let kind = match inst {
            SB => StoreKind::Byte,
            SH => StoreKind::Half,
            SW => StoreKind::Word,
            SD => StoreKind::DoubleWord,
            C_SW => StoreKind::Word,
            C_SD => StoreKind::DoubleWord,
            C_SWSP => StoreKind::Word,
            C_SDSP => StoreKind::DoubleWord,
            _ => return false,
        };

        let (rs1, rs2, imm) = match info {
            Info::S { rs1, rs2, imm } | Info::CS { rs1, rs2, imm } => (rs1, rs2, imm),
            Info::CSS { rs2, imm } => (2, rs2, imm),
            _ => debug_unreachable!(),
        };

        let v_rs1 = self.ir_builder.get_reg(rs1.into());
        let v_addr = self.ir_builder.compute(ComputeKind::Add, v_rs1, imm.into());
        let v_value = self.ir_builder.get_reg(rs2.into());
        let jit_ctx = self.trans_context.jit_context as usize;

        make_ir!(self.ir_builder, [
            call [cpu, trap] kind.helper(),
                [v_addr, v_value, jit_ctx, meta.instr_pc, meta.instr_cnt + 1];
        ], []);
        true
    }

    fn translate_mov(&mut self, inst: &InstrAndInfo, meta: &InstrMeta) -> bool {
        // we don't need to handle pseudo-instructions like `mv rd, rs => addi rd, rs, 0`,
        // cause it can be handled as `addi` instruction and do const folding.
        use RVInstrInfo as Info;
        use RiscvInstr::*;

        match *inst {
            (C_MV, Info::CR { rd_rs1, rs2 }) => {
                let value = self.ir_builder.get_reg(rs2.into());
                self.ir_builder.set_reg(rd_rs1.into(), value);
            }
            (C_LI | C_LUI, Info::CI { rd_rs1, imm }) => {
                self.ir_builder.set_reg(rd_rs1.into(), imm.into());
            }
            (LUI, Info::U { rd, imm }) => {
                self.ir_builder.set_reg(rd.into(), imm.into());
            }
            (AUIPC, Info::U { rd, imm }) => {
                let value = meta.instr_pc.wrapping_add(imm);
                self.ir_builder.set_reg(rd.into(), value.into());
            }
            (C_NOP, Info::CR { .. }) => {}
            _ => return false,
        }

        true
    }

    fn translate_common(&mut self, inst: &InstrAndInfo, meta: &InstrMeta) -> bool {
        false
            || self.translate_load(inst, meta)
            || self.translate_store(inst, meta)
            || self.translate_compute(inst, meta)
            || self.translate_mov(inst, meta)
    }

    pub fn translate(
        &mut self,
        cpu: &mut RVCPU,
        pc: WordType,
    ) -> Result<TranslateResult, BlockStopReason> {
        self.ir_builder = IRBuilder::new(pc);
        let mut meta = InstrMeta {
            instr_pc: pc,
            instr_cnt: 0,
        };
        let mut guest_bytes: u64 = 0;

        let fallback_build =
            |t: &mut Self, instr_pc: WordType| t.take_builder().build_jmp(instr_pc.into());

        let (mut ir_block, stop_reason) = loop {
            let Some(DecodeInstr { instr, info, .. }) = cpu.decode_at(meta.instr_pc) else {
                break (
                    fallback_build(self, meta.instr_pc),
                    BlockStopReason::DecodeFault,
                );
            };

            let inst_info = (instr, info);
            let ilen = instr.len();

            if let Some(block) = self.translate_jmp(&inst_info, &meta) {
                meta.instr_cnt += 1;
                guest_bytes += ilen;
                let reason = match block.insts.last() {
                    Some(IRInst::Terminator(Terminator::DynamicJmp { .. })) => {
                        BlockStopReason::DynamicJump(instr)
                    }
                    _ => BlockStopReason::DirectJump(instr),
                };
                break (block, reason);
            }

            if let Some(block) = self.translate_branch(&inst_info, &meta) {
                meta.instr_cnt += 1;
                guest_bytes += ilen;
                break (block, BlockStopReason::Branch(instr));
            }

            if !self.translate_common(&inst_info, &meta) {
                break (
                    fallback_build(self, meta.instr_pc),
                    BlockStopReason::Unsupported(instr),
                );
            }

            meta.instr_pc += instr.len();
            meta.instr_cnt += 1;
            guest_bytes += ilen;

            if meta.instr_cnt >= MAX_BLOCK_INSTRUCTIONS {
                break (
                    fallback_build(self, meta.instr_pc),
                    BlockStopReason::BlockLimit,
                );
            }
        };

        if meta.instr_cnt == 0 {
            Err(stop_reason)
        } else {
            ir_block.start_pc = pc;
            ir_block.instr_cnt = meta.instr_cnt as u32;
            Ok(TranslateResult {
                ir_block,
                instr_cnt: meta.instr_cnt,
                guest_bytes,
                stop_reason,
            })
        }
    }

    #[cfg(test)]
    pub fn translate_single_instruction(
        &mut self,
        pc: WordType,
        instr: RiscvInstr,
        info: RVInstrInfo,
    ) -> Option<IRBlock> {
        self.ir_builder = IRBuilder::new(pc);
        let meta = InstrMeta {
            instr_pc: pc,
            instr_cnt: 0,
        };
        let inst_info = (instr, info);
        if let Some(mut block) = None
            .or_else(|| self.translate_jmp(&inst_info, &meta))
            .or_else(|| self.translate_branch(&inst_info, &meta))
        {
            block.start_pc = pc;
            block.instr_cnt = 1;
            return Some(block);
        }
        if self.translate_common(&inst_info, &meta) {
            let next_pc = meta.instr_pc.wrapping_add(instr.len());
            let mut block = self.take_builder().build_jmp(next_pc.into());
            block.start_pc = pc;
            block.instr_cnt = 1;
            return Some(block);
        }
        self.ir_builder = IRBuilder::default();
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ialign_checks_jal_and_branch() {
        let mut translator_align4 = Translator::new(TranslatorContext {
            jit_context: std::ptr::null_mut(),
            ialign: 4,
        });

        let mut translator_align2 = Translator::new(TranslatorContext {
            jit_context: std::ptr::null_mut(),
            ialign: 2,
        });

        // 1. JAL aligned to 4 bytes: PC = 0x1000, imm = 8 => target = 0x1008 (aligned)
        let res = translator_align4.translate_single_instruction(
            0x1000,
            RiscvInstr::JAL,
            RVInstrInfo::J { rd: 1, imm: 8 },
        );
        assert!(res.is_some());

        // 2. JAL not aligned to 4 bytes: PC = 0x1000, imm = 2 => target = 0x1002 (misaligned when ialign=4)
        let res = translator_align4.translate_single_instruction(
            0x1000,
            RiscvInstr::JAL,
            RVInstrInfo::J { rd: 1, imm: 2 },
        );
        assert!(res.is_none());

        // 3. JAL with imm = 2 is valid when ialign = 2 (C extension)
        let res = translator_align2.translate_single_instruction(
            0x1000,
            RiscvInstr::JAL,
            RVInstrInfo::J { rd: 1, imm: 2 },
        );
        assert!(res.is_some());

        // 4. Branch not aligned to 4 bytes: PC = 0x1000, imm = 2 => target = 0x1002 (misaligned when ialign=4)
        let res = translator_align4.translate_single_instruction(
            0x1000,
            RiscvInstr::BEQ,
            RVInstrInfo::B {
                rs1: 1,
                rs2: 2,
                imm: 2,
            },
        );
        assert!(res.is_none());

        // 5. Branch aligned to 4 bytes: PC = 0x1000, imm = 4 => target = 0x1004 (valid)
        let res = translator_align4.translate_single_instruction(
            0x1000,
            RiscvInstr::BEQ,
            RVInstrInfo::B {
                rs1: 1,
                rs2: 2,
                imm: 4,
            },
        );
        assert!(res.is_some());
    }
}
