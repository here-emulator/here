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

use super::abi::{LoadKind, StoreKind};

/// Reason why translation of a basic block terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockStopReason {
    /// Conditional branch instruction (e.g. beq, bne, blt, bge).
    Branch(RiscvInstr),
    /// Direct unconditional jump (e.g. jal, c.j).
    DirectJump(RiscvInstr),
    /// Dynamic/indirect jump (e.g. jalr, c.jr, c.jalr).
    DynamicJump(RiscvInstr),
    /// Current JIT does not support this instruction; falls back to interpreter.
    Unsupported(RiscvInstr),
    /// Translation stopped because the maximum block length limit was reached.
    BlockLimit,
    /// Decoding failed (e.g. MMU fetch fault or invalid instruction).
    DecodeFault,
}

pub const MAX_BLOCK_INSTRUCTIONS: u32 = 1024;

pub struct TranslateResult {
    pub ir_block: IRBlock,
    pub meta: IRBlockMeta,
    // --- for stats ---
    pub guest_bytes: u64,
    pub stop_reason: BlockStopReason,
}

/// Info needed by translator for every instruction.
#[derive(Clone, Copy, Debug)]
pub struct InstrMeta {
    pub instr_pc: WordType,
    pub next_pc: WordType,
    /// Number of guest instructions translated in this basic block **before** the current one.
    pub guest_cnt: u32,
}

/// Info that is immutable during translation.
pub struct TranslatorContext {
    pub ialign: WordType,
}

/// Abstract instruction stream that supports peeking the next instruction.
pub trait InstrStream {
    /// Peeks the current decoded instruction without advancing the stream.
    fn peek(&mut self) -> Option<DecodeInstr>;

    /// Peeks the decoded instruction immediately following the current one without advancing the stream.
    fn peek_next(&mut self) -> Option<DecodeInstr>;

    /// Consumes the current instruction and advances the stream by one.
    fn consume_one(&mut self) -> Option<DecodeInstr>;

    /// Returns the guest PC of the current instruction at `peek()`.
    fn peek_pc(&self) -> WordType;

    /// Returns the guest PC immediately after the current instruction (i.e. `peek_pc + peek().len`).
    fn next_pc(&self) -> WordType;

    /// Returns the number of guest instruction bytes consumed so far (up to the current `peek`).
    fn guest_bytes(&self) -> u64;

    /// Returns the number of guest instructions consumed so far (up to the current `peek`).
    fn guest_cnt(&self) -> u32;
}

/// Decodes instructions on demand from CPU memory.
pub struct CpuInstrStream<'a> {
    cpu: &'a mut RVCPU,
    peek_pc: WordType,
    guest_cnt: u32,
    guest_bytes: u64,
    peeked: Option<DecodeInstr>,
    peeked_next: Option<DecodeInstr>,
}

impl<'a> CpuInstrStream<'a> {
    pub fn new(cpu: &'a mut RVCPU, pc: WordType) -> Self {
        Self {
            cpu,
            peek_pc: pc,
            guest_cnt: 0,
            guest_bytes: 0,
            peeked: None,
            peeked_next: None,
        }
    }
}

impl<'a> InstrStream for CpuInstrStream<'a> {
    fn peek(&mut self) -> Option<DecodeInstr> {
        if self.peeked.is_none() {
            self.peeked = self.cpu.decode_at(self.peek_pc);
        }
        self.peeked
    }

    fn peek_next(&mut self) -> Option<DecodeInstr> {
        let first = self.peek()?;
        if self.peeked_next.is_none() {
            let next_pc = self.peek_pc.wrapping_add(first.len as WordType);
            self.peeked_next = self.cpu.decode_at(next_pc);
        }
        self.peeked_next
    }

    fn consume_one(&mut self) -> Option<DecodeInstr> {
        let instr = self.peek()?;
        self.peek_pc = self.peek_pc.wrapping_add(instr.len as WordType);
        self.guest_bytes += instr.len as u64;
        self.guest_cnt += 1;
        self.peeked = self.peeked_next.take();
        self.peeked_next = None;
        Some(instr)
    }

    fn peek_pc(&self) -> WordType {
        self.peek_pc
    }

    fn next_pc(&self) -> WordType {
        let instr = self.peeked.unwrap();
        self.peek_pc.wrapping_add(instr.len as WordType)
    }

    fn guest_bytes(&self) -> u64 {
        self.guest_bytes
    }

    fn guest_cnt(&self) -> u32 {
        self.guest_cnt
    }
}

struct BranchTerminator {
    cmp: CmpKind,
    src1: Value,
    src2: Value,
    then_pc: u64,
    else_pc: u64,
}

pub struct Translator {
    trans_context: TranslatorContext,
}

// TODO: Remove `len` on `DecodeInstr`, and use that instead of `InstAndInfo`.
type InstrAndInfo = (RiscvInstr, RVInstrInfo);

impl Translator {
    pub fn new(trans_context: TranslatorContext) -> Self {
        Self { trans_context }
    }

    fn translate_jmp(
        &self,
        builder: &mut IRBuilder,
        inst: &InstrAndInfo,
        meta: &InstrMeta,
    ) -> Option<(Value, BlockStopReason)> {
        use RVInstrInfo as Info;
        use RiscvInstr::*;

        let (rd, offset, base) = match *inst {
            (JAL, Info::J { rd, imm }) => (Some(rd), imm, None),
            (JALR, Info::I { rs1, rd, imm }) => (Some(rd), imm, Some(rs1)),
            (C_J, Info::CJ { target }) => (None, target, None),
            (C_JR, Info::CR { rd_rs1, rs2: _ }) => (None, 0, Some(rd_rs1)),
            (C_JAL, Info::CJ { target }) => (Some(1), target, None),
            (C_JALR, Info::CR { rd_rs1, rs2: _ }) => (Some(1), 0, Some(rd_rs1)),
            _ => return None,
        };

        let (target_val, reason) = if let Some(base) = base {
            let base_reg = builder.get_reg(base.into());
            let v1 = builder.compute(ComputeKind::Add, base_reg, offset.into());
            let target = builder.compute(ComputeKind::And, v1, (!1).into()); // clear LSB, see the manual
            (target, BlockStopReason::DynamicJump(inst.0))
        } else {
            let target = meta.instr_pc.wrapping_add(offset);
            if (target & (self.trans_context.ialign - 1)) != 0 {
                return None;
            }
            (target.into(), BlockStopReason::DirectJump(inst.0))
        };

        if let Some(rd) = rd {
            if rd != 0 {
                builder.set_reg(rd.into(), meta.next_pc.into());
            }
        };

        Some((target_val, reason))
    }

    fn translate_branch(
        &self,
        builder: &mut IRBuilder,
        inst: &InstrAndInfo,
        meta: &InstrMeta,
    ) -> Option<BranchTerminator> {
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
        let fallthrough = meta.next_pc;

        if (target & (self.trans_context.ialign - 1)) != 0 {
            return None;
        }

        let rs1 = builder.get_reg(rs1.into());
        let rs2 = builder.get_reg(rs2.into());

        Some(BranchTerminator {
            cmp: kind,
            src1: rs1,
            src2: rs2,
            then_pc: target,
            else_pc: fallthrough,
        })
    }

    fn translate_compute(
        &self,
        builder: &mut IRBuilder,
        inst: &InstrAndInfo,
        _meta: &InstrMeta,
    ) -> bool {
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
                let v1 = builder.get_reg(rs1.into());
                let v2 = builder.get_reg(rs2.into());
                (rd, v1, v2)
            }
            (Info::I { rs1, rd, imm }, _) => {
                let v1 = builder.get_reg(rs1.into());
                (rd, v1, imm.into())
            }
            (Info::CR { rd_rs1, rs2 }, C_ADD) => {
                let v1 = builder.get_reg(rd_rs1.into());
                let v2 = builder.get_reg(rs2.into());
                (rd_rs1, v1, v2)
            }
            (Info::CA { rd_rs1, rs2 }, _) => {
                let v1 = builder.get_reg(rd_rs1.into());
                let v2 = builder.get_reg(rs2.into());
                (rd_rs1, v1, v2)
            }
            (Info::CI { rd_rs1, imm }, C_ADDI | C_ADDI16SP | C_ADDIW | C_SLLI) => {
                let v1 = builder.get_reg(rd_rs1.into());
                (rd_rs1, v1, imm.into())
            }
            (Info::CIW { rd, imm }, C_ADDI4SPN) => {
                let v1 = builder.get_reg(2.into());
                (rd, v1, imm.into())
            }
            (Info::CB { rd_rs1, imm }, C_ANDI | C_SRLI | C_SRAI) => {
                let v1 = builder.get_reg(rd_rs1.into());
                (rd_rs1, v1, imm.into())
            }
            _ => debug_unreachable!(),
        };

        let val = builder.compute(kind, v1, v2);
        builder.set_reg(rd.into(), val);

        true
    }

    fn translate_load(
        &self,
        builder: &mut IRBuilder,
        inst: &InstrAndInfo,
        meta: &InstrMeta,
    ) -> bool {
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

        let (rd, rs1, imm) = match info {
            Info::I { rs1, rd, imm } | Info::CL { rd, rs1, imm } => (rd, rs1, imm),
            Info::CI { rd_rs1, imm } => (rd_rs1, 2, imm),
            _ => debug_unreachable!(),
        };

        let v_rs1 = builder.get_reg(rs1.into());
        let v_addr = builder.compute(ComputeKind::Add, v_rs1, imm.into());

        let cpu = builder.cpu_vreg();
        let jit_ctx = builder.jit_context_vreg();

        let value = make_ir!(*builder, [
            value = call [trap] kind.helper(),
                [cpu, v_addr, jit_ctx, meta.instr_pc, (meta.guest_cnt + 1) as u64];
        ], [value]);

        builder.set_reg(rd.into(), value);
        true
    }

    fn translate_store(
        &self,
        builder: &mut IRBuilder,
        inst: &InstrAndInfo,
        meta: &InstrMeta,
    ) -> bool {
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

        let v_rs1 = builder.get_reg(rs1.into());
        let v_addr = builder.compute(ComputeKind::Add, v_rs1, imm.into());
        let v_value = builder.get_reg(rs2.into());
        let cpu = builder.cpu_vreg();
        let jit_ctx = builder.jit_context_vreg();

        make_ir!(*builder, [
            call [trap] kind.helper(),
                [cpu, v_addr, v_value, jit_ctx, meta.instr_pc, (meta.guest_cnt + 1) as u64];
        ], []);
        true
    }

    fn translate_mov(
        &self,
        builder: &mut IRBuilder,
        inst: &InstrAndInfo,
        meta: &InstrMeta,
    ) -> bool {
        // we don't need to handle pseudo-instructions like `mv rd, rs => addi rd, rs, 0`,
        // cause it can be handled as `addi` instruction and do const folding.
        use RVInstrInfo as Info;
        use RiscvInstr::*;

        match *inst {
            (C_MV, Info::CR { rd_rs1, rs2 }) => {
                let value = builder.get_reg(rs2.into());
                builder.set_reg(rd_rs1.into(), value);
            }
            (C_LI | C_LUI, Info::CI { rd_rs1, imm }) => {
                builder.set_reg(rd_rs1.into(), imm.into());
            }
            (LUI, Info::U { rd, imm }) => {
                builder.set_reg(rd.into(), imm.into());
            }
            (AUIPC, Info::U { rd, imm }) => {
                let value = meta.instr_pc.wrapping_add(imm);
                builder.set_reg(rd.into(), value.into());
            }
            (C_NOP, Info::CR { .. }) => {}
            _ => return false,
        }

        true
    }

    fn translate_common(
        &self,
        builder: &mut IRBuilder,
        inst: &InstrAndInfo,
        meta: &InstrMeta,
    ) -> bool {
        false
            || self.translate_load(builder, inst, meta)
            || self.translate_store(builder, inst, meta)
            || self.translate_compute(builder, inst, meta)
            || self.translate_mov(builder, inst, meta)
    }

    /// Try fusing `branch + mov => cmov`.
    ///
    /// Placeholder for architecture support. Currently always returns `false`.
    fn translate_cmov(
        &self,
        _builder: &mut IRBuilder,
        _branch: &DecodeInstr,
        _mov: &DecodeInstr,
        _meta: &InstrMeta,
    ) -> bool {
        // TODO
        false
    }

    pub fn translate_stream<S: InstrStream>(
        &self,
        stream: &mut S,
        pc: WordType,
        mut builder: IRBuilder,
    ) -> Result<TranslateResult, BlockStopReason> {
        let (ir_block, stop_reason) = loop {
            if stream.guest_cnt() >= MAX_BLOCK_INSTRUCTIONS {
                break (
                    builder.build_jmp(stream.peek_pc().into()),
                    BlockStopReason::BlockLimit,
                );
            }

            let Some(decode_instr) = stream.peek() else {
                break (
                    builder.build_jmp(stream.peek_pc().into()),
                    BlockStopReason::DecodeFault,
                );
            };

            let meta = InstrMeta {
                instr_pc: stream.peek_pc(),
                next_pc: stream.next_pc(),
                guest_cnt: stream.guest_cnt(),
            };

            let DecodeInstr {
                instr,
                info,
                len: _,
            } = decode_instr;
            let inst_info = (instr, info);

            if let Some(next) = stream.peek_next() {
                if self.translate_cmov(&mut builder, &decode_instr, &next, &meta) {
                    stream.consume_one();
                    stream.consume_one();
                    continue;
                }
            }

            if let Some((target, reason)) = self.translate_jmp(&mut builder, &inst_info, &meta) {
                stream.consume_one();
                break (builder.build_jmp(target), reason);
            }

            if let Some(b) = self.translate_branch(&mut builder, &inst_info, &meta) {
                stream.consume_one();
                break (
                    builder.build_branch(b.cmp, b.src1, b.src2, b.then_pc, b.else_pc),
                    BlockStopReason::Branch(instr),
                );
            }

            if !self.translate_common(&mut builder, &inst_info, &meta) {
                break (
                    builder.build_jmp(stream.peek_pc().into()),
                    BlockStopReason::Unsupported(instr),
                );
            }

            stream.consume_one();
        };

        if stream.guest_cnt() == 0 {
            return Err(stop_reason);
        }

        let meta = IRBlockMeta::new(pc, stream.guest_cnt());
        Ok(TranslateResult {
            ir_block,
            meta,
            guest_bytes: stream.guest_bytes(),
            stop_reason,
        })
    }

    pub fn translate(
        &self,
        cpu: &mut RVCPU,
        pc: WordType,
    ) -> Result<TranslateResult, BlockStopReason> {
        let mut stream = CpuInstrStream::new(cpu, pc);
        self.translate_stream(&mut stream, pc, IRBuilder::new())
    }
}

#[cfg(any(test, feature = "test-support"))]
pub struct MockInstrStream {
    insts: Vec<DecodeInstr>,
    idx: usize,
    peek_pc: WordType,
    guest_cnt: u32,
    guest_bytes: u64,
}

#[cfg(any(test, feature = "test-support"))]
impl MockInstrStream {
    pub fn new(pc: WordType, insts: Vec<DecodeInstr>) -> Self {
        Self {
            insts,
            idx: 0,
            peek_pc: pc,
            guest_cnt: 0,
            guest_bytes: 0,
        }
    }

    pub fn single(pc: WordType, instr: DecodeInstr) -> Self {
        Self::new(pc, vec![instr])
    }
}

#[cfg(any(test, feature = "test-support"))]
impl InstrStream for MockInstrStream {
    fn peek(&mut self) -> Option<DecodeInstr> {
        self.insts.get(self.idx).copied()
    }

    fn peek_next(&mut self) -> Option<DecodeInstr> {
        self.insts.get(self.idx + 1).copied()
    }

    fn consume_one(&mut self) -> Option<DecodeInstr> {
        let instr = *self.insts.get(self.idx)?;
        self.idx += 1;
        self.peek_pc = self.peek_pc.wrapping_add(instr.len as WordType);
        self.guest_bytes += instr.len as u64;
        self.guest_cnt += 1;
        Some(instr)
    }

    fn peek_pc(&self) -> WordType {
        self.peek_pc
    }

    fn next_pc(&self) -> WordType {
        let instr = self.insts.get(self.idx).expect("no instruction to peek");
        self.peek_pc.wrapping_add(instr.len as WordType)
    }

    fn guest_bytes(&self) -> u64 {
        self.guest_bytes
    }

    fn guest_cnt(&self) -> u32 {
        self.guest_cnt
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Translator {
    pub fn translate_single_instruction(
        &self,
        pc: WordType,
        decoded: DecodeInstr,
    ) -> Option<(IRBlock, IRBlockMeta)> {
        let mut stream = MockInstrStream::single(pc, decoded);
        self.translate_stream(&mut stream, pc, IRBuilder::new())
            .ok()
            .map(|res| (res.ir_block, res.meta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(instr: RiscvInstr, info: RVInstrInfo) -> DecodeInstr {
        DecodeInstr {
            instr,
            info,
            len: instr.len(),
        }
    }

    #[test]
    fn test_ialign_checks_jal_and_branch() {
        let translator_align4 = Translator::new(TranslatorContext { ialign: 4 });

        let translator_align2 = Translator::new(TranslatorContext { ialign: 2 });

        // 1. JAL aligned to 4 bytes: PC = 0x1000, imm = 8 => target = 0x1008 (aligned)
        let res = translator_align4.translate_single_instruction(
            0x1000,
            decoded(RiscvInstr::JAL, RVInstrInfo::J { rd: 1, imm: 8 }),
        );
        assert!(res.is_some());

        // 2. JAL not aligned to 4 bytes: PC = 0x1000, imm = 2 => target = 0x1002 (misaligned when ialign=4)
        let res = translator_align4.translate_single_instruction(
            0x1000,
            decoded(RiscvInstr::JAL, RVInstrInfo::J { rd: 1, imm: 2 }),
        );
        assert!(res.is_none());

        // 3. JAL with imm = 2 is valid when ialign = 2 (C extension)
        let res = translator_align2.translate_single_instruction(
            0x1000,
            decoded(RiscvInstr::JAL, RVInstrInfo::J { rd: 1, imm: 2 }),
        );
        assert!(res.is_some());

        // 4. Branch not aligned to 4 bytes: PC = 0x1000, imm = 2 => target = 0x1002 (misaligned when ialign=4)
        let res = translator_align4.translate_single_instruction(
            0x1000,
            decoded(
                RiscvInstr::BEQ,
                RVInstrInfo::B {
                    rs1: 1,
                    rs2: 2,
                    imm: 2,
                },
            ),
        );
        assert!(res.is_none());

        // 5. Branch aligned to 4 bytes: PC = 0x1000, imm = 4 => target = 0x1004 (valid)
        let res = translator_align4.translate_single_instruction(
            0x1000,
            decoded(
                RiscvInstr::BEQ,
                RVInstrInfo::B {
                    rs1: 1,
                    rs2: 2,
                    imm: 4,
                },
            ),
        );
        assert!(res.is_some());
    }

    #[test]
    fn test_translate_single_instruction_common_fallthrough() {
        let translator = Translator::new(TranslatorContext { ialign: 4 });

        let res = translator.translate_single_instruction(
            0x1000,
            decoded(
                RiscvInstr::ADDI,
                RVInstrInfo::I {
                    rd: 1,
                    rs1: 2,
                    imm: 42,
                },
            ),
        );
        let (block, meta) = res.expect("ADDI single instruction should translate successfully");
        assert_eq!(meta.start_pc, 0x1000);
        assert_eq!(meta.guest_cnt, 1);
        assert_eq!(
            block.terminator(),
            &Terminator::DirectJmp { target: 0x1004 }
        );
    }

    #[test]
    fn test_stream_multi_instruction_block() {
        let translator = Translator::new(TranslatorContext { ialign: 4 });

        let mut stream = MockInstrStream::new(
            0x1000,
            vec![
                DecodeInstr {
                    instr: RiscvInstr::ADDI,
                    info: RVInstrInfo::I {
                        rd: 1,
                        rs1: 0,
                        imm: 10,
                    },
                    len: 4,
                },
                DecodeInstr {
                    instr: RiscvInstr::ADDI,
                    info: RVInstrInfo::I {
                        rd: 2,
                        rs1: 1,
                        imm: 20,
                    },
                    len: 4,
                },
                DecodeInstr {
                    instr: RiscvInstr::JAL,
                    info: RVInstrInfo::J { rd: 0, imm: 16 },
                    len: 4,
                },
            ],
        );

        let res = translator
            .translate_stream(&mut stream, 0x1000, IRBuilder::new())
            .expect("should translate multi-instruction block");

        assert_eq!(res.meta.guest_cnt, 3);
        assert_eq!(res.guest_bytes, 12);
        assert_eq!(
            res.stop_reason,
            BlockStopReason::DirectJump(RiscvInstr::JAL)
        );
        assert_eq!(res.meta.start_pc, 0x1000);
        assert_eq!(
            res.ir_block.terminator(),
            &Terminator::DirectJmp { target: 0x1018 }
        );
    }
}
