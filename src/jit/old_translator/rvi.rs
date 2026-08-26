use super::*;
use crate::debug_unreachable;

impl X86CodeGen {
    pub(super) fn translate_rvi(
        &mut self,
        instr: RiscvInstr,
        info: RVInstrInfo,
        context: JitInfo,
    ) -> TranslateResult {
        use RiscvInstr::*;

        match instr {
            ADD | SUB | AND | OR | XOR | ADDW | SUBW => {
                let (rd, rs1, rs2) = r_args(info);
                let (op, commutative, width) = match instr {
                    ADD => (AluOp::Add, true, OperandWidth::Xlen),
                    SUB => (AluOp::Sub, false, OperandWidth::Xlen),
                    AND => (AluOp::And, true, OperandWidth::Xlen),
                    OR => (AluOp::Or, true, OperandWidth::Xlen),
                    XOR => (AluOp::Xor, true, OperandWidth::Xlen),
                    ADDW => (AluOp::Add, true, OperandWidth::Word),
                    SUBW => (AluOp::Sub, false, OperandWidth::Word),
                    _ => debug_unreachable!(),
                };
                self.translate_alu_r(rd, rs1, rs2, op, commutative, width);
            }
            ADDI | ANDI | ORI | XORI | ADDIW => {
                let (rd, rs1, imm) = i_args(info);
                let (op, width) = match instr {
                    ADDI => (AluOp::Add, OperandWidth::Xlen),
                    ANDI => (AluOp::And, OperandWidth::Xlen),
                    ORI => (AluOp::Or, OperandWidth::Xlen),
                    XORI => (AluOp::Xor, OperandWidth::Xlen),
                    ADDIW => (AluOp::Add, OperandWidth::Word),
                    _ => debug_unreachable!(),
                };
                self.translate_alu_i(rd, rs1, imm, op, width);
            }
            SLL | SRL | SRA | SLLW | SRLW | SRAW => {
                let (rd, rs1, rs2) = r_args(info);
                let (op, width) = match instr {
                    SLL => (ShiftOp::SHL, OperandWidth::Xlen),
                    SRL => (ShiftOp::SHR, OperandWidth::Xlen),
                    SRA => (ShiftOp::SAR, OperandWidth::Xlen),
                    SLLW => (ShiftOp::SHL, OperandWidth::Word),
                    SRLW => (ShiftOp::SHR, OperandWidth::Word),
                    SRAW => (ShiftOp::SAR, OperandWidth::Word),
                    _ => debug_unreachable!(),
                };
                self.translate_shift_r(rd, rs1, rs2, op, width);
            }
            SLLI | SRLI | SRAI | SLLIW | SRLIW | SRAIW => {
                let (rd, rs1, imm) = i_args(info);
                let (op, width) = match instr {
                    SLLI => (ShiftOp::SHL, OperandWidth::Xlen),
                    SRLI => (ShiftOp::SHR, OperandWidth::Xlen),
                    SRAI => (ShiftOp::SAR, OperandWidth::Xlen),
                    SLLIW => (ShiftOp::SHL, OperandWidth::Word),
                    SRLIW => (ShiftOp::SHR, OperandWidth::Word),
                    SRAIW => (ShiftOp::SAR, OperandWidth::Word),
                    _ => debug_unreachable!(),
                };
                self.translate_shift_i(rd, rs1, imm, op, width);
            }
            SLT | SLTU => {
                let (rd, rs1, rs2) = r_args(info);
                let condition = match instr {
                    SLT => ConditionCode::Less,
                    SLTU => ConditionCode::Below,
                    _ => debug_unreachable!(),
                };
                self.translate_compare_r(rd, rs1, rs2, condition);
            }
            SLTI | SLTIU => {
                let (rd, rs1, imm) = i_args(info);
                let condition = match instr {
                    SLTI => ConditionCode::Less,
                    SLTIU => ConditionCode::Below,
                    _ => debug_unreachable!(),
                };
                self.translate_compare_i(rd, rs1, imm, condition);
            }
            BEQ | BNE | BLT | BGE | BLTU | BGEU => {
                let (rs1, rs2, imm) = b_args(info);
                let condition = match instr {
                    BEQ => ConditionCode::Equal,
                    BNE => ConditionCode::NotEqual,
                    BLT => ConditionCode::Less,
                    BGE => ConditionCode::GreaterOrEqual,
                    BLTU => ConditionCode::Below,
                    BGEU => ConditionCode::AboveOrEqual,
                    _ => debug_unreachable!(),
                };
                self.translate_branch(rs1, Some(rs2), imm, condition, context);
                return TranslateResult::ControlFlow;
            }
            JAL => {
                let RVInstrInfo::J { rd, imm } = info else {
                    debug_unreachable!();
                };
                self.translate_jump_imm(Some(rd), imm, context);
                return TranslateResult::ControlFlow;
            }
            JALR => {
                let (rd, rs1, imm) = i_args(info);
                self.translate_jump_reg(Some(rd), rs1, imm, context);
                return TranslateResult::ControlFlow;
            }
            LUI | AUIPC => {
                let (rd, imm) = u_args(info);
                let value = match instr {
                    LUI => imm,
                    AUIPC => context.instr_pc.wrapping_add(imm),
                    _ => debug_unreachable!(),
                };
                self.translate_load_imm(rd, value);
            }
            LB | LBU | LH | LHU | LW | LWU | LD => {
                let (rd, rs1, imm) = i_args(info);
                let kind = match instr {
                    LB => LoadKind::SignedByte,
                    LBU => LoadKind::UnsignedByte,
                    LH => LoadKind::SignedHalf,
                    LHU => LoadKind::UnsignedHalf,
                    LW => LoadKind::SignedWord,
                    LWU => LoadKind::UnsignedWord,
                    LD => LoadKind::DoubleWord,
                    _ => debug_unreachable!(),
                };
                self.translate_load(rd, rs1, imm, kind, context);
            }
            SB | SH | SW | SD => {
                let RVInstrInfo::S { rs1, rs2, imm } = info else {
                    debug_unreachable!();
                };
                let kind = match instr {
                    SB => StoreKind::Byte,
                    SH => StoreKind::Half,
                    SW => StoreKind::Word,
                    SD => StoreKind::DoubleWord,
                    _ => debug_unreachable!(),
                };
                self.translate_store(rs1, rs2, imm, kind, context);
            }
            _ => return TranslateResult::Unsupported,
        }

        TranslateResult::Continue
    }
}

fn b_args(info: RVInstrInfo) -> (u8, u8, WordType) {
    let RVInstrInfo::B { rs1, rs2, imm } = info else {
        debug_unreachable!();
    };
    (rs1, rs2, imm)
}

fn r_args(info: RVInstrInfo) -> (u8, u8, u8) {
    let RVInstrInfo::R { rd, rs1, rs2 } = info else {
        debug_unreachable!();
    };
    (rd, rs1, rs2)
}

fn i_args(info: RVInstrInfo) -> (u8, u8, WordType) {
    let RVInstrInfo::I { rd, rs1, imm } = info else {
        debug_unreachable!();
    };
    (rd, rs1, imm)
}

fn u_args(info: RVInstrInfo) -> (u8, WordType) {
    let RVInstrInfo::U { rd, imm } = info else {
        debug_unreachable!();
    };
    (rd, imm)
}
