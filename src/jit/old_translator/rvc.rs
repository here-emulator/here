use super::*;
use crate::debug_unreachable;

impl X86CodeGen {
    pub(super) fn translate_rvc(
        &mut self,
        instr: RiscvInstr,
        info: RVInstrInfo,
        context: JitInfo,
    ) -> TranslateResult {
        use RiscvInstr::*;

        match instr {
            C_ADD | C_SUB | C_AND | C_OR | C_XOR | C_ADDW | C_SUBW => {
                let (rd, rs2) = compressed_r_args(info);
                let (op, commutative, width) = match instr {
                    C_ADD => (AluOp::Add, true, OperandWidth::Xlen),
                    C_SUB => (AluOp::Sub, false, OperandWidth::Xlen),
                    C_AND => (AluOp::And, true, OperandWidth::Xlen),
                    C_OR => (AluOp::Or, true, OperandWidth::Xlen),
                    C_XOR => (AluOp::Xor, true, OperandWidth::Xlen),
                    C_ADDW => (AluOp::Add, true, OperandWidth::Word),
                    C_SUBW => (AluOp::Sub, false, OperandWidth::Word),
                    _ => debug_unreachable!(),
                };
                self.translate_alu_r(rd, rd, rs2, op, commutative, width);
            }
            C_ADDI | C_ADDI16SP | C_ANDI | C_ADDIW => {
                let (rd, imm) = compressed_i_args(info);
                let (op, width) = match instr {
                    C_ADDI | C_ADDI16SP => (AluOp::Add, OperandWidth::Xlen),
                    C_ANDI => (AluOp::And, OperandWidth::Xlen),
                    C_ADDIW => (AluOp::Add, OperandWidth::Word),
                    _ => debug_unreachable!(),
                };
                self.translate_alu_i(rd, rd, imm, op, width);
            }
            C_ADDI4SPN => {
                let RVInstrInfo::CIW { rd, imm } = info else {
                    debug_unreachable!();
                };
                self.translate_alu_i(rd, 2, imm, AluOp::Add, OperandWidth::Xlen);
            }
            C_SLLI | C_SRLI | C_SRAI => {
                let (rd, imm) = compressed_i_args(info);
                let op = match instr {
                    C_SLLI => ShiftOp::SHL,
                    C_SRLI => ShiftOp::SHR,
                    C_SRAI => ShiftOp::SAR,
                    _ => debug_unreachable!(),
                };
                self.translate_shift_i(rd, rd, imm, op, OperandWidth::Xlen);
            }
            C_MV => {
                let RVInstrInfo::CR { rd_rs1: rd, rs2 } = info else {
                    debug_unreachable!();
                };
                self.translate_move(rd, rs2);
            }
            C_LI | C_LUI => {
                let (rd, imm) = compressed_i_args(info);
                self.translate_load_imm(rd, imm);
            }
            C_BEQZ | C_BNEZ => {
                let RVInstrInfo::CB { rd_rs1, imm } = info else {
                    debug_unreachable!();
                };
                let condition = match instr {
                    C_BEQZ => ConditionCode::Equal,
                    C_BNEZ => ConditionCode::NotEqual,
                    _ => debug_unreachable!(),
                };
                self.translate_branch(rd_rs1, None, imm, condition, context);
                return TranslateResult::ControlFlow;
            }
            C_J | C_JAL => {
                let RVInstrInfo::CJ { target } = info else {
                    debug_unreachable!();
                };
                let rd = (instr == C_JAL).then_some(1);
                self.translate_jump_imm(rd, target, context);
                return TranslateResult::ControlFlow;
            }
            C_JR | C_JALR => {
                let RVInstrInfo::CR {
                    rd_rs1: rs1,
                    rs2: _,
                } = info
                else {
                    debug_unreachable!();
                };
                let rd = (instr == C_JALR).then_some(1);
                self.translate_jump_reg(rd, rs1, 0, context);
                return TranslateResult::ControlFlow;
            }
            C_LW | C_LD => {
                let RVInstrInfo::CL { rd, rs1, imm } = info else {
                    debug_unreachable!();
                };
                let kind = match instr {
                    C_LW => LoadKind::SignedWord,
                    C_LD => LoadKind::DoubleWord,
                    _ => debug_unreachable!(),
                };
                self.translate_load(rd, rs1, imm, kind, context);
            }
            C_LWSP | C_LDSP => {
                let RVInstrInfo::CI { rd_rs1: rd, imm } = info else {
                    debug_unreachable!();
                };
                let kind = match instr {
                    C_LWSP => LoadKind::SignedWord,
                    C_LDSP => LoadKind::DoubleWord,
                    _ => debug_unreachable!(),
                };
                self.translate_load(rd, 2, imm, kind, context);
            }
            C_SW | C_SD => {
                let RVInstrInfo::CS { rs1, rs2, imm } = info else {
                    debug_unreachable!();
                };
                let kind = match instr {
                    C_SW => StoreKind::Word,
                    C_SD => StoreKind::DoubleWord,
                    _ => debug_unreachable!(),
                };
                self.translate_store(rs1, rs2, imm, kind, context);
            }
            C_SWSP | C_SDSP => {
                let RVInstrInfo::CSS { rs2, imm } = info else {
                    debug_unreachable!();
                };
                let kind = match instr {
                    C_SWSP => StoreKind::Word,
                    C_SDSP => StoreKind::DoubleWord,
                    _ => debug_unreachable!(),
                };
                self.translate_store(2, rs2, imm, kind, context);
            }
            C_NOP => {}
            _ => return TranslateResult::Unsupported,
        }

        TranslateResult::Continue
    }
}

fn compressed_r_args(info: RVInstrInfo) -> (u8, u8) {
    match info {
        RVInstrInfo::CR { rd_rs1, rs2 } | RVInstrInfo::CA { rd_rs1, rs2 } => (rd_rs1, rs2),
        _ => debug_unreachable!(),
    }
}

fn compressed_i_args(info: RVInstrInfo) -> (u8, WordType) {
    match info {
        RVInstrInfo::CI { rd_rs1, imm } | RVInstrInfo::CB { rd_rs1, imm } => (rd_rs1, imm),
        _ => debug_unreachable!(),
    }
}
