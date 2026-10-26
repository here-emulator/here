use super::*;
use here_types::pack_bits;

/// Opcode bytes. Deliberately excludes multi-byte integers: an unsuffixed
/// integer literal defaults to `i32`, which previously could silently emit
/// four opcode bytes.
trait Opcode {
    fn emit(self, buf: &mut CodeBuf);
}

impl Opcode for u8 {
    fn emit(self, buf: &mut CodeBuf) {
        buf.push(self);
    }
}

impl<const N: usize> Opcode for [u8; N] {
    fn emit(self, buf: &mut CodeBuf) {
        buf.extend_from_slice(&self);
    }
}

impl RegMem {
    #[inline]
    pub fn mode(&self) -> u8 {
        match *self {
            RegMem::Reg(_) => 0b11,
            // mod=00 with rbp/r13 encodes rip-relative in x86-64, so disp8 [reg+0] is used instead.
            RegMem::Mem(r) if matches!(r, X64Reg::RBP | X64Reg::R13) => 0b01,
            RegMem::Mem(_) => 0b00,
            RegMem::MemDisp8(_, _) => 0b01,
            RegMem::MemDisp32(_, _) => 0b10,
        }
    }

    pub fn emit_displacement(&self, buf: &mut CodeBuf) {
        match *self {
            RegMem::Reg(_) => {}
            // mod=01 requires 1-byte displacement (0x00 for [rbp]/[r13] with no offset).
            RegMem::Mem(r) if matches!(r, X64Reg::RBP | X64Reg::R13) => buf.push(0),
            RegMem::Mem(_) => {}
            RegMem::MemDisp8(_, byte) => buf.push(byte as u8),
            RegMem::MemDisp32(_, bytes) => buf.extend_from_slice(&bytes.to_le_bytes()),
        }
    }
}

/// REX prefix byte (`0100_WRXB`):
/// - W: 64-bit operand size promotion (0 = default/32-bit, 1 = 64-bit)
/// - R: ModR/M `reg` field extension (bit 3)
/// - X: SIB `index` field extension (bit 3)
/// - B: ModR/M `r/m`, SIB `base`, or opcode `reg` extension (bit 3)
fn emit_rex(buf: &mut CodeBuf, w: u8, r: u8, x: u8, b: u8, force: bool) {
    if force || (w | r | x | b) != 0 {
        let rex = pack_bits! {
            0b0100, 4;
            w, 1;
            r, 1;
            x, 1;
            b, 1;
        } as u8;
        buf.push(rex);
    }
}

/// ModR/M byte layout: `[ mod: 2 bits | reg/opcode_ext: 3 bits | r/m: 3 bits ]`.
fn emit_modrm(buf: &mut CodeBuf, mode: u8, reg_or_op: u8, rm: u8) {
    let modrm = pack_bits! {
        mode, 2;
        reg_or_op, 3;
        rm, 3;
    } as u8;
    buf.push(modrm);
}

/// spl, bpl, sil, dil (base 4..7) require a REX prefix to distinguish from legacy ah, ch, dh, bh.
#[inline]
fn needs_byte_rex(reg: X64Reg, rm: RegMem) -> bool {
    reg.base() >= 4 || matches!(rm, RegMem::Reg(r) if r.base() >= 4)
}

fn emit_modrm_encoding(
    buf: &mut CodeBuf,
    size: OperandSize,
    opcode: impl Opcode,
    reg_or_op: u8,
    reg_rex: u8,
    rm: RegMem,
    force_rex: bool,
) {
    // 0x66 prefix overrides default operand size to 16-bit word.
    if size.is_word() {
        buf.push(0x66);
    }
    let rm_reg = rm.reg();
    emit_rex(buf, size.rex_w(), reg_rex, 0, rm_reg.rex_bit(), force_rex);
    opcode.emit(buf);
    emit_modrm(buf, rm.mode(), reg_or_op, rm_reg.base());
    // r/m=4 in memory mode indicates presence of a SIB byte (0x24 = [rsp/r12] without index).
    // Note: General SIB (index * scale) is not supported.
    if rm_reg.base() == 4 && rm.mode() != 0b11 {
        buf.push(0x24);
    }
    rm.emit_displacement(buf);
}

/// Encodes `OP reg, r/m` (reg = dst, r/m = src).
fn emit_reg_rm(
    buf: &mut CodeBuf,
    size: OperandSize,
    opcode: impl Opcode,
    dst: X64Reg,
    src: RegMem,
) {
    let force_rex = size.is_byte() && needs_byte_rex(dst, src);
    emit_modrm_encoding(buf, size, opcode, dst.base(), dst.rex_bit(), src, force_rex);
}

/// Encodes `OP r/m, reg` (r/m = dst, reg = src).
fn emit_rm_reg(
    buf: &mut CodeBuf,
    size: OperandSize,
    opcode: impl Opcode,
    dst: RegMem,
    src: X64Reg,
) {
    let force_rex = size.is_byte() && needs_byte_rex(src, dst);
    emit_modrm_encoding(buf, size, opcode, src.base(), src.rex_bit(), dst, force_rex);
}

/// Group instruction: opcode extension in ModR/M `reg` field, operand in `r/m` field.
fn emit_group_rm(
    buf: &mut CodeBuf,
    size: OperandSize,
    opcode: impl Opcode,
    op_ext: u8,
    dst: RegMem,
) {
    let force_rex = size.is_byte() && matches!(dst, RegMem::Reg(r) if r.base() >= 4);
    emit_modrm_encoding(buf, size, opcode, op_ext, 0, dst, force_rex);
}

/// Short-form `mov reg, imm`: register index embedded in opcode (`0xb0+r` / `0xb8+r`).
fn emit_mov_reg_opcode(buf: &mut CodeBuf, size: OperandSize, dst: X64Reg) {
    if size.is_word() {
        buf.push(0x66);
    }
    let (opcode, force_rex) = match size {
        OperandSize::Byte => (0xb0_u8, dst.base() >= 4),
        _ => (0xb8_u8, false),
    };
    emit_rex(buf, size.rex_w(), 0, 0, dst.rex_bit(), force_rex);
    buf.push(opcode | dst.base());
}

/// Group 11 (`0xc6`/`0xc7` /0): `mov r/m, imm`.
fn emit_mov_rm_opcode(buf: &mut CodeBuf, size: OperandSize, dst: RegMem) {
    emit_group_rm(buf, size, size.opcode(0xc6), 0, dst);
}
impl X64Inst {
    pub fn emit(&self, buf: &mut CodeBuf) {
        match *self {
            // 0x88: mov r/m, r; 0x8a: mov r, r/m (bit 1 is direction bit 'd')
            Self::MovRmReg { size, dst, src } => {
                emit_rm_reg(buf, size, size.opcode(0x88), dst, src);
            }
            Self::MovRegRm { size, dst, src } => {
                emit_reg_rm(buf, size, size.opcode(0x8a), dst, src);
            }
            Self::MovRegImm32 { dst, imm } => {
                emit_mov_reg_opcode(buf, OperandSize::Dword, dst);
                buf.extend_from_slice(&imm.to_le_bytes());
            }
            Self::Movabs { dst, imm } => {
                emit_mov_reg_opcode(buf, OperandSize::Qword, dst);
                buf.extend_from_slice(&imm.to_le_bytes());
            }
            Self::MovRmImm32 { size, dst, imm } => match size {
                OperandSize::Dword | OperandSize::Qword => {
                    emit_mov_rm_opcode(buf, size, dst);
                    buf.extend_from_slice(&imm.to_le_bytes());
                }
                OperandSize::Byte | OperandSize::Word => {
                    panic!("MovRmImm only supports Dword and Qword operand sizes");
                }
            },
            Self::Movsxd { dst, src } => emit_reg_rm(buf, OperandSize::Qword, 0x63_u8, dst, src),
            // 32-bit destination zero-extends to 64-bit, saving a REX.W prefix byte.
            Self::Movzx8 { dst, src } => {
                let force_rex = matches!(src, RegMem::Reg(r) if r.base() >= 4);
                emit_modrm_encoding(
                    buf,
                    OperandSize::Dword,
                    [0x0f, 0xb6],
                    dst.base(),
                    dst.rex_bit(),
                    src,
                    force_rex,
                );
            }
            Self::AluRmReg { op, size, dst, src } => {
                emit_rm_reg(buf, size, op.rm_reg_opcode(size), dst, src);
            }
            Self::AluRegRm { op, size, dst, src } => {
                emit_reg_rm(buf, size, op.reg_rm_opcode(size), dst, src);
            }
            Self::AluRmImm8 { op, size, dst, imm } => {
                let opcode = if size.is_byte() { 0x80_u8 } else { 0x83_u8 };
                emit_group_rm(buf, size, opcode, op.op_ext(), dst);
                buf.push(imm as u8);
            }
            Self::AluRmImm32 { op, size, dst, imm } => {
                debug_assert!(
                    matches!(size, OperandSize::Dword | OperandSize::Qword),
                    "x86 opcode 0x81 only supports 32-bit and 64-bit operands with imm32"
                );
                emit_group_rm(buf, size, 0x81_u8, op.op_ext(), dst);
                buf.extend_from_slice(&imm.to_le_bytes());
            }
            // Group 2: 0xd2/0xd3 for shift by CL, 0xc0/0xc1 for shift by imm8
            Self::ShiftCl { op, size, dst } => {
                emit_group_rm(buf, size, size.opcode(0xd2), op.op_ext(), dst);
            }
            Self::ShiftImm { op, size, dst, imm } => {
                emit_group_rm(buf, size, size.opcode(0xc0), op.op_ext(), dst);
                buf.push(imm.get());
            }
            Self::Cmovcc {
                condition,
                dst,
                src,
            } => {
                emit_reg_rm(
                    buf,
                    OperandSize::Qword,
                    [0x0f, 0x40 | condition as u8],
                    dst,
                    src,
                );
            }
            Self::Setcc { condition, dst } => {
                emit_group_rm(
                    buf,
                    OperandSize::Byte,
                    [0x0f, 0x90 | condition as u8],
                    0,
                    dst.into(),
                );
            }
            // Group 5: 0xff /2 (call) defaults to 64-bit operand size in x86-64
            Self::Call { target } => {
                emit_group_rm(buf, OperandSize::Dword, 0xff_u8, 2, target.into())
            }
            // 0xe9: jmp rel32; 0x0f 0x80+cc: jcc rel32
            Self::JmpRel { offset } => {
                buf.push(0xe9);
                buf.extend_from_slice(&offset.to_le_bytes());
            }
            Self::Jcc { condition, offset } => {
                buf.extend_from_slice(&[0x0f, 0x80 | condition as u8]);
                buf.extend_from_slice(&offset.to_le_bytes());
            }
            Self::Ret => buf.push(0xc3),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use X64Reg::*;

    fn emit_inst(inst: X64Inst) -> Vec<u8> {
        let mut buf = Vec::new();
        inst.emit(&mut buf);
        buf
    }

    #[test]
    fn test_immediate_form_selection() {
        assert_eq!(
            X64Inst::alu_imm(AluOp::Add, OperandSize::Qword, RAX, 1),
            X64Inst::AluRmImm8 {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: RAX.into(),
                imm: 1,
            }
        );
        assert_eq!(
            X64Inst::alu_imm(AluOp::Add, OperandSize::Qword, RAX, 0x100),
            X64Inst::AluRmImm32 {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: RAX.into(),
                imm: 0x100,
            }
        );

        assert_eq!(ShiftCount::new(OperandSize::Dword, 31).get(), 31);
        assert_eq!(ShiftCount::new(OperandSize::Dword, 33).get(), 1);
        assert_eq!(ShiftCount::new(OperandSize::Qword, 64).get(), 0);
        assert_eq!(ShiftCount::new(OperandSize::Qword, 63).get(), 63);
    }
    #[test]
    fn test_encodings() {
        // Dword without REX prefix on legacy registers:
        assert_eq!(
            emit_inst(X64Inst::AluRmReg {
                op: AluOp::Add,
                size: OperandSize::Dword,
                dst: RAX.into(),
                src: RBX,
            }),
            [0x01, 0xd8]
        );
        // Dword with extended registers has REX prefix:
        assert_eq!(
            emit_inst(X64Inst::AluRmReg {
                op: AluOp::Add,
                size: OperandSize::Dword,
                dst: R8.into(),
                src: R9,
            }),
            [0x45, 0x01, 0xc8]
        );
        // Qword has REX.W:
        assert_eq!(
            emit_inst(X64Inst::AluRmReg {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: R8.into(),
                src: R9,
            }),
            [0x4d, 0x01, 0xc8]
        );
        // AluRegRm (add reg, r/m):
        assert_eq!(
            emit_inst(X64Inst::AluRegRm {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: RAX,
                src: RBX.into(),
            }),
            [0x48, 0x03, 0xc3]
        );
        // AluRmImm8 with small immediate (uses 0x83 imm8):
        assert_eq!(
            emit_inst(X64Inst::AluRmImm8 {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: RAX.into(),
                imm: 1,
            }),
            [0x48, 0x83, 0xc0, 0x01]
        );
        // AluRmImm32 with large immediate (uses 0x81 imm32):
        assert_eq!(
            emit_inst(X64Inst::AluRmImm32 {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: R8.into(),
                imm: 0x89ab_cdefu32 as i32,
            }),
            [0x49, 0x81, 0xc0, 0xef, 0xcd, 0xab, 0x89]
        );
        assert_eq!(
            emit_inst(X64Inst::AluRmImm8 {
                op: AluOp::Add,
                size: OperandSize::Byte,
                dst: RAX.into(),
                imm: 0x7f,
            }),
            [0x80, 0xc0, 0x7f]
        );
        assert_eq!(
            emit_inst(X64Inst::AluRmImm8 {
                op: AluOp::Add,
                size: OperandSize::Word,
                dst: RAX.into(),
                imm: -1,
            }),
            [0x66, 0x83, 0xc0, 0xff]
        );
        assert_eq!(
            emit_inst(X64Inst::AluRmImm8 {
                op: AluOp::Add,
                size: OperandSize::Dword,
                dst: RAX.into(),
                imm: -1,
            }),
            [0x83, 0xc0, 0xff]
        );
        assert_eq!(
            emit_inst(X64Inst::AluRmImm32 {
                op: AluOp::Add,
                size: OperandSize::Dword,
                dst: RAX.into(),
                imm: 0x1234_5678,
            }),
            [0x81, 0xc0, 0x78, 0x56, 0x34, 0x12]
        );
        assert_eq!(
            emit_inst(X64Inst::AluRmImm32 {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: RAX.into(),
                imm: 0x100,
            }),
            [0x48, 0x81, 0xc0, 0x00, 0x01, 0x00, 0x00]
        );
        // MovRegImm32 (fits in 32-bit: zero-extended 5 bytes)
        assert_eq!(
            emit_inst(X64Inst::MovRegImm32 {
                dst: RAX,
                imm: 0x1234,
            }),
            [0xb8, 0x34, 0x12, 0x00, 0x00]
        );
        // Movabs (full 64-bit: 10 bytes)
        assert_eq!(
            emit_inst(X64Inst::Movabs {
                dst: RAX,
                imm: 0x1234_5678_9abc_def0,
            }),
            [0x48, 0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12]
        );
        // mov_reg_imm helper with sign-extended i32 selects MovRmImm (7 bytes)
        assert_eq!(
            emit_inst(X64Inst::mov_reg_imm(RAX, -1i64 as u64)),
            [0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff]
        );
        assert_eq!(
            emit_inst(X64Inst::mov_reg_imm(R8, -1i64 as u64)),
            [0x49, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff]
        );
        // MovImm into memory (mov [rsp + 8], 0):
        assert_eq!(
            emit_inst(X64Inst::MovRmImm32 {
                size: OperandSize::Qword,
                dst: RegMem::from_disp(RSP, 8),
                imm: 0,
            }),
            [0x48, 0xc7, 0x44, 0x24, 0x08, 0x00, 0x00, 0x00, 0x00]
        );
        // MovImm into memory with sign-extended value (mov qword ptr [rsp + 8], -1):
        assert_eq!(
            emit_inst(X64Inst::MovRmImm32 {
                size: OperandSize::Qword,
                dst: RegMem::from_disp(RSP, 8),
                imm: -1,
            }),
            [0x48, 0xc7, 0x44, 0x24, 0x08, 0xff, 0xff, 0xff, 0xff]
        );
        // MovImm into memory (mov dword ptr [rsp + 8], 0x1234_5678):
        assert_eq!(
            emit_inst(X64Inst::MovRmImm32 {
                size: OperandSize::Dword,
                dst: RegMem::from_disp(RSP, 8),
                imm: 0x1234_5678,
            }),
            [0xc7, 0x44, 0x24, 0x08, 0x78, 0x56, 0x34, 0x12]
        );
        assert_eq!(
            emit_inst(X64Inst::ShiftImm {
                op: ShiftOp::Shl,
                size: OperandSize::Dword,
                dst: R8.into(),
                imm: ShiftCount::new(OperandSize::Dword, 31),
            }),
            [0x41, 0xc1, 0xe0, 0x1f]
        );
        assert_eq!(
            emit_inst(X64Inst::ShiftCl {
                op: ShiftOp::Sar,
                size: OperandSize::Qword,
                dst: RAX.into(),
            }),
            [0x48, 0xd3, 0xf8]
        );
        // Mov Dword on legacy regs (no REX):
        assert_eq!(
            emit_inst(X64Inst::MovRegRm {
                size: OperandSize::Dword,
                dst: RAX,
                src: RBX.into(),
            }),
            [0x8b, 0xc3]
        );
        // Mov Qword on legacy regs (REX.W):
        assert_eq!(
            emit_inst(X64Inst::MovRegRm {
                size: OperandSize::Qword,
                dst: RAX,
                src: RBX.into(),
            }),
            [0x48, 0x8b, 0xc3]
        );
        assert_eq!(
            emit_inst(X64Inst::MovRmReg {
                size: OperandSize::Qword,
                dst: RegMem::from_disp(RSP, 16),
                src: RAX,
            }),
            [0x48, 0x89, 0x44, 0x24, 0x10]
        );
        // RBP / R13 protection (must use disp8 0x00 instead of RIP-relative)
        assert_eq!(
            emit_inst(X64Inst::MovRegRm {
                size: OperandSize::Qword,
                dst: RAX,
                src: RegMem::Mem(RBP),
            }),
            [0x48, 0x8b, 0x45, 0x00]
        );
        assert_eq!(
            emit_inst(X64Inst::MovRegRm {
                size: OperandSize::Qword,
                dst: RAX,
                src: RegMem::from_disp(RBP, 0),
            }),
            [0x48, 0x8b, 0x45, 0x00]
        );
        assert_eq!(
            emit_inst(X64Inst::MovRegRm {
                size: OperandSize::Qword,
                dst: RAX,
                src: RegMem::Mem(R13),
            }),
            [0x49, 0x8b, 0x45, 0x00]
        );
        assert_eq!(
            emit_inst(X64Inst::Movsxd {
                dst: RAX,
                src: RDX.into(),
            }),
            [0x48, 0x63, 0xc2]
        );
        // Setcc with AL (base 0, no REX needed):
        assert_eq!(
            emit_inst(X64Inst::Setcc {
                condition: ConditionCode::Equal,
                dst: RAX,
            }),
            [0x0f, 0x94, 0xc0]
        );
        // Setcc with SIL (base 6, REX 0x40 required for byte register):
        assert_eq!(
            emit_inst(X64Inst::Setcc {
                condition: ConditionCode::Equal,
                dst: RSI,
            }),
            [0x40, 0x0f, 0x94, 0xc6]
        );
        // Setcc with R8B (extended register, REX 0x41):
        assert_eq!(
            emit_inst(X64Inst::Setcc {
                condition: ConditionCode::Equal,
                dst: R8,
            }),
            [0x41, 0x0f, 0x94, 0xc0]
        );
        // Movzx8 with AL (base 0, no REX needed):
        assert_eq!(
            emit_inst(X64Inst::Movzx8 {
                dst: RAX,
                src: RAX.into(),
            }),
            [0x0f, 0xb6, 0xc0]
        );
        // Movzx8 with DIL (base 7, REX 0x40 required for byte register):
        assert_eq!(
            emit_inst(X64Inst::Movzx8 {
                dst: RDI,
                src: RDI.into(),
            }),
            [0x40, 0x0f, 0xb6, 0xff]
        );
        // Movzx8 with SIL (base 6, REX 0x40 required for byte register):
        assert_eq!(
            emit_inst(X64Inst::Movzx8 {
                dst: RAX,
                src: RSI.into(),
            }),
            [0x40, 0x0f, 0xb6, 0xc6]
        );
        // Movzx8 with R8D destination (extended register, REX 0x44):
        assert_eq!(
            emit_inst(X64Inst::Movzx8 {
                dst: R8,
                src: RAX.into(),
            }),
            [0x44, 0x0f, 0xb6, 0xc0]
        );
        assert_eq!(
            emit_inst(X64Inst::Cmovcc {
                condition: ConditionCode::Equal,
                dst: RAX,
                src: RBX.into(),
            }),
            [0x48, 0x0f, 0x44, 0xc3]
        );
        // Call without REX for legacy reg:
        assert_eq!(emit_inst(X64Inst::Call { target: RAX }), [0xff, 0xd0]);
        assert_eq!(emit_inst(X64Inst::Call { target: R11 }), [0x41, 0xff, 0xd3]);
        assert_eq!(emit_inst(X64Inst::Ret), [0xc3]);
        // Byte store (sb semantics): mov byte ptr [rdi], sil
        assert_eq!(
            emit_inst(X64Inst::MovRmReg {
                size: OperandSize::Byte,
                dst: RegMem::Mem(RDI),
                src: RSI,
            }),
            [0x40, 0x88, 0x37]
        );
        // Byte load: mov sil, byte ptr [rdi]
        assert_eq!(
            emit_inst(X64Inst::MovRegRm {
                size: OperandSize::Byte,
                dst: RSI,
                src: RegMem::Mem(RDI),
            }),
            [0x40, 0x8a, 0x37]
        );
        // Word store (sh semantics): mov word ptr [rdi], si
        assert_eq!(
            emit_inst(X64Inst::MovRmReg {
                size: OperandSize::Word,
                dst: RegMem::Mem(RDI),
                src: RSI,
            }),
            [0x66, 0x89, 0x37]
        );
        // Word load: mov si, word ptr [rdi]
        assert_eq!(
            emit_inst(X64Inst::MovRegRm {
                size: OperandSize::Word,
                dst: RSI,
                src: RegMem::Mem(RDI),
            }),
            [0x66, 0x8b, 0x37]
        );
    }

    #[test]
    #[should_panic(expected = "MovRmImm only supports Dword and Qword operand sizes")]
    fn test_mov_rm_imm_byte_panics() {
        emit_inst(X64Inst::MovRmImm32 {
            size: OperandSize::Byte,
            dst: RegMem::Mem(RDI),
            imm: 0x42,
        });
    }

    #[test]
    #[should_panic(expected = "MovRmImm only supports Dword and Qword operand sizes")]
    fn test_mov_rm_imm_word_panics() {
        emit_inst(X64Inst::MovRmImm32 {
            size: OperandSize::Word,
            dst: RegMem::Mem(RDI),
            imm: 0x1234,
        });
    }
}
