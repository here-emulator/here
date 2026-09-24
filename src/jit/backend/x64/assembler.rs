use super::*;
use crate::pack_bits;

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
    opcode: impl Emitable,
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
    opcode.emit_to(buf);
    emit_modrm(buf, rm.mode(), reg_or_op, rm_reg.base());
    // r/m=4 in memory mode indicates presence of a SIB byte (0x24 = [rsp/r12] without index).
    if rm_reg.base() == 4 && rm.mode() != 0b11 {
        buf.push(0x24);
    }
    rm.emit_displacement(buf);
}

/// Encodes `OP reg, r/m` (reg = dst, r/m = src).
fn emit_reg_rm(
    buf: &mut CodeBuf,
    size: OperandSize,
    opcode: impl Emitable,
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
    opcode: impl Emitable,
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
    opcode: impl Emitable,
    op_ext: u8,
    dst: RegMem,
) {
    let force_rex = size.is_byte() && matches!(dst, RegMem::Reg(r) if r.base() >= 4);
    emit_modrm_encoding(buf, size, opcode, op_ext, 0, dst, force_rex);
}

/// Short-form `mov reg, imm`: register index embedded in opcode (`0xb0+r` / `0xb8+r`).
fn emit_mov_reg_imm(buf: &mut CodeBuf, size: OperandSize, dst: X64Reg, imm: u64) {
    if size.is_word() {
        buf.push(0x66);
    }
    let (opcode, force_rex) = match size {
        OperandSize::Byte => (0xb0_u8, dst.base() >= 4),
        _ => (0xb8_u8, false),
    };
    emit_rex(buf, size.rex_w(), 0, 0, dst.rex_bit(), force_rex);
    buf.push(opcode | dst.base());
    match size {
        OperandSize::Byte => (imm as u8).emit_to(buf),
        OperandSize::Word => (imm as u16).emit_to(buf),
        OperandSize::Dword => (imm as u32).emit_to(buf),
        OperandSize::Qword => imm.emit_to(buf),
    }
}

/// Group 11 (`0xc6`/`0xc7` /0): `mov r/m, imm`.
fn emit_mov_rm_imm(buf: &mut CodeBuf, size: OperandSize, dst: RegMem, imm: u64) {
    emit_group_rm(buf, size, size.opcode(0xc6), 0, dst);
    match size {
        OperandSize::Byte => (imm as u8).emit_to(buf),
        OperandSize::Word => (imm as u16).emit_to(buf),
        OperandSize::Dword => (imm as u32).emit_to(buf),
        OperandSize::Qword => (imm as u32).emit_to(buf),
    }
}

/// ModR/M group immediate: in x86-64, 64-bit operands only take a 32-bit sign-extended immediate.
// TODO: why not extract a function that can automatically shrink imm size?
fn emit_modrm_imm(buf: &mut CodeBuf, size: OperandSize, imm: u32) {
    match size {
        OperandSize::Byte => (imm as u8).emit_to(buf),
        OperandSize::Word => (imm as u16).emit_to(buf),
        _ => imm.emit_to(buf),
    }
}

impl X64Inst {
    pub fn emit(&self, buf: &mut CodeBuf) {
        match *self {
            // 0x88: mov r/m, r; 0x8a: mov r, r/m (bit 1 is direction bit 'd')
            Self::Mov { size, dst, src } => match (dst, src) {
                (RegMem::Reg(d), s) => emit_reg_rm(buf, size, size.opcode(0x8a), d, s),
                (d, RegMem::Reg(s)) => emit_rm_reg(buf, size, size.opcode(0x88), d, s),
                _ => panic!("memory to memory mov unsupported"),
            },
            Self::MovImm { size, dst, imm } => match dst {
                RegMem::Reg(d) => match size {
                    OperandSize::Qword => {
                        if imm <= u32::MAX as u64 {
                            // 32-bit mov reg, imm32 zero-extends into 64-bit register (5 bytes)
                            emit_mov_reg_imm(buf, OperandSize::Dword, d, imm);
                        } else if i32::try_from(imm as i64).is_ok() {
                            // 32-bit sign-extended mov r/m64, imm32 (7 bytes)
                            emit_mov_rm_imm(buf, size, dst, imm);
                        } else {
                            // Full 64-bit movabs reg64, imm64 (10 bytes)
                            emit_mov_reg_imm(buf, size, d, imm);
                        }
                    }
                    _ => emit_mov_reg_imm(buf, size, d, imm),
                },
                _ => emit_mov_rm_imm(buf, size, dst, imm),
            },
            Self::Movsxd { dst, src } => emit_reg_rm(buf, OperandSize::Qword, 0x63_u8, dst, src),
            Self::Movsx8 { dst, src } => {
                let force_rex = matches!(src, RegMem::Reg(r) if r.base() >= 4);
                emit_modrm_encoding(
                    buf,
                    OperandSize::Qword,
                    [0x0f, 0xbe],
                    dst.base(),
                    dst.rex_bit(),
                    src,
                    force_rex,
                );
            }
            Self::Movsx16 { dst, src } => {
                emit_reg_rm(buf, OperandSize::Qword, [0x0f, 0xbf], dst, src)
            }
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
            Self::Movzx16 { dst, src } => {
                emit_reg_rm(buf, OperandSize::Dword, [0x0f, 0xb7], dst, src)
            }
            Self::Alu { op, size, dst, src } => {
                emit_rm_reg(buf, size, op.rm_reg_opcode(size), dst, src);
            }
            Self::AluImm { op, size, dst, imm } => {
                let (opcode, is_imm8) = match size {
                    OperandSize::Byte => (0x80_u8, true),
                    _ if i8::try_from(imm as i32).is_ok() => (0x83_u8, true),
                    _ => (0x81_u8, false),
                };
                emit_group_rm(buf, size, opcode, op.op_ext(), dst);
                if is_imm8 {
                    buf.push(imm as u8);
                } else {
                    emit_modrm_imm(buf, size, imm);
                }
            }
            Self::Test { size, dst, src } => {
                emit_rm_reg(buf, size, size.opcode(0x84), dst, src);
            }
            // Group 3: 0xf6/0xf7 /0
            Self::TestImm { size, dst, imm } => {
                emit_group_rm(buf, size, size.opcode(0xf6), 0, dst);
                emit_modrm_imm(buf, size, imm);
            }
            // Group 2: 0xd2/0xd3 for shift by CL, 0xc0/0xc1 for shift by imm8
            Self::ShiftCl { op, size, dst } => {
                emit_group_rm(buf, size, size.opcode(0xd2), op.op_ext(), dst);
            }
            Self::ShiftImm { op, size, dst, imm } => {
                emit_group_rm(buf, size, size.opcode(0xc0), op.op_ext(), dst);
                buf.push(imm);
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
            // Group 5: 0xff /2 (call) and 0xff /4 (jmp) default to 64-bit operand size in x86-64
            Self::Call { target } => {
                emit_group_rm(buf, OperandSize::Dword, 0xff_u8, 2, target.into())
            }
            Self::Jmp { target } => {
                emit_group_rm(buf, OperandSize::Dword, 0xff_u8, 4, target.into())
            }
            // 0xe9: jmp rel32; 0x0f 0x80+cc: jcc rel32
            Self::JmpRel { offset } => {
                buf.push(0xe9);
                offset.emit_to(buf);
            }
            Self::Jcc { condition, offset } => {
                [0x0f, 0x80 | condition as u8].emit_to(buf);
                offset.emit_to(buf);
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
    fn test_encodings() {
        // Dword without REX prefix on legacy registers:
        assert_eq!(
            emit_inst(X64Inst::Alu {
                op: AluOp::Add,
                size: OperandSize::Dword,
                dst: RAX.into(),
                src: RBX,
            }),
            [0x01, 0xd8]
        );
        // Dword with extended registers has REX prefix:
        assert_eq!(
            emit_inst(X64Inst::Alu {
                op: AluOp::Add,
                size: OperandSize::Dword,
                dst: R8.into(),
                src: R9,
            }),
            [0x45, 0x01, 0xc8]
        );
        // Qword has REX.W:
        assert_eq!(
            emit_inst(X64Inst::Alu {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: R8.into(),
                src: R9,
            }),
            [0x4d, 0x01, 0xc8]
        );
        // AluImm with small immediate (uses 0x83 imm8):
        assert_eq!(
            emit_inst(X64Inst::AluImm {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: RAX.into(),
                imm: 1,
            }),
            [0x48, 0x83, 0xc0, 0x01]
        );
        // AluImm with large immediate (uses 0x81 imm32):
        assert_eq!(
            emit_inst(X64Inst::AluImm {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: R8.into(),
                imm: 0x89ab_cdef,
            }),
            [0x49, 0x81, 0xc0, 0xef, 0xcd, 0xab, 0x89]
        );
        // Test instruction
        assert_eq!(
            emit_inst(X64Inst::Test {
                size: OperandSize::Qword,
                dst: RAX.into(),
                src: RAX,
            }),
            [0x48, 0x85, 0xc0]
        );
        assert_eq!(
            emit_inst(X64Inst::Test {
                size: OperandSize::Dword,
                dst: RAX.into(),
                src: RAX,
            }),
            [0x85, 0xc0]
        );
        // TestImm instruction
        assert_eq!(
            emit_inst(X64Inst::TestImm {
                size: OperandSize::Qword,
                dst: RAX.into(),
                imm: 1,
            }),
            [0x48, 0xf7, 0xc0, 0x01, 0x00, 0x00, 0x00]
        );
        // MovImm (fits in 32-bit: zero-extended 5 bytes)
        assert_eq!(
            emit_inst(X64Inst::MovImm {
                size: OperandSize::Qword,
                dst: RAX.into(),
                imm: 0x1234,
            }),
            [0xb8, 0x34, 0x12, 0x00, 0x00]
        );
        // MovImm (full 64-bit: 10 bytes)
        assert_eq!(
            emit_inst(X64Inst::MovImm {
                size: OperandSize::Qword,
                dst: RAX.into(),
                imm: 0x1234_5678_9abc_def0,
            }),
            [0x48, 0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12]
        );
        // MovImm (sign-extended i32: 7 bytes)
        assert_eq!(
            emit_inst(X64Inst::MovImm {
                size: OperandSize::Qword,
                dst: RAX.into(),
                imm: -1i64 as u64,
            }),
            [0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff]
        );
        // MovImm into memory (mov [rsp + 8], 0):
        assert_eq!(
            emit_inst(X64Inst::MovImm {
                size: OperandSize::Qword,
                dst: RegMem::from_disp(RSP, 8),
                imm: 0,
            }),
            [0x48, 0xc7, 0x44, 0x24, 0x08, 0x00, 0x00, 0x00, 0x00]
        );
        // MovImm into memory with sign-extended value (mov qword ptr [rsp + 8], -1):
        assert_eq!(
            emit_inst(X64Inst::MovImm {
                size: OperandSize::Qword,
                dst: RegMem::from_disp(RSP, 8),
                imm: -1i64 as u64,
            }),
            [0x48, 0xc7, 0x44, 0x24, 0x08, 0xff, 0xff, 0xff, 0xff]
        );
        // MovImm into memory (mov dword ptr [rsp + 8], 0x1234_5678):
        assert_eq!(
            emit_inst(X64Inst::MovImm {
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
                imm: 31,
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
            emit_inst(X64Inst::Mov {
                size: OperandSize::Dword,
                dst: RAX.into(),
                src: RBX.into(),
            }),
            [0x8b, 0xc3]
        );
        // Mov Qword on legacy regs (REX.W):
        assert_eq!(
            emit_inst(X64Inst::Mov {
                size: OperandSize::Qword,
                dst: RAX.into(),
                src: RBX.into(),
            }),
            [0x48, 0x8b, 0xc3]
        );
        assert_eq!(
            emit_inst(X64Inst::Mov {
                size: OperandSize::Qword,
                dst: RegMem::from_disp(RSP, 16),
                src: RAX.into(),
            }),
            [0x48, 0x89, 0x44, 0x24, 0x10]
        );
        // RBP / R13 protection (must use disp8 0x00 instead of RIP-relative)
        assert_eq!(
            emit_inst(X64Inst::Mov {
                size: OperandSize::Qword,
                dst: RAX.into(),
                src: RegMem::Mem(RBP),
            }),
            [0x48, 0x8b, 0x45, 0x00]
        );
        assert_eq!(
            emit_inst(X64Inst::Mov {
                size: OperandSize::Qword,
                dst: RAX.into(),
                src: RegMem::from_disp(RBP, 0),
            }),
            [0x48, 0x8b, 0x45, 0x00]
        );
        assert_eq!(
            emit_inst(X64Inst::Mov {
                size: OperandSize::Qword,
                dst: RAX.into(),
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
        // Jmp without REX for legacy reg:
        assert_eq!(emit_inst(X64Inst::Jmp { target: RAX }), [0xff, 0xe0]);
        assert_eq!(emit_inst(X64Inst::Jmp { target: R11 }), [0x41, 0xff, 0xe3]);
        assert_eq!(emit_inst(X64Inst::Ret), [0xc3]);
        // Byte store (sb semantics): mov byte ptr [rdi], sil
        assert_eq!(
            emit_inst(X64Inst::Mov {
                size: OperandSize::Byte,
                dst: RegMem::Mem(RDI),
                src: RSI.into(),
            }),
            [0x40, 0x88, 0x37]
        );
        // Byte load: mov sil, byte ptr [rdi]
        assert_eq!(
            emit_inst(X64Inst::Mov {
                size: OperandSize::Byte,
                dst: RSI.into(),
                src: RegMem::Mem(RDI),
            }),
            [0x40, 0x8a, 0x37]
        );
        // Word store (sh semantics): mov word ptr [rdi], si
        assert_eq!(
            emit_inst(X64Inst::Mov {
                size: OperandSize::Word,
                dst: RegMem::Mem(RDI),
                src: RSI.into(),
            }),
            [0x66, 0x89, 0x37]
        );
        // Word load: mov si, word ptr [rdi]
        assert_eq!(
            emit_inst(X64Inst::Mov {
                size: OperandSize::Word,
                dst: RSI.into(),
                src: RegMem::Mem(RDI),
            }),
            [0x66, 0x8b, 0x37]
        );
        // Byte immediate store: mov byte ptr [rdi], 0x42
        assert_eq!(
            emit_inst(X64Inst::MovImm {
                size: OperandSize::Byte,
                dst: RegMem::Mem(RDI),
                imm: 0x42,
            }),
            [0xc6, 0x07, 0x42]
        );
        // Word immediate store: mov word ptr [rdi], 0x1234
        assert_eq!(
            emit_inst(X64Inst::MovImm {
                size: OperandSize::Word,
                dst: RegMem::Mem(RDI),
                imm: 0x1234,
            }),
            [0x66, 0xc7, 0x07, 0x34, 0x12]
        );
        // Byte immediate to reg: mov sil, 0x42
        assert_eq!(
            emit_inst(X64Inst::MovImm {
                size: OperandSize::Byte,
                dst: RSI.into(),
                imm: 0x42,
            }),
            [0x40, 0xb6, 0x42]
        );
        // Word immediate to reg: mov si, 0x1234
        assert_eq!(
            emit_inst(X64Inst::MovImm {
                size: OperandSize::Word,
                dst: RSI.into(),
                imm: 0x1234,
            }),
            [0x66, 0xbe, 0x34, 0x12]
        );
    }
}
