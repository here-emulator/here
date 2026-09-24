use std::fmt::{self, Write};

use super::*;

impl X64Reg {
    pub fn name(self, size: OperandSize) -> &'static str {
        match size {
            OperandSize::Byte => self.name_byte(),
            OperandSize::Word => self.name_word(),
            OperandSize::Dword => match self {
                Self::RAX => "eax",
                Self::RCX => "ecx",
                Self::RDX => "edx",
                Self::RBX => "ebx",
                Self::RSP => "esp",
                Self::RBP => "ebp",
                Self::RSI => "esi",
                Self::RDI => "edi",
                Self::R8 => "r8d",
                Self::R9 => "r9d",
                Self::R10 => "r10d",
                Self::R11 => "r11d",
                Self::R12 => "r12d",
                Self::R13 => "r13d",
                Self::R14 => "r14d",
                Self::R15 => "r15d",
            },
            OperandSize::Qword => match self {
                Self::RAX => "rax",
                Self::RCX => "rcx",
                Self::RDX => "rdx",
                Self::RBX => "rbx",
                Self::RSP => "rsp",
                Self::RBP => "rbp",
                Self::RSI => "rsi",
                Self::RDI => "rdi",
                Self::R8 => "r8",
                Self::R9 => "r9",
                Self::R10 => "r10",
                Self::R11 => "r11",
                Self::R12 => "r12",
                Self::R13 => "r13",
                Self::R14 => "r14",
                Self::R15 => "r15",
            },
        }
    }

    pub fn name_word(self) -> &'static str {
        match self {
            Self::RAX => "ax",
            Self::RCX => "cx",
            Self::RDX => "dx",
            Self::RBX => "bx",
            Self::RSP => "sp",
            Self::RBP => "bp",
            Self::RSI => "si",
            Self::RDI => "di",
            Self::R8 => "r8w",
            Self::R9 => "r9w",
            Self::R10 => "r10w",
            Self::R11 => "r11w",
            Self::R12 => "r12w",
            Self::R13 => "r13w",
            Self::R14 => "r14w",
            Self::R15 => "r15w",
        }
    }

    pub fn name_byte(self) -> &'static str {
        match self {
            Self::RAX => "al",
            Self::RCX => "cl",
            Self::RDX => "dl",
            Self::RBX => "bl",
            Self::RSP => "spl",
            Self::RBP => "bpl",
            Self::RSI => "sil",
            Self::RDI => "dil",
            Self::R8 => "r8b",
            Self::R9 => "r9b",
            Self::R10 => "r10b",
            Self::R11 => "r11b",
            Self::R12 => "r12b",
            Self::R13 => "r13b",
            Self::R14 => "r14b",
            Self::R15 => "r15b",
        }
    }
}

impl ConditionCode {
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Equal => "e",
            Self::NotEqual => "ne",
            Self::Less => "l",
            Self::GreaterOrEqual => "ge",
            Self::Below => "b",
            Self::AboveOrEqual => "ae",
        }
    }
}

impl AluOp {
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Or => "or",
            Self::Adc => "adc",
            Self::Sbb => "sbb",
            Self::And => "and",
            Self::Sub => "sub",
            Self::Xor => "xor",
            Self::Cmp => "cmp",
        }
    }
}

impl ShiftOp {
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::Sar => "sar",
        }
    }
}

fn write_rm(w: &mut impl Write, rm: RegMem, size: OperandSize) -> fmt::Result {
    match rm {
        RegMem::Reg(r) => w.write_str(r.name(size)),
        RegMem::Mem(r) => write!(w, "[{}]", r.name(OperandSize::Qword)),
        RegMem::MemDisp8(r, d) => {
            let base = r.name(OperandSize::Qword);
            if d >= 0 {
                write!(w, "[{base} + {:#x}]", d)
            } else {
                write!(w, "[{base} - {:#x}]", (-d as i16))
            }
        }
        RegMem::MemDisp32(r, d) => {
            let base = r.name(OperandSize::Qword);
            if d >= 0 {
                write!(w, "[{base} + {:#x}]", d)
            } else {
                write!(w, "[{base} - {:#x}]", (-d as i64))
            }
        }
    }
}

fn write_mov_operand(
    w: &mut impl Write,
    rm: RegMem,
    size: OperandSize,
    include_ptr: bool,
) -> fmt::Result {
    match rm {
        RegMem::Reg(r) => w.write_str(r.name(size)),
        _ => {
            if include_ptr {
                let ptr = match size {
                    OperandSize::Byte => "byte ptr ",
                    OperandSize::Word => "word ptr ",
                    OperandSize::Dword => "dword ptr ",
                    OperandSize::Qword => "qword ptr ",
                };
                w.write_str(ptr)?;
            }
            write_rm(w, rm, size)
        }
    }
}

fn write_rm_sized(w: &mut impl Write, rm: RegMem, ptr_size: &str, reg_str: &str) -> fmt::Result {
    match rm {
        RegMem::Reg(_) => w.write_str(reg_str),
        _ => {
            write!(w, "{ptr_size} ptr ")?;
            write_rm(w, rm, OperandSize::Qword)
        }
    }
}

impl X64Inst {
    pub fn put_asm_text(&self, w: &mut impl Write) -> fmt::Result {
        match *self {
            Self::Mov { size, dst, src } => {
                w.write_str("mov ")?;
                write_mov_operand(w, dst, size, true)?;
                w.write_str(", ")?;
                write_mov_operand(w, src, size, matches!(dst, RegMem::Reg(_)))
            }
            Self::MovImm { size, dst, imm } => {
                w.write_str("mov ")?;
                write_mov_operand(w, dst, size, true)?;
                write!(w, ", {:#x}", imm)
            }
            Self::Movsxd { dst, src } => {
                write!(w, "movsxd {}, ", dst.name(OperandSize::Qword))?;
                write_rm_sized(w, src, "dword", src.reg().name(OperandSize::Dword))
            }
            Self::Movsx8 { dst, src } => {
                write!(w, "movsx {}, ", dst.name(OperandSize::Qword))?;
                write_rm_sized(w, src, "byte", src.reg().name_byte())
            }
            Self::Movsx16 { dst, src } => {
                write!(w, "movsx {}, ", dst.name(OperandSize::Qword))?;
                write_rm_sized(w, src, "word", src.reg().name_word())
            }
            Self::Movzx8 { dst, src } => {
                write!(w, "movzx {}, ", dst.name(OperandSize::Dword))?;
                write_rm_sized(w, src, "byte", src.reg().name_byte())
            }
            Self::Movzx16 { dst, src } => {
                write!(w, "movzx {}, ", dst.name(OperandSize::Dword))?;
                write_rm_sized(w, src, "word", src.reg().name_word())
            }
            Self::Alu { op, size, dst, src } => {
                write!(w, "{} ", op.mnemonic())?;
                write_rm(w, dst, size)?;
                write!(w, ", {}", src.name(size))
            }
            Self::AluImm { op, size, dst, imm } => {
                write!(w, "{} ", op.mnemonic())?;
                write_rm(w, dst, size)?;
                write!(w, ", {:#x}", imm)
            }
            Self::Test { size, dst, src } => {
                w.write_str("test ")?;
                write_rm(w, dst, size)?;
                write!(w, ", {}", src.name(size))
            }
            Self::TestImm { size, dst, imm } => {
                w.write_str("test ")?;
                write_rm(w, dst, size)?;
                write!(w, ", {:#x}", imm)
            }
            Self::ShiftCl { op, size, dst } => {
                write!(w, "{} ", op.mnemonic())?;
                write_rm(w, dst, size)?;
                w.write_str(", cl")
            }
            Self::ShiftImm { op, size, dst, imm } => {
                write!(w, "{} ", op.mnemonic())?;
                write_rm(w, dst, size)?;
                write!(w, ", {}", imm)
            }
            Self::Cmovcc {
                condition,
                dst,
                src,
            } => {
                write!(
                    w,
                    "cmov{} {}, ",
                    condition.mnemonic(),
                    dst.name(OperandSize::Qword),
                )?;
                write_rm(w, src, OperandSize::Qword)
            }
            Self::Setcc { condition, dst } => {
                write!(w, "set{} {}", condition.mnemonic(), dst.name_byte())
            }
            Self::Call { target } => {
                write!(w, "call {}", target.name(OperandSize::Qword))
            }
            Self::Jmp { target } => {
                write!(w, "jmp {}", target.name(OperandSize::Qword))
            }
            Self::JmpRel { offset } => {
                if offset >= 0 {
                    write!(w, "jmp .+{:#x}", offset)
                } else {
                    write!(w, "jmp .-{:#x}", (-offset as i64))
                }
            }
            Self::Jcc { condition, offset } => {
                if offset >= 0 {
                    write!(w, "j{} .+{:#x}", condition.mnemonic(), offset)
                } else {
                    write!(w, "j{} .-{:#x}", condition.mnemonic(), (-offset as i64))
                }
            }
            Self::Ret => w.write_str("ret"),
        }
    }

    pub fn to_asm_string(&self) -> String {
        let mut s = String::new();
        let _ = self.put_asm_text(&mut s);
        s
    }
}

impl fmt::Display for X64Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.put_asm_text(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asm_text() {
        assert_eq!(
            X64Inst::Mov {
                size: OperandSize::Qword,
                dst: X64Reg::RAX.into(),
                src: X64Reg::RBX.into(),
            }
            .to_asm_string(),
            "mov rax, rbx"
        );
        assert_eq!(
            X64Inst::Mov {
                size: OperandSize::Qword,
                dst: X64Reg::RAX.into(),
                src: RegMem::from_disp(X64Reg::RBP, -8),
            }
            .to_asm_string(),
            "mov rax, qword ptr [rbp - 0x8]"
        );
        assert_eq!(
            X64Inst::Mov {
                size: OperandSize::Qword,
                dst: X64Reg::RAX.into(),
                src: RegMem::Mem(X64Reg::RBP),
            }
            .to_asm_string(),
            "mov rax, qword ptr [rbp]"
        );
        assert_eq!(
            X64Inst::MovImm {
                size: OperandSize::Qword,
                dst: X64Reg::RAX.into(),
                imm: 0x1234,
            }
            .to_asm_string(),
            "mov rax, 0x1234"
        );
        assert_eq!(
            X64Inst::MovImm {
                size: OperandSize::Qword,
                dst: RegMem::from_disp(X64Reg::RBP, -8),
                imm: 0,
            }
            .to_asm_string(),
            "mov qword ptr [rbp - 0x8], 0x0"
        );
        assert_eq!(
            X64Inst::Alu {
                op: AluOp::Add,
                size: OperandSize::Qword,
                dst: X64Reg::RAX.into(),
                src: X64Reg::RBX,
            }
            .to_asm_string(),
            "add rax, rbx"
        );
        assert_eq!(
            X64Inst::AluImm {
                op: AluOp::Sub,
                size: OperandSize::Qword,
                dst: X64Reg::RAX.into(),
                imm: 10,
            }
            .to_asm_string(),
            "sub rax, 0xa"
        );
        assert_eq!(
            X64Inst::Test {
                size: OperandSize::Qword,
                dst: X64Reg::RAX.into(),
                src: X64Reg::RAX,
            }
            .to_asm_string(),
            "test rax, rax"
        );
        assert_eq!(
            X64Inst::TestImm {
                size: OperandSize::Qword,
                dst: X64Reg::RAX.into(),
                imm: 1,
            }
            .to_asm_string(),
            "test rax, 0x1"
        );
        assert_eq!(
            X64Inst::ShiftCl {
                op: ShiftOp::Shl,
                size: OperandSize::Qword,
                dst: X64Reg::RAX.into(),
            }
            .to_asm_string(),
            "shl rax, cl"
        );
        assert_eq!(
            X64Inst::Setcc {
                condition: ConditionCode::Equal,
                dst: X64Reg::RAX,
            }
            .to_asm_string(),
            "sete al"
        );
        assert_eq!(
            X64Inst::Call {
                target: X64Reg::R11
            }
            .to_asm_string(),
            "call r11"
        );
        assert_eq!(X64Inst::Ret.to_asm_string(), "ret");
        assert_eq!(
            X64Inst::Mov {
                size: OperandSize::Byte,
                dst: RegMem::Mem(X64Reg::RDI),
                src: X64Reg::RSI.into(),
            }
            .to_asm_string(),
            "mov byte ptr [rdi], sil"
        );
        assert_eq!(
            X64Inst::Mov {
                size: OperandSize::Word,
                dst: RegMem::Mem(X64Reg::RDI),
                src: X64Reg::RSI.into(),
            }
            .to_asm_string(),
            "mov word ptr [rdi], si"
        );
        assert_eq!(
            X64Inst::MovImm {
                size: OperandSize::Byte,
                dst: RegMem::Mem(X64Reg::RDI),
                imm: 0x42,
            }
            .to_asm_string(),
            "mov byte ptr [rdi], 0x42"
        );
        assert_eq!(
            X64Inst::MovImm {
                size: OperandSize::Word,
                dst: RegMem::Mem(X64Reg::RDI),
                imm: 0x1234,
            }
            .to_asm_string(),
            "mov word ptr [rdi], 0x1234"
        );
    }
}
