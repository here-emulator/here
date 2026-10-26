use std::fmt;
use std::fmt::Write;

use crate::backend::AsmPatch;
use crate::backend::write_edge_label;

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

fn write_signed_hex(w: &mut impl Write, value: i64) -> fmt::Result {
    if value < 0 {
        write!(w, "-{:#x}", value.unsigned_abs())
    } else {
        write!(w, "{value:#x}")
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

impl fmt::Display for X64Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::MovRmReg { size, dst, src } => {
                f.write_str("mov ")?;
                write_mov_operand(f, dst, size, true)?;
                write!(f, ", {}", src.name(size))
            }
            Self::MovRegRm { size, dst, src } => {
                f.write_str("mov ")?;
                write!(f, "{}, ", dst.name(size))?;
                write_mov_operand(f, src, size, true)
            }
            Self::MovRegImm32 { dst, imm } => {
                write!(f, "mov {}, {imm:#x}", dst.name(OperandSize::Qword))
            }
            Self::Movabs { dst, imm } => {
                write!(f, "movabs {}, {imm:#x}", dst.name(OperandSize::Qword))
            }
            Self::MovRmImm32 { size, dst, imm } => {
                f.write_str("mov ")?;
                write_mov_operand(f, dst, size, true)?;
                f.write_str(", ")?;
                write_signed_hex(f, imm as i64)
            }
            Self::Movsxd { dst, src } => {
                write!(f, "movsxd {}, ", dst.name(OperandSize::Qword))?;
                write_rm_sized(f, src, "dword", src.reg().name(OperandSize::Dword))
            }
            Self::Movzx8 { dst, src } => {
                write!(f, "movzx {}, ", dst.name(OperandSize::Dword))?;
                write_rm_sized(f, src, "byte", src.reg().name_byte())
            }
            Self::AluRmReg { op, size, dst, src } => {
                write!(f, "{} ", op.mnemonic())?;
                write_rm(f, dst, size)?;
                write!(f, ", {}", src.name(size))
            }
            Self::AluRegRm { op, size, dst, src } => {
                write!(f, "{} {}, ", op.mnemonic(), dst.name(size))?;
                write_rm(f, src, size)
            }
            Self::AluRmImm8 { op, size, dst, imm } => {
                write!(f, "{} ", op.mnemonic())?;
                write_rm(f, dst, size)?;
                f.write_str(", ")?;
                write_signed_hex(f, imm as i64)
            }
            Self::AluRmImm32 { op, size, dst, imm } => {
                write!(f, "{} ", op.mnemonic())?;
                write_rm(f, dst, size)?;
                f.write_str(", ")?;
                write_signed_hex(f, imm as i64)
            }
            Self::ShiftCl { op, size, dst } => {
                write!(f, "{} ", op.mnemonic())?;
                write_rm(f, dst, size)?;
                f.write_str(", cl")
            }
            Self::ShiftImm { op, size, dst, imm } => {
                write!(f, "{} ", op.mnemonic())?;
                write_rm(f, dst, size)?;
                write!(f, ", {}", imm.get())
            }
            Self::Cmovcc {
                condition,
                dst,
                src,
            } => {
                write!(
                    f,
                    "cmov{} {}, ",
                    condition.mnemonic(),
                    dst.name(OperandSize::Qword),
                )?;
                write_rm(f, src, OperandSize::Qword)
            }
            Self::Setcc { condition, dst } => {
                write!(f, "set{} {}", condition.mnemonic(), dst.name_byte())
            }
            Self::Call { target } => {
                write!(f, "call {}", target.name(OperandSize::Qword))
            }
            Self::JmpRel { offset } => {
                if offset >= 0 {
                    write!(f, "jmp .+{:#x}", offset)
                } else {
                    write!(f, "jmp .-{:#x}", (-offset as i64))
                }
            }
            Self::Jcc { condition, offset } => {
                if offset >= 0 {
                    write!(f, "j{} .+{:#x}", condition.mnemonic(), offset)
                } else {
                    write!(f, "j{} .-{:#x}", condition.mnemonic(), (-offset as i64))
                }
            }
            Self::Ret => f.write_str("ret"),
        }
    }
}

/// Renders one dump line, substituting a value that was only resolved by a
/// patch after the instruction had been emitted.
pub(super) fn render_inst(inst: &X64Inst, out: &mut String, patch: Option<AsmPatch>) {
    match (inst, patch) {
        (X64Inst::JmpRel { .. }, Some(AsmPatch::Target { edge, .. })) => {
            out.push_str("jmp ");
            write_edge_label(out, edge);
        }
        (X64Inst::Jcc { condition, .. }, Some(AsmPatch::Target { edge, .. })) => {
            let _ = write!(out, "j{} ", condition.mnemonic());
            write_edge_label(out, edge);
        }
        (X64Inst::AluRmImm32 { op, dst, .. }, Some(AsmPatch::Imm(imm))) => {
            let _ = write!(out, "{} ", op.mnemonic());
            let _ = write_rm(out, *dst, OperandSize::Qword);
            let _ = write!(out, ", {imm:#x}");
        }
        _ => {
            let _ = write!(out, "{inst}");
        }
    }
}
