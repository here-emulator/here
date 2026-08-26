#![cfg(target_arch = "x86_64")]

use super::*;
use crate::pack_bits;

mod assembler;
pub use assembler::*;

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum X86Reg {
    RAX = 0, RCX = 1, RDX = 2,  RBX = 3,  RSP = 4,  RBP = 5,  RSI = 6,  RDI = 7,
    R8 = 8,  R9 = 9,  R10 = 10, R11 = 11, R12 = 12, R13 = 13, R14 = 14, R15 = 15,
}

impl From<X86Reg> for u8 {
    fn from(value: X86Reg) -> Self {
        value as u8
    }
}

impl X86Reg {
    #[inline]
    pub fn base(self) -> u8 {
        (self as u8) & 7
    }

    #[inline]
    pub fn rex_bit(self) -> u8 {
        (self as u8) >> 3
    }

    #[inline]
    pub const fn caller_save(self) -> bool {
        // [T]::iter as a const fn is not yet stable...
        let mut index = 0;
        while index < Self::CALLER_SAVES.len() {
            if Self::CALLER_SAVES[index] as u8 == self as u8 {
                return true;
            }
            index += 1;
        }
        false
    }

    #[inline]
    pub const fn callee_save(self) -> bool {
        !self.caller_save()
    }

    #[cfg(target_family = "windows")]
    pub const CALLER_SAVES: &[X86Reg] = &[
        Self::RAX,
        Self::RCX,
        Self::RDX,
        Self::R8,
        Self::R9,
        Self::R10,
        Self::R11,
    ];

    #[cfg(not(target_family = "windows"))]
    pub const CALLER_SAVES: &[X86Reg] = &[
        Self::RAX,
        Self::RCX,
        Self::RDX,
        Self::RSI,
        Self::RDI,
        Self::R8,
        Self::R9,
        Self::R10,
        Self::R11,
    ];
}

use X86Reg as Reg;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CallArg {
    Reg(Reg),
    Imm(u64),
}

impl From<Reg> for CallArg {
    fn from(reg: Reg) -> Self {
        Self::Reg(reg)
    }
}

impl From<u64> for CallArg {
    fn from(value: u64) -> Self {
        Self::Imm(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandSize {
    Dword,
    Qword,
}

impl OperandSize {
    /// NOTE:
    /// `REX.W` = 0 actually means using default operand instead of 32-bit operand,
    /// some special instructions like `jmp (near)` default to 64-bit operand in long mode.
    #[inline]
    fn rex_w(self) -> u8 {
        match self {
            Self::Dword => 0,
            Self::Qword => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add = 0b000,
    Or = 0b001,
    /// Add with carry
    Adc = 0b010,
    /// Sub with borrow
    Sbb = 0b011,
    And = 0b100,
    Sub = 0b101,
    Xor = 0b110,
    Cmp = 0b111,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionCode {
    Below = 0x2,
    AboveOrEqual = 0x3,
    Equal = 0x4,
    NotEqual = 0x5,
    Less = 0xc,
    GreaterOrEqual = 0xd,
}

#[derive(Debug, Clone, Copy)]
pub enum RegMem {
    Reg(Reg),
    Mem(Reg),
    MemDisp8(Reg, i8),
    MemDisp32(Reg, i32),
}

impl From<X86Reg> for RegMem {
    fn from(r: X86Reg) -> Self {
        RegMem::Reg(r)
    }
}

impl RegMem {
    #[inline]
    pub fn from_disp(reg: X86Reg, disp: impl Into<i32>) -> RegMem {
        let disp: i32 = disp.into();
        if disp == 0 && !matches!(reg, X86Reg::RBP | X86Reg::R13) {
            RegMem::Mem(reg)
        } else if i8::MIN as i32 <= disp && disp <= i8::MAX as i32 {
            RegMem::MemDisp8(reg, disp as i8)
        } else {
            RegMem::MemDisp32(reg, disp as i32)
        }
    }

    /// mod bits for ModR/M byte
    fn mode(&self) -> u8 {
        match self {
            RegMem::Reg(_) => 0b11,
            RegMem::Mem(_) => 0b00,
            RegMem::MemDisp8(_, _) => 0b01,
            RegMem::MemDisp32(_, _) => 0b10,
        }
    }

    fn reg(&self) -> Reg {
        use RegMem::*;

        // sanity check
        if let Mem(r) = *self {
            assert!(
                r != X86Reg::RBP && r != X86Reg::R13,
                "it leads to [RIP+disp32] instead of [r/m] on x64, maybe not what you want"
            );
        }

        match *self {
            Reg(_) => {}
            Mem(r) | MemDisp8(r, _) | MemDisp32(r, _) => {
                assert!(r != X86Reg::RSP && r != X86Reg::R12, "SIB is unsupported");
            }
        }

        match self {
            Reg(r) | Mem(r) | MemDisp8(r, _) | MemDisp32(r, _) => {
                return *r;
            }
        }
    }

    fn emit_displacement(&self, buf: &mut CodeBuf) {
        use RegMem::*;
        match self {
            Reg(_) | Mem(_) => {}
            MemDisp8(_, byte) => {
                byte.emit_to(buf);
            }
            MemDisp32(_, bytes) => {
                bytes.emit_to(buf);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftOp {
    SAR,
    SHL,
    SHR,
}

impl ShiftOp {
    /// in ModR/m byte
    fn op3(&self) -> u8 {
        use ShiftOp::*;
        match self {
            SHL => 4,
            SAR => 7,
            SHR => 5,
        }
    }
}
