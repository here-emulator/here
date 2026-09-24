#![cfg(target_arch = "x86_64")]

use super::*;

pub mod asm_text;
pub mod assembler;
pub mod codegen;

pub use codegen::*;

// TODO: rename to x64
pub struct X64;

impl ISA for X64 {
    type Reg = X64Reg;
    type Inst = X64Inst;
    type Backend = X64CodeGen;
    type PatchKind = X64PatchKind;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X64PatchKind {
    /// 32-bit immediate in `sub reg, imm32` (specifically `sub rsp, imm32`: `48 81 ec <imm32>`).
    SubImm32,
    /// 32-bit relative displacement in near jump (`jmp rel32`: `e9 <disp32>`).
    JmpRel32,
    /// 32-bit relative displacement in near conditional branch (`jcc rel32`: `0f 8x <disp32>`).
    JccRel32,
}

impl PatchLoc<X64PatchKind> {
    #[inline]
    pub unsafe fn patch(&self, buf: *mut u8, imm: i64) {
        unsafe {
            let ptr = buf.add(self.offset);
            match self.kind {
                X64PatchKind::SubImm32 => {
                    debug_assert!(
                        *ptr == 0x48
                            && (*ptr.add(1) == 0x81 || *ptr.add(1) == 0x83)
                            && *ptr.add(2) == 0xec
                    );
                    *ptr.add(1) = 0x81;
                    std::ptr::copy_nonoverlapping(
                        (imm as u32).to_le_bytes().as_ptr(),
                        ptr.add(3),
                        4,
                    );
                }
                X64PatchKind::JmpRel32 => {
                    debug_assert_eq!(*ptr, 0xe9);
                    let next_pc = ptr.add(5);
                    let disp = imm - (next_pc as isize as i64);
                    debug_assert!(
                        i32::try_from(disp).is_ok(),
                        "jump target out of rel32 range"
                    );
                    std::ptr::copy_nonoverlapping(
                        (disp as i32).to_le_bytes().as_ptr(),
                        ptr.add(1),
                        4,
                    );
                }
                X64PatchKind::JccRel32 => {
                    debug_assert!(*ptr == 0x0f && (*ptr.add(1) & 0xf0) == 0x80);
                    let next_pc = ptr.add(6);
                    let disp = imm - (next_pc as isize as i64);
                    debug_assert!(
                        i32::try_from(disp).is_ok(),
                        "branch target out of rel32 range"
                    );
                    std::ptr::copy_nonoverlapping(
                        (disp as i32).to_le_bytes().as_ptr(),
                        ptr.add(2),
                        4,
                    );
                }
            }
        }
    }
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum X64Reg {
    RAX = 0, RCX = 1, RDX = 2,  RBX = 3,  RSP = 4,  RBP = 5,  RSI = 6,  RDI = 7,
    R8 = 8,  R9 = 9,  R10 = 10, R11 = 11, R12 = 12, R13 = 13, R14 = 14, R15 = 15,
}

impl From<X64Reg> for u8 {
    fn from(value: X64Reg) -> Self {
        value as u8
    }
}

impl From<u8> for X64Reg {
    fn from(value: u8) -> Self {
        debug_assert!(value <= 15);
        unsafe { std::mem::transmute(value) }
    }
}

impl X64Reg {
    #[inline]
    pub fn base(self) -> u8 {
        (self as u8) & 7
    }

    #[inline]
    pub fn rex_bit(self) -> u8 {
        (self as u8) >> 3
    }

    pub const CALLER_SAVE_MASK: u32 = caller_save_mask();

    #[inline]
    pub const fn caller_save(self) -> bool {
        (Self::CALLER_SAVE_MASK & (1 << (self as u8))) != 0
    }

    #[inline]
    pub const fn callee_save(self) -> bool {
        !self.caller_save()
    }

    #[cfg(target_family = "windows")]
    pub const CALLER_SAVES: &[X64Reg] = &[
        Self::RAX,
        Self::RCX,
        Self::RDX,
        Self::R8,
        Self::R9,
        Self::R10,
        Self::R11,
    ];

    #[cfg(not(target_family = "windows"))]
    pub const CALLER_SAVES: &[X64Reg] = &[
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

const fn caller_save_mask() -> u32 {
    let mut rst = 0;
    let mut index = 0;
    while index < X64Reg::CALLER_SAVES.len() {
        rst |= 1 << (X64Reg::CALLER_SAVES[index] as u8);
        index += 1;
    }
    rst
}

impl TargetReg for X64Reg {
    const REG_MASK: u32 = u16::MAX as u32;
    const CALLER_SAVE_MASK: u32 = X64Reg::CALLER_SAVE_MASK;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperandSize {
    Byte,
    Word,
    Dword,
    Qword,
}

impl OperandSize {
    #[inline]
    pub const fn is_byte(self) -> bool {
        matches!(self, Self::Byte)
    }

    #[inline]
    pub const fn is_word(self) -> bool {
        matches!(self, Self::Word)
    }

    /// The x64 opcode width bit (`w`): bit 0 is 0 for 8-bit, 1 for full-width (16/32/64-bit).
    #[inline]
    pub const fn w_bit(self) -> u8 {
        match self {
            Self::Byte => 0,
            _ => 1,
        }
    }

    /// Applies the operand width bit (`w`) to a byte-sized base opcode.
    #[inline]
    pub const fn opcode(self, base_byte_op: u8) -> u8 {
        base_byte_op | self.w_bit()
    }

    /// REX.W bit (bit 3 of REX prefix): promotes default 32-bit operand size to 64-bit.
    #[inline]
    pub const fn rex_w(self) -> u8 {
        match self {
            Self::Qword => 1,
            _ => 0,
        }
    }
}

/// Group 1 opcode extension encoded in the ModR/M `reg` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AluOp {
    Add = 0b000,
    Or = 0b001,
    Adc = 0b010,
    Sbb = 0b011,
    And = 0b100,
    Sub = 0b101,
    Xor = 0b110,
    Cmp = 0b111,
}

impl AluOp {
    #[inline]
    pub const fn op_ext(self) -> u8 {
        self as u8
    }

    /// Primary opcode for `OP r/m, reg` (ALU opcodes are spaced by 8: op << 3 | w).
    #[inline]
    pub const fn rm_reg_opcode(self, size: OperandSize) -> u8 {
        size.opcode((self as u8) << 3)
    }
}

/// Group 2 opcode extension encoded in the ModR/M `reg` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShiftOp {
    Shl = 4,
    Shr = 5,
    Sar = 7,
}

impl ShiftOp {
    #[inline]
    pub const fn op_ext(self) -> u8 {
        self as u8
    }
}

/// 4-bit condition code appended to 0x0f 0x80 (jcc), 0x0f 0x40 (cmov), 0x0f 0x90 (setcc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ConditionCode {
    Below = 0x2,
    AboveOrEqual = 0x3,
    Equal = 0x4,
    NotEqual = 0x5,
    Less = 0xc,
    GreaterOrEqual = 0xd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegMem {
    Reg(X64Reg),
    Mem(X64Reg),
    MemDisp8(X64Reg, i8),
    MemDisp32(X64Reg, i32),
}

impl From<X64Reg> for RegMem {
    #[inline]
    fn from(r: X64Reg) -> Self {
        RegMem::Reg(r)
    }
}

impl RegMem {
    #[inline]
    pub fn from_disp(reg: X64Reg, disp: impl Into<i32>) -> Self {
        let disp: i32 = disp.into();
        if disp == 0 {
            RegMem::Mem(reg)
        } else if i8::MIN as i32 <= disp && disp <= i8::MAX as i32 {
            RegMem::MemDisp8(reg, disp as i8)
        } else {
            RegMem::MemDisp32(reg, disp)
        }
    }

    #[inline]
    pub fn reg(&self) -> X64Reg {
        match *self {
            RegMem::Reg(r) | RegMem::Mem(r) | RegMem::MemDisp8(r, _) | RegMem::MemDisp32(r, _) => r,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X64Inst {
    // --- Data movement ---
    /// `mov dst, src`
    Mov {
        size: OperandSize,
        dst: RegMem,
        src: RegMem,
    },
    /// `mov dst, imm`
    MovImm {
        size: OperandSize,
        dst: RegMem,
        imm: u64,
    },
    /// `movsxd dst, src` (sign-extend 32-bit to 64-bit)
    Movsxd { dst: X64Reg, src: RegMem },
    /// `movsx dst, src` (sign-extend 8-bit to 64-bit)
    Movsx8 { dst: X64Reg, src: RegMem },
    /// `movsx dst, src` (sign-extend 16-bit to 64-bit)
    Movsx16 { dst: X64Reg, src: RegMem },
    /// `movzx dst, src` (zero-extend 8-bit to 32/64-bit)
    Movzx8 { dst: X64Reg, src: RegMem },
    /// `movzx dst, src` (zero-extend 16-bit to 32/64-bit)
    Movzx16 { dst: X64Reg, src: RegMem },

    // --- Arithmetic & Logic ---
    /// `op dst, src`
    Alu {
        op: AluOp,
        size: OperandSize,
        dst: RegMem,
        src: X64Reg,
    },
    /// `op dst, imm`
    AluImm {
        op: AluOp,
        size: OperandSize,
        dst: RegMem,
        imm: u32,
    },
    /// `test dst, src`
    Test {
        size: OperandSize,
        dst: RegMem,
        src: X64Reg,
    },
    /// `test dst, imm`
    TestImm {
        size: OperandSize,
        dst: RegMem,
        imm: u32,
    },

    // --- Shift ---
    /// `op dst, cl`
    ShiftCl {
        op: ShiftOp,
        size: OperandSize,
        dst: RegMem,
    },
    /// `op dst, imm`
    ShiftImm {
        op: ShiftOp,
        size: OperandSize,
        dst: RegMem,
        imm: u8,
    },

    // --- Conditional ---
    /// `cmovcc dst, src`
    Cmovcc {
        condition: ConditionCode,
        dst: X64Reg,
        src: RegMem,
    },
    /// `setcc dst`
    Setcc {
        condition: ConditionCode,
        dst: X64Reg,
    },

    // --- Control flow ---
    /// `call target`
    Call { target: X64Reg },
    /// `jmp target`
    Jmp { target: X64Reg },
    /// `jmp offset` (relative)
    JmpRel { offset: i32 },
    /// `jcc offset` (relative)
    Jcc {
        condition: ConditionCode,
        offset: i32,
    },
    /// `ret`
    Ret,
}
