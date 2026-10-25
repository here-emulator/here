#![cfg(target_arch = "x86_64")]

use super::*;

pub mod asm_text;
pub mod assembler;
pub mod codegen;

pub use codegen::*;

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
                    debug_assert!(*ptr == 0x48 && *ptr.add(1) == 0x81 && *ptr.add(2) == 0xec);
                    ptr.add(3)
                        .cast::<u32>()
                        .write_unaligned((imm as u32).to_le());
                }
                X64PatchKind::JmpRel32 => {
                    debug_assert_eq!(*ptr, 0xe9);
                    let next_pc = ptr.add(5);
                    let disp = imm - (next_pc as isize as i64);
                    debug_assert!(
                        i32::try_from(disp).is_ok(),
                        "jump target out of rel32 range"
                    );
                    ptr.add(1)
                        .cast::<i32>()
                        .write_unaligned((disp as i32).to_le());
                }
                X64PatchKind::JccRel32 => {
                    debug_assert!(*ptr == 0x0f && (*ptr.add(1) & 0xf0) == 0x80);
                    let next_pc = ptr.add(6);
                    let disp = imm - (next_pc as isize as i64);
                    debug_assert!(
                        i32::try_from(disp).is_ok(),
                        "branch target out of rel32 range"
                    );
                    ptr.add(2)
                        .cast::<i32>()
                        .write_unaligned((disp as i32).to_le());
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

    /// Registers clobbered by a helper call.
    ///
    /// Helper calls always use the SysV x86-64 ABI, regardless of host OS, so
    /// this is the SysV caller-saved set on every platform. In particular
    /// `RDI`/`RSI` are caller-saved even on Windows.
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

    fn index(self) -> usize {
        self as usize
    }

    fn from_index(index: usize) -> Self {
        debug_assert!(index < Self::REG_COUNT);
        X64Reg::from(index as u8)
    }
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

    /// Primary opcode for `OP reg, r/m` (ALU opcodes are spaced by 8: (op << 3) | 2 | w).
    #[inline]
    pub const fn reg_rm_opcode(self, size: OperandSize) -> u8 {
        size.opcode(((self as u8) << 3) | 2)
    }
}

impl TryFrom<ComputeKind> for AluOp {
    type Error = ();

    #[inline]
    fn try_from(kind: ComputeKind) -> Result<Self, Self::Error> {
        match kind {
            ComputeKind::Add | ComputeKind::AddW => Ok(AluOp::Add),
            ComputeKind::Sub | ComputeKind::SubW => Ok(AluOp::Sub),
            ComputeKind::And => Ok(AluOp::And),
            ComputeKind::Or => Ok(AluOp::Or),
            ComputeKind::Xor => Ok(AluOp::Xor),
            _ => Err(()),
        }
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

impl TryFrom<ComputeKind> for ShiftOp {
    type Error = ();

    #[inline]
    fn try_from(kind: ComputeKind) -> Result<Self, Self::Error> {
        match kind {
            ComputeKind::Shl | ComputeKind::ShlW => Ok(ShiftOp::Shl),
            ComputeKind::Shr | ComputeKind::ShrW => Ok(ShiftOp::Shr),
            ComputeKind::Sar | ComputeKind::SarW => Ok(ShiftOp::Sar),
            _ => Err(()),
        }
    }
}

/// Shift count, already masked to the width the hardware uses (5 bits for
/// byte/word/dword operands, 6 bits for qword operands).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShiftCount(u8);

impl ShiftCount {
    #[inline]
    pub const fn new(size: OperandSize, count: u64) -> Self {
        let mask = match size {
            OperandSize::Qword => 0x3f,
            _ => 0x1f,
        };
        Self((count as u8) & mask)
    }

    #[inline]
    pub const fn get(self) -> u8 {
        self.0
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

impl From<CmpKind> for ConditionCode {
    #[inline]
    fn from(cmp: CmpKind) -> Self {
        match cmp {
            CmpKind::Eq => ConditionCode::Equal,
            CmpKind::NotEq => ConditionCode::NotEqual,
            CmpKind::GtEq => ConditionCode::GreaterOrEqual,
            CmpKind::GtEqUnsigned => ConditionCode::AboveOrEqual,
            CmpKind::Lt => ConditionCode::Less,
            CmpKind::LtUnsigned => ConditionCode::Below,
        }
    }
}

/// Register or `[base + displacement]` memory operand.
///
/// Note: General SIB (`index * scale`) is not supported.
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

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X64Inst {
    // --- Data movement ---
    /// `mov rm, reg`
    MovRmReg {
        size: OperandSize,
        dst: RegMem,
        src: X64Reg,
    },

    /// `mov reg, rm`
    MovRegRm {
        size: OperandSize,
        dst: X64Reg,
        src: RegMem,
    },

    /// `mov reg, imm32` (`0xb8+r id`, 5 bytes, zero-extends to 64-bit).
    MovRegImm32 { dst: X64Reg, imm: u32 },

    /// `movabs reg, imm64` (`REX.W 0xb8+r io`, 10 bytes).
    Movabs { dst: X64Reg, imm: u64 },

    /// `mov r/m, imm32` (`0xc7 /0`, 32-bit sign-extended immediate).
    ///
    /// Note: Only Dword and Qword operand sizes are supported;
    MovRmImm32 {
        size: OperandSize,
        dst: RegMem,
        imm: i32,
    },

    /// `movsxd dst, src` (sign-extend 32-bit to 64-bit).
    Movsxd { dst: X64Reg, src: RegMem },

    /// `movzx dst, src` (zero-extend 8-bit to 32/64-bit).
    Movzx8 { dst: X64Reg, src: RegMem },

    // --- Arithmetic & Logic ---
    /// `op r/m, reg`
    AluRmReg {
        op: AluOp,
        size: OperandSize,
        dst: RegMem,
        src: X64Reg,
    },
    /// `op reg, r/m`
    AluRegRm {
        op: AluOp,
        size: OperandSize,
        dst: X64Reg,
        src: RegMem,
    },
    /// `op r/m, imm8` (`0x83 /ext ib`, 8-bit sign-extended immediate).
    AluRmImm8 {
        op: AluOp,
        size: OperandSize,
        dst: RegMem,
        imm: i8,
    },

    /// `op r/m, imm32` (`0x81 /ext id`, 32-bit sign-extended immediate).
    AluRmImm32 {
        op: AluOp,
        size: OperandSize,
        dst: RegMem,
        imm: i32,
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
        imm: ShiftCount,
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

#[allow(dead_code)]
impl X64Inst {
    #[inline]
    pub fn mov_rr(size: OperandSize, dst: X64Reg, src: X64Reg) -> Self {
        Self::MovRegRm {
            size,
            dst,
            src: RegMem::Reg(src),
        }
    }

    #[inline]
    pub fn mov_reg_rm(size: OperandSize, dst: X64Reg, src: impl Into<RegMem>) -> Self {
        Self::MovRegRm {
            size,
            dst,
            src: src.into(),
        }
    }

    #[inline]
    pub fn mov_rm_reg(size: OperandSize, dst: impl Into<RegMem>, src: X64Reg) -> Self {
        Self::MovRmReg {
            size,
            dst: dst.into(),
            src,
        }
    }

    #[inline]
    pub fn mov_reg_imm(dst: X64Reg, imm: u64) -> Self {
        if imm == 0 {
            Self::AluRmReg {
                op: AluOp::Xor,
                size: OperandSize::Dword,
                dst: dst.into(),
                src: dst,
            }
        } else if let Ok(imm32) = u32::try_from(imm) {
            Self::MovRegImm32 { dst, imm: imm32 }
        } else if let Ok(imm32) = i32::try_from(imm as i64) {
            Self::MovRmImm32 {
                size: OperandSize::Qword,
                dst: RegMem::Reg(dst),
                imm: imm32,
            }
        } else {
            Self::Movabs { dst, imm }
        }
    }

    #[inline]
    pub fn mov_rm_imm(size: OperandSize, dst: impl Into<RegMem>, imm: i32) -> Self {
        Self::MovRmImm32 {
            size,
            dst: dst.into(),
            imm,
        }
    }

    #[inline]
    pub fn alu_rm_reg(op: AluOp, size: OperandSize, dst: impl Into<RegMem>, src: X64Reg) -> Self {
        Self::AluRmReg {
            op,
            size,
            dst: dst.into(),
            src,
        }
    }
    #[inline]
    pub fn alu_reg_rm(op: AluOp, size: OperandSize, dst: X64Reg, src: impl Into<RegMem>) -> Self {
        Self::AluRegRm {
            op,
            size,
            dst,
            src: src.into(),
        }
    }
    /// Automatically chooses shortest immediate form.
    #[inline]
    pub fn alu_imm(op: AluOp, size: OperandSize, dst: impl Into<RegMem>, imm: i32) -> Self {
        let dst = dst.into();
        if let Ok(imm8) = i8::try_from(imm) {
            Self::AluRmImm8 {
                op,
                size,
                dst,
                imm: imm8,
            }
        } else {
            Self::AluRmImm32 { op, size, dst, imm }
        }
    }

    /// Forces 32-bit immediate form (`0x81 /ext id`), e.g. for stack allocation patch slot.
    #[inline]
    pub fn alu_imm32(op: AluOp, size: OperandSize, dst: impl Into<RegMem>, imm: i32) -> Self {
        Self::AluRmImm32 {
            op,
            size,
            dst: dst.into(),
            imm,
        }
    }

    #[inline]
    pub fn shift_cl(op: ShiftOp, size: OperandSize, dst: impl Into<RegMem>) -> Self {
        Self::ShiftCl {
            op,
            size,
            dst: dst.into(),
        }
    }

    #[inline]
    pub fn shift_imm(
        op: ShiftOp,
        size: OperandSize,
        dst: impl Into<RegMem>,
        imm: ShiftCount,
    ) -> Self {
        Self::ShiftImm {
            op,
            size,
            dst: dst.into(),
            imm,
        }
    }
}
