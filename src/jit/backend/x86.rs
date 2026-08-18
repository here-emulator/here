#![cfg(target_arch = "x86_64")]

use crate::pack_bits;

use super::*;

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum X86Reg {
    RAX = 0, RCX = 1, RDX = 2,  RBX = 3,  RSP = 4,  RBP = 5,  RSI = 6,  RDI = 7,
    R8 = 8,  R9 = 9,  R10 = 10, R11 = 11, R12 = 12, R13 = 13, R14 = 14, R15 = 15,
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

pub struct X86Assembler {
    buf: Vec<u8>,
}

impl X86Assembler {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(64),
        }
    }

    pub fn build(self) -> Vec<u8> {
        self.buf
    }

    fn emit<T: Emitable>(&mut self, bytes: T) {
        bytes.emit_to(&mut self.buf);
    }

    /// - w: 64-bit operand when `1`, otherwise default operand size (32-bit for most instructions).
    /// - r: MODRM.reg
    /// - x: SIB.index
    /// - b: MODRM.rm or SIB.base, or something else...
    fn emit_rex(&mut self, w: u8, r: u8, x: u8, b: u8) {
        let rex = pack_bits! {
            0b0100, 4;
            w, 1;
            r, 1;
            x, 1;
            b, 1;
        } as u8;

        self.emit(rex);
    }

    /// - mode: addressing mode, `0b11` for register-direct.
    /// - reg_or_op: register or 3 bit opcode extension. (high in REX.R)
    /// - rm: register operand with optionally with a displacement. (high in REX.B)
    fn emit_modrm(&mut self, mode: u8, reg_or_op: u8, rm: u8) {
        let modrm = pack_bits! {
            mode, 2;
            reg_or_op, 3;
            rm, 3;
        } as u8;

        self.emit(modrm);
    }

    fn emit_displacement(&mut self, reg_mem: RegMem) {
        reg_mem.emit_displacement(&mut self.buf);
    }

    fn emit_reg_rm(
        &mut self,
        size: OperandSize,
        opcode: impl Emitable,
        dst_reg: Reg,
        src_rm: RegMem,
    ) {
        let src_reg = src_rm.reg();
        self.emit_rex(size.rex_w(), dst_reg.rex_bit(), 0, src_reg.rex_bit());
        self.emit(opcode);
        self.emit_modrm(src_rm.mode(), dst_reg.base(), src_reg.base());
        self.emit_displacement(src_rm);
    }

    fn emit_rm_reg(
        &mut self,
        size: OperandSize,
        opcode: impl Emitable,
        dst_rm: RegMem,
        src_reg: Reg,
    ) {
        let dst_reg = dst_rm.reg();
        self.emit_rex(size.rex_w(), src_reg.rex_bit(), 0, dst_reg.rex_bit());
        self.emit(opcode);
        self.emit_modrm(dst_rm.mode(), src_reg.base(), dst_reg.base());
        self.emit_displacement(dst_rm);
    }

    fn emit_group_rm(
        &mut self,
        size: OperandSize,
        opcode: impl Emitable,
        opcode_extension: u8,
        dst_rm: RegMem,
    ) {
        let dst_reg = dst_rm.reg();
        self.emit_rex(size.rex_w(), 0, 0, dst_reg.rex_bit());
        self.emit(opcode);
        self.emit_modrm(dst_rm.mode(), opcode_extension, dst_reg.base());
        self.emit_displacement(dst_rm);
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
            use X86Reg::*;
            assert!(
                r != RBP && r != R13,
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

    fn emit_displacement(&self, buf: &mut Vec<u8>) {
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

impl X86Assembler {
    fn alu_rm_r(&mut self, size: OperandSize, op: AluOp, dst: RegMem, src: Reg) {
        self.emit_rm_reg(size, ((op as u8) << 3) | 1, dst, src);
    }

    fn alu_rm_imm32(&mut self, size: OperandSize, op: AluOp, dst: RegMem, imm: u32) {
        self.emit_group_rm(size, 0x81_u8, op as u8, dst);
        self.emit(imm);
    }

    /// Move rm32 to r64 with sign-extension.
    pub fn movsxd_r64_rm32(&mut self, dst: Reg, src: RegMem) {
        self.emit_reg_rm(OperandSize::Qword, 0x63_u8, dst, src);
    }

    pub fn mov_r64_imm64(&mut self, dst: Reg, imm: u64) {
        self.emit_rex(1, 0, 0, dst.rex_bit());
        self.emit(0xB8 | dst.base());
        self.emit(imm);
    }

    pub fn mov_r_rm(&mut self, size: OperandSize, dst: Reg, src: RegMem) {
        self.emit_reg_rm(size, 0x8b_u8, dst, src);
    }

    pub fn mov_rm_r(&mut self, size: OperandSize, dst: RegMem, src: Reg) {
        self.emit_rm_reg(size, 0x89_u8, dst, src);
    }

    pub fn ret(&mut self) {
        self.emit(0xC3 as u8);
    }

    pub fn call_rm64(&mut self, target: Reg) {
        self.emit_rex(0, 0, 0, target.rex_bit());
        self.emit(0xff_u8);
        self.emit_modrm(0b11, 2, target.base());
    }

    /// Calls a JIT helper using the host C ABI.
    ///
    /// This only prepares the registers needed for arguments and the call target:
    ///
    /// - This preserves no caller-saved state:
    /// - Argument moves are emitted in order without resolving register overlaps.
    /// - If helper may exit, you must call [`RegAssign::emit_flush`] or [`RegAssign::emit_writeback`] (if in a branch).
    pub fn call_helper(&mut self, target: u64, args: &[CallArg]) {
        #[cfg(not(target_family = "windows"))]
        self.call_helper_sysv(target, args);

        #[cfg(target_family = "windows")]
        self.call_helper_windows(target, args);
    }

    #[cfg(not(target_family = "windows"))]
    fn call_helper_sysv(&mut self, target: u64, args: &[CallArg]) {
        const ARG_REGS: [Reg; 6] = [Reg::RDI, Reg::RSI, Reg::RDX, Reg::RCX, Reg::R8, Reg::R9];
        assert!(
            args.len() <= ARG_REGS.len(),
            "too many SysV helper arguments"
        );
        assert!(
            args.iter().all(|arg| match arg {
                CallArg::Reg(src) => !ARG_REGS.contains(src),
                CallArg::Imm(_) => true,
            }),
            "helper argument register moves must not overlap"
        );

        for (&arg, &dst) in args.iter().zip(&ARG_REGS) {
            match arg {
                CallArg::Reg(src) => self.mov_r64_rm64(dst, src),
                CallArg::Imm(value) => self.mov_r64_imm64(dst, value),
            }
        }

        self.mov_r64_imm64(Reg::R11, target);
        self.call_rm64(Reg::R11);
    }

    #[cfg(target_family = "windows")]
    fn call_helper_windows(&mut self, target: u64, args: &[CallArg]) {
        const ARG_REGS: [Reg; 4] = [Reg::RCX, Reg::RDX, Reg::R8, Reg::R9];
        const SHADOW_SPACE: usize = 32;

        assert!(args.len() <= 6, "too many Windows helper arguments");
        assert!(
            args.iter().all(|arg| match arg {
                CallArg::Reg(src) => !ARG_REGS.contains(src),
                CallArg::Imm(_) => true,
            }),
            "helper argument register moves must not overlap"
        );

        let stack_arg_count = args.len().saturating_sub(ARG_REGS.len());
        let unaligned_stack_bytes = SHADOW_SPACE + stack_arg_count * size_of::<u64>();
        let stack_bytes = if unaligned_stack_bytes % 16 == 8 {
            unaligned_stack_bytes
        } else {
            unaligned_stack_bytes + 8
        };
        self.alu_rm64_imm32(AluOp::Sub, Reg::RSP, stack_bytes as u32);

        for (index, &arg) in args.iter().enumerate() {
            if let Some(&dst) = ARG_REGS.get(index) {
                match arg {
                    CallArg::Reg(src) => self.mov_r64_rm64(dst, src),
                    CallArg::Imm(value) => self.mov_r64_imm64(dst, value),
                }
                continue;
            }

            let offset = SHADOW_SPACE + (index - ARG_REGS.len()) * size_of::<u64>();
            match arg {
                CallArg::Reg(src) => self.mov_rsp_disp64_r64(offset as i8, src),
                CallArg::Imm(value) => {
                    self.mov_r64_imm64(Reg::R10, value);
                    self.mov_rsp_disp64_r64(offset as i8, Reg::R10);
                }
            }
        }

        self.mov_r64_imm64(Reg::R11, target);
        self.call_rm64(Reg::R11);
        self.alu_rm64_imm32(AluOp::Add, Reg::RSP, stack_bytes as u32);
    }

    pub fn mov_rsp_disp64_r64(&mut self, disp: i8, src: Reg) {
        self.emit_rex(1, src.rex_bit(), 0, 0);
        self.emit(0x89_u8);
        self.emit_modrm(0b01, src.base(), Reg::RSP.base());
        self.emit(0x24_u8);
        self.emit(disp);
    }

    pub fn setcc_r8(&mut self, condition: ConditionCode, dst: Reg) {
        self.emit_rex(0, 0, 0, dst.rex_bit());
        self.emit([0x0f, 0x90 | condition as u8]);
        self.emit_modrm(0b11, 0, dst.base());
    }

    pub fn cmovcc_r64_rm64(&mut self, condition: ConditionCode, dst: Reg, src: RegMem) {
        self.emit_reg_rm(OperandSize::Qword, [0x0f, 0x40 | condition as u8], dst, src);
    }

    pub fn movzx_r32_r8(&mut self, dst: Reg, src: Reg) {
        self.emit_reg_rm(OperandSize::Dword, [0x0f, 0xb6], dst, src.into());
    }

    pub fn movsx_r64_rm8(&mut self, dst: Reg, src: RegMem) {
        self.emit_reg_rm(OperandSize::Qword, [0x0f, 0xbe], dst, src);
    }

    pub fn movsx_r64_rm16(&mut self, dst: Reg, src: RegMem) {
        self.emit_reg_rm(OperandSize::Qword, [0x0f, 0xbf], dst, src);
    }

    pub fn movzx_r32_rm16(&mut self, dst: Reg, src: RegMem) {
        self.emit_reg_rm(OperandSize::Dword, [0x0f, 0xb7], dst, src);
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

impl X86Assembler {
    /// Shift r/m by CL times, which is the lowest byte of [`Reg::RCX`].
    pub fn shift_rm_cl(&mut self, size: OperandSize, shift: ShiftOp, dst: RegMem) {
        self.emit_group_rm(size, 0xd3_u8, shift.op3(), dst);
    }

    pub fn shift_rm_imm8(&mut self, size: OperandSize, shift: ShiftOp, dst: RegMem, imm: u8) {
        self.emit_group_rm(size, 0xc1_u8, shift.op3(), dst);
        self.emit(imm);
    }
}

impl X86Assembler {
    #[inline]
    pub fn alu_rm64_r64(&mut self, op: AluOp, dst: impl Into<RegMem>, src: Reg) {
        self.alu_rm_r(OperandSize::Qword, op, dst.into(), src);
    }

    #[inline]
    pub fn alu_rm32_r32(&mut self, op: AluOp, dst: impl Into<RegMem>, src: Reg) {
        self.alu_rm_r(OperandSize::Dword, op, dst.into(), src);
    }

    #[inline]
    pub fn alu_rm64_imm32(&mut self, op: AluOp, dst: impl Into<RegMem>, imm: u32) {
        self.alu_rm_imm32(OperandSize::Qword, op, dst.into(), imm);
    }

    #[inline]
    pub fn alu_rm32_imm32(&mut self, op: AluOp, dst: impl Into<RegMem>, imm: u32) {
        self.alu_rm_imm32(OperandSize::Dword, op, dst.into(), imm);
    }

    #[inline]
    pub fn sext(&mut self, dst: Reg) {
        self.movsxd_r64_rm32(dst, RegMem::Reg(dst));
    }

    #[inline]
    pub fn mov_r64_rm64(&mut self, dst: Reg, src: impl Into<RegMem>) {
        self.mov_r_rm(OperandSize::Qword, dst, src.into());
    }

    #[inline]
    pub fn mov_r32_rm32(&mut self, dst: Reg, src: impl Into<RegMem>) {
        self.mov_r_rm(OperandSize::Dword, dst, src.into());
    }

    #[inline]
    pub fn mov_rm64_r64(&mut self, dst: impl Into<RegMem>, src: Reg) {
        self.mov_rm_r(OperandSize::Qword, dst.into(), src);
    }

    #[inline]
    pub fn mov_rm32_r32(&mut self, dst: impl Into<RegMem>, src: Reg) {
        self.mov_rm_r(OperandSize::Dword, dst.into(), src);
    }
}

#[cfg(test)]
#[path = "x86_test.rs"]
mod test;
