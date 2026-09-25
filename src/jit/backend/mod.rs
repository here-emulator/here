use super::*;

pub mod codegen;
mod cpu_access;
mod reg_alloc;
pub mod x64;

#[cfg(test)]
mod tests;

use crate::config::arch_config::WordType;
pub use codegen::*;
use pass::*;

pub trait ISA: Sized {
    type Reg: TargetReg;
    type Inst: TargetInst<Reg = Self::Reg>;
    type Backend: CodeGenBackend<Self>;
    type PatchKind: Copy + std::fmt::Debug + 'static;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchLoc<K> {
    pub offset: usize,
    pub kind: K,
}

impl<K> PatchLoc<K> {
    #[inline]
    pub const fn new(offset: usize, kind: K) -> Self {
        Self { offset, kind }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainPatch<K> {
    pub loc: PatchLoc<K>,
    pub target_pc: WordType,
}

impl<K> ChainPatch<K> {
    #[inline]
    pub const fn new(loc: PatchLoc<K>, target_pc: WordType) -> Self {
        Self { loc, target_pc }
    }
}

pub trait TargetReg: Into<u8> + From<u8> + Copy {
    const REG_MASK: u32;
    const CALLER_SAVE_MASK: u32;
    const CALLEE_SAVE_MASK: u32 = (!Self::CALLER_SAVE_MASK) & Self::REG_MASK;
}

pub trait TargetInst: std::fmt::Display + Sized {
    type Reg: TargetReg;

    fn emit_to(&self, buf: &mut CodeBuf);

    fn mov(dst: Self::Reg, src: Self::Reg) -> Self;
    fn mov_imm(dst: Self::Reg, imm: u64) -> Self;
    fn spill(slot: u32, src: Self::Reg) -> Self;
    fn reload(dst: Self::Reg, slot: u32) -> Self;
}

pub trait CodeGenBackend<I: ISA> {
    #[must_use]
    fn codegen(ctx: &mut CodeGenContext<I>, insts: &[IRInst]) -> usize;

    fn config() -> CodeGenConfig;

    /// Reserves space for spill frame allocation (stack adjustment) at the start of code buffer.
    fn emit_stack_alloc(ctx: &mut CodeGenContext<I>) -> PatchLoc<I::PatchKind>;

    /// Patches spill frame allocation with the actual slot count.
    fn patch_stack_alloc(
        buf: &mut CodeBuf,
        asm_buf: Option<&mut String>,
        loc: PatchLoc<I::PatchKind>,
        slots: u32,
    );

    /// Emits stack frame deallocation (`add rsp, bytes` if stack was allocated).
    fn emit_stack_free(ctx: &mut CodeGenContext<I>);

    /// Emits block exit returning `next_pc` in register `rax` and returning.
    /// Returns the buffer offset where the block exit entry point starts.
    fn emit_block_exit(ctx: &mut CodeGenContext<I>, next_pc: WordType) -> usize;

    /// Emits dynamic jump setting return register `rax` from `hreg` and returning.
    fn emit_dynamic_jmp(ctx: &mut CodeGenContext<I>, hreg: I::Reg);

    /// Emits a static jump instruction, returning a patch location for its displacement.
    fn emit_static_jmp(ctx: &mut CodeGenContext<I>) -> PatchLoc<I::PatchKind>;

    /// Emits comparison and conditional/unconditional branches, returning `(then_patch, else_patch)`.
    fn emit_branch(
        ctx: &mut CodeGenContext<I>,
        cmp: CmpKind,
        src1: I::Reg,
        src2: I::Reg,
    ) -> (PatchLoc<I::PatchKind>, PatchLoc<I::PatchKind>);

    /// Patches the instruction at `loc` (relative to `buf`) with `imm`.
    ///
    /// For immediate patches (e.g. stack allocation), `imm` is the raw value.
    /// For jump/branch targets, `imm` is the target absolute address, and the backend computes the displacement.
    ///
    /// # Safety
    /// `buf` must point to valid code memory.
    unsafe fn patch(buf: *mut u8, loc: PatchLoc<I::PatchKind>, imm: i64);

    /// Emits budget check at the start of a block, returning a patch location for the out-of-budget branch.
    fn emit_check_budget(ctx: &mut CodeGenContext<I>, block_insts: u32) -> PatchLoc<I::PatchKind>;

    /// Emits budget decrement before block exit.
    fn emit_update_budget(ctx: &mut CodeGenContext<I>, block_insts: u32);
}

pub type CodeBuf = Vec<u8>;

pub trait CodeBufExt {
    fn emit_zeros(&mut self, len: usize);
}

impl CodeBufExt for CodeBuf {
    #[inline]
    fn emit_zeros(&mut self, len: usize) {
        self.resize(self.len() + len, 0);
    }
}

trait Emitable {
    fn emit_to(self, buf: &mut CodeBuf);
}

impl Emitable for u8 {
    fn emit_to(self, buf: &mut CodeBuf) {
        buf.push(self);
    }
}

impl Emitable for u16 {
    fn emit_to(self, buf: &mut CodeBuf) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}

impl Emitable for u32 {
    fn emit_to(self, buf: &mut CodeBuf) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}

impl Emitable for u64 {
    fn emit_to(self, buf: &mut CodeBuf) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}

impl Emitable for &[u8] {
    fn emit_to(self, buf: &mut CodeBuf) {
        buf.extend_from_slice(self);
    }
}

impl<const N: usize> Emitable for [u8; N] {
    fn emit_to(self, buf: &mut CodeBuf) {
        buf.extend_from_slice(&self);
    }
}

macro_rules! define_emitable_signed {
    ($t:ty) => {
        impl Emitable for $t {
            fn emit_to(self, buf: &mut CodeBuf) {
                self.cast_unsigned().emit_to(buf);
            }
        }
    };
}

define_emitable_signed! { i8 }
define_emitable_signed! { i16 }
define_emitable_signed! { i32 }
define_emitable_signed! { i64 }
