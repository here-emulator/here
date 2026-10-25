//! Host code generation and register allocation.
//!
//! - A compiled block begins with a budget check.
//! - Terminators update (decrement) the remaining budget before jumping.
//! - Block exits jump to exit stubs or chained target blocks.

use super::*;

pub mod asm;
pub mod codegen;
mod reg_alloc;
pub mod x64;

#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) mod tests;

use crate::config::arch_config::WordType;
pub use asm::*;
pub use codegen::*;
use pass::*;
pub use reg_alloc::RegAllocConfig;

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

pub trait TargetReg: Copy + Eq + std::fmt::Debug + 'static {
    const REG_MASK: u32;
    const CALLER_SAVE_MASK: u32;
    const CALLEE_SAVE_MASK: u32 = (!Self::CALLER_SAVE_MASK) & Self::REG_MASK;
    /// Number of registers tracked by the allocator.
    const REG_COUNT: usize = Self::REG_MASK.count_ones() as usize;

    /// Index of this register, in `0..REG_COUNT`.
    fn index(self) -> usize;

    /// Inverse of [`Self::index`].
    fn from_index(index: usize) -> Self;

    /// Allocator mask bit for this register.
    fn bit(self) -> u32 {
        let index = self.index();
        debug_assert!(index < 32);
        1u32 << index
    }
}

pub trait TargetInst: std::fmt::Display + Copy + Sized {
    type Reg: TargetReg;

    fn emit_to(&self, buf: &mut CodeBuf);

    fn mov(dst: Self::Reg, src: Self::Reg) -> Self;
    fn mov_imm(dst: Self::Reg, imm: u64) -> Self;
    fn spill(slot: u32, src: Self::Reg) -> Self;
    fn reload(dst: Self::Reg, slot: u32) -> Self;

    /// Renders the dump line, substituting a value that was only known after
    /// emission (branch target or patched immediate).
    fn render_asm(&self, out: &mut String, _patch: Option<AsmPatch>) {
        use std::fmt::Write;
        let _ = write!(out, "{self}");
    }
}

pub trait CodeGenBackend<I: ISA> {
    #[must_use]
    fn codegen(ctx: &mut CodeGenContext<I>, insts: &[IRInst]) -> usize;

    /// Host registers reserved or permanently bound for this backend.
    fn reg_alloc_config() -> RegAllocConfig<I::Reg>;

    /// Reserves space for spill frame allocation (stack adjustment) at the start of code buffer.
    fn emit_stack_alloc(ctx: &mut CodeGenContext<I>) -> PatchLoc<I::PatchKind>;

    /// Patches spill frame allocation with the actual slot count and records
    /// the resolved immediate for the assembly dump.
    fn patch_stack_alloc(ctx: &mut CodeGenContext<I>, loc: PatchLoc<I::PatchKind>, slots: u32);

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
    fn emit_check_budget(ctx: &mut CodeGenContext<I>, guest_cnt: u32) -> PatchLoc<I::PatchKind>;

    /// Emits budget decrement before block exit.
    fn emit_update_budget(ctx: &mut CodeGenContext<I>, guest_cnt: u32);
}

pub type CodeBuf = Vec<u8>;
