use super::*;

pub mod codegen;
mod cpu_access;
mod reg_alloc;
pub mod x64;

#[cfg(test)]
mod tests;

pub use codegen::*;
use pass::*;

pub trait ISA: Sized {
    type Reg: TargetReg;
    type Inst: TargetInst<Reg = Self::Reg>;
    type Backend: CodeGenBackend<Self>;
    type PatchKind: Copy + std::fmt::Debug;
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

pub trait TargetReg: Into<u8> + From<u8> + Copy {
    const REG_MASK: u32;
    const CALLER_SAVE_MASK: u32;
    const CALLEE_SAVE_MASK: u32 = (!Self::CALLER_SAVE_MASK) & Self::REG_MASK;
}

pub trait TargetInst: Sized {
    type Reg: TargetReg;

    fn emit_to(&self, buf: &mut CodeBuf);
    fn put_asm_text(&self, w: &mut impl std::fmt::Write) -> std::fmt::Result;

    fn mov(dst: Self::Reg, src: Self::Reg) -> Self;
    fn mov_imm(dst: Self::Reg, imm: u64) -> Self;
    fn spill(slot: u32, src: Self::Reg) -> Self;
    fn reload(dst: Self::Reg, slot: u32) -> Self;
}

pub trait CodeGenBackend<I: ISA> {
    #[must_use]
    fn codegen(ctx: &mut CodeGenContext<I>, insts: &[IRInst]) -> usize;

    fn codegen_terminator(ctx: &mut CodeGenContext<I>, term: &Terminator);

    fn config() -> CodeGenConfig;

    /// Reserves space for spill frame allocation (stack adjustment) at the start of code buffer.
    fn reserve_spill_frame(ctx: &mut CodeGenContext<I>) -> PatchLoc<I::PatchKind>;

    /// Patches spill frame allocation with the actual slot count.
    fn patch_spill_frame(
        buf: &mut CodeBuf,
        asm_buf: Option<&mut String>,
        loc: PatchLoc<I::PatchKind>,
        slots: u32,
    );

    /// Emits block exit (restoring spill stack frame if needed, and returning).
    fn emit_block_exit(ctx: &mut CodeGenContext<I>);
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
