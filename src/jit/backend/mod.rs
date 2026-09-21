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
    type Backend<'a>: TargetBackend<'a, Self>;
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

pub trait TargetBackend<'a, I: ISA>: Sized {
    fn new() -> Self;

    #[must_use]
    fn codegen(&mut self, ctx: &mut CodeGenContext<I>, insts: &[IRInst]) -> usize;

    fn config(&mut self) -> CodeGenConfig;

    /// Emits the function prologue allocating stack space for spill slots.
    /// Only called in Pass 2 when `slots > 0`.
    fn emit_prologue(&mut self, ctx: &mut CodeGenContext<I>, slots: u32);

    /// Emits the function epilogue restoring stack space before block exit.
    fn emit_epilogue(&mut self, ctx: &mut CodeGenContext<I>);

    /// Resets any internal backend state between compilation passes.
    fn reset(&mut self) {}

    // TODO: add function to patch ret -> jmp
}

pub type CodeBuf = Vec<u8>;

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
