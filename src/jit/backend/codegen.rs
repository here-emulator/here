use std::marker::PhantomData;

use super::reg_alloc::*;
use super::*;

pub trait CodeGenTrait: Sized {
    fn try_new() -> Option<Self>;
    fn reset(&mut self);
    fn compile<'a>(&'a mut self, block: &IRBlock) -> CodeBlockRef<'a>;
}

/// A struct that provides data needed by each implementation of [`TargetBackend`],
/// like data for register allocation.
pub struct CodeGenContext<'a, I: ISA> {
    pub(super) buf: &'a mut CodeBuf,
    pub(super) asm_buf: Option<&'a mut String>,
    pub(super) state: &'a mut RegAllocState,
    pub(super) liveness: &'a [liveness::Liveness],
    pub(super) _phantom: PhantomData<I>,
}

impl<'a, I: ISA> CodeGenContext<'a, I> {
    pub fn emit(&mut self, inst: &I::Inst) {
        inst.emit_to(self.buf);
        if let Some(asm) = &mut self.asm_buf {
            let _ = inst.put_asm_text(*asm);
            asm.push('\n');
        }
    }

    pub fn buf(&mut self) -> &mut CodeBuf {
        self.buf
    }

    pub fn asm_buf(&mut self) -> Option<&mut String> {
        self.asm_buf.as_deref_mut()
    }
}

pub struct CodeGenConfig {
    /// Allocatable register mask.
    pub alloc_reg_mask: u32,
}

impl CodeGenConfig {
    #[inline]
    pub const fn new(alloc_reg_mask: u32) -> Self {
        Self { alloc_reg_mask }
    }
}

pub struct CodeBlockRef<'a> {
    pub buf: &'a [u8],
    pub asm_text: Option<&'a str>,
}

pub struct CodeGen<I: ISA> {
    code_buf: CodeBuf,
    asm_buf: Option<String>,
    liveness: Vec<liveness::Liveness>,
    state: RegAllocState,
    phantom: PhantomData<I>,
}

impl<I: ISA> CodeGen<I> {
    pub fn set_dump_asm(&mut self, enable: bool) {
        if enable {
            if self.asm_buf.is_none() {
                self.asm_buf = Some(String::new());
            }
        } else {
            self.asm_buf = None;
        }
    }

    pub fn with_dump_asm(mut self, enable: bool) -> Self {
        self.set_dump_asm(enable);
        self
    }
}

impl<I: ISA> CodeGenTrait for CodeGen<I> {
    fn try_new() -> Option<Self> {
        Some(Self {
            code_buf: CodeBuf::new(),
            asm_buf: None,
            liveness: Vec::new(),
            state: RegAllocState::default(),
            phantom: PhantomData,
        })
    }

    fn reset(&mut self) {
        self.code_buf.clear();
        if let Some(asm) = &mut self.asm_buf {
            asm.clear();
        }
        self.liveness.clear();
    }

    fn compile<'a>(&'a mut self, block: &IRBlock) -> CodeBlockRef<'a> {
        self.code_buf.clear();
        if let Some(asm) = &mut self.asm_buf {
            asm.clear();
        }
        liveness::analyze(block, &mut self.liveness);

        let mut backend = I::Backend::new();
        let config = backend.config();
        self.state.init(block.vreg_count(), config.alloc_reg_mask);

        let mut ctx = CodeGenContext::<I>::new(
            &mut self.code_buf,
            self.asm_buf.as_mut(),
            &mut self.state,
            &self.liveness,
        );

        let stack_frame_offset = backend.emit_stack_frame(&mut ctx);

        let mut curr = 0;
        while curr < block.insts.len() {
            let window = &block.insts[curr..];

            let cnt = backend.codegen(&mut ctx, window);
            assert!(
                cnt > 0,
                "backend.codegen must consume at least one instruction"
            );

            for ir in window[..cnt].iter() {
                if ir.is_terminator() {
                    break;
                }

                // If defined a vreg that has no subsequent uses, release it immediately
                if let Some(def_vreg) = ir.def() {
                    if self.liveness[def_vreg.index()].last_use().is_none() {
                        ctx.unuse_vreg(def_vreg);
                    }
                }

                // mark vreg unused if its last use is reached in or before this batch
                ir.for_each_use(|vreg| {
                    let last_use = self.liveness[vreg.index()].last_use();
                    if last_use.is_none_or(|val| (val as usize) < curr + cnt) {
                        ctx.unuse_vreg(vreg);
                    }
                });
            }

            curr += cnt;
        }

        let slots = self.state.current_use_slot();
        backend.patch_stack_frame(&mut self.code_buf, stack_frame_offset, slots);

        CodeBlockRef {
            buf: self.code_buf.as_slice(),
            asm_text: self.asm_buf.as_deref(),
        }
    }
}

pub type NativeCodeGen = cfg_select! {
    target_arch = "x86_64" => {
        CodeGen<x64::X86>
    }

    _ => {
        DummyCodeGen
    }
};

struct DummyCodeGen {}

impl CodeGenTrait for DummyCodeGen {
    fn try_new() -> Option<Self> {
        None
    }

    fn reset(&mut self) {
        panic!()
    }

    fn compile<'a>(&'a mut self, _block: &IRBlock) -> CodeBlockRef<'a> {
        panic!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::x64::{X64Inst, X86};

    #[test]
    fn test_codegen_context_asm_text() {
        let mut code_buf = CodeBuf::new();
        let mut asm_buf = String::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b11);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx =
            CodeGenContext::<X86>::new(&mut code_buf, Some(&mut asm_buf), &mut state, &liveness);

        ctx.emit(&X64Inst::Ret);

        let v0 = VReg::from(0);
        let v1 = VReg::from(1);
        let v2 = VReg::from(2);
        let _g0 = ctx.def_vreg(v0);
        let _g1 = ctx.def_vreg(v1);
        drop(_g0);
        drop(_g1);
        let _g2 = ctx.def_vreg(v2);

        assert!(!code_buf.is_empty());
        assert!(asm_buf.contains("ret\n"));
        assert!(asm_buf.contains("mov qword ptr [rsp"));
    }

    #[test]
    fn test_codegen_try_new_and_reset() {
        let mut codegen = CodeGen::<X86>::try_new().expect("try_new should succeed for X86");
        codegen.set_dump_asm(true);
        codegen.reset();
    }
}
