use std::marker::PhantomData;

pub use super::reg_alloc::CallArg;
use super::reg_alloc::*;
use super::*;

pub trait CodeGenTrait: Sized {
    fn try_new() -> Option<Self>;
    fn reset(&mut self);
    fn compile<'a>(&'a mut self, block: &IRBlock) -> CodeBlockRef<'a>;
    fn set_dump_asm(&mut self, enable: bool);
}

/// A struct that provides data needed by each implementation of [`TargetBackend`],
/// like data for register allocation.
pub struct CodeGenContext<'a, I: ISA> {
    buf: Option<&'a mut CodeBuf>,
    asm_buf: Option<&'a mut String>,
    pub(super) state: &'a mut RegAllocState,
    pub(super) liveness: &'a [liveness::Liveness],
    _phantom: PhantomData<I>,
}

impl<'a, I: ISA> CodeGenContext<'a, I> {
    pub fn new(
        buf: &'a mut CodeBuf,
        asm_buf: Option<&'a mut String>,
        state: &'a mut RegAllocState,
        liveness: &'a [liveness::Liveness],
    ) -> Self {
        Self {
            buf: Some(buf),
            asm_buf,
            state,
            liveness,
            _phantom: PhantomData,
        }
    }

    pub fn new_dry_run(state: &'a mut RegAllocState, liveness: &'a [liveness::Liveness]) -> Self {
        Self {
            buf: None,
            asm_buf: None,
            state,
            liveness,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn is_dry_run(&self) -> bool {
        self.buf.is_none()
    }

    #[inline]
    pub fn emit(&mut self, inst: &I::Inst) {
        if let Some(buf) = &mut self.buf {
            inst.emit_to(buf);
            if let Some(asm) = &mut self.asm_buf {
                let _ = inst.put_asm_text(*asm);
                asm.push('\n');
            }
        }
    }

    #[inline]
    pub fn buf(&mut self) -> Option<&mut CodeBuf> {
        self.buf.as_deref_mut()
    }

    #[inline]
    pub fn asm_buf(&mut self) -> Option<&mut String> {
        self.asm_buf.as_deref_mut()
    }
}

pub struct CodeGenConfig {
    /// Allocatable register mask.
    alloc_reg_mask: u32,
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

    fn run_pass(
        backend: &mut I::Backend<'_>,
        ctx: &mut CodeGenContext<I>,
        block: &IRBlock,
        liveness: &[liveness::Liveness],
    ) {
        let mut curr = 0;
        while curr < block.insts.len() {
            let window = &block.insts[curr..];
            let cnt = backend.codegen(ctx, window);
            assert!(
                cnt > 0,
                "backend.codegen must consume at least one instruction"
            );

            for ir in window[..cnt].iter() {
                // If defined a vreg that has no subsequent uses, release it immediately
                if let Some(def_vreg) = ir.def() {
                    if liveness[def_vreg.index()].last_use().is_none() {
                        ctx.unuse_vreg(def_vreg);
                    }
                }

                // mark vreg unused if its last use is reached in or before this batch
                ir.for_each_use(|vreg| {
                    let last_use = liveness[vreg.index()].last_use();
                    if last_use.is_none_or(|val| (val as usize) < curr + cnt) {
                        ctx.unuse_vreg(vreg);
                    }
                });

                if ir.is_terminator() {
                    break;
                }
            }

            curr += cnt;
        }
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

        // Pass 1: Dry-run to calculate exact spill slots
        self.state.init(block.vreg_count(), config.alloc_reg_mask);
        let mut dry_ctx = CodeGenContext::<I>::new_dry_run(&mut self.state, &self.liveness);
        Self::run_pass(&mut backend, &mut dry_ctx, block, &self.liveness);
        let total_slots = self.state.current_use_slot();

        // Pass 2: Real emission with known total_slots
        backend.reset();
        self.state.init(block.vreg_count(), config.alloc_reg_mask);
        let mut ctx = CodeGenContext::<I>::new(
            &mut self.code_buf,
            self.asm_buf.as_mut(),
            &mut self.state,
            &self.liveness,
        );
        backend.emit_prologue(&mut ctx, total_slots);
        Self::run_pass(&mut backend, &mut ctx, block, &self.liveness);

        debug_assert_eq!(self.state.current_use_slot(), total_slots);

        CodeBlockRef {
            buf: self.code_buf.as_slice(),
            asm_text: self.asm_buf.as_deref(),
        }
    }

    fn set_dump_asm(&mut self, enable: bool) {
        CodeGen::set_dump_asm(self, enable);
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

    fn set_dump_asm(&mut self, _enable: bool) {
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
        let _r0 = ctx.def_vreg(v0);
        let _r1 = ctx.def_vreg(v1);
        ctx.end_instruction();
        let _r2 = ctx.def_vreg(v2);
        ctx.end_instruction();

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

    #[test]
    #[should_panic(expected = "use of unallocated/dead virtual register")]
    fn test_use_unallocated_vreg_panics() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b11);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X86>::new(&mut code_buf, None, &mut state, &liveness);
        let v0 = VReg::from(0);
        let _ = ctx.use_vreg(v0);
    }

    #[test]
    fn test_prepare_call_args() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b1111);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X86>::new(&mut code_buf, None, &mut state, &liveness);
        let v0 = VReg::from(0);
        let _ = ctx.def_vreg(v0);
        ctx.end_instruction();

        ctx.prepare_call_args([(crate::jit::backend::x64::X86Reg::RCX, v0.into())]);
        assert!(!code_buf.is_empty());
    }

    #[test]
    #[should_panic(expected = "call argument register must be caller-saved")]
    fn test_prepare_call_args_rejects_callee_saved() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b1111);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X86>::new(&mut code_buf, None, &mut state, &liveness);
        let v0 = VReg::from(0);
        let _ = ctx.def_vreg(v0);
        ctx.end_instruction();

        ctx.prepare_call_args([(crate::jit::backend::x64::X86Reg::RBX, v0.into())]);
    }

    #[test]
    #[should_panic(expected = "is already pinned")]
    fn test_prepare_call_args_rejects_duplicate_target() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b1111);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X86>::new(&mut code_buf, None, &mut state, &liveness);
        let v0 = VReg::from(0);
        let v1 = VReg::from(1);
        let _ = ctx.def_vreg(v0);
        let _ = ctx.def_vreg(v1);
        ctx.end_instruction();

        ctx.prepare_call_args([
            (crate::jit::backend::x64::X86Reg::RCX, v0.into()),
            (crate::jit::backend::x64::X86Reg::RCX, v1.into()),
        ]);
    }
}
