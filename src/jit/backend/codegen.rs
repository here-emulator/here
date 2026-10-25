use smallvec::SmallVec;
use std::marker::PhantomData;

use super::reg_alloc::*;
use super::*;

pub trait CodeGenTrait: Sized {
    type PatchKind: Copy + std::fmt::Debug + 'static;

    fn try_new() -> Option<Self>;
    fn compile<'a>(
        &'a mut self,
        block: &IRBlock,
        meta: &IRBlockMeta,
    ) -> CodeBlockRef<'a, Self::PatchKind>;
    unsafe fn patch(buf: *mut u8, loc: PatchLoc<Self::PatchKind>, imm: i64);

    /// Enables assembly dumps for subsequent compilations.
    fn set_dump_asm(&mut self, _enable: bool) {}
}

/// A struct that provides data needed by each implementation of [`CodeGenBackend`],
/// like data for register allocation.
pub struct CodeGenContext<'a, I: ISA> {
    buf: &'a mut CodeBuf,
    asm: Option<&'a mut AsmBuf<I::Inst>>,
    pub(super) state: &'a mut RegAllocState<I::Reg>,
    pub(super) liveness: &'a LivenessPass,
    _phantom: PhantomData<I>,
}

impl<'a, I: ISA> CodeGenContext<'a, I> {
    pub fn new(
        buf: &'a mut CodeBuf,
        asm: Option<&'a mut AsmBuf<I::Inst>>,
        state: &'a mut RegAllocState<I::Reg>,
        liveness: &'a LivenessPass,
    ) -> Self {
        Self {
            buf,
            asm,
            state,
            liveness,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn emit(&mut self, inst: &I::Inst) {
        let offset = self.buf.len();
        inst.emit_to(self.buf);
        if let Some(asm) = &mut self.asm {
            asm.push(offset, *inst);
        }
    }

    /// Records a value resolved after [`Self::emit`] for the dump.
    pub fn record_patch(&mut self, offset: usize, patch: AsmPatch) {
        if let Some(asm) = &mut self.asm {
            asm.patch(offset, patch);
        }
    }

    #[inline]
    pub fn buf(&mut self) -> &mut CodeBuf {
        self.buf
    }

    #[inline]
    pub fn current_use_slot(&self) -> u32 {
        self.state.current_use_slot()
    }

    #[inline]
    pub fn patch_jmp(&mut self, loc: PatchLoc<I::PatchKind>, target_offset: usize, edge: AsmEdge) {
        unsafe {
            let target_addr = self.buf.as_ptr().add(target_offset) as usize as i64;
            I::Backend::patch(self.buf.as_mut_ptr(), loc, target_addr);
        }
        if let Some(asm) = &mut self.asm {
            asm.patch(
                loc.offset,
                AsmPatch::Target {
                    offset: target_offset,
                    edge,
                },
            );
        }
    }
}

pub struct CodeBlockRef<'a, K: 'static> {
    pub buf: &'a [u8],
    pub asm_text: Option<&'a str>,
    pub patches: SmallVec<[ChainPatch<K>; 2]>,
}

/// Indentation prefixed to every dumped instruction line.
const ASM_INDENT: &str = "\t";

/// Generic basic-block code generator for a target [`ISA`].
pub struct CodeGen<I: ISA> {
    code_buf: CodeBuf,
    asm: Option<AsmBuf<I::Inst>>,
    asm_text: Option<String>,
    liveness: LivenessPass,
    state: RegAllocState<I::Reg>,
    phantom: PhantomData<I>,
}

impl<I: ISA> CodeGen<I> {
    pub fn set_dump_asm(&mut self, enable: bool) {
        if enable {
            if self.asm.is_none() {
                self.asm = Some(AsmBuf::default());
                self.asm_text = Some(String::new());
            }
        } else {
            self.asm = None;
            self.asm_text = None;
        }
    }

    fn emit_terminator(
        ctx: &mut CodeGenContext<I>,
        term: &Terminator,
        guest_cnt: u32,
    ) -> SmallVec<[ChainPatch<I::PatchKind>; 2]> {
        // TODO: don't use small vec
        let mut chain_patches = SmallVec::new();
        match term {
            Terminator::DirectJmp { target } => {
                I::Backend::emit_update_budget(ctx, guest_cnt);
                I::Backend::emit_stack_free(ctx);
                let patch = I::Backend::emit_static_jmp(ctx);
                let exit_offset = I::Backend::emit_block_exit(ctx, *target as WordType);
                ctx.patch_jmp(
                    patch,
                    exit_offset,
                    AsmEdge::Direct {
                        target: *target as WordType,
                    },
                );
                chain_patches.push(ChainPatch::new(patch, *target as WordType));
            }
            Terminator::DynamicJmp { target } => {
                // we must pin `hreg` *before* emitting other instructions.
                let hreg = ctx.use_vreg(*target);
                I::Backend::emit_update_budget(ctx, guest_cnt);
                I::Backend::emit_stack_free(ctx);
                I::Backend::emit_dynamic_jmp(ctx, hreg);
                ctx.clear_reg_pins();
            }
            Terminator::Branch {
                cmp,
                src1,
                src2,
                then_pc,
                else_pc,
            } => {
                let r_src1 = ctx.use_vreg(*src1);
                let r_src2 = ctx.use_vreg(*src2);
                I::Backend::emit_update_budget(ctx, guest_cnt);
                I::Backend::emit_stack_free(ctx);
                let (then_patch, else_patch) = I::Backend::emit_branch(ctx, *cmp, r_src1, r_src2);
                ctx.clear_reg_pins();

                let then_exit = I::Backend::emit_block_exit(ctx, *then_pc as WordType);
                ctx.patch_jmp(
                    then_patch,
                    then_exit,
                    AsmEdge::Then {
                        target: *then_pc as WordType,
                    },
                );
                chain_patches.push(ChainPatch::new(then_patch, *then_pc as WordType));

                let else_exit = I::Backend::emit_block_exit(ctx, *else_pc as WordType);
                ctx.patch_jmp(
                    else_patch,
                    else_exit,
                    AsmEdge::Else {
                        target: *else_pc as WordType,
                    },
                );
                chain_patches.push(ChainPatch::new(else_patch, *else_pc as WordType));
            }
        }
        chain_patches
    }

    fn emit_insts(
        ctx: &mut CodeGenContext<I>,
        block: &IRBlock,
        guest_cnt: u32,
        liveness: &LivenessPass,
    ) -> SmallVec<[ChainPatch<I::PatchKind>; 2]> {
        let mut curr = 0;
        while curr < block.insts().len() {
            let window = &block.insts()[curr..];

            let cnt = I::Backend::codegen(ctx, window);
            debug_assert!(cnt > 0);
            ctx.clear_reg_pins();

            for ir in window[..cnt].iter() {
                // If defined a vreg that has no subsequent uses, release it immediately
                if let Some(def_vreg) = ir.def() {
                    if liveness[def_vreg].last_use().is_none() {
                        ctx.release_vreg(def_vreg);
                    }
                }

                // mark vreg unused if its last use is reached in or before this batch
                ir.for_each_use(|vreg| {
                    let last_use = liveness[vreg].last_use();
                    if last_use.is_none_or(|val| val < curr + cnt) {
                        ctx.release_vreg(vreg);
                    }
                });
            }

            curr += cnt;
        }
        Self::emit_terminator(ctx, block.terminator(), guest_cnt)
    }
}

impl<I: ISA> CodeGenTrait for CodeGen<I> {
    fn try_new() -> Option<Self> {
        Some(Self {
            code_buf: CodeBuf::new(),
            asm: None,
            asm_text: None,
            liveness: LivenessPass::default(),
            state: RegAllocState::default(),
            phantom: PhantomData,
        })
    }

    type PatchKind = I::PatchKind;

    fn compile<'a>(
        &'a mut self,
        block: &IRBlock,
        meta: &IRBlockMeta,
    ) -> CodeBlockRef<'a, Self::PatchKind> {
        self.code_buf.clear();
        if let Some(asm) = &mut self.asm {
            asm.clear();
        }
        self.liveness.run(block);

        let reg_config = I::Backend::reg_alloc_config();
        self.state.init(block.vreg_count(), &reg_config);
        let mut ctx = CodeGenContext::<I>::new(
            &mut self.code_buf,
            self.asm.as_mut(),
            &mut self.state,
            &self.liveness,
        );
        let oob_patch = I::Backend::emit_check_budget(&mut ctx, meta.guest_cnt);
        let spill_loc = I::Backend::emit_stack_alloc(&mut ctx);

        let patches = Self::emit_insts(&mut ctx, block, meta.guest_cnt, &self.liveness);

        let oob_exit = I::Backend::emit_block_exit(&mut ctx, meta.start_pc);
        ctx.patch_jmp(oob_patch, oob_exit, AsmEdge::OutOfBudget);

        let total_slots = ctx.current_use_slot();
        I::Backend::patch_stack_alloc(&mut ctx, spill_loc, total_slots);
        drop(ctx);

        if let Some(asm) = &self.asm {
            let text = self
                .asm_text
                .as_mut()
                .expect("asm text buffer is allocated together with asm entries");
            text.clear();
            asm.render(text, ASM_INDENT);
        }

        CodeBlockRef {
            buf: self.code_buf.as_slice(),
            asm_text: self.asm_text.as_deref(),
            patches,
        }
    }

    unsafe fn patch(buf: *mut u8, loc: PatchLoc<Self::PatchKind>, imm: i64) {
        unsafe {
            I::Backend::patch(buf, loc, imm);
        }
    }
}

pub type NativeCodeGen = cfg_select! {
    target_arch = "x86_64" => {
        CodeGen<x64::X64>
    }

    _ => {
        DummyCodeGen
    }
};

#[allow(dead_code)]
pub(crate) struct DummyCodeGen {}

impl CodeGenTrait for DummyCodeGen {
    type PatchKind = ();

    fn try_new() -> Option<Self> {
        None
    }

    fn compile<'a>(
        &'a mut self,
        _block: &IRBlock,
        _meta: &IRBlockMeta,
    ) -> CodeBlockRef<'a, Self::PatchKind> {
        panic!()
    }

    unsafe fn patch(_buf: *mut u8, _loc: PatchLoc<Self::PatchKind>, _imm: i64) {
        panic!()
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::jit::backend::x64::{X64, X64Inst, X64Reg};

    fn test_liveness(vreg_count: u32) -> LivenessPass {
        let mut pass = LivenessPass::default();
        pass.run(&IRBlock::new(
            vreg_count,
            Vec::new(),
            Terminator::DirectJmp { target: 0 },
        ));
        pass
    }

    struct TestCtx {
        code_buf: CodeBuf,
        asm: Option<AsmBuf<X64Inst>>,
        state: RegAllocState<X64Reg>,
        liveness: LivenessPass,
    }

    impl TestCtx {
        fn new(vreg_count: u32) -> Self {
            Self::with_config(vreg_count, false, &RegAllocConfig::default())
        }

        fn with_asm(vreg_count: u32) -> Self {
            Self::with_config(vreg_count, true, &RegAllocConfig::default())
        }

        fn with_config(vreg_count: u32, enable_asm: bool, config: &RegAllocConfig<X64Reg>) -> Self {
            let mut state = RegAllocState::default();
            state.init(vreg_count, config);
            let liveness = test_liveness(vreg_count);
            Self {
                code_buf: CodeBuf::new(),
                asm: if enable_asm {
                    Some(AsmBuf::default())
                } else {
                    None
                },
                state,
                liveness,
            }
        }

        fn ctx(&mut self) -> CodeGenContext<'_, X64> {
            CodeGenContext::<X64>::new(
                &mut self.code_buf,
                self.asm.as_mut(),
                &mut self.state,
                &self.liveness,
            )
        }
    }

    #[test]
    fn test_codegen_context_asm_text() {
        let mut t = TestCtx::with_asm(17);
        let mut ctx = t.ctx();
        ctx.emit(&X64Inst::Ret);

        // Fill every host register, then define one more to force a spill.
        for i in 0..17 {
            let _ = ctx.def_vreg(VReg::new(i));
            ctx.clear_reg_pins();
        }
        drop(ctx);

        let mut text = String::new();
        t.asm.unwrap().render(&mut text, ASM_INDENT);

        assert!(!t.code_buf.is_empty());
        assert!(text.contains("\tret\n"));
        assert!(text.contains("mov qword ptr [rsp"));
    }

    #[test]
    #[should_panic(expected = "use of unallocated/dead virtual register")]
    fn test_use_unallocated_vreg_panics() {
        let mut t = TestCtx::new(4);
        let mut ctx = t.ctx();
        let _ = ctx.use_vreg(VReg::from(0));
    }

    #[test]
    fn test_prepare_call_args() {
        let mut t = TestCtx::new(4);
        let mut ctx = t.ctx();
        let v0 = VReg::from(0);
        let _ = ctx.def_vreg(v0);
        ctx.clear_reg_pins();

        ctx.prepare_call_args([(X64Reg::RCX, v0)]);
        assert!(!ctx.buf().is_empty());
    }

    #[test]
    #[should_panic(expected = "call argument register must be caller-saved")]
    fn test_prepare_call_args_rejects_callee_saved() {
        let mut t = TestCtx::new(4);
        let mut ctx = t.ctx();
        let v0 = VReg::from(0);
        let _ = ctx.def_vreg(v0);
        ctx.clear_reg_pins();

        ctx.prepare_call_args([(X64Reg::RBX, v0)]);
    }

    #[test]
    #[should_panic]
    fn test_prepare_call_args_rejects_duplicate_target() {
        let mut t = TestCtx::new(4);
        let mut ctx = t.ctx();
        let v0 = VReg::from(0);
        let v1 = VReg::from(1);
        let _ = ctx.def_vreg(v0);
        let _ = ctx.def_vreg(v1);
        ctx.clear_reg_pins();

        ctx.prepare_call_args([(X64Reg::RCX, v0), (X64Reg::RCX, v1)]);
    }
}
