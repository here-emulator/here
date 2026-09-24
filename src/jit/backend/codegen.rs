use smallvec::SmallVec;
use std::marker::PhantomData;

pub use super::reg_alloc::CallArg;
use super::reg_alloc::*;
use super::*;

pub trait CodeGenTrait: Sized {
    type PatchKind: Copy + std::fmt::Debug + 'static;

    fn try_new() -> Option<Self>;
    fn reset(&mut self);
    fn compile<'a>(&'a mut self, block: &IRBlock) -> CodeBlockRef<'a, Self::PatchKind>;
    fn set_dump_asm(&mut self, enable: bool);
    unsafe fn patch(buf: *mut u8, loc: PatchLoc<Self::PatchKind>, imm: i64);
}

/// A struct that provides data needed by each implementation of [`CodeGenBackend`],
/// like data for register allocation.
pub struct CodeGenContext<'a, I: ISA> {
    buf: &'a mut CodeBuf,
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
            buf,
            asm_buf,
            state,
            liveness,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn emit(&mut self, inst: &I::Inst) {
        inst.emit_to(self.buf);
        if let Some(asm) = &mut self.asm_buf {
            let _ = inst.put_asm_text(*asm);
            asm.push('\n');
        }
    }

    #[inline]
    pub fn buf(&mut self) -> &mut CodeBuf {
        self.buf
    }

    #[inline]
    pub fn emit_zeros(&mut self, len: usize) {
        self.buf.emit_zeros(len);
    }

    #[inline]
    pub fn current_use_slot(&self) -> u32 {
        self.state.current_use_slot()
    }

    #[inline]
    pub fn asm_buf(&mut self) -> Option<&mut String> {
        self.asm_buf.as_deref_mut()
    }

    #[inline]
    pub fn patch_jmp(&mut self, loc: PatchLoc<I::PatchKind>, target_offset: usize) {
        unsafe {
            let target_addr = self.buf.as_ptr().add(target_offset) as usize as i64;
            I::Backend::patch(self.buf.as_mut_ptr(), loc, target_addr);
        }
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

pub struct CodeBlockRef<'a, K: 'static> {
    pub buf: &'a [u8],
    pub asm_text: Option<&'a str>,
    pub patches: SmallVec<[ChainPatch<K>; 2]>,
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

    fn emit_terminator(
        ctx: &mut CodeGenContext<I>,
        term: &Terminator,
        block_insts: u32,
    ) -> SmallVec<[ChainPatch<I::PatchKind>; 2]> {
        let mut patches = SmallVec::new();
        match term {
            Terminator::DirectJmp { target } => {
                I::Backend::emit_update_budget(ctx, block_insts);
                I::Backend::emit_stack_free(ctx);
                let patch = I::Backend::emit_static_jmp(ctx);
                let exit_offset = I::Backend::emit_block_exit(ctx, *target as WordType);
                ctx.patch_jmp(patch, exit_offset);
                patches.push(ChainPatch::new(patch, *target as WordType));
            }
            Terminator::DynamicJmp { target } => {
                let hreg = ctx.use_vreg(*target);
                I::Backend::emit_update_budget(ctx, block_insts);
                I::Backend::emit_stack_free(ctx);
                I::Backend::emit_dynamic_jmp(ctx, hreg);
                ctx.end_instruction();
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
                I::Backend::emit_update_budget(ctx, block_insts);
                I::Backend::emit_stack_free(ctx);
                let (then_patch, else_patch) = I::Backend::emit_branch(ctx, *cmp, r_src1, r_src2);
                ctx.end_instruction();

                let then_exit = I::Backend::emit_block_exit(ctx, *then_pc as WordType);
                ctx.patch_jmp(then_patch, then_exit);
                patches.push(ChainPatch::new(then_patch, *then_pc as WordType));

                let else_exit = I::Backend::emit_block_exit(ctx, *else_pc as WordType);
                ctx.patch_jmp(else_patch, else_exit);
                patches.push(ChainPatch::new(else_patch, *else_pc as WordType));
            }
        }
        patches
    }

    fn emit_insts(
        ctx: &mut CodeGenContext<I>,
        block: &IRBlock,
        liveness: &[liveness::Liveness],
    ) -> SmallVec<[ChainPatch<I::PatchKind>; 2]> {
        let mut curr = 0;
        let mut patches = SmallVec::new();
        while curr < block.insts.len() {
            let window = &block.insts[curr..];
            if let Some(term) = window[0].get_terminator() {
                patches = Self::emit_terminator(ctx, &term, block.instr_cnt);
                break;
            }

            let cnt = I::Backend::codegen(ctx, window);
            debug_assert!(cnt > 0);

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
            }

            curr += cnt;
        }
        patches
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

    type PatchKind = I::PatchKind;

    fn compile<'a>(&'a mut self, block: &IRBlock) -> CodeBlockRef<'a, Self::PatchKind> {
        self.code_buf.clear();
        if let Some(asm) = &mut self.asm_buf {
            asm.clear();
        }
        liveness::analyze(block, &mut self.liveness);

        let config = I::Backend::config();

        self.state.init(block.vreg_count(), config.alloc_reg_mask);
        let mut ctx = CodeGenContext::<I>::new(
            &mut self.code_buf,
            self.asm_buf.as_mut(),
            &mut self.state,
            &self.liveness,
        );
        let oob_patch = I::Backend::emit_check_budget(&mut ctx, block.instr_cnt);
        let spill_loc = I::Backend::emit_stack_alloc(&mut ctx);

        let patches = Self::emit_insts(&mut ctx, block, &self.liveness);

        let oob_exit = I::Backend::emit_block_exit(&mut ctx, block.start_pc);
        ctx.patch_jmp(oob_patch, oob_exit);

        let total_slots = ctx.current_use_slot();
        drop(ctx);

        I::Backend::patch_stack_alloc(
            &mut self.code_buf,
            self.asm_buf.as_mut(),
            spill_loc,
            total_slots,
        );

        CodeBlockRef {
            buf: self.code_buf.as_slice(),
            asm_text: self.asm_buf.as_deref(),
            patches,
        }
    }

    fn set_dump_asm(&mut self, enable: bool) {
        CodeGen::set_dump_asm(self, enable);
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

struct DummyCodeGen {}

impl CodeGenTrait for DummyCodeGen {
    type PatchKind = ();

    fn try_new() -> Option<Self> {
        None
    }

    fn reset(&mut self) {
        panic!()
    }

    fn compile<'a>(&'a mut self, _block: &IRBlock) -> CodeBlockRef<'a, Self::PatchKind> {
        panic!()
    }

    fn set_dump_asm(&mut self, _enable: bool) {
        panic!()
    }

    unsafe fn patch(_buf: *mut u8, _loc: PatchLoc<Self::PatchKind>, _imm: i64) {
        panic!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::x64::{X64, X64Inst};

    #[test]
    fn test_codegen_context_asm_text() {
        let mut code_buf = CodeBuf::new();
        let mut asm_buf = String::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b11);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx =
            CodeGenContext::<X64>::new(&mut code_buf, Some(&mut asm_buf), &mut state, &liveness);

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
        let mut codegen = CodeGen::<X64>::try_new().unwrap();
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

        let mut ctx = CodeGenContext::<X64>::new(&mut code_buf, None, &mut state, &liveness);
        let v0 = VReg::from(0);
        let _ = ctx.use_vreg(v0);
    }

    #[test]
    fn test_prepare_call_args() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b1111);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X64>::new(&mut code_buf, None, &mut state, &liveness);
        let v0 = VReg::from(0);
        let _ = ctx.def_vreg(v0);
        ctx.end_instruction();

        ctx.prepare_call_args([(crate::jit::backend::x64::X64Reg::RCX, v0.into())]);
        assert!(!code_buf.is_empty());
    }

    #[test]
    #[should_panic(expected = "call argument register must be caller-saved")]
    fn test_prepare_call_args_rejects_callee_saved() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b1111);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X64>::new(&mut code_buf, None, &mut state, &liveness);
        let v0 = VReg::from(0);
        let _ = ctx.def_vreg(v0);
        ctx.end_instruction();

        ctx.prepare_call_args([(crate::jit::backend::x64::X64Reg::RBX, v0.into())]);
    }

    #[test]
    #[should_panic(expected = "is already pinned")]
    fn test_prepare_call_args_rejects_duplicate_target() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b1111);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X64>::new(&mut code_buf, None, &mut state, &liveness);
        let v0 = VReg::from(0);
        let v1 = VReg::from(1);
        let _ = ctx.def_vreg(v0);
        let _ = ctx.def_vreg(v1);
        ctx.end_instruction();

        ctx.prepare_call_args([
            (crate::jit::backend::x64::X64Reg::RCX, v0.into()),
            (crate::jit::backend::x64::X64Reg::RCX, v1.into()),
        ]);
    }

    #[test]
    #[should_panic]
    fn test_codegen_terminator_arm_unreachable() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b1111);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X64>::new(&mut code_buf, None, &mut state, &liveness);
        let inst = IRInst::Terminator(Terminator::DirectJmp { target: 0x1000 });
        let _ = <X64 as ISA>::Backend::codegen(&mut ctx, std::slice::from_ref(&inst));
    }

    #[test]
    fn test_emit_static_jmp_and_patch() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b1111);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X64>::new(&mut code_buf, None, &mut state, &liveness);
        let patch = <X64 as ISA>::Backend::emit_static_jmp(&mut ctx);
        let exit_offset = <X64 as ISA>::Backend::emit_block_exit(&mut ctx, 0x1234);
        ctx.patch_jmp(patch, exit_offset);

        assert_eq!(code_buf[patch.offset], 0xe9);
        let disp = i32::from_le_bytes(
            code_buf[patch.offset + 1..patch.offset + 5]
                .try_into()
                .unwrap(),
        );
        assert_eq!(disp, 0);
    }

    #[test]
    fn test_emit_branch_and_patch() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b1111);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X64>::new(&mut code_buf, None, &mut state, &liveness);
        let (then_patch, else_patch) = <X64 as ISA>::Backend::emit_branch(
            &mut ctx,
            CmpKind::Eq,
            crate::jit::backend::x64::X64Reg::RAX,
            crate::jit::backend::x64::X64Reg::RCX,
        );
        let then_exit = <X64 as ISA>::Backend::emit_block_exit(&mut ctx, 0x1111);
        ctx.patch_jmp(then_patch, then_exit);

        let else_exit = <X64 as ISA>::Backend::emit_block_exit(&mut ctx, 0x2222);
        ctx.patch_jmp(else_patch, else_exit);

        assert_eq!(code_buf[then_patch.offset], 0x0f);
        assert_eq!(code_buf[then_patch.offset + 1], 0x84);
        let then_disp = i32::from_le_bytes(
            code_buf[then_patch.offset + 2..then_patch.offset + 6]
                .try_into()
                .unwrap(),
        );
        assert_eq!(then_patch.offset + 6 + (then_disp as usize), then_exit);

        assert_eq!(code_buf[else_patch.offset], 0xe9);
        let else_disp = i32::from_le_bytes(
            code_buf[else_patch.offset + 1..else_patch.offset + 5]
                .try_into()
                .unwrap(),
        );
        assert_eq!(else_patch.offset + 5 + (else_disp as usize), else_exit);
    }

    #[test]
    fn test_emit_check_and_update_budget() {
        let mut code_buf = CodeBuf::new();
        let mut state = RegAllocState::default();
        state.init(4, 0b1111);
        let liveness = vec![liveness::Liveness::default(); 4];

        let mut ctx = CodeGenContext::<X64>::new(&mut code_buf, None, &mut state, &liveness);
        let oob_patch = <X64 as ISA>::Backend::emit_check_budget(&mut ctx, 10);
        <X64 as ISA>::Backend::emit_update_budget(&mut ctx, 10);
        let oob_exit = <X64 as ISA>::Backend::emit_block_exit(&mut ctx, 0x5000);
        ctx.patch_jmp(oob_patch, oob_exit);

        assert_eq!(code_buf[oob_patch.offset], 0x0f);
        assert_eq!(code_buf[oob_patch.offset + 1], 0x82); // ConditionCode::Below
        let disp = i32::from_le_bytes(
            code_buf[oob_patch.offset + 2..oob_patch.offset + 6]
                .try_into()
                .unwrap(),
        );
        assert_eq!(oob_patch.offset + 6 + (disp as usize), oob_exit);
    }
}
