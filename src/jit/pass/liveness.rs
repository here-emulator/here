//! Used at the beginning of codegen, for register allcation.

use super::*;

#[derive(Clone, Copy)]
pub enum VRegHint {
    PreferCallee,
    PreferCaller,
}

#[derive(Default, Clone)]
pub struct Liveness {
    def: usize,
    last_use: Option<usize>,
    /// Whether there's a call between its def-use.
    across_call: bool,
}

#[allow(dead_code)]
impl Liveness {
    pub fn def(&self) -> usize {
        self.def
    }

    pub fn last_use(&self) -> Option<usize> {
        self.last_use
    }

    pub fn across_call(&self) -> bool {
        self.across_call
    }

    pub fn reg_hint(&self) -> VRegHint {
        if self.across_call() {
            VRegHint::PreferCallee
        } else {
            VRegHint::PreferCaller
        }
    }
}

#[derive(Default)]
pub struct LivenessPass {
    vregs: Vec<Liveness>,
}
impl LivenessPass {
    pub fn run(&mut self, block: &IRBlock) {
        let vreg_cnt = block.vreg_count() as usize;
        self.vregs.clear();
        self.vregs.resize(vreg_cnt, Liveness::default());

        let mut last_call: Option<usize> = None;
        for (idx, inst) in block.insts().iter().enumerate() {
            inst.for_each_use(|r| {
                let info = &mut self.vregs[r.index()];
                info.last_use = Some(idx);
                if let Some(lst) = last_call {
                    if lst > info.def {
                        info.across_call = true;
                    }
                }
            });

            if matches!(inst, IRInst::Call { .. }) {
                last_call = Some(idx);
            }

            if let Some(vreg) = inst.def() {
                self.vregs[vreg.index()].def = idx;
            }
        }

        let term_idx = block.insts().len();
        block.terminator().for_each_use(|r| {
            let info = &mut self.vregs[r.index()];
            info.last_use = Some(term_idx);
            if let Some(lst) = last_call {
                if lst > info.def {
                    info.across_call = true;
                }
            }
        });
    }
}

impl std::ops::Index<VReg> for LivenessPass {
    type Output = Liveness;

    #[inline]
    fn index(&self, vreg: VReg) -> &Self::Output {
        &self.vregs[vreg.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liveness_basic_def_use() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let v2 = VReg::new(2);
        let insts = vec![
            IRInst::Const { dst: v0, imm: 42 },
            IRInst::Const { dst: v1, imm: 100 },
            IRInst::Compute {
                kind: ComputeKind::Add,
                dst: v2,
                src1: v0,
                src2: v1,
            },
        ];
        let block = IRBlock::new(3, insts, Terminator::DirectJmp { target: 0x1000 });

        let mut pass = LivenessPass::default();
        pass.run(&block);

        assert_eq!(pass[v0].def(), 0);
        assert_eq!(pass[v0].last_use(), Some(2));
        assert!(!pass[v0].across_call());

        assert_eq!(pass[v1].def(), 1);
        assert_eq!(pass[v1].last_use(), Some(2));
        assert!(!pass[v1].across_call());

        assert_eq!(pass[v2].def(), 2);
        assert_eq!(pass[v2].last_use(), None);
    }

    #[test]
    fn test_liveness_across_call() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let insts = vec![
            IRInst::Const { dst: v0, imm: 1 },
            IRInst::Call {
                can_trap: false,
                func_ptr: 0x1234,
                args: vec![],
                ret: None,
            },
            IRInst::Compute {
                kind: ComputeKind::Add,
                dst: v1,
                src1: v0,
                src2: v0,
            },
        ];
        let block = IRBlock::new(2, insts, Terminator::DirectJmp { target: 0x1000 });

        let mut pass = LivenessPass::default();
        pass.run(&block);

        assert!(pass[v0].across_call());
        assert!(matches!(pass[v0].reg_hint(), VRegHint::PreferCallee));
    }

    #[test]
    fn test_liveness_terminator_use() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let insts = vec![
            IRInst::Const { dst: v0, imm: 10 },
            IRInst::Const { dst: v1, imm: 20 },
        ];
        let term = Terminator::Branch {
            cmp: CmpKind::Eq,
            src1: v0,
            src2: v1,
            then_pc: 0x2000,
            else_pc: 0x3000,
        };
        let block = IRBlock::new(2, insts, term);

        let mut pass = LivenessPass::default();
        pass.run(&block);

        assert_eq!(pass[v0].last_use(), Some(2));
        assert_eq!(pass[v1].last_use(), Some(2));
    }
}
