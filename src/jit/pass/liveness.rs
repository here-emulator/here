//! Analyze when a vreg last used and whether it's def-use across function call.
//!
//! Used at the beginning of codegen, to allocate registers.

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
        if self.across_call {
            VRegHint::PreferCallee
        } else {
            VRegHint::PreferCaller
        }
    }
}

pub fn analyze(block: &IRBlock, liveness: &mut Vec<Liveness>) {
    let vreg_cnt = block.vreg_count() as usize;
    liveness.clear();
    liveness.resize(vreg_cnt, Liveness::default());

    let mut last_call: Option<usize> = None;
    for (idx, inst) in block.insts.iter().enumerate() {
        inst.for_each_use(|r| {
            let info = &mut liveness[r.index()];
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
            liveness[vreg.index()].def = idx;
        }
    }
}
