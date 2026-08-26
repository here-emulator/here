use super::*;

#[derive(Default, Clone)]
struct Liveness {
    def: InstId,
    last_use: Option<InstId>,
    cross_call: bool,
}

struct LivenessPass;

impl LivenessPass {
    fn analyze(block: &IRBlock) -> Vec<Liveness> {
        let vreg_cnt = block.vreg_count() as usize;
        let mut info_table = vec![Liveness::default(); vreg_cnt];
        let mut last_call: Option<InstId> = None;

        for (idx, inst) in block.insts.iter().enumerate() {
            inst.for_each_use(|r| {
                let info = &mut info_table[r.index()];
                info.last_use = Some(idx as InstId);
                if let Some(lst) = last_call
                    && lst > info.def
                {
                    info.cross_call = true;
                }
            });

            if let Some(vreg) = inst.def() {
                info_table[vreg.index()].def = idx as InstId;
            }

            if matches!(inst, IRInst::Call { .. }) {
                last_call = Some(idx as InstId);
            }
        }

        info_table
    }
}
