use crate::config::arch_config::REGFILE_CNT;

use super::*;

pub struct IRBuilder {
    guest_map: [Option<Value>; REGFILE_CNT],
    insts: Vec<IRInst>,
    next: <VReg as IRReg>::IndexType,
}

impl IRBuilder {
    fn alloc_vreg(&mut self) -> VReg {
        let reg = VReg { idx: self.next };
        self.next += 1;
        reg
    }

    pub fn get_reg(&mut self, guest: GuestReg) -> Value {
        if guest.index() == 0 {
            return 0.into();
        }

        if let Some(value) = self.guest_map[guest.index()] {
            return value;
        }

        let vreg = self.alloc_vreg();
        self.insts.push(IRInst::GetReg {
            dst: vreg,
            src: guest,
        });

        vreg.into()
    }

    pub fn set_reg(&mut self, guest: GuestReg, value: Value) {
        if guest.index() == 0 {
            return;
        }

        self.guest_map[guest.index()] = Some(value);
    }

    pub fn compute(&mut self, kind: ALUKind, src1: Value, src2: Value) -> Value {
        use Value::*;
        match (src1, src2, kind.commutative()) {
            (Imm(lhs), Imm(rhs), _) => Imm(kind.compute(lhs, rhs)),

            (VReg(vreg), Imm(imm), _) | (Imm(imm), VReg(vreg), true) => {
                if Some(imm) == kind.right_identity() {
                    Value::VReg(vreg)
                } else {
                    let dst = self.alloc_vreg();
                    self.insts.push(IRInst::ComputeImm {
                        kind,
                        dst,
                        src1: vreg,
                        src2: imm,
                    });
                    Value::VReg(dst)
                }
            }

            (Imm(imm), VReg(rhs), false) => {
                if Some(imm) == kind.left_identity() {
                    Value::VReg(rhs)
                } else {
                    let lhs = self.alloc_vreg();
                    self.insts.push(IRInst::Const { dst: lhs, src: imm });
                    let dst = self.alloc_vreg();
                    self.insts.push(IRInst::Compute {
                        kind,
                        dst,
                        src1: lhs,
                        src2: rhs,
                    });
                    Value::VReg(dst)
                }
            }

            (VReg(lhs), VReg(rhs), _) => {
                let dst = self.alloc_vreg();
                self.insts.push(IRInst::Compute {
                    kind,
                    dst,
                    src1: lhs,
                    src2: rhs,
                });
                Value::VReg(dst)
            }
        }
    }

    pub fn build(self) -> IRBlock {
        debug_assert!(self.insts.last().unwrap().is_terminator());

        IRBlock {
            insts: self.insts,
            vreg_count: self.next,
        }
    }
}
