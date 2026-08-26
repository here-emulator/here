use crate::config::arch_config::WordType;

#[macro_use]
mod macros;

mod ir_builder;
pub use ir_builder::*;

pub type InstId = u32;

#[derive(Debug, Clone, Copy)]
pub enum Value {
    VReg(VReg),
    Imm(u64),
}

impl From<VReg> for Value {
    fn from(value: VReg) -> Self {
        Self::VReg(value)
    }
}

impl From<u64> for Value {
    fn from(value: u64) -> Self {
        Self::Imm(value)
    }
}

/// Never construct it by hand, use [`IRBuilder`].
pub enum IRInst {
    Const {
        dst: VReg,
        src: u64,
    },
    GetReg {
        dst: VReg,
        src: GuestReg,
    },
    SetReg {
        dst: GuestReg,
        src: VReg,
    },
    /// If only one operand is imm, put it on `src2`.
    Compute {
        kind: ALUKind,
        dst: VReg,
        src1: VReg,
        src2: VReg,
    },
    ComputeImm {
        kind: ALUKind,
        dst: VReg,
        src1: VReg,
        src2: u64,
    },
    Call {
        can_trap: bool,
        func_ptr: usize,
        args: Vec<VReg>,
        ret: Option<VReg>,
    },
    DirectJmp {
        target_pc: WordType,
    },
    DynamicJmp {
        target_pc: VReg,
    },
    Branch {
        cond_bool: VReg,
        then_pc: WordType,
        else_pc: WordType,
    },
}

impl IRInst {
    pub fn is_terminator(&self) -> bool {
        use IRInst::*;
        matches!(
            self,
            DirectJmp { .. } | DynamicJmp { .. } | Branch { .. } | Call { can_trap: true, .. }
        )
    }

    pub fn def(&self) -> Option<VReg> {
        use IRInst::*;
        match self {
            GetReg { dst, .. }
            | Compute { dst, .. }
            | ComputeImm { dst, .. }
            | Const { dst, .. } => {
                return Some(*dst);
            }
            Call { ret, .. } => {
                return *ret;
            }
            SetReg { .. } | DirectJmp { .. } | DynamicJmp { .. } | Branch { .. } => {
                return None;
            }
        }
    }

    pub fn for_each_use(&self, mut f: impl FnMut(VReg)) {
        use IRInst::*;
        match self {
            SetReg { src, .. } => {
                f(*src);
            }
            Compute { src1, src2, .. } => {
                f(*src1);
                f(*src2);
            }
            ComputeImm { src1, .. } => f(*src1),
            Call { args, .. } => {
                for val in args {
                    f(*val);
                }
            }
            DynamicJmp { target_pc } => {
                f(*target_pc);
            }
            Branch { cond_bool, .. } => {
                f(*cond_bool);
            }
            GetReg { .. } | DirectJmp { .. } | Const { .. } => {}
        }
    }
}

pub struct IRBlock {
    vreg_count: InstId,
    pub insts: Vec<IRInst>,
}

impl IRBlock {
    pub fn vreg_count(&self) -> InstId {
        self.vreg_count
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ALUKind {
    Add,
    Sub,
    And,
    Or,
    Xor,
}

impl ALUKind {
    fn compute(self, lhs: u64, rhs: u64) -> u64 {
        match self {
            ALUKind::Add => lhs.wrapping_add(rhs),
            ALUKind::Sub => lhs.wrapping_sub(rhs),
            ALUKind::And => lhs & rhs,
            ALUKind::Or => lhs | rhs,
            ALUKind::Xor => lhs ^ rhs,
        }
    }

    fn commutative(self) -> bool {
        use ALUKind::*;
        match self {
            Sub => false,
            Add | And | Or | Xor => true,
        }
    }

    fn left_identity(self) -> Option<u64> {
        use ALUKind::*;
        match self {
            Add | Or | Xor => Some(0),
            And => Some(u64::MAX),
            Sub => None,
        }
    }

    fn right_identity(self) -> Option<u64> {
        use ALUKind::*;
        match self {
            Add | Sub | Or | Xor => Some(0),
            And => Some(u64::MAX),
        }
    }
}

pub trait IRReg {
    type IndexType;
    fn index(self) -> usize;
}

macro_rules! define_index {
    ($name:ident, $type:ty) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name {
            idx: $type,
        }

        impl IRReg for $name {
            type IndexType = $type;
            fn index(self) -> usize {
                self.idx as usize
            }
        }
    };
}

define_index! { GuestReg, u8 }
define_index! { VReg, InstId }
define_index! { HostReg, u8 }
