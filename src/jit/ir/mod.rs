#[macro_use]
mod macros;

mod ir_builder;
pub use ir_builder::*;

pub type VRegId = u32;

/// Never construct it by hand, always use [`IRBuilder`] or [`make_ir`].
///
/// `branch` and `jump` can only exist as the last instruction.
pub enum IRInst {
    Const {
        dst: VReg,
        imm: u64,
    },
    GetReg {
        dst: VReg,
        src: GuestReg,
    },
    // TODO: Trace vreg to guest reg mapping, instead of a dedicated Inst.
    SetReg {
        dst: GuestReg,
        src: VReg,
    },
    Compute {
        kind: ComputeKind,
        dst: VReg,
        src1: VReg,
        src2: VReg,
    },
    Call {
        can_trap: bool,
        /// Pass the current `*mut RVCPU` as the helper's first ABI argument.
        ///
        /// The CPU pointer is supplied by the backend and is not an SSA value.
        pass_cpu: bool,
        func_ptr: usize,
        args: Vec<VReg>,
        ret: Option<VReg>,
    },
    // TODO: move terminator into a seperate enum.
    DirectJmp {
        target: u64,
    },
    DynamicJmp {
        target: VReg,
    },
    Branch {
        cmp: CmpKind,
        src1: VReg,
        src2: VReg,
        then_pc: u64,
        else_pc: u64,
    },
}

impl IRInst {
    pub fn is_terminator(&self) -> bool {
        use IRInst::*;
        matches!(self, DirectJmp { .. } | DynamicJmp { .. } | Branch { .. })
    }

    pub fn def(&self) -> Option<VReg> {
        use IRInst::*;
        match self {
            GetReg { dst, .. } | Compute { dst, .. } | Const { dst, .. } => {
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
            Compute { src1, src2, .. } | Branch { src1, src2, .. } => {
                f(*src1);
                f(*src2);
            }
            Call { args, .. } => {
                for val in args {
                    f(*val);
                }
            }
            DynamicJmp { target: target_pc } => {
                f(*target_pc);
            }
            GetReg { .. } | DirectJmp { .. } | Const { .. } => {}
        }
    }
}

pub struct IRBlock {
    vreg_count: VRegId,
    pub insts: Vec<IRInst>,
}

impl IRBlock {
    pub fn vreg_count(&self) -> VRegId {
        self.vreg_count
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ComputeKind {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Sar,
    Slt,
    Sltu,
    AddW,
    SubW,
    ShlW,
    ShrW,
    SarW,
}

impl ComputeKind {
    fn compute(self, lhs: u64, rhs: u64) -> u64 {
        match self {
            ComputeKind::Add => lhs.wrapping_add(rhs),
            ComputeKind::Sub => lhs.wrapping_sub(rhs),
            ComputeKind::And => lhs & rhs,
            ComputeKind::Or => lhs | rhs,
            ComputeKind::Xor => lhs ^ rhs,
            ComputeKind::Shl => lhs.wrapping_shl(rhs as u32),
            ComputeKind::Shr => lhs.wrapping_shr(rhs as u32),
            ComputeKind::Sar => (lhs as i64).wrapping_shr(rhs as u32) as u64,
            ComputeKind::Slt => ((lhs as i64) < (rhs as i64)) as u64,
            ComputeKind::Sltu => (lhs < rhs) as u64,
            ComputeKind::AddW => sext_word((lhs as u32).wrapping_add(rhs as u32)),
            ComputeKind::SubW => sext_word((lhs as u32).wrapping_sub(rhs as u32)),
            ComputeKind::ShlW => sext_word((lhs as u32).wrapping_shl(rhs as u32)),
            ComputeKind::ShrW => sext_word((lhs as u32).wrapping_shr(rhs as u32)),
            ComputeKind::SarW => sext_word(((lhs as u32 as i32).wrapping_shr(rhs as u32)) as u32),
        }
    }

    /// whether x op y == y op x
    fn commutative(self) -> bool {
        use ComputeKind::*;
        match self {
            Add | And | Or | Xor | AddW => true,
            Sub | Shl | Shr | Sar | Slt | Sltu | SubW | ShlW | ShrW | SarW => false,
        }
    }

    fn left_identity(self) -> Option<u64> {
        use ComputeKind::*;
        match self {
            Add | Or | Xor => Some(0),
            And => Some(u64::MAX),
            Sub | Shl | Shr | Sar | Slt | Sltu | AddW | SubW | ShlW | ShrW | SarW => None,
        }
    }

    fn right_identity(self) -> Option<u64> {
        use ComputeKind::*;
        match self {
            Add | Sub | Or | Xor => Some(0),
            And => Some(u64::MAX),
            Shl | Shr | Sar => Some(0),
            Slt | Sltu | AddW | SubW | ShlW | ShrW | SarW => None,
        }
    }
}

fn sext_word(value: u32) -> u64 {
    value as i32 as i64 as u64
}

#[derive(Debug, Clone, Copy)]
pub enum CmpKind {
    Eq,
    NotEq,
    GtEq,
    GtEqUnsigned,
    Lt,
    LtUnsigned,
}

pub trait IRReg: From<Self::IndexType> {
    type IndexType;
    fn index(self) -> usize;
}

macro_rules! define_index {
    ($name:ident, $type:ty) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name {
            idx: $type,
        }

        impl From<$type> for $name {
            fn from(value: $type) -> Self {
                Self { idx: value }
            }
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
define_index! { VReg, VRegId }
define_index! { HostReg, u8 }
