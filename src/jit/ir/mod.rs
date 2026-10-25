//! Intermediate Representation (IR) for basic blocks.
//!
//! - An [`IRBlock`] is in Static Single Assignment (SSA) form,
//!   which means a [`VReg`] can only be defined once.
//! - An [`IRBlock`] has a single entry and exits only via [`Terminator`] at the end,
//!   with one **exception**: [helper functions](`super::abi::helpers`) can abort.

#[macro_use]
mod macros;

mod ir_builder;
mod vreg_allocator;
pub use ir_builder::*;
pub use vreg_allocator::*;

use crate::config::arch_config::WordType;

pub type VRegId = u32;

/// Never construct it by hand, always use [`IRBuilder`] or [`make_ir`].
pub enum IRInst {
    Const {
        dst: VReg,
        imm: u64,
    },
    GetReg {
        dst: VReg,
        src: GuestReg,
    },
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
    /// Call a runtime helper by address, with at most 6 integer arguments.
    ///
    /// Helpers marked with `can_trap` can abort (cold path).
    /// See [`super::abi::helpers`].
    Call {
        can_trap: bool,
        func_ptr: usize,
        args: Vec<VReg>,
        ret: Option<VReg>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminator {
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

impl Terminator {
    pub fn for_each_use(&self, mut f: impl FnMut(VReg)) {
        match self {
            Terminator::DirectJmp { .. } => {}
            Terminator::DynamicJmp { target } => {
                f(*target);
            }
            Terminator::Branch { src1, src2, .. } => {
                f(*src1);
                f(*src2);
            }
        }
    }
}

impl IRInst {
    pub fn def(&self) -> Option<VReg> {
        use IRInst::*;
        match self {
            GetReg { dst, .. } | Compute { dst, .. } | Const { dst, .. } => Some(*dst),
            Call { ret, .. } => *ret,
            SetReg { .. } => None,
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
            Call { args, .. } => {
                for val in args {
                    f(*val);
                }
            }
            GetReg { .. } | Const { .. } => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IRBlockMeta {
    pub start_pc: WordType,
    pub guest_cnt: u32,
}

impl IRBlockMeta {
    pub const fn new(start_pc: WordType, guest_cnt: u32) -> Self {
        Self {
            start_pc,
            guest_cnt,
        }
    }
}

pub struct IRBlock {
    vreg_count: VRegId,
    insts: Vec<IRInst>,
    terminator: Terminator,
}

impl IRBlock {
    pub fn new(vreg_count: VRegId, insts: Vec<IRInst>, terminator: Terminator) -> Self {
        Self {
            vreg_count,
            insts,
            terminator,
        }
    }

    pub fn vreg_count(&self) -> VRegId {
        self.vreg_count
    }

    pub fn insts(&self) -> &[IRInst] {
        &self.insts
    }

    pub fn terminator(&self) -> &Terminator {
        &self.terminator
    }

    #[allow(dead_code)]
    pub fn terminator_mut(&mut self) -> &mut Terminator {
        &mut self.terminator
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
    pub const fn compute(self, lhs: u64, rhs: u64) -> u64 {
        use ComputeKind::*;
        match self {
            Add => lhs.wrapping_add(rhs),
            Sub => lhs.wrapping_sub(rhs),
            And => lhs & rhs,
            Or => lhs | rhs,
            Xor => lhs ^ rhs,
            Shl => lhs.wrapping_shl(rhs as u32),
            Shr => lhs.wrapping_shr(rhs as u32),
            Sar => (lhs as i64).wrapping_shr(rhs as u32) as u64,
            Slt => ((lhs as i64) < (rhs as i64)) as u64,
            Sltu => (lhs < rhs) as u64,
            AddW => sext_word((lhs as u32).wrapping_add(rhs as u32)),
            SubW => sext_word((lhs as u32).wrapping_sub(rhs as u32)),
            ShlW => sext_word((lhs as u32).wrapping_shl(rhs as u32)),
            ShrW => sext_word((lhs as u32).wrapping_shr(rhs as u32)),
            SarW => sext_word(((lhs as u32 as i32).wrapping_shr(rhs as u32)) as u32),
        }
    }

    /// whether x op y == y op x
    pub const fn commutative(self) -> bool {
        use ComputeKind::*;
        match self {
            Add | And | Or | Xor | AddW => true,
            Sub | Shl | Shr | Sar | Slt | Sltu | SubW | ShlW | ShrW | SarW => false,
        }
    }

    pub const fn is_32bit(self) -> bool {
        use ComputeKind::*;
        match self {
            AddW | SubW | ShlW | ShrW | SarW => true,
            Add | Sub | And | Or | Xor | Shl | Shr | Sar | Slt | Sltu => false,
        }
    }

    pub const fn left_identity(self) -> Option<u64> {
        use ComputeKind::*;
        match self {
            Add | Or | Xor => Some(0),
            And => Some(u64::MAX),
            Sub | Shl | Shr | Sar | Slt | Sltu | AddW | SubW | ShlW | ShrW | SarW => None,
        }
    }

    pub const fn right_identity(self) -> Option<u64> {
        use ComputeKind::*;
        match self {
            Add | Sub | Or | Xor => Some(0),
            And => Some(u64::MAX),
            Shl | Shr | Sar => Some(0),
            Slt | Sltu | AddW | SubW | ShlW | ShrW | SarW => None,
        }
    }
}

const fn sext_word(value: u32) -> u64 {
    value as i32 as i64 as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    ($name:ident, $type:ty, $prefix:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name {
            idx: $type,
        }

        #[allow(dead_code)]
        impl $name {
            pub const fn new(idx: $type) -> Self {
                Self { idx }
            }
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

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, concat!($prefix, "{}"), self.idx)
            }
        }
    };
}

define_index! { GuestReg, u8, "x" }
define_index! { VReg, VRegId, "v" }

impl std::fmt::Display for ComputeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::fmt::Display for CmpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::fmt::Display for IRInst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IRInst::Const { dst, imm } => {
                if *imm > 9 {
                    write!(f, "{dst} = const {imm:#x} ({imm})")
                } else {
                    write!(f, "{dst} = const {imm}")
                }
            }
            IRInst::GetReg { dst, src } => write!(f, "{dst} = get_reg {src}"),
            IRInst::SetReg { dst, src } => write!(f, "set_reg {dst}, {src}"),
            IRInst::Compute {
                kind,
                dst,
                src1,
                src2,
            } => {
                write!(f, "{dst} = compute {kind}, {src1}, {src2}")
            }
            IRInst::Call {
                can_trap,
                func_ptr,
                args,
                ret,
            } => {
                if let Some(r) = ret {
                    write!(f, "{r} = ")?;
                }
                write!(f, "call [")?;
                if *can_trap {
                    write!(f, "trap")?;
                }
                write!(f, "] {func_ptr:#x}, [")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, "]")
            }
        }
    }
}

impl std::fmt::Display for Terminator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Terminator::DirectJmp { target } => write!(f, "jmp {target:#x}"),
            Terminator::DynamicJmp { target } => write!(f, "jmp {target}"),
            Terminator::Branch {
                cmp,
                src1,
                src2,
                then_pc,
                else_pc,
            } => {
                write!(
                    f,
                    "branch {cmp}, {src1}, {src2}, then={then_pc:#x}, else={else_pc:#x}"
                )
            }
        }
    }
}

impl std::fmt::Display for IRBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for inst in &self.insts {
            writeln!(f, "    {inst}")?;
        }
        write!(f, "    {}", self.terminator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminator_def_use_and_methods() {
        let v1 = VReg::from(1);
        let v2 = VReg::from(2);

        let direct = Terminator::DirectJmp { target: 0x1000 };
        let mut uses = Vec::new();
        direct.for_each_use(|u| uses.push(u));
        assert!(uses.is_empty());

        let dynamic = Terminator::DynamicJmp { target: v1 };
        let mut uses = Vec::new();
        dynamic.for_each_use(|u| uses.push(u));
        assert_eq!(uses, vec![v1]);

        let branch = Terminator::Branch {
            cmp: CmpKind::Eq,
            src1: v1,
            src2: v2,
            then_pc: 0x2000,
            else_pc: 0x3000,
        };
        let mut uses = Vec::new();
        branch.for_each_use(|u| uses.push(u));
        assert_eq!(uses, vec![v1, v2]);
    }
}
