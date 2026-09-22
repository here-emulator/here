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
    Terminator(Terminator),
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
    pub fn is_terminator(&self) -> bool {
        matches!(self, IRInst::Terminator(_))
    }

    pub fn get_terminator(&self) -> Option<Terminator> {
        match self {
            IRInst::Terminator(term) => Some(*term),
            _ => None,
        }
    }

    pub fn def(&self) -> Option<VReg> {
        use IRInst::*;
        match self {
            GetReg { dst, .. } | Compute { dst, .. } | Const { dst, .. } => Some(*dst),
            Call { ret, .. } => *ret,
            SetReg { .. } | Terminator(_) => None,
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
            Terminator(term) => {
                term.for_each_use(f);
            }
            GetReg { .. } | Const { .. } => {}
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

    pub fn get_terminator(&self) -> Option<Terminator> {
        self.insts.last().and_then(IRInst::get_terminator)
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
define_index! { HostReg, u8, "h" }

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
                pass_cpu,
                func_ptr,
                args,
                ret,
            } => {
                if let Some(r) = ret {
                    write!(f, "{r} = ")?;
                }
                write!(f, "call [")?;
                let mut first = true;
                if *pass_cpu {
                    write!(f, "cpu")?;
                    first = false;
                }
                if *can_trap {
                    if !first {
                        write!(f, ", ")?;
                    }
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
            IRInst::Terminator(t) => write!(f, "{t}"),
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
        for (i, inst) in self.insts.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "    {inst}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_display() {
        let mut builder = IRBuilder::default();
        make_ir!(builder, [
            v0 = const 42;
            v1 = get_reg 1;
            v2 = compute Add, v0, v1;
            set_reg 2, v2;
        ], []);
        let block = builder.build_jmp(0x1000u64.into());
        let s = format!("{block}");
        assert!(s.contains("const 0x2a (42)"));
        assert!(s.contains("get_reg x1"));
        assert!(s.contains("compute Add"));
        assert!(s.contains("set_reg x2"));
        assert!(s.contains("jmp 0x1000"));
    }

    #[test]
    fn test_terminator_def_use_and_methods() {
        let v1 = VReg::from(1);
        let v2 = VReg::from(2);

        let direct = IRInst::Terminator(Terminator::DirectJmp { target: 0x1000 });
        assert!(direct.is_terminator());
        assert_eq!(direct.def(), None);
        let mut uses = Vec::new();
        direct.for_each_use(|u| uses.push(u));
        assert!(uses.is_empty());
        assert_eq!(
            direct.get_terminator(),
            Some(Terminator::DirectJmp { target: 0x1000 })
        );

        let dynamic = IRInst::Terminator(Terminator::DynamicJmp { target: v1 });
        assert!(dynamic.is_terminator());
        assert_eq!(dynamic.def(), None);
        let mut uses = Vec::new();
        dynamic.for_each_use(|u| uses.push(u));
        assert_eq!(uses, vec![v1]);
        assert_eq!(
            dynamic.get_terminator(),
            Some(Terminator::DynamicJmp { target: v1 })
        );

        let branch = IRInst::Terminator(Terminator::Branch {
            cmp: CmpKind::Eq,
            src1: v1,
            src2: v2,
            then_pc: 0x2000,
            else_pc: 0x3000,
        });
        assert!(branch.is_terminator());
        assert_eq!(branch.def(), None);
        let mut uses = Vec::new();
        branch.for_each_use(|u| uses.push(u));
        assert_eq!(uses, vec![v1, v2]);
        assert_eq!(
            branch.get_terminator(),
            Some(Terminator::Branch {
                cmp: CmpKind::Eq,
                src1: v1,
                src2: v2,
                then_pc: 0x2000,
                else_pc: 0x3000,
            })
        );
    }
}
