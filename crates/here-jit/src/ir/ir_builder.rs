use crate::config::arch_config::REGFILE_CNT;

use super::*;

/// [`VReg`] or immidiate.
///
/// Only used for [`ir_builder`] (const folding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl From<usize> for Value {
    fn from(value: usize) -> Self {
        Self::Imm(value as u64)
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Self::Imm(value as i64 as u64)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Imm(value as u64)
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Self::Imm(value as u64)
    }
}

/// ABI properties of an IR helper call.
#[derive(Debug, Clone, Copy, Default)]
pub struct CallOptions {
    can_trap: bool,
}

impl CallOptions {
    #[must_use]
    pub const fn trapping(mut self) -> Self {
        self.can_trap = true;
        self
    }
}

/// Build [`IRBlock`]. Check out [`make_ir`] for a convenient usage.
///
/// To reduce the amount of [`IRInst`], instead of relying on subsequent IR passes, [`ir_builder`] will:
/// - do simple constant folding;
/// - track the guest register states, trying to omit [`Self::set_reg`];
pub struct IRBuilder {
    guest_map: [Option<Value>; REGFILE_CNT],
    last_saved_guest: [Option<Value>; REGFILE_CNT],
    // TODO: reuse buffer
    insts: Vec<IRInst>,
    vreg_allocator: VRegAllocator,
}

impl Default for IRBuilder {
    fn default() -> Self {
        Self {
            guest_map: [None; REGFILE_CNT],
            last_saved_guest: [None; REGFILE_CNT],
            insts: Vec::new(),
            vreg_allocator: VRegAllocator::default(),
        }
    }
}

impl IRBuilder {
    fn alloc_vreg(&mut self) -> VReg {
        self.vreg_allocator.alloc()
    }

    fn save_guest_regs(&mut self) {
        for idx in 0..REGFILE_CNT {
            let (curr, last) = (self.guest_map[idx], self.last_saved_guest[idx]);
            if curr == last {
                continue;
            }

            self.last_saved_guest[idx] = curr;

            if let Some(src) = curr {
                let src = self.value_to_vreg(src);
                self.insts.push(IRInst::SetReg {
                    dst: (idx as u8).into(),
                    src,
                });
            }
        }
    }

    fn value_to_vreg(&mut self, value: Value) -> VReg {
        match value {
            Value::VReg(reg) => reg,
            Value::Imm(imm) => {
                // TODO: record previously generated const values
                let reg = self.alloc_vreg();
                self.insts.push(IRInst::Const { dst: reg, imm });
                reg
            }
        }
    }

    fn emit_call(
        &mut self,
        options: CallOptions,
        func_ptr: usize,
        args: Vec<Value>,
        ret: Option<VReg>,
    ) {
        if options.can_trap {
            self.save_guest_regs();
        }

        let args = args
            .into_iter()
            .map(|value| self.value_to_vreg(value))
            .collect();

        self.insts.push(IRInst::Call {
            can_trap: options.can_trap,
            func_ptr,
            args,
            ret,
        });
    }

    fn build(self, terminator: Terminator) -> IRBlock {
        IRBlock::new(self.vreg_allocator.vreg_count(), self.insts, terminator)
    }
}

#[allow(dead_code)]
impl IRBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub const fn cpu_vreg(&self) -> VReg {
        self.vreg_allocator.cpu_vreg()
    }

    #[inline]
    pub const fn jit_context_vreg(&self) -> VReg {
        self.vreg_allocator.jit_context_vreg()
    }

    #[must_use]
    pub fn const_value(&mut self, imm: u64) -> Value {
        Value::Imm(imm)
    }

    #[must_use]
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

    #[must_use]
    pub fn compute(&mut self, kind: ComputeKind, src1: Value, src2: Value) -> Value {
        use Value::*;
        match (src1, src2, kind.commutative()) {
            (Imm(lhs), Imm(rhs), _) => return Imm(kind.compute(lhs, rhs)),

            (VReg(lhs), Imm(imm), _) | (Imm(imm), VReg(lhs), true) => {
                if Some(imm) == kind.right_identity() {
                    return Value::VReg(lhs);
                }
            }

            (Imm(imm), VReg(rhs), false) => {
                if Some(imm) == kind.left_identity() {
                    return Value::VReg(rhs);
                }
            }

            _ => {}
        }

        let dst = self.alloc_vreg();
        let src1 = self.value_to_vreg(src1);
        let src2 = self.value_to_vreg(src2);
        self.insts.push(IRInst::Compute {
            kind,
            dst,
            src1,
            src2,
        });
        Value::VReg(dst)
    }

    /// Emit a helper call without a return value.
    pub fn call(&mut self, func_ptr: usize, args: Vec<Value>, options: CallOptions) {
        self.emit_call(options, func_ptr, args, None);
    }

    /// Emit a helper call with one return value.
    #[must_use]
    pub fn call_ret(&mut self, func_ptr: usize, args: Vec<Value>, options: CallOptions) -> Value {
        let ret = self.alloc_vreg();
        self.emit_call(options, func_ptr, args, Some(ret));
        ret.into()
    }

    /// `target`: must be absolute address.
    pub fn build_jmp(mut self, target: Value) -> IRBlock {
        self.save_guest_regs();

        let terminator = match target {
            Value::Imm(value) => Terminator::DirectJmp { target: value },
            Value::VReg(reg) => Terminator::DynamicJmp { target: reg },
        };

        self.build(terminator)
    }

    /// `target`: must be absolute address.
    pub fn build_branch(
        mut self,
        cmp: CmpKind,
        src1: Value,
        src2: Value,
        then_pc: u64,
        else_pc: u64,
    ) -> IRBlock {
        self.save_guest_regs();

        let src1 = self.value_to_vreg(src1);
        let src2 = self.value_to_vreg(src2);

        let terminator = Terminator::Branch {
            cmp,
            src1,
            src2,
            then_pc,
            else_pc,
        };
        self.build(terminator)
    }
}
