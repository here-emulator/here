use super::{VReg, VRegId};

pub const CPU_VREG: VReg = VReg::new(0);
pub const JIT_CONTEXT_VREG: VReg = VReg::new(1);
const FIRST_DYNAMIC_VREG: VRegId = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VRegAllocator {
    next: VRegId,
}

impl VRegAllocator {
    pub const CPU_VREG: VReg = CPU_VREG;
    pub const JIT_CONTEXT_VREG: VReg = JIT_CONTEXT_VREG;

    #[inline]
    pub const fn new() -> Self {
        Self {
            next: FIRST_DYNAMIC_VREG,
        }
    }

    #[inline]
    pub const fn cpu_vreg(&self) -> VReg {
        Self::CPU_VREG
    }

    #[inline]
    pub const fn jit_context_vreg(&self) -> VReg {
        Self::JIT_CONTEXT_VREG
    }

    #[inline]
    pub fn alloc(&mut self) -> VReg {
        let reg = VReg::new(self.next);
        self.next += 1;
        reg
    }

    #[inline]
    pub fn vreg_count(&self) -> VRegId {
        self.next
    }
}

impl Default for VRegAllocator {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
