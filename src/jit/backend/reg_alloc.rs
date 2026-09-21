//! At most 32 target register is supported right now.

use super::*;

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegLocation {
    #[default]
    None,
    Reg(u8),
    Spill(u32),
    Const(u64),
    ConstInReg(u8, u64),
    // TODO: add Guest(u8),
}

impl RegLocation {
    #[inline]
    pub const fn get_reg(&self) -> Option<u8> {
        match *self {
            Self::Reg(reg) | Self::ConstInReg(reg, _) => Some(reg),
            _ => None,
        }
    }

    #[inline]
    pub const fn get_const(&self) -> Option<u64> {
        match *self {
            Self::Const(imm) | Self::ConstInReg(_, imm) => Some(imm),
            _ => None,
        }
    }

    #[inline]
    pub const fn get_spill(&self) -> Option<u32> {
        match *self {
            Self::Spill(slot) => Some(slot),
            _ => None,
        }
    }

    #[inline]
    pub const fn is_const(&self) -> bool {
        self.get_const().is_some()
    }
}

/// At most 32 registier is supported, which should be enough.
#[derive(Default)]
pub struct RegAllocState {
    hreg_to_vreg: [Option<VReg>; 32],
    vreg_to_loc: Vec<RegLocation>,

    allocatable_mask: u32,
    /// Whether the target register holds a `VReg`, it doesn't indicate whether it's pinned.
    occupied_mask: u32,
    /// Physical registers pinned for the duration of the current instruction.
    inst_pinned_mask: u32,

    free_spill_slot: Vec<u32>,
    slot_count: u32,
}

impl RegAllocState {
    pub fn init(&mut self, vreg_cnt: VRegId, allocatable_mask: u32) {
        self.hreg_to_vreg.fill(None);
        self.vreg_to_loc.clear();
        self.vreg_to_loc
            .resize(vreg_cnt as usize, RegLocation::default());
        self.allocatable_mask = allocatable_mask;
        self.occupied_mask = 0;
        self.inst_pinned_mask = 0;
        self.free_spill_slot.clear();
        self.slot_count = 0;
    }

    #[inline]
    fn pin(&mut self, reg: u8) {
        self.inst_pinned_mask |= 1 << reg;
    }

    #[inline]
    fn unpin(&mut self, reg: u8) {
        self.inst_pinned_mask &= !(1 << reg);
    }

    #[inline]
    fn clear_inst_pins(&mut self) {
        self.inst_pinned_mask = 0;
    }

    #[inline]
    fn is_pinned(&self, reg: u8) -> bool {
        (self.inst_pinned_mask & (1 << reg)) != 0
    }

    #[inline]
    fn pinned_mask(&self) -> u32 {
        self.inst_pinned_mask
    }

    #[inline]
    fn spillable_mask(&self) -> u32 {
        self.occupied_mask & !self.pinned_mask()
    }

    #[inline]
    fn free_mask(&self) -> u32 {
        self.allocatable_mask & !self.occupied_mask & !self.pinned_mask()
    }

    #[inline]
    fn const_mask(&self) -> u32 {
        let mut mask = 0u32;
        let mut occupied = self.occupied_mask;
        while occupied != 0 {
            let r = occupied.trailing_zeros();
            if let Some(vreg) = self.hreg_to_vreg[r as usize] {
                if self.vreg_to_loc[vreg.index()].is_const() {
                    mask |= 1 << r;
                }
            }
            occupied &= occupied - 1;
        }
        mask
    }

    fn unbind(&mut self, vreg: VReg) {
        let loc = self.vreg_to_loc[vreg.index()];
        if let Some(old_reg) = loc.get_reg() {
            self.hreg_to_vreg[old_reg as usize] = None;
            self.occupied_mask &= !(1 << old_reg);
        } else if let Some(slot) = loc.get_spill() {
            self.free_spill_slot.push(slot);
        }

        self.vreg_to_loc[vreg.index()] = RegLocation::None;
    }

    fn bind(&mut self, vreg: VReg, reg: u8) {
        let const_val = self.vreg_to_loc[vreg.index()].get_const();
        self.unbind(vreg);

        debug_assert!(
            self.hreg_to_vreg[reg as usize].is_none(),
            "Reg {} is currently occupied by {:?}.",
            reg,
            self.hreg_to_vreg[reg as usize]
        );

        self.hreg_to_vreg[reg as usize] = Some(vreg);
        self.occupied_mask |= 1 << reg;
        self.vreg_to_loc[vreg.index()] = match const_val {
            Some(imm) => RegLocation::ConstInReg(reg, imm),
            None => RegLocation::Reg(reg),
        };
    }

    pub fn def_const(&mut self, vreg: VReg, imm: u64) {
        self.unbind(vreg);
        self.vreg_to_loc[vreg.index()] = RegLocation::Const(imm);
    }

    #[inline]
    pub fn get_const(&self, vreg: VReg) -> Option<u64> {
        self.vreg_to_loc[vreg.index()].get_const()
    }

    pub fn current_use_slot(&self) -> u32 {
        self.slot_count
    }

    fn alloc_spill_slot(&mut self) -> u32 {
        self.free_spill_slot.pop().unwrap_or_else(|| {
            let slot = self.slot_count;
            self.slot_count += 1;
            slot
        })
    }

    #[must_use]
    fn evict(&mut self, reg: u8) -> Option<(VReg, u32)> {
        if self.is_pinned(reg) {
            panic!("trying to evict pinned register {:?}", reg);
        }

        let occupant = self.hreg_to_vreg[reg as usize]?;

        self.hreg_to_vreg[reg as usize] = None;
        self.occupied_mask &= !(1 << reg);

        if let Some(imm) = self.vreg_to_loc[occupant.index()].get_const() {
            self.vreg_to_loc[occupant.index()] = RegLocation::Const(imm);
            None
        } else {
            let slot = self.alloc_spill_slot();
            self.vreg_to_loc[occupant.index()] = RegLocation::Spill(slot);
            Some((occupant, slot))
        }
    }

    #[inline]
    fn pick_in(&self, mask: u32) -> Option<u8> {
        if mask != 0 {
            Some(mask.trailing_zeros() as u8)
        } else {
            None
        }
    }

    #[must_use]
    fn pick_free_in(&self, mask: u32) -> Option<u8> {
        self.pick_in(self.free_mask() & mask)
    }

    #[must_use]
    fn pick_free(&self) -> Option<u8> {
        self.pick_free_in(u32::MAX)
    }

    #[must_use]
    fn pick_free_prefer(&self, prefer: u32) -> Option<u8> {
        self.pick_free_in(prefer).or_else(|| self.pick_free())
    }

    #[must_use]
    fn pick_victim_in(&self, mask: u32) -> Option<u8> {
        self.pick_in(self.spillable_mask() & mask)
    }

    fn pick_victim_prefer(&self, prefer: u32) -> u8 {
        self.pick_victim_in(prefer)
            .or_else(|| self.pick_victim_in(u32::MAX))
            .expect("no spillable register available")
    }

    fn pick_victim(&self) -> u8 {
        self.pick_victim_prefer(self.const_mask())
    }
}

impl<'b, I: ISA> CodeGenContext<'b, I> {
    #[inline]
    pub fn emit_mov(&mut self, dst: I::Reg, src: I::Reg) {
        self.emit(&I::Inst::mov(dst, src));
    }

    #[inline]
    pub fn emit_mov_imm(&mut self, dst: I::Reg, imm: u64) {
        self.emit(&I::Inst::mov_imm(dst, imm));
    }

    #[inline]
    pub fn def_const(&mut self, vreg: VReg, imm: u64) {
        self.state.def_const(vreg, imm);
    }

    #[inline]
    pub fn try_get_const(&self, vreg: VReg) -> Option<u64> {
        self.state.get_const(vreg)
    }

    #[inline]
    pub fn try_get_const_i32(&self, vreg: VReg) -> Option<i32> {
        self.try_get_const(vreg)
            .and_then(|val| i32::try_from(val as i64).ok())
    }

    #[inline]
    pub fn emit_spill(&mut self, slot: u32, src: I::Reg) {
        self.emit(&I::Inst::spill(slot, src));
    }

    #[inline]
    pub fn emit_reload(&mut self, dst: I::Reg, slot: u32) {
        self.emit(&I::Inst::reload(dst, slot));
    }

    fn evict(&mut self, target: I::Reg) {
        if let Some((_, slot)) = self.state.evict(target.into()) {
            self.emit_spill(slot, target);
        }
    }

    fn evict_one_victim(&mut self) -> I::Reg {
        let victim = self.state.pick_victim().into();
        self.evict(victim);
        victim
    }

    fn pick_reg(&mut self, vreg: VReg) -> I::Reg {
        if let Some(reg) = self.state.vreg_to_loc[vreg.index()].get_reg() {
            return reg.into();
        }

        let prefer = match self.liveness[vreg.index()].reg_hint() {
            liveness::VRegHint::PreferCallee => I::Reg::CALLEE_SAVE_MASK,
            liveness::VRegHint::PreferCaller => I::Reg::CALLER_SAVE_MASK,
        };

        self.state
            .pick_free_prefer(prefer)
            .map(|r| r.into())
            .unwrap_or_else(|| self.evict_one_victim())
    }

    #[inline]
    fn load_vreg_value_to_reg(&mut self, dst: I::Reg, vreg: VReg) {
        let loc = self.state.vreg_to_loc[vreg.index()];
        if let Some(old_reg) = loc.get_reg() {
            if old_reg != dst.into() {
                self.emit_mov(dst, old_reg.into());
            }
        } else if let Some(slot) = loc.get_spill() {
            self.emit_reload(dst, slot);
        } else if let Some(imm) = loc.get_const() {
            self.emit_mov_imm(dst, imm);
        } else {
            panic!("use of unallocated/dead virtual register {:?}", vreg);
        }
    }

    pub fn def_vreg(&mut self, vreg: VReg) -> I::Reg {
        let hreg = self.pick_reg(vreg);
        self.def_vreg_fixed(vreg, hreg)
    }

    pub fn def_vreg_fixed(&mut self, vreg: VReg, hreg: I::Reg) -> I::Reg {
        let hreg_u8: u8 = hreg.into();
        if self.state.vreg_to_loc[vreg.index()].get_reg() == Some(hreg_u8) {
            self.state.pin(hreg_u8);
            return hreg;
        }

        self.evict(hreg);
        self.state.unbind(vreg);
        self.state.bind(vreg, hreg_u8);
        self.state.pin(hreg_u8);
        hreg
    }

    pub fn use_vreg(&mut self, vreg: VReg) -> I::Reg {
        let hreg = self.pick_reg(vreg);
        self.use_vreg_fixed(vreg, hreg)
    }

    pub fn use_vreg_fixed(&mut self, vreg: VReg, hreg: I::Reg) -> I::Reg {
        let hreg_u8: u8 = hreg.into();
        if self.state.vreg_to_loc[vreg.index()].get_reg() == Some(hreg_u8) {
            self.state.pin(hreg_u8);
            return hreg;
        }

        self.evict(hreg);
        self.load_vreg_value_to_reg(hreg, vreg);
        self.state.bind(vreg, hreg_u8);
        self.state.pin(hreg_u8);
        hreg
    }

    /// Guarentee that it's safe to do redundant calls.
    pub fn unuse_vreg(&mut self, vreg: VReg) {
        self.state.unbind(vreg);
    }

    pub fn alloc_scratch(&mut self) -> I::Reg {
        let hreg: I::Reg = self
            .state
            .pick_free()
            .map(|r| r.into())
            .unwrap_or_else(|| self.evict_one_victim());
        self.state.pin(hreg.into());
        hreg
    }

    pub fn alloc_scratch_fixed(&mut self, hreg: I::Reg) -> I::Reg {
        self.evict(hreg);
        self.state.pin(hreg.into());
        hreg
    }

    #[inline]
    pub fn end_instruction(&mut self) {
        self.state.clear_inst_pins();
    }

    pub fn prepare_call_args(&mut self, args: impl IntoIterator<Item = (I::Reg, CallArg<I::Reg>)>) {
        self.spill_caller_saved();

        for (dst, arg) in args {
            let dst_u8: u8 = dst.into();
            debug_assert!(
                (I::Reg::CALLER_SAVE_MASK & (1 << dst_u8)) != 0,
                "call argument register must be caller-saved"
            );
            debug_assert!(
                !self.state.is_pinned(dst_u8),
                "call argument register {:?} is already pinned",
                dst_u8
            );
            match arg {
                CallArg::VReg(vreg) => self.load_vreg_value_to_reg(dst, vreg),
                CallArg::Reg(src) => {
                    if dst_u8 != src.into() {
                        self.emit_mov(dst, src);
                    }
                }
            }
            self.state.pin(dst_u8);
        }
    }

    pub fn spill_given(&mut self, mut mask: u32) {
        mask &= self.state.occupied_mask;
        while mask != 0 {
            // TODO: use `isolate_lowest_one` once stable.
            let curr = mask & mask.wrapping_neg();
            self.evict((curr.trailing_zeros() as u8).into());
            mask ^= curr;
        }
    }

    pub fn spill_caller_saved(&mut self) {
        self.spill_given(I::Reg::CALLER_SAVE_MASK);
    }

    pub fn spill_all(&mut self) {
        self.spill_given(I::Reg::REG_MASK);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallArg<R> {
    VReg(VReg),
    Reg(R),
}

impl<R> From<VReg> for CallArg<R> {
    fn from(v: VReg) -> Self {
        CallArg::VReg(v)
    }
}
