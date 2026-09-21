//! At most 32 target register is supported right now.

use super::*;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum RegLocation {
    #[default]
    None,
    Reg(u8),
    Spill(u32),
    // TODO: add Guest(u8),
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

    fn unbind(&mut self, vreg: VReg) {
        match self.vreg_to_loc[vreg.index()] {
            RegLocation::Reg(old_reg) => {
                self.hreg_to_vreg[old_reg as usize] = None;
                self.occupied_mask &= !(1 << old_reg);
            }
            RegLocation::Spill(slot) => {
                self.free_spill_slot.push(slot);
            }
            RegLocation::None => {}
        }

        self.vreg_to_loc[vreg.index()] = RegLocation::None;
    }

    fn bind(&mut self, vreg: VReg, reg: u8) {
        self.unbind(vreg);

        debug_assert!(
            self.hreg_to_vreg[reg as usize].is_none(),
            "Reg {} is currently occupied by {:?}.",
            reg,
            self.hreg_to_vreg[reg as usize]
        );

        self.hreg_to_vreg[reg as usize] = Some(vreg);
        self.occupied_mask |= 1 << reg;
        self.vreg_to_loc[vreg.index()] = RegLocation::Reg(reg);
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

        let slot = self.alloc_spill_slot();
        self.hreg_to_vreg[reg as usize] = None;
        self.occupied_mask &= !(1 << reg);
        self.vreg_to_loc[occupant.index()] = RegLocation::Spill(slot);

        Some((occupant, slot))
    }

    #[must_use]
    fn pick_free(&mut self) -> Option<u8> {
        self.pick_free_in(u32::MAX)
    }

    #[must_use]
    fn pick_free_in(&mut self, mask: u32) -> Option<u8> {
        let free = self.free_mask() & mask;
        if free != 0 {
            Some(free.trailing_zeros() as u8)
        } else {
            None
        }
    }

    #[must_use]
    fn pick_free_prefer(&mut self, prefer: u32) -> Option<u8> {
        self.pick_free_in(prefer).or_else(|| self.pick_free())
    }

    fn pick_victim(&mut self) -> u8 {
        let mask = self.spillable_mask();
        assert_ne!(mask, 0, "no spillable register available");
        mask.trailing_zeros() as u8
    }
}

impl<'b, I: ISA> CodeGenContext<'b, I> {
    #[inline]
    pub fn emit_mov(&mut self, dst: I::Reg, src: I::Reg) {
        self.emit(&I::Inst::mov(dst, src));
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
        if let RegLocation::Reg(reg) = self.state.vreg_to_loc[vreg.index()] {
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

    pub fn def_vreg(&mut self, vreg: VReg) -> I::Reg {
        if let RegLocation::Reg(reg) = self.state.vreg_to_loc[vreg.index()] {
            self.state.pin(reg);
            return reg.into();
        }

        let hreg = self.pick_reg(vreg);
        self.def_vreg_fixed(vreg, hreg)
    }

    pub fn def_vreg_fixed(&mut self, vreg: VReg, hreg: I::Reg) -> I::Reg {
        let hreg_u8: u8 = hreg.into();
        if self.state.vreg_to_loc[vreg.index()] == RegLocation::Reg(hreg_u8) {
            self.state.pin(hreg_u8);
            return hreg;
        }

        self.evict(hreg);
        self.state.bind(vreg, hreg_u8);
        self.state.pin(hreg_u8);
        hreg
    }

    pub fn use_vreg(&mut self, vreg: VReg) -> I::Reg {
        if let RegLocation::Reg(reg) = self.state.vreg_to_loc[vreg.index()] {
            self.state.pin(reg);
            return reg.into();
        }

        let hreg = self.pick_reg(vreg);
        self.use_vreg_fixed(vreg, hreg)
    }

    #[inline]
    fn load_vreg_value_to_reg(&mut self, dst: I::Reg, vreg: VReg) {
        match self.state.vreg_to_loc[vreg.index()] {
            RegLocation::Reg(old_reg) => {
                if old_reg != dst.into() {
                    self.emit_mov(dst, old_reg.into());
                }
            }
            RegLocation::Spill(slot) => self.emit_reload(dst, slot),
            RegLocation::None => panic!("use of unallocated/dead virtual register {:?}", vreg),
        }
    }

    pub fn use_vreg_fixed(&mut self, vreg: VReg, hreg: I::Reg) -> I::Reg {
        let hreg_u8: u8 = hreg.into();
        let current_loc = self.state.vreg_to_loc[vreg.index()];

        if current_loc == RegLocation::Reg(hreg_u8) {
            self.state.pin(hreg_u8);
            return hreg;
        }

        self.evict(hreg);
        self.load_vreg_value_to_reg(hreg, vreg);
        self.state.bind(vreg, hreg_u8);
        self.state.pin(hreg_u8);
        hreg
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

    /// Guarentee that it's safe to do redundant calls.
    pub fn unuse_vreg(&mut self, vreg: VReg) {
        self.state.unbind(vreg);
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

    fn spill_caller_saved(&mut self) {
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
