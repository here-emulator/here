use std::cell::Cell;

use super::*;

pub enum RegHint {
    PreferCallee,
    PreferCaller,
}

#[derive(Default, Clone, Copy)]
enum RegLocation {
    #[default]
    None,
    Reg(u8),
    Spill(u32),
}

pub struct RegAllocState {
    hreg_to_vreg: [Option<VReg>; 32],
    vreg_to_loc: Vec<RegLocation>,

    allocatable_mask: u32,
    /// Whether the target register holds a VReg, it doesn't indicate whether it's pinned.
    occupied_mask: u32,
    pinned_cell: Cell<u32>,

    free_spill_slot: Vec<u32>,
    slot_count: u32,
}

impl RegAllocState {
    pub fn new(vreg_cnt: InstId, allocatable_mask: u32) -> Self {
        Self {
            hreg_to_vreg: [None; 32],
            vreg_to_loc: vec![RegLocation::default(); vreg_cnt as usize],
            allocatable_mask,
            occupied_mask: 0,
            pinned_cell: Cell::new(0),
            free_spill_slot: vec![],
            slot_count: 0,
        }
    }

    #[inline]
    fn pinned_mask(&self) -> u32 {
        self.pinned_cell.get()
    }

    #[inline]
    fn spillable_mask(&self) -> u32 {
        self.occupied_mask & !self.pinned_mask()
    }

    #[inline]
    fn free_mask(&mut self) -> u32 {
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

    fn alloc_spill_slot(&mut self) -> u32 {
        self.free_spill_slot.pop().unwrap_or_else(|| {
            let slot = self.slot_count;
            self.slot_count += 1;
            slot
        })
    }

    fn pin(&self, reg: u8) {
        self.pinned_cell.set(self.pinned_mask() | (1 << reg));
    }

    fn unpin(&self, reg: u8) {
        self.pinned_cell.set(self.pinned_mask() & !(1 << reg));
    }

    #[must_use]
    fn evict(&mut self, reg: u8) -> Option<(VReg, u32)> {
        if self.pinned_mask() & (1 << reg) != 0 {
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
        None.or_else(|| self.pick_free_in(prefer))
            .or_else(|| self.pick_free())
    }

    fn pick_victim(&mut self) -> u8 {
        let mask = self.spillable_mask();
        assert_ne!(mask, 0);
        mask.trailing_zeros() as u8
    }
}

pub trait TargetReg: Into<u8> + From<u8> + Copy {
    const CALLER_SAVE_MASK: u32;
    const CALLEE_SAVE_MASK: u32;

    fn arg_reg(idx: u8) -> Option<u8>;
}

pub trait RegAllocBase {
    type Reg: TargetReg;

    fn state(&mut self) -> &mut RegAllocState;
    fn emit_mov(&mut self, dst: Self::Reg, src: Self::Reg);
    fn emit_spill(&mut self, slot: u32, src: Self::Reg);
    fn emit_reload(&mut self, dst: Self::Reg, slot: u32);
}

trait RegAllocHelper: RegAllocBase {
    fn evict(&mut self, target: Self::Reg) {
        if let Some((_, slot)) = self.state().evict(target.into()) {
            self.emit_spill(slot, target);
        }
    }

    fn evict_one_occupied(&mut self) -> Self::Reg {
        let victim = self.state().pick_victim().into();
        self.evict(victim);
        victim
    }

    fn prepare_host_reg(&mut self, vreg: VReg, hint: RegHint) -> Self::Reg {
        if let RegLocation::Reg(reg) = self.state().vreg_to_loc[vreg.index()] {
            return reg.into();
        }

        let prefer = match hint {
            RegHint::PreferCallee => Self::Reg::CALLEE_SAVE_MASK,
            RegHint::PreferCaller => Self::Reg::CALLER_SAVE_MASK,
        };

        self.state()
            .pick_free_prefer(prefer)
            .map(|r| r.into())
            .unwrap_or_else(|| self.evict_one_occupied())
    }
}

pub trait RegAllocOps: RegAllocHelper {
    fn def_vreg<'a>(&'a mut self, vreg: VReg, hint: RegHint) -> RegGuard<'a, Self::Reg> {
        let reg = self.prepare_host_reg(vreg, hint);
        RegGuard::new(reg, self.state())
    }

    fn unuse_vreg(&mut self, vreg: VReg) {
        self.state().unbind(vreg);
    }

    #[must_use]
    fn use_vreg<'a>(&'a mut self, vreg: VReg, hint: RegHint) -> RegGuard<'a, Self::Reg> {
        let hreg = self.prepare_host_reg(vreg, hint);
        self.use_vreg_fixed(vreg, hreg)
    }

    #[must_use]
    fn use_vreg_fixed<'a>(&'a mut self, vreg: VReg, hreg: Self::Reg) -> RegGuard<'a, Self::Reg> {
        self.evict(hreg);

        match self.state().vreg_to_loc[vreg.index()] {
            RegLocation::Reg(old_reg) => {
                if old_reg != hreg.into() {
                    self.emit_mov(hreg, old_reg.into());
                }
            }
            RegLocation::Spill(slot) => {
                self.emit_reload(hreg, slot);
            }
            RegLocation::None => {}
        }

        self.state().bind(vreg, hreg.into());

        RegGuard::new(hreg, self.state())
    }

    fn spill_given(&mut self, mut mask: u32) {
        while mask != 0 {
            // TODO: use isolate_lowest_one once it's stabled.
            let curr = mask & mask.wrapping_neg();
            self.evict((curr.count_zeros() as u8).into());
            mask ^= curr;
        }
    }

    fn spill_caller_saved(&mut self) {
        self.spill_given(Self::Reg::CALLER_SAVE_MASK);
    }

    fn spill_all(&mut self) {
        self.spill_given(u32::MAX);
    }
}

impl<R: RegAllocBase> RegAllocHelper for R {}
impl<R: RegAllocHelper> RegAllocOps for R {}

pub struct RegGuard<'a, R: TargetReg> {
    reg: R,
    state: &'a RegAllocState,
}

impl<'a, R: TargetReg> Drop for RegGuard<'a, R> {
    fn drop(&mut self) {
        self.state.unpin(self.reg().into());
    }
}

impl<'a, R: TargetReg> RegGuard<'a, R> {
    fn new(reg: R, state: &'a RegAllocState) -> Self {
        state.pin(reg.into());
        Self { reg, state }
    }

    pub fn reg(&self) -> R {
        self.reg
    }
}
