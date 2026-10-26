use super::*;

/// Storage location of a virtual register.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Loc<R: TargetReg> {
    None,
    Reg(R),
    Spill(u32),
    /// Permanently bound to a host register; never released or evicted.
    Fixed(R),
}

impl<R: TargetReg> Loc<R> {
    #[inline]
    pub const fn get_reg(&self) -> Option<R> {
        match *self {
            Self::Reg(reg) | Self::Fixed(reg) => Some(reg),
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
    pub const fn is_fixed(&self) -> bool {
        matches!(*self, Self::Fixed(_))
    }
}

/// Owns both directions of the `hreg <-> vreg` mapping.
struct Placements<R: TargetReg> {
    hreg_to_vreg: Vec<Option<VReg>>,
    vreg_to_loc: Vec<Loc<R>>,
}

impl<R: TargetReg> Default for Placements<R> {
    fn default() -> Self {
        Self {
            hreg_to_vreg: Vec::new(),
            vreg_to_loc: Vec::new(),
        }
    }
}

impl<R: TargetReg> Placements<R> {
    fn reset(&mut self, vreg_cnt: usize) {
        self.hreg_to_vreg.clear();
        self.hreg_to_vreg.resize(R::REG_COUNT, None);
        self.vreg_to_loc.clear();
        self.vreg_to_loc.resize(vreg_cnt, Loc::None);
    }

    #[inline]
    fn loc(&self, vreg: VReg) -> Loc<R> {
        self.vreg_to_loc[vreg.index()]
    }

    #[inline]
    fn occupant(&self, hreg: R) -> Option<VReg> {
        self.hreg_to_vreg[hreg.index()]
    }

    /// Bitmask of host registers that currently hold a vreg.
    ///
    /// Derived on demand so it cannot disagree with `hreg_to_vreg`.
    #[inline]
    fn occupied_mask(&self) -> u32 {
        let mut mask = 0;
        for (index, occupant) in self.hreg_to_vreg.iter().enumerate() {
            if occupant.is_some() {
                mask |= 1u32 << index;
            }
        }
        mask
    }

    /// Permanently bind `vreg` to `hreg`. Only valid during initialization.
    fn place_fixed(&mut self, vreg: VReg, hreg: R) {
        debug_assert!(self.occupant(hreg).is_none());
        debug_assert!(matches!(self.loc(vreg), Loc::None));
        self.hreg_to_vreg[hreg.index()] = Some(vreg);
        self.vreg_to_loc[vreg.index()] = Loc::Fixed(hreg);
    }

    /// Remove and return the occupant of `hreg`.
    fn take(&mut self, hreg: R) -> Option<VReg> {
        let vreg = self.occupant(hreg)?;
        self.hreg_to_vreg[hreg.index()] = None;
        self.vreg_to_loc[vreg.index()] = Loc::None;
        Some(vreg)
    }

    /// Set the location of `vreg`, returning its previous location.
    ///
    /// Callers release a spill slot returned in the old location.
    fn place(&mut self, vreg: VReg, loc: Loc<R>) -> Loc<R> {
        let old = self.loc(vreg);
        debug_assert!(!old.is_fixed());
        if let Some(old_hreg) = old.get_reg() {
            self.hreg_to_vreg[old_hreg.index()] = None;
        }
        if let Some(new_hreg) = loc.get_reg() {
            debug_assert!(self.occupant(new_hreg).is_none());
            self.hreg_to_vreg[new_hreg.index()] = Some(vreg);
        }
        self.vreg_to_loc[vreg.index()] = loc;
        old
    }

    /// Release `vreg`'s location, returning it.
    ///
    /// Fixed bindings are left untouched and reported as `Loc::None`.
    fn release(&mut self, vreg: VReg) -> Loc<R> {
        let old = self.loc(vreg);
        if old.is_fixed() {
            return Loc::None;
        }
        if let Some(hreg) = old.get_reg() {
            self.hreg_to_vreg[hreg.index()] = None;
        }
        self.vreg_to_loc[vreg.index()] = Loc::None;
        old
    }
}

/// Host registers reserved or permanently bound by a backend.
#[derive(Clone, Copy)]
pub struct RegAllocConfig<R: TargetReg> {
    /// Registers the allocator must never hand out (e.g. the stack pointer).
    pub reserved: &'static [R],
    /// Virtual registers permanently bound to host registers.
    pub fixed: &'static [(VReg, R)],
}

impl<R: TargetReg> Default for RegAllocConfig<R> {
    fn default() -> Self {
        Self {
            reserved: &[],
            fixed: &[],
        }
    }
}

/// Per-block register allocator state.
pub struct RegAllocState<R: TargetReg> {
    placements: Placements<R>,

    /// Host registers the allocator may hand out. Immutable after [`Self::init`].
    allocatable_mask: u32,

    pinned_mask: u32,

    /// Value of each vreg when known, independent of where it is stored.
    consts: Vec<Option<u64>>,

    free_spill_slot: Vec<u32>,
    slot_count: u32,
}

impl<R: TargetReg> Default for RegAllocState<R> {
    fn default() -> Self {
        Self {
            placements: Placements::default(),
            allocatable_mask: 0,
            pinned_mask: 0,
            consts: Vec::new(),
            free_spill_slot: Vec::new(),
            slot_count: 0,
        }
    }
}

impl<R: TargetReg> RegAllocState<R> {
    /// Reset for a block with `vreg_cnt` virtual registers; `config` is applied once.
    pub fn init(&mut self, vreg_cnt: VRegId, config: &RegAllocConfig<R>) {
        assert!(
            R::REG_COUNT <= 32,
            "register allocator masks are u32, got {} registers",
            R::REG_COUNT
        );
        self.placements.reset(vreg_cnt as usize);

        let mut allocatable = R::REG_MASK;
        for &hreg in config.reserved {
            allocatable &= !hreg.bit();
        }
        for &(vreg, hreg) in config.fixed {
            allocatable &= !hreg.bit();
            self.placements.place_fixed(vreg, hreg);
        }
        self.allocatable_mask = allocatable;
        self.pinned_mask = 0;
        self.consts.clear();
        self.consts.resize(vreg_cnt as usize, None);

        self.free_spill_slot.clear();
        self.slot_count = 0;
    }

    #[inline]
    fn loc(&self, vreg: VReg) -> Loc<R> {
        self.placements.loc(vreg)
    }

    /// Pin a host register for the current instruction.
    ///
    /// Pinning an empty register is valid: it protects a scratch register.
    #[inline]
    fn pin(&mut self, reg: R) {
        self.pinned_mask |= reg.bit();
    }

    #[inline]
    fn clear_pins(&mut self) {
        self.pinned_mask = 0;
    }

    #[inline]
    fn is_pinned(&self, reg: R) -> bool {
        (self.pinned_mask & reg.bit()) != 0
    }

    #[inline]
    fn pinned_mask(&self) -> u32 {
        self.pinned_mask
    }

    #[inline]
    fn spillable_mask(&self) -> u32 {
        self.placements.occupied_mask() & self.allocatable_mask & !self.pinned_mask()
    }

    #[inline]
    fn free_mask(&self) -> u32 {
        self.allocatable_mask & !self.placements.occupied_mask() & !self.pinned_mask()
    }

    #[inline]
    fn const_mask(&self) -> u32 {
        let mut mask = 0u32;
        let mut occupied = self.placements.occupied_mask();
        while occupied != 0 {
            let index = occupied.trailing_zeros() as usize;
            let is_const = self
                .placements
                .occupant(R::from_index(index))
                .is_some_and(|vreg| self.consts[vreg.index()].is_some());
            if is_const {
                mask |= 1u32 << index;
            }
            occupied &= occupied - 1;
        }
        mask
    }

    /// Move `vreg` off its current storage, keeping any known constant.
    fn unbind(&mut self, vreg: VReg) {
        let old = self.placements.release(vreg);
        if let Some(slot) = old.get_spill() {
            self.free_spill_slot.push(slot);
        }
    }

    /// Drop a dead vreg's storage and known constant.
    fn release(&mut self, vreg: VReg) {
        self.unbind(vreg);
        self.consts[vreg.index()] = None;
    }

    fn bind(&mut self, vreg: VReg, reg: R) {
        self.unbind(vreg);
        self.placements.place(vreg, Loc::Reg(reg));
    }

    pub fn def_const(&mut self, vreg: VReg, imm: u64) {
        debug_assert!(!self.placements.loc(vreg).is_fixed());
        self.unbind(vreg);
        self.consts[vreg.index()] = Some(imm);
    }

    #[inline]
    pub fn get_const(&self, vreg: VReg) -> Option<u64> {
        self.consts[vreg.index()]
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
    fn evict(&mut self, reg: R) -> Option<(VReg, u32)> {
        if self.is_pinned(reg) {
            panic!("trying to evict pinned register {reg:?}");
        }
        if (self.allocatable_mask & reg.bit()) == 0 {
            let kind = if self.placements.occupant(reg).is_some() {
                "fixed"
            } else {
                "reserved"
            };
            panic!("trying to evict {kind} register {reg:?}");
        }

        let occupant = self.placements.take(reg)?;
        if self.consts[occupant.index()].is_some() {
            None
        } else {
            let slot = self.alloc_spill_slot();
            self.placements.place(occupant, Loc::Spill(slot));
            Some((occupant, slot))
        }
    }

    #[inline]
    fn pick_in(&self, mask: u32) -> Option<R> {
        if mask == 0 {
            return None;
        }
        let index = mask.trailing_zeros() as usize;
        debug_assert!(index < R::REG_COUNT);
        Some(R::from_index(index))
    }

    #[must_use]
    fn pick_free_in(&self, mask: u32) -> Option<R> {
        self.pick_in(self.free_mask() & mask)
    }

    #[must_use]
    fn pick_free(&self) -> Option<R> {
        self.pick_free_in(u32::MAX)
    }

    #[must_use]
    fn pick_free_prefer(&self, prefer: u32) -> Option<R> {
        self.pick_free_in(prefer).or_else(|| self.pick_free())
    }

    #[must_use]
    fn pick_victim_in(&self, mask: u32) -> Option<R> {
        self.pick_in(self.spillable_mask() & mask)
    }

    fn pick_victim_prefer(&self, prefer: u32) -> R {
        self.pick_victim_in(prefer)
            .or_else(|| self.pick_victim_in(u32::MAX))
            .expect("no spillable register available")
    }

    fn pick_victim(&self) -> R {
        self.pick_victim_prefer(self.const_mask())
    }
}

impl<'b, I: ISA> CodeGenContext<'b, I> {
    #[inline]
    pub fn emit_mov(&mut self, dst: I::Reg, src: I::Reg) {
        if dst != src {
            self.emit(&I::Inst::mov(dst, src));
        }
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
        if let Some((_, slot)) = self.state.evict(target) {
            self.emit_spill(slot, target);
        }
    }

    fn evict_one_victim(&mut self) -> I::Reg {
        let victim = self.state.pick_victim();
        self.evict(victim);
        victim
    }

    fn pick_reg(&mut self, vreg: VReg) -> I::Reg {
        if let Some(reg) = self.state.loc(vreg).get_reg() {
            return reg;
        }

        let prefer = match self.liveness[vreg].reg_hint() {
            liveness::VRegHint::PreferCallee => I::Reg::CALLEE_SAVE_MASK,
            liveness::VRegHint::PreferCaller => I::Reg::CALLER_SAVE_MASK,
        };

        self.state
            .pick_free_prefer(prefer)
            .unwrap_or_else(|| self.evict_one_victim())
    }

    #[inline]
    fn load_vreg_value_to_reg(&mut self, dst: I::Reg, vreg: VReg) {
        let loc = self.state.loc(vreg);
        if let Some(old_reg) = loc.get_reg() {
            self.emit_mov(dst, old_reg);
        } else if let Some(slot) = loc.get_spill() {
            self.emit_reload(dst, slot);
        } else if let Some(imm) = self.state.get_const(vreg) {
            self.emit_mov_imm(dst, imm);
        } else {
            panic!("use of unallocated/dead virtual register {:?}", vreg);
        }
    }
}

#[allow(dead_code)]
impl<'b, I: ISA> CodeGenContext<'b, I> {
    /// Allocate a register for `vreg`, evicting another binding if needed.
    pub fn def_vreg(&mut self, vreg: VReg) -> I::Reg {
        let hreg = self.pick_reg(vreg);
        self.def_vreg_fixed(vreg, hreg)
    }

    /// Allocate `hreg` for `vreg`, evicting its current occupant.
    pub fn def_vreg_fixed(&mut self, vreg: VReg, hreg: I::Reg) -> I::Reg {
        if self.state.loc(vreg).get_reg() == Some(hreg) {
            self.state.pin(hreg);
            return hreg;
        }

        self.evict(hreg);
        self.state.bind(vreg, hreg);
        self.state.pin(hreg);
        hreg
    }

    /// Bring `vreg` into a register, evicting another binding if needed.
    pub fn use_vreg(&mut self, vreg: VReg) -> I::Reg {
        let hreg = self.pick_reg(vreg);
        self.use_vreg_fixed(vreg, hreg)
    }

    /// Load `vreg` into `hreg` and make it the vreg's home.
    pub fn use_vreg_fixed(&mut self, vreg: VReg, hreg: I::Reg) -> I::Reg {
        if self.state.loc(vreg).get_reg() == Some(hreg) {
            self.state.pin(hreg);
            return hreg;
        }

        self.evict(hreg);
        self.load_vreg_value_to_reg(hreg, vreg);
        self.state.bind(vreg, hreg);
        self.state.pin(hreg);
        hreg
    }

    /// Release a dead vreg. Idempotent; fixed vregs stay bound.
    pub fn release_vreg(&mut self, vreg: VReg) {
        self.state.release(vreg);
    }

    /// Reserve a scratch register, evicting another binding if necessary.
    pub fn alloc_scratch(&mut self) -> I::Reg {
        let hreg = self
            .state
            .pick_free()
            .unwrap_or_else(|| self.evict_one_victim());
        self.state.pin(hreg);
        hreg
    }

    /// Reserve `hreg` as a scratch register, evicting its current occupant.
    pub fn alloc_scratch_fixed(&mut self, hreg: I::Reg) -> I::Reg {
        self.evict(hreg);
        self.state.pin(hreg);
        hreg
    }

    /// Clear pins. Every emitted instruction must end with this call.
    #[inline]
    pub fn clear_reg_pins(&mut self) {
        self.state.clear_pins();
    }

    /// Spill caller-saved bindings and load each vreg into its argument register.
    /// Targets must be caller-saved, unique, and unpinned.
    pub fn prepare_call_args(&mut self, args: impl IntoIterator<Item = (I::Reg, VReg)>) {
        self.spill_caller_saved();

        for (dst, vreg) in args {
            debug_assert!(
                (I::Reg::CALLER_SAVE_MASK & dst.bit()) != 0,
                "call argument register must be caller-saved"
            );
            debug_assert!(!self.state.is_pinned(dst));
            self.load_vreg_value_to_reg(dst, vreg);
            self.state.pin(dst);
        }
    }

    pub fn spill_mask(&mut self, mut mask: u32) {
        mask &= self.state.spillable_mask();
        while mask != 0 {
            // TODO: use `isolate_lowest_one` once stable.
            let curr = mask & mask.wrapping_neg();
            self.evict(I::Reg::from_index(curr.trailing_zeros() as usize));
            mask ^= curr;
        }
    }

    /// Spill every unpinned, non-fixed binding held in a caller-saved register.
    pub fn spill_caller_saved(&mut self) {
        self.spill_mask(I::Reg::CALLER_SAVE_MASK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct TestReg(u8);

    impl TargetReg for TestReg {
        const REG_MASK: u32 = 0b11;
        const CALLER_SAVE_MASK: u32 = 0b01;

        fn index(self) -> usize {
            self.0 as usize
        }

        fn from_index(index: usize) -> Self {
            Self(index as u8)
        }
    }

    const R0: TestReg = TestReg(0);
    const R1: TestReg = TestReg(1);

    #[test]
    fn spill_slots_are_reused() {
        let mut state: RegAllocState<TestReg> = RegAllocState::default();
        state.init(6, &RegAllocConfig::default());

        state.bind(VReg::new(2), R0);
        state.bind(VReg::new(3), R1);
        assert_eq!(state.evict(R0), Some((VReg::new(2), 0)));
        assert_eq!(state.evict(R1), Some((VReg::new(3), 1)));
        assert_eq!(state.current_use_slot(), 2);

        // Releasing twice must not duplicate the slot in the free list.
        state.release(VReg::new(2));
        state.release(VReg::new(2));
        assert_eq!(state.free_spill_slot, vec![0]);

        state.bind(VReg::new(4), R0);
        assert_eq!(state.evict(R0), Some((VReg::new(4), 0)));
        assert_eq!(state.current_use_slot(), 2);
    }

    #[test]
    fn const_in_register_evicts_without_spilling() {
        let mut state: RegAllocState<TestReg> = RegAllocState::default();
        state.init(2, &RegAllocConfig::default());

        state.def_const(VReg::new(0), 42);
        state.bind(VReg::new(0), R0);
        assert_eq!(state.get_const(VReg::new(0)), Some(42));
        assert_eq!(state.loc(VReg::new(0)), Loc::Reg(R0));

        assert_eq!(state.evict(R0), None);
        assert_eq!(state.current_use_slot(), 0);
        assert_eq!(state.loc(VReg::new(0)), Loc::None);
        assert_eq!(state.get_const(VReg::new(0)), Some(42));

        state.release(VReg::new(0));
        assert_eq!(state.get_const(VReg::new(0)), None);
    }

    #[test]
    fn fixed_vreg_is_occupied_but_never_allocatable() {
        const RESERVED: &[TestReg] = &[R0];
        const FIXED: &[(VReg, TestReg)] = &[(VReg::new(0), R1)];
        let config = RegAllocConfig {
            reserved: RESERVED,
            fixed: FIXED,
        };

        let mut state: RegAllocState<TestReg> = RegAllocState::default();
        state.init(4, &config);

        assert_eq!(state.loc(VReg::new(0)), Loc::Fixed(R1));
        assert_eq!(state.placements.occupant(R1), Some(VReg::new(0)));

        // Occupancy includes the fixed register; neither free nor spillable does.
        assert_ne!(state.placements.occupied_mask() & R1.bit(), 0);
        assert_eq!(state.free_mask() & R1.bit(), 0);
        assert_eq!(state.spillable_mask() & R1.bit(), 0);
        assert_eq!(state.free_mask() & R0.bit(), 0);

        // Releasing a fixed vreg is a no-op.
        state.release(VReg::new(0));
        assert_eq!(state.loc(VReg::new(0)), Loc::Fixed(R1));
        assert_eq!(state.placements.occupant(R1), Some(VReg::new(0)));
    }

    #[test]
    #[should_panic(expected = "pinned")]
    fn evicting_pinned_register_panics() {
        let mut state: RegAllocState<TestReg> = RegAllocState::default();
        state.init(2, &RegAllocConfig::default());
        state.pin(R0);
        let _ = state.evict(R0);
    }

    #[test]
    #[should_panic(expected = "fixed")]
    fn evicting_fixed_register_panics() {
        const FIXED: &[(VReg, TestReg)] = &[(VReg::new(0), R1)];
        let config = RegAllocConfig {
            reserved: &[],
            fixed: FIXED,
        };

        let mut state: RegAllocState<TestReg> = RegAllocState::default();
        state.init(2, &config);
        let _ = state.evict(R1);
    }
}
