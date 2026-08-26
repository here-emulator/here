use itertools::Either;
use smallvec::SmallVec;
use std::{cell::Cell, ops::Deref};

use super::*;
use crate::jit::old_backend::x86::{X86Assembler, X86Reg};

#[derive(Clone, Copy)]
struct HostRegState {
    host: X86Reg,
    guest: Option<u8>,
    dirty: bool,
    /// cannot reassign this host register to other guest register if refcount != 0.
    refcount: u8,
}

impl HostRegState {
    fn new(host: X86Reg) -> Self {
        HostRegState {
            host,
            guest: None,
            dirty: false,
            refcount: 0,
        }
    }

    fn spill(&self, asm: &mut X86Assembler) {
        if !self.dirty {
            return;
        }

        let guest = self.guest.expect("a dirty host register must have a guest");
        asm.mov_rm64_r64(reg_on_mem(guest), self.host);
    }
}

#[derive(Clone, Copy)]
enum GuestAccess {
    /// Load the guest value, ensure it won't be changed.
    ReadOnly,
    /// Won't load the guest value.
    WriteOnly,
    /// Load the guest value and write the updated value back.
    ReadWrite,
}

impl GuestAccess {
    fn reads(self) -> bool {
        matches!(self, Self::ReadOnly | Self::ReadWrite)
    }

    fn writes(self) -> bool {
        matches!(self, Self::WriteOnly | Self::ReadWrite)
    }
}

#[derive(Clone, Copy)]
enum RegAssignment {
    Scratch,
    Guest { index: u8, access: GuestAccess },
}

pub struct RegGuard<'a> {
    host: X86Reg,
    state: &'a Cell<HostRegState>,
}

impl<'a> Drop for RegGuard<'a> {
    fn drop(&mut self) {
        let mut state = self.state.get();
        state.refcount -= 1;
        self.state.set(state);
    }
}

impl<'a> RegGuard<'a> {
    pub fn reg(&self) -> X86Reg {
        self.host
    }
}

impl<'a> Deref for RegGuard<'a> {
    type Target = X86Reg;

    fn deref(&self) -> &Self::Target {
        &self.host
    }
}

// TODO: use ArrayVec instead of SmallVec
type StateVec = SmallVec<[Cell<HostRegState>; 16]>;

#[derive(Debug, Clone, Copy)]
enum RegKind {
    Callee,
    CallerOrCallee,
}

pub struct RegAssign {
    /// Registers that can alloc freely, must be callee-saved.
    callee_saves: StateVec,
    /// Registers that can alloc freely unless crossing function call.
    caller_saves: StateVec,
    // TODO: think of a better name
    /// Registers that can only be allocated when requested.
    traces: StateVec,
}

impl RegAssign {
    pub fn new() -> Self {
        let from_reg_slice = |s: &[X86Reg]| -> StateVec {
            s.into_iter()
                .map(|host| Cell::new(HostRegState::new(*host)))
                .collect()
        };

        Self {
            callee_saves: from_reg_slice(GUEST_CALLEE_REGS),
            caller_saves: from_reg_slice(GUEST_CALLER_REGS),
            traces: from_reg_slice(GUEST_ASSIGN_REGS),
        }
    }

    fn all_states(&self) -> impl Iterator<Item = &Cell<HostRegState>> {
        self.callee_saves
            .iter()
            .chain(&self.caller_saves)
            .chain(&self.traces)
    }

    fn guest_state(&self, guest: u8) -> Option<&Cell<HostRegState>> {
        let mut found = None;
        for cell in self.all_states() {
            if cell.get().guest != Some(guest) {
                continue;
            }

            assert!(
                found.is_none(),
                "guest register {guest} has multiple host mappings"
            );
            found = Some(cell);
        }
        found
    }

    fn host_state(&self, host: X86Reg) -> Option<&Cell<HostRegState>> {
        self.all_states().find(|state| state.get().host == host)
    }

    fn states_of(&self, kind: RegKind) -> impl Iterator<Item = &Cell<HostRegState>> {
        match kind {
            RegKind::Callee => Either::Left(self.callee_saves.iter()),
            RegKind::CallerOrCallee => {
                Either::Right(self.caller_saves.iter().chain(&self.callee_saves))
            }
        }
    }

    ///  Alloc a register on given condition.
    ///
    /// Panic if changing guest register while refcount != 0.
    fn do_alloc<'a, C>(
        &'a self,
        kind: RegKind,
        asm: &mut X86Assembler,
        cond: C,
        assignment: RegAssignment,
    ) -> Option<RegGuard<'a>>
    where
        C: Fn(&HostRegState) -> bool,
    {
        let (new_guest, reads, writes) = match assignment {
            RegAssignment::Scratch => (None, false, false),
            RegAssignment::Guest { index, access } => {
                (Some(index), access.reads(), access.writes())
            }
        };

        for cell in self.states_of(kind) {
            let mut state = cell.get();
            if cond(&state) {
                if state.guest != new_guest {
                    assert!(state.refcount == 0);
                    state.spill(asm);
                    state.guest = new_guest;
                    state.dirty = writes;
                    if reads {
                        let guest = new_guest.expect("a guest read must have a guest register");
                        asm.mov_r64_rm64(state.host, reg_on_mem(guest));
                    }
                } else if writes {
                    state.dirty = true;
                }

                state.refcount += 1;
                cell.set(state);

                return Some(RegGuard {
                    host: state.host,
                    state: cell,
                });
            }
        }

        None
    }

    fn do_alloc_free<'a>(
        &'a self,
        kind: RegKind,
        asm: &mut X86Assembler,
        assignment: RegAssignment,
    ) -> Option<RegGuard<'a>> {
        self.do_alloc(kind, asm, |s| s.refcount == 0, assignment)
    }

    #[must_use]
    pub fn scratch<'a>(&'a self, asm: &mut X86Assembler) -> RegGuard<'a> {
        self.do_alloc_free(RegKind::CallerOrCallee, asm, RegAssignment::Scratch)
            .unwrap()
    }

    #[must_use]
    fn guest_with<'a>(
        &'a self,
        guest: u8,
        kind: RegKind,
        access: GuestAccess,
        asm: &mut X86Assembler,
    ) -> RegGuard<'a> {
        let assignment = RegAssignment::Guest {
            index: guest,
            access,
        };

        if let Some(cell) = self.guest_state(guest) {
            if matches!(kind, RegKind::Callee) && cell.get().host.caller_save() {
                let mut state = cell.get();
                assert_eq!(
                    state.refcount, 0,
                    "cannot move a live caller-saved guest register to callee-saved storage"
                );
                state.spill(asm);
                state.guest = None;
                state.dirty = false;
                cell.set(state);
            }
        }

        self.do_alloc(kind, asm, |state| state.guest == Some(guest), assignment)
            .or_else(|| self.do_alloc_free(kind, asm, assignment))
            .unwrap()
    }

    #[must_use]
    pub fn guest_read<'a>(&'a self, guest: u8, asm: &mut X86Assembler) -> RegGuard<'a> {
        self.guest_with(guest, RegKind::CallerOrCallee, GuestAccess::ReadOnly, asm)
    }

    #[must_use]
    pub fn guest_read_callee<'a>(&'a self, guest: u8, asm: &mut X86Assembler) -> RegGuard<'a> {
        self.guest_with(guest, RegKind::Callee, GuestAccess::ReadOnly, asm)
    }

    #[must_use]
    pub fn guest_write<'a>(&'a self, guest: u8, asm: &mut X86Assembler) -> RegGuard<'a> {
        self.guest_with(guest, RegKind::CallerOrCallee, GuestAccess::WriteOnly, asm)
    }

    #[must_use]
    pub fn guest_write_callee<'a>(&'a self, guest: u8, asm: &mut X86Assembler) -> RegGuard<'a> {
        self.guest_with(guest, RegKind::Callee, GuestAccess::WriteOnly, asm)
    }

    #[must_use]
    pub fn guest_read_write<'a>(&'a self, guest: u8, asm: &mut X86Assembler) -> RegGuard<'a> {
        self.guest_with(guest, RegKind::CallerOrCallee, GuestAccess::ReadWrite, asm)
    }

    #[must_use]
    pub fn guest_read_write_callee<'a>(
        &'a self,
        guest: u8,
        asm: &mut X86Assembler,
    ) -> RegGuard<'a> {
        self.guest_with(guest, RegKind::Callee, GuestAccess::ReadWrite, asm)
    }

    #[must_use]
    pub fn host<'a>(&'a self, host: X86Reg, asm: &mut X86Assembler) -> RegGuard<'a> {
        let cell = self
            .host_state(host)
            .unwrap_or_else(|| panic!("host register {host:?} is not tracked by RegAssign"));
        let mut state = cell.get();

        assert_eq!(
            state.refcount, 0,
            "host register {host:?} is already occupied"
        );

        state.spill(asm);
        state.refcount += 1;
        state.guest = None;
        state.dirty = false;

        cell.set(state);
        RegGuard { host, state: cell }
    }

    /// Write back all dirty guest mappings when leaving a basic block or before
    /// calling a helper that may leave the generated block through an exception.
    pub fn flush(&self, asm: &mut X86Assembler) {
        for cell in self.all_states() {
            let mut state = cell.get();
            state.spill(asm);
            if state.refcount == 0 {
                state.dirty = false;
            }
            cell.set(state);
        }
    }

    pub fn flush_for_call(&self, asm: &mut X86Assembler) {
        for cell in self.caller_saves.iter() {
            let mut state = cell.get();
            assert_eq!(state.refcount, 0);
            state.spill(asm);
            state.guest = None;
            state.dirty = false;
            cell.set(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_guest_access(access: GuestAccess) -> (Vec<u8>, X86Reg) {
        let regs = RegAssign::new();
        let mut asm = X86Assembler::new();
        let host;
        {
            let guest = match access {
                GuestAccess::ReadOnly => regs.guest_read(1, &mut asm),
                GuestAccess::WriteOnly => regs.guest_write(1, &mut asm),
                GuestAccess::ReadWrite => regs.guest_read_write(1, &mut asm),
            };
            host = guest.reg();
        }
        regs.flush(&mut asm);
        (asm.build(), host)
    }

    #[test]
    fn guest_access_controls_load_and_writeback() {
        let mut readonly = X86Assembler::new();
        let (code, host) = build_guest_access(GuestAccess::ReadOnly);
        readonly.mov_r64_rm64(host, reg_on_mem(1));
        assert_eq!(code, readonly.build());

        let mut writeonly = X86Assembler::new();
        let (code, host) = build_guest_access(GuestAccess::WriteOnly);
        writeonly.mov_rm64_r64(reg_on_mem(1), host);
        assert_eq!(code, writeonly.build());

        let mut readwrite = X86Assembler::new();
        let (code, host) = build_guest_access(GuestAccess::ReadWrite);
        readwrite.mov_r64_rm64(host, reg_on_mem(1));
        readwrite.mov_rm64_r64(reg_on_mem(1), host);
        assert_eq!(code, readwrite.build());
    }

    #[test]
    fn flush_writes_back_and_retains_cached_guest() {
        let regs = RegAssign::new();
        let mut asm = X86Assembler::new();
        let host = {
            let guest = regs.guest_write(1, &mut asm);
            guest.reg()
        };

        regs.flush(&mut asm);
        let guest = regs.guest_read(1, &mut asm);
        drop(guest);
        regs.flush(&mut asm);

        let mut expected = X86Assembler::new();
        expected.mov_rm64_r64(reg_on_mem(1), host);
        assert_eq!(asm.build(), expected.build());
    }

    #[test]
    fn callee_access_moves_a_caller_guest() {
        let regs = RegAssign::new();
        let mut asm = X86Assembler::new();

        let caller_host = {
            let guest = regs.guest_write(1, &mut asm);
            guest.reg()
        };
        let callee_host = {
            let guest = regs.guest_write_callee(1, &mut asm);
            guest.reg()
        };

        let state = regs.guest_state(1).unwrap().get();
        assert_eq!(state.host, callee_host);
        assert_ne!(caller_host, callee_host);

        let mut expected = X86Assembler::new();
        expected.mov_rm64_r64(reg_on_mem(1), caller_host);
        assert_eq!(asm.build(), expected.build());

        regs.flush_for_call(&mut X86Assembler::new());
        assert_eq!(regs.guest_state(1).unwrap().get().host, callee_host);
    }

    #[test]
    fn flush_for_call_discards_only_caller_saved_mappings() {
        let regs = RegAssign::new();
        let mut asm = X86Assembler::new();

        let caller = regs.guest_write(1, &mut asm);
        drop(caller);
        let callee = regs.guest_write_callee(2, &mut asm);
        drop(callee);

        regs.flush_for_call(&mut asm);

        assert!(regs.guest_state(1).is_none());
        assert_eq!(regs.guest_state(2).unwrap().get().guest, Some(2));
    }
}
