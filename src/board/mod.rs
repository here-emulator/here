use crate::{
    device::plic::types::PlicContextId,
    isa::riscv::executor::{BatchResult, ExecutionHook, RVCPU},
};

pub mod builder;
pub mod virt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoardStatus {
    Running,
    Halt,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub(crate) enum VirtBoardPlicContextId {
    Cpu0MachineMode,
    Cpu0SuperviserMode,
}

impl Into<PlicContextId> for VirtBoardPlicContextId {
    fn into(self) -> PlicContextId {
        self as PlicContextId
    }
}

pub trait Board {
    const STEP_BATCH_CYCLES: u64 = 1024;

    fn status(&self) -> BoardStatus;

    fn cpu(&self) -> &RVCPU;
    fn cpu_mut(&mut self) -> &mut RVCPU;

    fn loader(&self) -> Option<&crate::load::ELFLoader>;

    fn step_batch_with_hook<H: ExecutionHook>(&mut self, cycles: u64, hook: &mut H) -> BatchResult;
    fn step_batch(&mut self, cycles: u64) -> BatchResult;

    fn run_cycles_with<F>(&mut self, cycles: u64, mut step_fn: F) -> BatchResult
    where
        F: FnMut(&mut Self, u64) -> BatchResult,
    {
        let mut executed = 0;
        let mut hook_stopped = false;

        while executed < cycles && self.status() == BoardStatus::Running {
            let batch_cycles = (cycles - executed).min(Self::STEP_BATCH_CYCLES);
            let result = step_fn(self, batch_cycles);
            executed += result.cycles;

            if result.hook_stopped {
                hook_stopped = true;
                break;
            }
        }

        BatchResult {
            cycles: executed,
            hook_stopped,
        }
    }

    #[inline]
    fn run_cycles_hooked<H: ExecutionHook>(&mut self, cycles: u64, hook: &mut H) -> BatchResult {
        self.run_cycles_with(cycles, |board, c| board.step_batch_with_hook(c, hook))
    }

    #[inline]
    /// Execute exactly `cycles` CPU cycles unless the board halts first.
    fn run_cycles(&mut self, cycles: u64) -> u64 {
        self.run_cycles_with(cycles, |board, c| board.step_batch(c))
            .cycles
    }

    #[inline]
    /// Execute one cycle. This is slower than batching; prefer [`Self::run_cycles`]
    /// or [`Self::run_cycles_hooked`] when possible.
    fn step(&mut self) {
        self.run_cycles(1);
    }

    fn run(&mut self) {
        self.run_cycles(u64::MAX);
    }
}
