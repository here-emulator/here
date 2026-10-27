#![feature(unsafe_cell_access)]

use std::path::PathBuf;
use std::str::FromStr;

use here_device::plic::types::PlicContextId;
use here_device::virtio::virtio_device::VirtIODeviceEnum;
use here_isa::riscv::executor::BatchResult;
use here_isa::riscv::executor::ExecutionHook;
use here_isa::riscv::executor::RVCPU;

pub mod builder;
#[cfg(feature = "native")]
pub mod byte_io;
pub mod load;
#[cfg(feature = "native")]
pub mod task_spawner;
pub mod virt;

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub dev_type: VirtIODeviceEnum,
    pub path: PathBuf,
}

impl FromStr for DeviceConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split(':');
        let dev_type = match parts.next() {
            Some("virtio-block") => VirtIODeviceEnum::VirtIOBlock,
            Some("virtio-network") => VirtIODeviceEnum::VirtIONet,
            Some(other) => return Err(format!("Unknown device type: {}", other)),
            None => return Err("Invalid device arguments.".into()),
        };
        let path = PathBuf::from(parts.next().ok_or("Need input a device path.")?);
        Ok(DeviceConfig { dev_type, path })
    }
}

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
    const STEP_BATCH_CYCLES: u64 = 4096;

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

impl<B: Board> Board for &mut B {
    const STEP_BATCH_CYCLES: u64 = B::STEP_BATCH_CYCLES;

    fn status(&self) -> BoardStatus {
        (**self).status()
    }

    fn cpu(&self) -> &RVCPU {
        (**self).cpu()
    }

    fn cpu_mut(&mut self) -> &mut RVCPU {
        (**self).cpu_mut()
    }

    fn loader(&self) -> Option<&crate::load::ELFLoader> {
        (**self).loader()
    }

    fn step_batch_with_hook<H: ExecutionHook>(&mut self, cycles: u64, hook: &mut H) -> BatchResult {
        (**self).step_batch_with_hook(cycles, hook)
    }

    fn step_batch(&mut self, cycles: u64) -> BatchResult {
        (**self).step_batch(cycles)
    }
}
