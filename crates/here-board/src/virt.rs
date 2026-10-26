use std::cell::Cell;
use std::cell::UnsafeCell;
use std::hint::cold_path;
use std::rc::Rc;
use std::sync::atomic::Ordering;

pub use here_device::uart16550a::UartIoError;
pub use here_device::uart16550a::UartIoMode;

use crate::Board;
use crate::BoardStatus;
use crate::VirtBoardPlicContextId;
use crate::load::ELFLoader;
use here_device::aclint::Clint;
use here_device::clock::Timer;
use here_device::clock::VirtualClock;
use here_device::device_manager::DeviceArena;
use here_device::device_manager::DeviceHandle;
use here_device::plic::PLIC;
use here_device::power_manager::POWER_OFF_CODE;
use here_device::power_manager::POWER_STATUS;
use here_device::uart16550a::UartBytePort;
use here_isa::riscv::executor::BatchResult;
use here_isa::riscv::executor::ExecutionHook;
use here_isa::riscv::executor::ExecutorBackend;
use here_isa::riscv::executor::RVCPU;

#[cfg(feature = "native")]
use crate::task_spawner::TaskHandle;

#[allow(unused)]
pub mod config {
    use here_types::WordType;

    pub const UART_BASE: WordType = 0x1000_0000;
    pub const UART_SIZE: WordType = 8;
    /// UART PLIC interrupt source ID, must match DTS `interrupts = <0xa>`
    pub const UART_IRQ: u32 = 10;

    pub const VIRTIO_MMIO_BASE: WordType = 0x1000_1000;
    pub const VIRTIO_MMIO_SIZE: WordType = 0x1000;
    /// First VirtIO PLIC source ID. Subsequent MMIO transports use consecutive IDs.
    pub const VIRTIO_IRQ_BASE: u32 = 1;

    pub const POWER_MANAGER_BASE: WordType = 0x10_0000;
    pub const CLINT_BASE: WordType = 0x200_0000;
    pub const PLIC_BASE: WordType = 0xc00_0000;
    pub const SAMPLE_TIMER_BASE: WordType = 0x10_1000;
}

pub use config::*;

pub struct VirtBoard {
    // Stop device tasks before dropping the state they may depend on.
    #[cfg(feature = "native")]
    pub task_handle: TaskHandle,

    pub loader: Option<ELFLoader>,

    pub cpu: Box<RVCPU>,
    pub(crate) executor_backend: Box<dyn ExecutorBackend>,
    pub cycles: Rc<Cell<u64>>,
    pub clock: VirtualClock,
    pub timer: Rc<UnsafeCell<Timer<VirtualClock>>>,

    pub clint: DeviceHandle<Clint>,
    pub plic: DeviceHandle<PLIC>,
    pub uart_io: UartIoMode,
    pub uart_port: Option<UartBytePort>,
    #[cfg(feature = "native")]
    pub uart_stdin_handle: Option<crate::byte_io::StdinHandle>,

    pub status: BoardStatus,

    // Must remain after `cpu`: MemoryMapIO stores a non-owning pointer into this arena.
    pub device_arena: Box<DeviceArena>,
}

impl VirtBoard {
    pub fn device<D: here_device::DeviceTrait>(&self, handle: DeviceHandle<D>) -> &D {
        self.device_arena.device(handle)
    }

    pub fn device_mut<D: here_device::DeviceTrait>(&mut self, handle: DeviceHandle<D>) -> &mut D {
        self.device_arena.device_mut(handle)
    }

    pub fn uart_port(&mut self) -> Result<&mut UartBytePort, UartIoError> {
        self.uart_port
            .as_mut()
            .ok_or(UartIoError::Unavailable(self.uart_io))
    }

    pub fn take_uart_port(&mut self) -> Result<UartBytePort, UartIoError> {
        self.uart_port
            .take()
            .ok_or(UartIoError::Unavailable(self.uart_io))
    }

    #[cfg(feature = "native")]
    pub fn uart_stdin_handle(&self) -> Option<crate::byte_io::StdinHandle> {
        self.uart_stdin_handle
    }

    pub fn cycles(&self) -> u64 {
        self.cycles.get()
    }

    /// Prints executor statistics, if the backend has any.
    pub fn report(&self) {
        self.executor_backend.report();
    }

    fn prepare_cpu_batch(&mut self) {
        unsafe { self.timer.as_mut_unchecked() }.tick();
        let plic = self.plic;
        self.device_mut(plic).update_context_irq_lines(&[
            VirtBoardPlicContextId::Cpu0MachineMode.into(),
            VirtBoardPlicContextId::Cpu0SuperviserMode.into(),
        ]);
    }

    fn finish_cpu_batch(&mut self, cycles: u64) {
        self.cycles.set(self.cycles.get().wrapping_add(cycles));
        // TODO: We can simply read from `PowerManager` if VirtBoard owns `PowerManager`.
        if POWER_STATUS.load(Ordering::Acquire).eq(&POWER_OFF_CODE) {
            cold_path();
            self.cpu.power_off();
            self.status = BoardStatus::Halt;
            log::info!("Total cycles: {}", self.cycles());
        }
    }

    #[inline]
    fn execute_cpu_batch<F>(&mut self, execute: F) -> BatchResult
    where
        F: FnOnce(&mut Self) -> BatchResult,
    {
        if self.status != BoardStatus::Running {
            return BatchResult {
                cycles: 0,
                hook_stopped: false,
            };
        }

        self.prepare_cpu_batch();
        let result = execute(self);
        self.finish_cpu_batch(result.cycles);
        result
    }
}

impl Board for VirtBoard {
    fn status(&self) -> BoardStatus {
        self.status
    }

    fn cpu(&self) -> &RVCPU {
        &self.cpu
    }

    fn cpu_mut(&mut self) -> &mut RVCPU {
        &mut self.cpu
    }

    fn loader(&self) -> Option<&crate::load::ELFLoader> {
        self.loader.as_ref()
    }

    #[inline]
    fn step_batch_with_hook<H: ExecutionHook>(&mut self, cycles: u64, hook: &mut H) -> BatchResult {
        self.execute_cpu_batch(|board| board.cpu.step_batch_with_hook(cycles, hook))
    }

    fn step_batch(&mut self, cycles: u64) -> BatchResult {
        self.execute_cpu_batch(|board| {
            board
                .executor_backend
                .step_batch(board.cpu.as_mut(), cycles);
            BatchResult {
                cycles,
                hook_stopped: false,
            }
        })
    }
}
