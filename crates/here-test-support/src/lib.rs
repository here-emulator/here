//! Board fixtures shared by the workspace's tests.
//!
//! Deliberately not published: production crates must not depend on this, so
//! test-only wiring (JIT backends, program loading, standard device sets)
//! stays out of every shipped dependency graph.

use here_board::builder::RVBoardBuilder;
use here_board::load::ELFLoader;
use here_board::load::load_bin;
use here_board::virt::UartIoMode;
use here_board::virt::VirtBoard;
use here_board::virt::config::*;
use here_device::aclint::CLINT_SIZE;
use here_device::plic::PLIC_SIZE;
use here_device::power_manager::POWER_MANAGER_SIZE;
use here_device::ram::Ram;
use here_isa::riscv::executor::ExecutorBackend;

/// Adds the standard virt machine's base devices to `ram`.
///
/// Callers append optional devices (VirtIO, sample timer, ...) themselves.
pub fn standard_builder(ram: Ram, uart_io: UartIoMode) -> RVBoardBuilder {
    RVBoardBuilder::new(ram)
        .add_power_manager(POWER_MANAGER_BASE, POWER_MANAGER_SIZE)
        .add_clint(CLINT_BASE, CLINT_SIZE)
        .add_plic(PLIC_BASE, PLIC_SIZE)
        .add_uart(UART_BASE, UART_SIZE, UART_IRQ, uart_io)
}

/// Board with the standard devices and empty RAM.
pub fn empty_board(uart_io: UartIoMode) -> VirtBoard {
    board_from_ram(Ram::new(), uart_io)
}

pub fn board_from_ram(ram: Ram, uart_io: UartIoMode) -> VirtBoard {
    standard_builder(ram, uart_io).build()
}

pub fn board_from_binary(bytes: &[u8], uart_io: UartIoMode) -> VirtBoard {
    let mut ram = Ram::new();
    load_bin(&mut ram, bytes);
    board_from_ram(ram, uart_io)
}

/// Loads an ELF image and keeps its loader attached for symbol lookups.
pub fn board_from_elf(bytes: &[u8], uart_io: UartIoMode) -> Option<VirtBoard> {
    let loader = ELFLoader::try_new(bytes.to_vec())?;
    let mut ram = Ram::new();
    loader.load_to_ram(&mut ram);
    let mut board = board_from_ram(ram, uart_io);
    board.loader = Some(loader);
    Some(board)
}

/// Binary board running on the JIT backend; `None` where JIT is unsupported.
pub fn jit_board_from_binary(bytes: &[u8]) -> Option<VirtBoard> {
    let backend: Box<dyn ExecutorBackend> = Box::new(here_jit::JitRuntime::try_new().ok()?);
    let mut ram = Ram::new();
    load_bin(&mut ram, bytes);
    Some(
        standard_builder(ram, UartIoMode::External)
            .with_executor_backend(backend)
            .build(),
    )
}
