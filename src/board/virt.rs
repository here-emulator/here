use std::{
    cell::{Cell, UnsafeCell},
    hint::cold_path,
    rc::Rc,
    sync::atomic::Ordering,
};

pub use crate::device::uart16550a::{UartIoError, UartIoMode};

use crate::{
    DeviceConfig,
    board::{Board, BoardStatus, VirtBoardPlicContextId, builder::RVBoardBuilder},
    clock::{Timer, VirtualClock},
    config::arch_config::WordType,
    device::{
        self,
        aclint::{CLINT_SIZE, Clint},
        device_manager::{DeviceArena, DeviceHandle},
        plic::{PLIC, PLIC_SIZE},
        power_manager::{POWER_MANAGER_SIZE, POWER_OFF_CODE, POWER_STATUS},
        uart16550a::UartBytePort,
    },
    isa::riscv::{
        decoder::Decoder,
        executor::{BatchResult, ExecutionHook, RVCPU},
        trap::Interrupt,
    },
    load::{ELFLoader, load_bin},
    ram::Ram,
    ram_config,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::task_spawner::TaskHandle;

#[cfg(all(feature = "test-device", not(target_arch = "wasm32")))]
use crate::device::sample_timer::{SAMPLE_TIMER_INTERRUPT_ID, SAMPLE_TIMER_SIZE};

#[allow(unused)]
pub mod config {
    use crate::config::arch_config::WordType;

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

pub trait RiscvIRQHandler {
    fn handle_irq(&mut self, interrupt: Interrupt, level: bool);
}

pub trait RiscvIRQSource {
    fn set_irq_line(&mut self, line: IRQLine, id: usize);
}

#[derive(Debug)]
pub struct MemoryImage {
    pub address: WordType,
    pub data: Vec<u8>,
}

impl MemoryImage {
    pub fn new(address: WordType, data: Vec<u8>) -> Self {
        Self { address, data }
    }
}

pub struct VirtBoardConfig {
    decoder: Option<Decoder>,
    virtio_devices: Vec<DeviceConfig>,
    memory_images: Vec<MemoryImage>,
    initial_registers: Vec<(u8, WordType)>,
    uart_io: UartIoMode,
}

impl Default for VirtBoardConfig {
    fn default() -> Self {
        Self {
            decoder: None,
            virtio_devices: Vec::new(),
            memory_images: Vec::new(),
            initial_registers: Vec::new(),
            uart_io: UartIoMode::External,
        }
    }
}

impl VirtBoardConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_decoder(mut self, decoder: Decoder) -> Self {
        self.decoder = Some(decoder);
        self
    }

    pub fn with_virtio_devices(mut self, devices: Vec<DeviceConfig>) -> Self {
        self.virtio_devices = devices;
        self
    }

    /// Load an additional image into guest physical RAM **after** the primary ELF/binary is loaded.
    pub fn with_memory_image(mut self, image: MemoryImage) -> Self {
        self.memory_images.push(image);
        self
    }

    pub fn with_reg(mut self, register: u8, value: WordType) -> Self {
        self.initial_registers.push((register, value));
        self
    }

    pub fn with_uart_io(mut self, mode: UartIoMode) -> Self {
        self.uart_io = mode;
        self
    }
}

/// NOTE: Only used in single-threaded contexts.
pub struct IRQLine {
    target: *mut dyn RiscvIRQHandler,
    interrupt_nr: Interrupt,
}

impl IRQLine {
    pub fn new(target: *mut dyn RiscvIRQHandler, interrupt_nr: Interrupt) -> Self {
        Self {
            target,
            interrupt_nr,
        }
    }

    pub fn set_irq(&mut self, level: bool) {
        unsafe { &mut *self.target }.handle_irq(self.interrupt_nr, level);
    }
}

pub struct VirtBoard {
    // Stop device tasks before dropping the state they may depend on.
    #[cfg(not(target_arch = "wasm32"))]
    pub task_handle: TaskHandle,

    pub loader: Option<ELFLoader>,

    pub cpu: Box<RVCPU>,
    pub cycles: Rc<Cell<u64>>,
    pub clock: VirtualClock,
    pub timer: Rc<UnsafeCell<Timer<VirtualClock>>>,

    pub clint: DeviceHandle<Clint>,
    pub plic: DeviceHandle<PLIC>,
    pub uart_io: UartIoMode,
    pub uart_port: Option<UartBytePort>,
    #[cfg(feature = "native-cli")]
    pub uart_stdin_handle: Option<crate::byte_io::StdinHandle>,

    pub status: BoardStatus,

    // Must remain after `cpu`: MemoryMapIO stores a non-owning pointer into this arena.
    pub device_arena: Box<DeviceArena>,
}

const STEP_BATCH_CYCLES: u64 = 1024;

impl VirtBoard {
    pub fn device<D: device::DeviceTrait>(&self, handle: DeviceHandle<D>) -> &D {
        self.device_arena.device(handle)
    }

    pub fn device_mut<D: device::DeviceTrait>(&mut self, handle: DeviceHandle<D>) -> &mut D {
        self.device_arena.device_mut(handle)
    }

    pub fn from_binary_with(bytes: &[u8], config: VirtBoardConfig) -> Result<Self, String> {
        let mut ram = Ram::new();
        load_bin(&mut ram, bytes);
        Self::from_ram_with(ram, config)
    }

    pub fn from_elf(bytes: Vec<u8>) -> Result<Self, String> {
        Self::from_elf_with(bytes, VirtBoardConfig::new())
    }

    pub fn from_elf_with(bytes: Vec<u8>, config: VirtBoardConfig) -> Result<Self, String> {
        let mut ram = Ram::new();
        let loader = ELFLoader::try_new(bytes).ok_or_else(|| "Invalid ELF file".to_string())?;
        loader.load_to_ram(&mut ram);
        let mut board = Self::from_ram_with(ram, config)?;
        board.loader = Some(loader);
        Ok(board)
    }

    pub fn from_ram_with(mut ram: Ram, config: VirtBoardConfig) -> Result<Self, String> {
        let VirtBoardConfig {
            decoder,
            virtio_devices,
            memory_images,
            initial_registers,
            uart_io,
        } = config;

        for image in memory_images {
            let offset = image
                .address
                .checked_sub(ram_config::BASE_ADDR)
                .ok_or_else(|| {
                    format!(
                        "memory image address 0x{:x} is below RAM base 0x{:x}",
                        image.address,
                        ram_config::BASE_ADDR
                    )
                })?;
            ram.try_insert_section(&image.data, offset)
                .map_err(|error| {
                    format!(
                        "failed to load memory image at 0x{:x}: {error}",
                        image.address
                    )
                })?;
        }

        let mut builder = RVBoardBuilder::new(ram);

        if let Some(decoder) = decoder {
            builder = builder.with_decoder(decoder);
        }

        builder = builder
            .with_initial_registers(initial_registers)
            .add_power_manager(config::POWER_MANAGER_BASE, POWER_MANAGER_SIZE)
            .add_clint(config::CLINT_BASE, CLINT_SIZE)
            .add_plic(config::PLIC_BASE, PLIC_SIZE)
            .add_uart(
                config::UART_BASE,
                config::UART_SIZE,
                config::UART_IRQ,
                uart_io,
            );

        #[cfg(not(target_arch = "wasm32"))]
        {
            for virtio_device in virtio_devices {
                builder = builder.add_virtio_device(
                    config::VIRTIO_MMIO_BASE,
                    config::VIRTIO_MMIO_SIZE,
                    config::VIRTIO_IRQ_BASE,
                    virtio_device,
                );
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = virtio_devices;
        }

        #[cfg(all(feature = "test-device", not(target_arch = "wasm32")))]
        {
            builder = builder.add_sample_timer(
                config::SAMPLE_TIMER_BASE,
                SAMPLE_TIMER_SIZE,
                SAMPLE_TIMER_INTERRUPT_ID,
            );
        }

        builder.build()
    }

    #[cfg(test)]
    pub fn push_uart_input(&self, bytes: &[u8]) -> Result<(), UartIoError> {
        self.uart_port
            .as_ref()
            .ok_or(UartIoError::Unavailable(self.uart_io))?
            .push_input(bytes)
    }

    #[cfg(test)]
    pub fn take_uart_output(&mut self) -> Result<Vec<u8>, UartIoError> {
        Ok(self
            .uart_port
            .as_mut()
            .ok_or(UartIoError::Unavailable(self.uart_io))?
            .take_output())
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

    #[cfg(feature = "native-cli")]
    pub fn uart_stdin_handle(&self) -> Option<crate::byte_io::StdinHandle> {
        self.uart_stdin_handle
    }

    pub fn cycles(&self) -> u64 {
        self.cycles.get()
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
    const STEP_BATCH_CYCLES: u64 = 1024;

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
            board.cpu.step_batch(cycles);
            BatchResult {
                cycles,
                hook_stopped: false,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Clock;
    use crate::config::arch_config::XLEN;
    use crate::device::DeviceTrait;
    use crate::isa::DebugTarget;
    use crate::isa::riscv::csr_reg::csr_macro::Mcause;
    use crate::isa::riscv::csr_reg::{NamedCsrReg, csr_index};
    use crate::ram_config;

    fn create_test_board() -> VirtBoard {
        let mut ram = Ram::new();
        for i in 0..=0x100000 {
            ram.write::<u32>(4 * i, 0x13).unwrap(); // NOP
        }

        let mut board = VirtBoard::from_ram_with(ram, VirtBoardConfig::new()).unwrap();
        board.cpu.debug_csr(csr_index::mtvec, Some(0x8000_2000));
        board
    }

    #[test]
    fn test_step_cycles_advances_board_clock() {
        let mut board = create_test_board();
        let requested = <VirtBoard as Board>::STEP_BATCH_CYCLES + 3;

        let executed = board.run_cycles(requested);

        assert_eq!(executed, requested);
        assert_eq!(board.cycles(), requested);
        assert_eq!(board.clock.now(), requested >> 3);
        assert_eq!(board.cpu.read_pc(), ram_config::BASE_ADDR + requested * 4);
    }

    #[test]
    fn test_memory_image_and_initial_register_config() {
        use crate::isa::riscv::debugger::Address;

        let image_address = ram_config::BASE_ADDR + 0x2000;
        let image_offset = image_address - ram_config::BASE_ADDR;
        let image = vec![0xd0, 0x0d, 0xfe, 0xed];
        let original = vec![0xaa; image.len()];
        let mut ram = Ram::new();
        ram.try_insert_section(&original, image_offset).unwrap();

        let config = VirtBoardConfig::new()
            .with_memory_image(MemoryImage::new(image_address, image.clone()))
            .with_reg(11, image_address);

        let mut board = VirtBoard::from_ram_with(ram, config).unwrap();

        assert_eq!(board.cpu.read_reg(11), image_address);
        for (offset, expected) in image.into_iter().enumerate() {
            assert_eq!(
                board
                    .cpu
                    .read_memory::<u8>(Address::Phys(image_address as u64 + offset as u64))
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn external_uart_exposes_owned_port() {
        let mut board = VirtBoard::from_binary_with(&[], VirtBoardConfig::new()).unwrap();

        board.push_uart_input(b"a").unwrap();
        board.uart_port().unwrap().push_input(b"b").unwrap();
        assert!(board.take_uart_output().unwrap().is_empty());
    }

    #[test]
    fn virtboard_mmio_uses_uart16550a() {
        use crate::isa::riscv::debugger::Address;

        let mut board = VirtBoard::from_binary_with(&[], VirtBoardConfig::new()).unwrap();

        board
            .cpu
            .write_memory(Address::Phys(config::UART_BASE), b'a')
            .unwrap();
        assert_eq!(board.take_uart_output().unwrap(), vec![b'a']);

        board.push_uart_input(b"b").unwrap();
        board
            .cpu
            .write_memory(Address::Phys(config::UART_BASE + 1), 0x01u8)
            .unwrap();
        assert_eq!(
            board
                .cpu
                .read_memory::<u8>(Address::Phys(config::UART_BASE))
                .unwrap(),
            b'b'
        );
    }

    #[test]
    fn none_uart_rejects_external_operations() {
        let mut board =
            VirtBoard::from_binary_with(&[], VirtBoardConfig::new().with_uart_io(UartIoMode::None))
                .unwrap();

        assert_eq!(
            board.push_uart_input(b"a"),
            Err(UartIoError::Unavailable(UartIoMode::None))
        );
        assert_eq!(
            board.take_uart_output(),
            Err(UartIoError::Unavailable(UartIoMode::None))
        );
        assert!(matches!(
            board.uart_port(),
            Err(UartIoError::Unavailable(UartIoMode::None))
        ));
    }

    #[cfg(feature = "native-cli")]
    #[test]
    fn stdio_uart_exposes_only_router_handle() {
        let mut board = VirtBoard::from_binary_with(
            &[],
            VirtBoardConfig::new().with_uart_io(UartIoMode::Stdio),
        )
        .unwrap();

        assert!(board.uart_stdin_handle().is_some());
        assert_eq!(
            board.push_uart_input(b"a"),
            Err(UartIoError::Unavailable(UartIoMode::Stdio))
        );
        assert_eq!(
            board.take_uart_output(),
            Err(UartIoError::Unavailable(UartIoMode::Stdio))
        );
    }

    #[test]
    fn test_clint_mmio_access() {
        let mut board = create_test_board();

        // 直接测试 CLINT 设备
        let clint_handle = board.clint;
        let clint = board.device_mut(clint_handle);
        // 测试 mtime 读取
        let _ = clint.read_u64(0xbff8).unwrap();

        // 测试 mtime 写入
        let test_time = 0x123456789abcdef0u64;
        let write_result = clint.write_u64(0xbff8, test_time);
        assert!(
            write_result.is_ok(),
            "Failed to write to mtime: {:?}",
            write_result
        );

        // 验证写入后的读取
        let read_time: u64 = clint.read_u64(0xbff8).unwrap();
        assert_eq!(read_time, test_time, "mtime write/read mismatch");

        // 测试 mtimecmp 访问
        let timecmp_value = 0xfedcba9876543210u64;
        let write_result = clint.write_u64(0x4000, timecmp_value);
        assert!(
            write_result.is_ok(),
            "Failed to write to mtimecmp: {:?}",
            write_result
        );

        let read_timecmp: u64 = clint.read_u64(0x4000).unwrap();
        assert_eq!(read_timecmp, timecmp_value, "mtimecmp write/read mismatch");
    }

    #[test]
    fn test_clint_timer_interrupt() {
        let mut board = create_test_board();

        let interrupt_handler_addr = ram_config::BASE_ADDR + 0x1000;
        board
            .cpu_mut()
            .debug_csr(csr_index::mtvec, Some(interrupt_handler_addr));

        // Enable MIE in mstatus
        board.cpu_mut().debug_csr(csr_index::mstatus, Some(1 << 3));

        // Enable MTIE
        board.cpu_mut().debug_csr(csr_index::mie, Some(1 << 7));

        let target_time = 5;
        {
            let clint_handle = board.clint;
            let clint = board.device_mut(clint_handle);
            clint.write_u64(0x4000, target_time).unwrap();
        }

        println!("Running board steps to test timer interrupt...");

        let mut reach_mtvec = false;
        for i in 0..128 {
            board.step();

            let pc = board.cpu_mut().read_pc();

            if pc == interrupt_handler_addr {
                println!("PC jumped to interrupt handler at step {}!", i);
                reach_mtvec = true;
                break;
            }
        }

        assert!(reach_mtvec);
        assert_eq!(
            board.cpu_mut().debug_csr(csr_index::mip, None),
            Some(1 << 7)
        );
        assert!(board.clock.now() >= target_time);

        // Test MSIP (software interrupt)
        board.cpu_mut().write_pc(ram_config::BASE_ADDR);

        // Re-enable MIE in mstatus
        board.cpu_mut().debug_csr(csr_index::mstatus, Some(1 << 3));

        // Disable MTIE and enable MSIE
        board.cpu_mut().debug_csr(csr_index::mie, Some(1 << 3));

        {
            let clint_handle = board.clint;
            let clint = board.device_mut(clint_handle);
            clint.write_u64(0x0, 1).unwrap();
        }

        board.step();
        assert!(board.cpu_mut().read_pc() == interrupt_handler_addr);

        let mcause = board
            .cpu_mut()
            .debug_csr(Mcause::get_index(), None)
            .unwrap();
        assert_eq!(mcause, (1u64 << (XLEN - 1)) | 0b11)
    }

    #[cfg(feature = "test-device")]
    #[test]
    fn sample_timer_rearms_after_control_reset() {
        use std::{
            thread::sleep,
            time::{Duration, Instant},
        };

        use crate::board::virt::config::SAMPLE_TIMER_BASE;
        use crate::device::sample_timer::SAMPLE_TIMER_INTERRUPT_ID;
        use crate::{config::arch_config::WordType, isa::riscv::debugger::Address};
        const CONTEXT_ENABLE_BIT_OFFSET: WordType = 0x002000;
        const CONTEXT_ENABLE_BIT_SIZE: WordType = 0x80;
        const CONTEXT_CONFIG_OFFSET: WordType = 0x200000;
        const CONTEXT_CONFIG_SIZE: WordType = 0x1000;
        const CLAIM_COMPLETE_OFFSET: WordType =
            CONTEXT_CONFIG_OFFSET + (0 * CONTEXT_CONFIG_SIZE) + 4;

        fn wait_for_claim(board: &mut VirtBoard, claim_addr: WordType) -> u32 {
            let deadline = Instant::now() + Duration::from_secs(1);
            loop {
                board.step();
                let plic_handle = board.plic;
                let claimed_id = board.device_mut(plic_handle).read_u32(claim_addr).unwrap();
                if claimed_id != 0 {
                    return claimed_id;
                }
                assert!(
                    Instant::now() < deadline,
                    "sample timer interrupt timed out"
                );
                sleep(Duration::from_millis(1));
            }
        }

        let mut board = create_test_board();

        {
            let plic_handle = board.plic;
            let plic = board.device_mut(plic_handle);
            // priority_threshold
            let addr = CONTEXT_CONFIG_OFFSET + (0 * CONTEXT_CONFIG_SIZE);
            plic.write_u32(addr, 1).unwrap();

            // Sample timer interrupt priority.
            plic.write_u32(SAMPLE_TIMER_INTERRUPT_ID as WordType * 4, 5)
                .unwrap();

            // interrupt enable.
            let addr = CONTEXT_ENABLE_BIT_OFFSET + (0 * CONTEXT_ENABLE_BIT_SIZE) + 4;
            plic.write_u32(addr, 0xffffffff).unwrap();
        }

        // Mask writes cancel an outstanding deadline, so configure it before
        // the interval registers that schedule the timer.
        board
            .cpu
            .write_memory(
                Address::Phys(SAMPLE_TIMER_BASE + size_of::<u32>() as WordType),
                1u32,
            )
            .unwrap();
        board
            .cpu
            .write_memory(
                Address::Phys(SAMPLE_TIMER_BASE + 2 * size_of::<u32>() as WordType),
                10u32,
            )
            .unwrap();
        board
            .cpu
            .write_memory(
                Address::Phys(SAMPLE_TIMER_BASE + 3 * size_of::<u32>() as WordType),
                0u32,
            )
            .unwrap();

        let first_claim = wait_for_claim(&mut board, CLAIM_COMPLETE_OFFSET);
        assert_eq!(first_claim, SAMPLE_TIMER_INTERRUPT_ID);

        // Clear the device before completing the level-triggered PLIC interrupt.
        board
            .cpu
            .write_memory(Address::Phys(SAMPLE_TIMER_BASE), 1u32)
            .unwrap();
        board
            .device_mut(board.plic)
            .write_u32(CLAIM_COMPLETE_OFFSET, first_claim)
            .unwrap();
        assert_eq!(
            board
                .device_mut(board.plic)
                .read_u32(CLAIM_COMPLETE_OFFSET)
                .unwrap(),
            0
        );

        let second_claim = wait_for_claim(&mut board, CLAIM_COMPLETE_OFFSET);
        assert_eq!(second_claim, SAMPLE_TIMER_INTERRUPT_ID);
    }
}
