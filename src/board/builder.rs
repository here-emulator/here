use std::{
    any::TypeId,
    cell::{Cell, UnsafeCell},
    collections::HashMap,
    ptr::NonNull,
    rc::Rc,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::task_spawner::TaskSpawner;
use crate::{
    board::{
        BoardStatus, VirtBoardPlicContextId,
        virt::{IRQLine, RiscvIRQHandler, RiscvIRQSource, UartIoMode, VirtBoard},
    },
    clock::{Timer, VirtualClock},
    config::arch_config::WordType,
    device::{
        DeviceTrait, IdAllocator, PlicDevice,
        aclint::Clint,
        device_manager::{DeviceArena, DeviceArenaBuilder, DeviceHandle},
        mmio::{MemoryMapIO, MemoryMapItem},
        plic::{PLIC, PeriphIrqId},
        power_manager::PowerManager,
        uart16550a::{Uart16550A, UartBytePort},
    },
    isa::{
        DebugTarget,
        riscv::{decoder::Decoder, executor::RVCPU, mmu::VirtAddrManager, trap::Interrupt},
    },
    ram::Ram,
};

pub struct RVBoardBuilder {
    ram_ref: Rc<UnsafeCell<Ram>>,
    plic: Box<PLIC>,
    arena_builder: DeviceArenaBuilder,
    id_allocators: HashMap<TypeId, IdAllocator>,

    // CPU
    mmio_items: Vec<MemoryMapItem>,
    decoder: Option<Decoder>,
    initial_registers: Vec<(u8, WordType)>,

    // UART
    uart_io: UartIoMode,
    uart_port: Option<UartBytePort>,
    #[cfg(feature = "native-cli")]
    uart_stdin_handle: Option<crate::byte_io::StdinHandle>,

    // mmio info
    clint_mmio: Option<(WordType, WordType)>,
    plic_mmio: Option<(WordType, WordType)>,

    #[cfg(not(target_arch = "wasm32"))]
    pub spawner: TaskSpawner,
}

impl RVBoardBuilder {
    const MTIME_OFFSET: u64 = 0xbff8;
    const MTIMECMP_OFFSET: u64 = 0x4000;

    // Loader may init Ram before RVBoard init.
    pub fn new(ram: Ram) -> Self {
        Self {
            ram_ref: Rc::new(UnsafeCell::new(ram)),
            plic: Box::new(PLIC::new()),
            arena_builder: DeviceArenaBuilder::new(),
            mmio_items: Vec::new(),
            id_allocators: HashMap::new(),

            // CPU
            decoder: None,
            initial_registers: Vec::new(),

            // UART
            uart_io: UartIoMode::External,
            uart_port: None,
            #[cfg(feature = "native-cli")]
            uart_stdin_handle: None,

            #[cfg(not(target_arch = "wasm32"))]
            spawner: TaskSpawner::new(),
            clint_mmio: None,
            plic_mmio: None,
        }
    }

    pub fn with_decoder(mut self, decoder: Decoder) -> Self {
        self.decoder = Some(decoder);
        self
    }

    pub fn with_initial_registers(mut self, registers: Vec<(u8, WordType)>) -> Self {
        self.initial_registers = registers;
        self
    }

    pub fn add_power_manager(self, base: WordType, size: WordType) -> Self {
        self.add_plain_mmio_device(Box::new(PowerManager::new()), base, size)
    }

    pub fn add_clint(mut self, base: WordType, size: WordType) -> Self {
        self.clint_mmio = Some((base, size));
        self
    }

    pub fn add_plic(mut self, base: WordType, size: WordType) -> Self {
        self.plic_mmio = Some((base, size));
        self
    }

    pub fn add_uart(
        mut self,
        base: WordType,
        size: WordType,
        irq: PeriphIrqId,
        mode: UartIoMode,
    ) -> Self {
        self.uart_io = mode;
        let (uart1, uart_port1) = Uart16550A::new();
        self.uart_port = Some(uart_port1);
        self.add_plic_device(Box::new(uart1), base, size, irq)
    }

    #[cfg(all(feature = "test-device", not(target_arch = "wasm32")))]
    pub fn add_sample_timer(mut self, base: WordType, size: WordType, irq: PeriphIrqId) -> Self {
        use crate::device::sample_timer::SampleTimerDevice;

        let (device, task) = SampleTimerDevice::new();
        self.spawner.register(task);
        self.add_plic_device(Box::new(device), base, size, irq)
    }

    pub fn add_plic_device<D: PlicDevice + 'static>(
        mut self,
        mut device: Box<D>,
        base: WordType,
        size: WordType,
        interrupt_id: PeriphIrqId,
    ) -> Self {
        let type_id = TypeId::of::<D>();
        let allocator = self.id_allocators.entry(type_id).or_insert_with(|| {
            IdAllocator::new(0, stringify!(D).to_string(), base, size, Some(interrupt_id))
        });

        let info = allocator.get();
        let irq = info.irq.unwrap_or(interrupt_id);
        let device_ptr = NonNull::from(&mut *device as &mut dyn PlicDevice);
        self.plic.register_device(device_ptr, irq);

        self.add_plain_mmio_device(device, info.base, info.size)
    }

    pub fn add_plain_mmio_device<D: DeviceTrait>(
        mut self,
        device: Box<D>,
        base: WordType,
        size: WordType,
    ) -> Self {
        let handle = self.arena_builder.register(device);
        self.mmio_items.push(MemoryMapItem::new(base, size, handle));
        self
    }

    fn register_rv_interrupt(
        cpu: &mut Box<RVCPU>,
        arena: &mut DeviceArena,
        clint: &DeviceHandle<Clint>,
        plic: &DeviceHandle<PLIC>,
        clint_base: WordType,
    ) {
        let cpu_ptr = cpu.as_mut() as *mut dyn RiscvIRQHandler;

        // Machine Timer Interrupt
        arena
            .device_mut(*clint)
            .set_irq_line(IRQLine::new(cpu_ptr, Interrupt::MachineTimer), 0);

        // Machine Soft Interrupt
        arena
            .device_mut(*clint)
            .set_irq_line(IRQLine::new(cpu_ptr, Interrupt::MachineSoft), 1);

        cpu.time_addr = Some(clint_base + Self::MTIME_OFFSET);

        // PLIC External Interrupt.
        arena.device_mut(*plic).set_irq_line(
            IRQLine::new(cpu_ptr, Interrupt::MachineExternal),
            VirtBoardPlicContextId::Cpu0MachineMode.into(),
        );
        arena.device_mut(*plic).set_irq_line(
            IRQLine::new(cpu_ptr, Interrupt::SupervisorExternal),
            VirtBoardPlicContextId::Cpu0SuperviserMode.into(),
        );
    }

    pub fn build(mut self) -> Result<VirtBoard, String> {
        let cycles = Rc::new(Cell::new(0));
        let clock = VirtualClock::new(cycles.clone());
        let timer = Rc::new(UnsafeCell::new(Timer::new(clock.clone())));
        let ram_ref = self.ram_ref.clone();

        let mut uart_port = None;
        #[cfg(feature = "native-cli")]
        let mut uart_stdin_handle = None;
        if let Some(uart_port1) = self.uart_port.take() {
            match self.uart_io {
                UartIoMode::None => drop(uart_port1),
                UartIoMode::External => uart_port = Some(uart_port1),
                #[cfg(feature = "native-cli")]
                UartIoMode::Stdio => {
                    use crate::byte_io::StdinRouter;

                    let input = uart_port1.input_sender();
                    self.spawner.register(uart_port1);

                    let router = StdinRouter::global();
                    let handle = router.register(input);
                    router.switch_to(handle);
                    uart_stdin_handle = Some(handle);
                }
            }
        }

        // Verify that the required devices have been added.
        let (clint_base, clint_size) = match self.clint_mmio {
            Some(range) => range,
            None => {
                panic!("Clint MMIO address is not configured");
            }
        };
        let (plic_base, plic_size) = match self.plic_mmio {
            Some(range) => range,
            None => {
                panic!("Plic MMIO address is not configured");
            }
        };

        let clint = Box::new(Clint::new(
            1,
            0,
            Self::MTIME_OFFSET,
            Self::MTIMECMP_OFFSET,
            clock.clone(),
            timer.clone(),
        ));

        // add power_manager, clint, plic.
        let clint = self.arena_builder.register(clint);
        let plic = self.arena_builder.register(self.plic);
        self.mmio_items.extend([
            MemoryMapItem::new(clint_base, clint_size, clint),
            MemoryMapItem::new(plic_base, plic_size, plic),
        ]);

        let mut device_arena = Box::new(self.arena_builder.build());

        // build vaddr manager (MMIO).
        let device_arena_ptr = NonNull::from(device_arena.as_mut());
        let mmio = unsafe {
            MemoryMapIO::from_mmio_items(ram_ref.clone(), device_arena_ptr, self.mmio_items)
        };
        let vaddr_manager = VirtAddrManager::from_ram_and_mmio(ram_ref.clone(), mmio);

        // build rvcpu.
        let decoder = self.decoder.take().unwrap_or_else(Decoder::new);
        let mut cpu = Box::new(RVCPU::from_decoder(decoder, vaddr_manager));

        // init cpu regfile.
        for (register, value) in self.initial_registers {
            cpu.write_reg(register, value);
        }

        // register rv interrupt.
        Self::register_rv_interrupt(&mut cpu, device_arena.as_mut(), &clint, &plic, clint_base);

        #[cfg(not(target_arch = "wasm32"))]
        let task_handle = self.spawner.start();

        Ok(VirtBoard {
            #[cfg(not(target_arch = "wasm32"))]
            task_handle,
            loader: None,
            cpu,
            cycles,
            clock,
            timer,

            clint,
            plic,
            uart_io: self.uart_io,
            uart_port,
            #[cfg(feature = "native-cli")]
            uart_stdin_handle,

            status: BoardStatus::Running,
            device_arena,
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;

    use crate::{
        DeviceConfig,
        device::virtio::{
            virtio_block::VirtIOBlkDeviceBuilder, virtio_device::VirtIODeviceEnum,
            virtio_mmio::VirtIOMMIO,
        },
    };

    impl RVBoardBuilder {
        pub fn add_virtio_device(
            mut self,
            base: WordType,
            size: WordType,
            irq: PeriphIrqId,
            config: DeviceConfig,
        ) -> Self {
            let virtio_device = match config.dev_type {
                VirtIODeviceEnum::VirtIOBlock => {
                    let ram_base = unsafe { &mut self.ram_ref.as_mut_unchecked()[0] as *mut u8 };
                    VirtIOBlkDeviceBuilder::new(
                        ram_base,
                        String::from(config.path.to_str().unwrap()),
                    )
                    .host_feature(
                        crate::device::virtio::virtio_block::VirtIOBlockFeature::BlockSize,
                    )
                    .host_feature(crate::device::virtio::virtio_block::VirtIOBlockFeature::Flush)
                    .get_and_spawner_task(&mut self.spawner)
                }
                dev_type => {
                    panic!("unsupport device: {:#?}", dev_type);
                }
            };
            let virtio_mmio_device = VirtIOMMIO::new(Box::new(UnsafeCell::new(virtio_device)));
            self.add_plic_device(Box::new(virtio_mmio_device), base, size, irq)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::virt::config::*;
    use crate::device::{aclint::CLINT_SIZE, plic::PLIC_SIZE, power_manager::POWER_MANAGER_SIZE};
    #[test]
    #[should_panic]
    fn test_rvboard_builder_missing_clint() {
        let ram = Ram::new();
        let _ = RVBoardBuilder::new(ram)
            .add_power_manager(POWER_MANAGER_BASE, POWER_MANAGER_SIZE)
            .add_plic(PLIC_BASE, PLIC_SIZE)
            .build();
    }

    #[test]
    #[should_panic]
    fn test_rvboard_builder_missing_plic() {
        let ram = Ram::new();
        let _ = RVBoardBuilder::new(ram)
            .add_power_manager(POWER_MANAGER_BASE, POWER_MANAGER_SIZE)
            .add_clint(CLINT_BASE, CLINT_SIZE)
            .build();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod native_tests {
    use super::*;
    use crate::board::virt::*;
    use crate::device::virtio::virtio_block::init_block_file;
    use crate::device::{aclint::CLINT_SIZE, plic::PLIC_SIZE, power_manager::POWER_MANAGER_SIZE};
    use crate::{DeviceConfig, device::virtio::virtio_device::VirtIODeviceEnum};

    fn add_default_mmio_devices(builder: RVBoardBuilder) -> RVBoardBuilder {
        builder
            .add_power_manager(POWER_MANAGER_BASE, POWER_MANAGER_SIZE)
            .add_clint(CLINT_BASE, CLINT_SIZE)
            .add_plic(PLIC_BASE, PLIC_SIZE)
    }

    #[test]
    fn test_rvboard_builder_basic() {
        let ram = Ram::new();
        let board = add_default_mmio_devices(RVBoardBuilder::new(ram)).build();
        assert!(board.is_ok());
    }

    #[test]
    fn test_rvboard_builder_multiple_virtio_devices() {
        let temp_dir = std::env::temp_dir();
        let file_path1 = temp_dir.join("test_builder_virtio_block_1.img");
        let file_path2 = temp_dir.join("test_builder_virtio_block_2.img");
        let buf = [0u8; 512];
        let _ = init_block_file(file_path1.to_str().unwrap(), 1, |_| &buf);
        let _ = init_block_file(file_path2.to_str().unwrap(), 1, |_| &buf);

        let ram = Ram::new();
        let devices = vec![
            DeviceConfig {
                dev_type: VirtIODeviceEnum::VirtIOBlock,
                path: file_path1.clone(),
            },
            DeviceConfig {
                dev_type: VirtIODeviceEnum::VirtIOBlock,
                path: file_path2.clone(),
            },
        ];

        let mut builder = add_default_mmio_devices(RVBoardBuilder::new(ram));
        for config in devices {
            builder = builder.add_virtio_device(
                VIRTIO_MMIO_BASE,
                VIRTIO_MMIO_SIZE,
                VIRTIO_IRQ_BASE,
                config,
            );
        }
        let board = builder.build();
        assert!(board.is_ok());

        let _ = std::fs::remove_file(file_path1);
        let _ = std::fs::remove_file(file_path2);
    }

    #[test]
    fn test_rvboard_builder_add_device() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_builder_add_device_virtio.img");
        let buf = [0u8; 512];
        let _ = init_block_file(file_path.to_str().unwrap(), 1, |_| &buf);

        let ram = Ram::new();
        let board = add_default_mmio_devices(RVBoardBuilder::new(ram))
            .add_uart(UART_BASE, UART_SIZE, UART_IRQ, UartIoMode::External)
            .add_virtio_device(
                VIRTIO_MMIO_BASE,
                VIRTIO_MMIO_SIZE,
                VIRTIO_IRQ_BASE,
                DeviceConfig {
                    dev_type: VirtIODeviceEnum::VirtIOBlock,
                    path: file_path.clone(),
                },
            )
            .build();
        assert!(board.is_ok());

        let _ = std::fs::remove_file(file_path);
    }
}
