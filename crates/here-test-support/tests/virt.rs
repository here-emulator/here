//! Board behaviour tests that exercise the public board API end to end.

use here_board::Board;
use here_board::virt::UartIoError;
use here_board::virt::UartIoMode;
use here_board::virt::VirtBoard;
use here_board::virt::config::*;
use here_device::DeviceTrait;
use here_device::clock::Clock;
use here_device::ram::Ram;
use here_isa::DebugTarget;
use here_isa::riscv::address::Address;
use here_isa::riscv::csr_reg::NamedCsrReg;
use here_isa::riscv::csr_reg::csr_index;
use here_isa::riscv::csr_reg::csr_macro::Mcause;
use here_isa::riscv::csr_reg::csr_macro::Mcycle;
use here_isa::riscv::csr_reg::csr_macro::Minstret;
use here_isa::riscv::csr_reg::csr_macro::Satp;
use here_isa::riscv::executor::ExecutionHook;
use here_isa::riscv::executor::RVCPU;
use here_test_support::board_from_binary;
use here_test_support::jit_board_from_binary;
use here_test_support::standard_builder;
use here_types::WordType;
use here_types::XLEN;
use here_types::ram_config;

fn create_test_board() -> VirtBoard {
    let mut ram = Ram::new();
    for i in 0..=0x100000 {
        ram.write::<u32>(4 * i, 0x13).unwrap(); // NOP
    }

    let mut board = standard_builder(ram, UartIoMode::External)
        .add_sample_timer(
            SAMPLE_TIMER_BASE,
            here_device::sample_timer::SAMPLE_TIMER_SIZE,
            here_device::sample_timer::SAMPLE_TIMER_INTERRUPT_ID,
        )
        .build();
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
fn jit_backend_matches_interpreter_for_supported_block() {
    let program = [
        0x0010_8093_u32, // addi x1, x1, 1
        0x0010_8093_u32, // addi x1, x1, 1
        0x0020_80b3_u32, // add x1, x1, x2
        0x0020_8133_u32, // add x2, x1, x2
        0x0020_81b3_u32, // add x3, x1, x2
        0x001f_8f93_u32, // addi x31, x31, 1
    ];
    let bytes = program
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    let mut interpreter = board_from_binary(&bytes, UartIoMode::External);
    let Some(mut jit) = jit_board_from_binary(&bytes) else {
        return;
    };
    interpreter.cpu_mut().write_reg(2, 7);
    jit.cpu_mut().write_reg(2, 7);

    assert_eq!(interpreter.run_cycles(6), 6);
    assert_eq!(jit.run_cycles(6), 6);

    assert_eq!(jit.cpu.read_pc(), interpreter.cpu.read_pc());
    assert_eq!(jit.cpu.read_reg(1), interpreter.cpu.read_reg(1));
    assert_eq!(jit.cpu.read_reg(2), interpreter.cpu.read_reg(2));
    assert_eq!(jit.cpu.read_reg(3), interpreter.cpu.read_reg(3));
    assert_eq!(jit.cpu.read_reg(31), interpreter.cpu.read_reg(31));
    assert_eq!(
        jit.cpu_mut().debug_csr(Mcycle::get_index(), None),
        interpreter.cpu_mut().debug_csr(Mcycle::get_index(), None)
    );
    assert_eq!(
        jit.cpu_mut().debug_csr(Minstret::get_index(), None),
        interpreter.cpu_mut().debug_csr(Minstret::get_index(), None)
    );
}

#[test]
fn jit_fallback_executes_exactly_one_cycle_before_resuming_jit() {
    let program = [
        0x0220_80b3_u32, // mul x1, x1, x2
        0x0010_8093_u32, // addi x1, x1, 1
    ];
    let bytes = program
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let Some(mut board) = jit_board_from_binary(&bytes) else {
        return;
    };
    board.cpu_mut().write_reg(1, 3);
    board.cpu_mut().write_reg(2, 4);

    assert_eq!(board.run_cycles(2), 2);
    assert_eq!(board.cpu.read_pc(), ram_config::BASE_ADDR + 8);
    assert_eq!(board.cpu.read_reg(1), 13);
    assert_eq!(
        board.cpu_mut().debug_csr(Mcycle::get_index(), None),
        Some(2)
    );
    assert_eq!(
        board.cpu_mut().debug_csr(Minstret::get_index(), None),
        Some(2)
    );
}

#[test]
fn satp_write_invalidates_cached_jit_blocks() {
    let bytes = 0x0010_8093_u32.to_le_bytes(); // addi x1, x1, 1
    let Some(mut board) = jit_board_from_binary(&bytes) else {
        return;
    };

    assert_eq!(board.run_cycles(1), 1);
    board
        .cpu_mut()
        .write_memory(Address::Virt(ram_config::BASE_ADDR), 0x0020_8093_u32)
        .unwrap(); // addi x1, x1, 2
    board.cpu_mut().write_csr(Satp::get_index(), 1).unwrap();
    board.cpu_mut().write_pc(ram_config::BASE_ADDR);
    assert_eq!(board.run_cycles(1), 1);

    assert_eq!(board.cpu.read_reg(1), 3);
}

#[test]
fn hooked_execution_uses_per_instruction_interpreter_path_with_jit_configured() {
    struct StopAfterTwo(Vec<WordType>);

    impl ExecutionHook for StopAfterTwo {
        type CycleContext = WordType;

        fn before_step(&mut self, cpu: &mut RVCPU) -> Self::CycleContext {
            cpu.read_pc()
        }

        fn on_interrupt_taken(&mut self, interrupted_pc: WordType) -> Self::CycleContext {
            interrupted_pc
        }

        fn after_step(&mut self, pc: WordType, _cpu: &mut RVCPU) -> bool {
            self.0.push(pc);
            self.0.len() == 2
        }
    }

    let bytes = [0x0010_8093_u32; 3]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let Some(mut board) = jit_board_from_binary(&bytes) else {
        return;
    };
    let mut hook = StopAfterTwo(Vec::new());

    let result = board.run_cycles_hooked(3, &mut hook);

    assert_eq!(result.cycles, 2);
    assert!(result.hook_stopped);
    assert_eq!(hook.0, [ram_config::BASE_ADDR, ram_config::BASE_ADDR + 4]);
    assert_eq!(board.cycles(), 2);
    assert_eq!(board.cpu.read_reg(1), 2);
}

#[test]
fn builder_loads_memory_image_and_initial_registers() {
    let image_address = ram_config::BASE_ADDR + 0x2000;
    let image_offset = image_address - ram_config::BASE_ADDR;
    let image = vec![0xd0, 0x0d, 0xfe, 0xed];
    let mut ram = Ram::new();
    ram.try_insert_section(&image, image_offset).unwrap();

    let mut board = standard_builder(ram, UartIoMode::External)
        .with_initial_registers(vec![(11, image_address)])
        .build();

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
    let mut board = board_from_binary(&[], UartIoMode::External);

    board.uart_port().unwrap().push_input(b"a").unwrap();
    board.uart_port().unwrap().push_input(b"b").unwrap();
    assert!(board.uart_port().unwrap().take_output().is_empty());
}

#[test]
fn virtboard_mmio_uses_uart16550a() {
    let mut board = board_from_binary(&[], UartIoMode::External);

    board
        .cpu
        .write_memory(Address::Phys(UART_BASE), b'a')
        .unwrap();
    assert_eq!(board.uart_port().unwrap().take_output(), vec![b'a']);

    board.uart_port().unwrap().push_input(b"b").unwrap();
    board
        .cpu
        .write_memory(Address::Phys(UART_BASE + 1), 0x01u8)
        .unwrap();
    assert_eq!(
        board
            .cpu
            .read_memory::<u8>(Address::Phys(UART_BASE))
            .unwrap(),
        b'b'
    );
}

#[test]
fn none_uart_rejects_external_operations() {
    let mut board = board_from_binary(&[], UartIoMode::None);

    assert!(matches!(
        board.uart_port(),
        Err(UartIoError::Unavailable(UartIoMode::None))
    ));
}

#[test]
fn stdio_uart_exposes_only_router_handle() {
    let mut board = board_from_binary(&[], UartIoMode::Stdio);

    assert!(board.uart_stdin_handle().is_some());
    assert!(matches!(
        board.uart_port(),
        Err(UartIoError::Unavailable(UartIoMode::Stdio))
    ));
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

#[test]
fn sample_timer_rearms_after_control_reset() {
    use std::thread::sleep;
    use std::time::Duration;
    use std::time::Instant;

    use here_device::sample_timer::SAMPLE_TIMER_INTERRUPT_ID;
    const CONTEXT_ENABLE_BIT_OFFSET: WordType = 0x002000;
    const CONTEXT_ENABLE_BIT_SIZE: WordType = 0x80;
    const CONTEXT_CONFIG_OFFSET: WordType = 0x200000;
    const CONTEXT_CONFIG_SIZE: WordType = 0x1000;
    const CLAIM_COMPLETE_OFFSET: WordType = CONTEXT_CONFIG_OFFSET + (0 * CONTEXT_CONFIG_SIZE) + 4;

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
