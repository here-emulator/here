#![cfg_attr(debug_assertions, allow(dead_code))]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod logging;
mod welcome;

use std::fs;
use std::io::{IsTerminal, Write};
use std::time::Instant;

use clap::Parser;
use here::board::Board;
use here::byte_io::StdinRouter;
use here::gdb;
use here::isa::DebugTarget;
use here::isa::riscv::debugger::Address;
use here::isa::riscv::decoder::Decoder;
use here::isa::riscv::isa_builder::DEFAULT_ISA;
use here::rvdb::{Printer, RvdbSession, SyncREPL};
use here::{
    DeviceConfig,
    board::virt::{ExecutorBackendKind, MemoryImage, UartIoMode, VirtBoard, VirtBoardConfig},
    config::arch_config::WordType,
};

use crate::{logging::LogLevel, welcome::display_welcome_message};

#[derive(Debug, Copy, Clone, PartialEq, Eq, clap::ValueEnum)]
enum TargetFormat {
    Auto,
    Elf,
    Bin,
}

const DEFAULT_DTB_ADDRESS: WordType = 0x9f00_0000;
const DEFAULT_GDB_PORT: u16 = 1234;

struct RawModeGuard {
    enabled: bool,
}

impl RawModeGuard {
    fn new(enabled: bool) -> Result<Self, String> {
        if enabled {
            log::debug!("enabling raw mode");
            crossterm::terminal::enable_raw_mode().map_err(|error| error.to_string())?;
        }
        Ok(Self { enabled })
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        if self.enabled {
            log::debug!("disabling raw mode");
            if let Err(error) = crossterm::terminal::disable_raw_mode() {
                eprintln!("failed to restore terminal mode: {error}");
            }
        }
    }
}

fn parse_address(value: &str) -> Result<WordType, String> {
    let value = value.trim();
    let normalized = value.replace('_', "");
    let parsed = if let Some(hex) = normalized
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16)
    } else {
        normalized.parse::<u64>()
    }
    .map_err(|error| format!("invalid address {value:?}: {error}"))?;

    WordType::try_from(parsed).map_err(|_| format!("address {value:?} does not fit WordType"))
}

fn display_device_list(devices: &[DeviceConfig]) {
    println!("\x1b[{}mdevice list:", 34);
    for device in devices {
        println!("\t{:#?}: {:#?}", device.dev_type, device.path);
    }
    println!("\x1b[0m");
}

#[derive(Parser, Debug)]
#[command(
    version,
    next_line_help = true,
    about = "An educational full-system RISC-V emulator written in Rust.",
    after_help = "Terminal controls:\n  Ctrl+A, then x  Exit the emulator."
)]
struct Args {
    /// RISC-V ELF executable or raw binary image to run.
    path: std::path::PathBuf,

    /// Choose the input format; auto will check by filename extension.
    #[arg(
        value_enum,
        short,
        long,
        value_name = "FORMAT",
        default_value_t = TargetFormat::Auto
    )]
    format: TargetFormat,

    /// Start the built-in rvdb debugger.
    #[arg(
        short = 'g',
        long = "debug",
        conflicts_with = "gdb",
        default_value_t = false
    )]
    debug: bool,

    /// Enable GDB remote debugging server (TCP 127.0.0.1:1234 by default).
    #[arg(short = 'G', long = "gdb", default_value_t = false)]
    gdb: bool,

    /// TCP port for the GDB server (conflicts with --gdb-uds).
    #[arg(long = "gdb-port", value_name = "PORT", requires = "gdb")]
    #[cfg_attr(unix, arg(conflicts_with = "gdb_uds"))]
    gdb_port: Option<u16>,

    /// Use a Unix domain socket for the GDB server (conflicts with --gdb-port).
    #[cfg(unix)]
    #[arg(
        long = "gdb-uds",
        value_name = "PATH",
        requires = "gdb",
        conflicts_with = "gdb_port"
    )]
    gdb_uds: Option<std::path::PathBuf>,

    /// Run rvdb commands from this file before entering the interactive debugger. Requires --debug.
    #[arg(short = 'S', long = "script", value_name = "FILE", requires = "debug")]
    script: Option<std::path::PathBuf>,

    /// Print additional startup details.
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    /// Set the logging level.
    #[arg(
        value_enum,
        long = "loglevel",
        value_name = "LEVEL",
        default_value_t = LogLevel::Info
    )]
    log_level: LogLevel,

    /// Attach a VirtIO block device; may be repeated. Format: virtio-block:PATH
    #[arg(
        long = "device",
        value_name = "TYPE:PATH",
        action = clap::ArgAction::Append
    )]
    devices: Vec<DeviceConfig>,

    /// Configure the decoder with an ISA string such as RV64GC or RV64GCV.
    #[arg(long = "isa", value_name = "ISA", default_value = DEFAULT_ISA)]
    isa: String,

    /// Write the RISC-V architecture-test signature to this file on exit.
    #[arg(long = "signature", value_name = "FILE")]
    signature: Option<std::path::PathBuf>,

    /// Set the architecture-test signature word size in bytes (4 or 8). Requires --signature.
    #[arg(
        long = "signature-granularity",
        value_name = "BYTES",
        requires = "signature",
        default_value_t = 4
    )]
    signature_granularity: u32,

    /// Stop after this many emulated cycles; 0 disables the limit.
    #[arg(long = "max-cycles", value_name = "COUNT", default_value_t = 0)]
    max_cycles: u64,

    /// Select the CPU executor backend.
    #[arg(
        long = "executor",
        value_enum,
        value_name = "BACKEND",
        default_value_t = ExecutorBackendKind::Interpreter
    )]
    executor: ExecutorBackendKind,

    /// Load a DTB and pass its guest address to OpenSBI in register a1.
    #[arg(long = "dtb", value_name = "FILE")]
    dtb: Option<std::path::PathBuf>,

    /// Load --dtb at this guest physical address (default: 0x9f000000).
    #[arg(
        long = "dtb-address",
        value_name = "ADDRESS",
        requires = "dtb",
        value_parser = parse_address
    )]
    dtb_address: Option<WordType>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RvdbMode {
    Disabled,
    Interactive,
    ScriptOnly,
    MissingScript,
}

fn rvdb_mode(args: &Args, stdin_terminal: bool, stdout_terminal: bool) -> RvdbMode {
    if !args.debug {
        RvdbMode::Disabled
    } else if stdin_terminal && stdout_terminal {
        RvdbMode::Interactive
    } else if args.script.is_some() {
        RvdbMode::ScriptOnly
    } else {
        RvdbMode::MissingScript
    }
}

fn uart_io_mode(args: &Args) -> UartIoMode {
    if args.gdb {
        UartIoMode::None
    } else {
        UartIoMode::Stdio
    }
}

fn should_enable_raw_mode(args: &Args, _mode: RvdbMode, _stdin_term: bool) -> bool {
    args.gdb == false
}

fn gdb_config(args: &Args) -> Option<gdb::Config> {
    if !args.gdb {
        return None;
    }

    #[cfg(unix)]
    if let Some(path) = &args.gdb_uds {
        return Some(gdb::Config::Uds(path.clone()));
    }

    Some(gdb::Config::Tcp(args.gdb_port.unwrap_or(DEFAULT_GDB_PORT)))
}

/// Used for riscv-arch-test.
fn dump_signature(
    board: &mut VirtBoard,
    out_path: &std::path::Path,
    granularity: u32,
) -> Result<(), String> {
    let loader = board
        .loader()
        .ok_or_else(|| "ELF loader not available; cannot resolve signature symbols".to_string())?;

    let symtab = loader.get_symbol_table().ok_or_else(|| {
        "No .symtab found in ELF; cannot resolve begin_signature/end_signature".to_string()
    })?;

    let begin = symtab
        .func_addr_by_name("begin_signature")
        .ok_or_else(|| "Symbol begin_signature not found".to_string())?;
    let end = symtab
        .func_addr_by_name("end_signature")
        .ok_or_else(|| "Symbol end_signature not found".to_string())?;

    if end <= begin {
        return Err(format!(
            "Invalid signature range: begin=0x{:x}, end=0x{:x}",
            begin, end
        ));
    }

    let size = end - begin;
    let step = match granularity {
        4 => 4u64,
        8 => 8u64,
        other => return Err(format!("Unsupported signature granularity: {}", other)),
    };

    if size % step != 0 {
        return Err(format!(
            "Signature size 0x{:x} not aligned to granularity {}",
            size, step
        ));
    }

    let file = std::fs::File::create(out_path).map_err(|e| {
        format!(
            "Failed to create signature file {}: {}",
            out_path.display(),
            e
        )
    })?;
    let mut w = std::io::BufWriter::new(file);

    let mut addr = begin;
    while addr < end {
        match step {
            4 => {
                let v = board
                    .cpu
                    .read_memory::<u32>(Address::Phys(addr))
                    .map_err(|e| format!("Failed to read signature @0x{:x}: {:?}", addr, e))?;
                use std::io::Write;
                writeln!(w, "{:08x}", v)
                    .map_err(|e| format!("Failed to write signature: {}", e))?;
            }
            8 => {
                let v = board
                    .cpu
                    .read_memory::<u64>(Address::Phys(addr))
                    .map_err(|e| format!("Failed to read signature @0x{:x}: {:?}", addr, e))?;
                use std::io::Write;
                writeln!(w, "{:016x}", v)
                    .map_err(|e| format!("Failed to write signature: {}", e))?;
            }
            _ => unreachable!(),
        }
        addr += step;
    }

    Ok(())
}

fn main() {
    let cli_args = Args::parse();
    let _logger_handle = logging::init(cli_args.log_level);

    display_welcome_message();

    let stdin_terminal = std::io::stdin().is_terminal();
    let stdout_terminal = std::io::stdout().is_terminal();
    let rvdb_mode = rvdb_mode(&cli_args, stdin_terminal, stdout_terminal);
    if rvdb_mode == RvdbMode::MissingScript {
        eprintln!("rvdb requires --script when stdin or stdout is not a terminal");
        std::process::exit(2);
    }

    let uart_io = uart_io_mode(&cli_args);
    let raw_mode = should_enable_raw_mode(&cli_args, rvdb_mode, stdin_terminal);
    let raw_mode_guard = RawModeGuard::new(raw_mode).unwrap_or_else(|error| {
        eprintln!("failed to enable terminal raw mode: {error}");
        std::process::exit(2);
    });

    if cli_args.verbose {
        println!(
            "path = {:?}, debug = {}, verbose = {}, log_level = {:?}.\r",
            cli_args.path, cli_args.debug, cli_args.verbose, cli_args.log_level
        );
        display_device_list(&cli_args.devices);
    }

    let decoder = Decoder::from_isa_str(&cli_args.isa).unwrap_or_else(|e| {
        eprintln!("Invalid ISA string {:?}: {}", cli_args.isa, e);
        std::process::exit(2);
    });
    let mut board_config = VirtBoardConfig::new()
        .with_decoder(decoder)
        .with_virtio_devices(cli_args.devices.clone())
        .with_uart_io(uart_io)
        .with_executor_backend(cli_args.executor);

    if let Some(dtb_path) = &cli_args.dtb {
        let dtb_address = cli_args.dtb_address.unwrap_or(DEFAULT_DTB_ADDRESS);
        assert!(
            dtb_address.is_multiple_of(8),
            "DTB address 0x{dtb_address:x} must be 8-byte aligned"
        );
        let dtb = std::fs::read(dtb_path)
            .unwrap_or_else(|error| panic!("Failed to read DTB {}: {error}", dtb_path.display()));
        board_config = board_config
            .with_memory_image(MemoryImage::new(dtb_address, dtb))
            .with_reg(11, dtb_address);

        if cli_args.verbose {
            println!(
                "DTB {} will be loaded at 0x{dtb_address:x} and passed in a1\r",
                dtb_path.display()
            );
        }
    }

    let ext = cli_args
        .path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("<unknown>");

    let mut board = match (cli_args.format, ext) {
        (TargetFormat::Elf, _) | (TargetFormat::Auto, "elf") => {
            if cli_args.verbose {
                println!("ELF file detected\r");
            }
            let bytes = std::fs::read(cli_args.path.clone()).expect("Failed to read target file");
            VirtBoard::from_elf_with(bytes, board_config).expect("ELF load failed")
        }

        (TargetFormat::Bin, _) | (TargetFormat::Auto, "bin") => {
            if cli_args.verbose {
                println!("Binary file detected\r");
            }
            let bytes = std::fs::read(cli_args.path.clone()).expect("Failed to read target file");
            VirtBoard::from_binary_with(&bytes, board_config).expect("Binary load failed")
        }

        _ => {
            log::error!("Format is not supported at present.");
            panic!();
        }
    };

    // Stdio boards initialize the router while wiring UART. GDB has no UART
    // target, but still needs the router's host control sequence handling.
    StdinRouter::global();

    if cli_args.debug {
        let script_lines = cli_args.script.as_ref().map(|script| {
            std::fs::read_to_string(script)
                .unwrap_or_else(|error| {
                    panic!("failed to read script {}: {error}", script.display())
                })
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        });

        if rvdb_mode == RvdbMode::Interactive {
            let uart_handle = board
                .uart_stdin_handle()
                .expect("Stdio board must expose its UART stdin handle");
            let session = RvdbSession::with_printer(board, Printer::ansi_color());
            let mut repl = SyncREPL::new(session, uart_handle);
            if let Some(lines) = &script_lines
                && repl.run_script(lines)
            {
                return;
            }
            repl.run();
        } else {
            let mut session = RvdbSession::with_printer(board, Printer::plain());
            let lines = script_lines.as_deref().expect("script checked above");
            let mut stdout = std::io::stdout().lock();
            let _ = session
                .run_script(lines, |output| {
                    stdout.write_all(output.as_bytes())?;
                    stdout.flush()
                })
                .unwrap();
        }
        return;
    } else if let Some(config) = gdb_config(&cli_args) {
        if let Err(e) = gdb::event_loop(board, config) {
            log::error!("{:?}", e);
            panic!();
        }
    } else {
        if let Some(sig_path) = &cli_args.signature {
            // Create the signature file before running the emulator to ensure the file exists even if the emulator crashes.
            fs::File::create(sig_path).expect("Failed to create signature file");
        }

        let now = Instant::now();
        if cli_args.max_cycles == 0 {
            board.run();
        } else {
            board.run_cycles(cli_args.max_cycles);
            if board.status() != here::board::BoardStatus::Halt {
                log::error!(
                    "Max cycles reached: {} at pc {}",
                    cli_args.max_cycles,
                    board.cpu().read_pc()
                );
            }
        }
        if let Some(sig_path) = &cli_args.signature {
            if let Err(e) = dump_signature(
                &mut board,
                sig_path.as_path(),
                cli_args.signature_granularity,
            ) {
                log::error!("Failed to dump signature: {}", e);
            }
        }

        drop(board);
        drop(raw_mode_guard);

        println!("Used time: {}s", now.elapsed().as_secs_f32());
    }
}

#[cfg(test)]
mod arg_tests {
    use super::*;

    #[test]
    fn gdb_defaults_to_tcp_port_1234() {
        let args = Args::try_parse_from(["here", "program.elf", "-G"]).unwrap();

        assert_eq!(gdb_config(&args), Some(gdb::Config::Tcp(1234)));
    }

    #[test]
    fn non_terminal_rvdb_is_script_only() {
        let args = Args::try_parse_from([
            "here",
            "program.bin",
            "--debug",
            "--script",
            "commands.rvdb",
        ])
        .unwrap();

        assert_eq!(uart_io_mode(&args), UartIoMode::Stdio);
        assert_eq!(rvdb_mode(&args, false, false), RvdbMode::ScriptOnly);
    }

    #[test]
    fn non_terminal_rvdb_without_script_is_rejected() {
        let args = Args::try_parse_from(["here", "program.bin", "--debug"]).unwrap();

        assert_eq!(rvdb_mode(&args, false, false), RvdbMode::MissingScript);
    }

    #[test]
    fn terminal_rvdb_stays_interactive_after_optional_script() {
        let args = Args::try_parse_from([
            "here",
            "program.bin",
            "--debug",
            "--script",
            "commands.rvdb",
        ])
        .unwrap();

        assert_eq!(rvdb_mode(&args, true, true), RvdbMode::Interactive);
    }

    #[test]
    fn gdb_tcp_port_is_configurable() {
        let args =
            Args::try_parse_from(["here", "program.elf", "-G", "--gdb-port", "4321"]).unwrap();

        assert_eq!(gdb_config(&args), Some(gdb::Config::Tcp(4321)));
    }

    #[cfg(unix)]
    #[test]
    fn gdb_uds_path_is_configurable() {
        let args = Args::try_parse_from([
            "here",
            "program.elf",
            "-G",
            "--gdb-uds",
            "/tmp/riscv-gdb.sock",
        ])
        .unwrap();

        assert_eq!(
            gdb_config(&args),
            Some(gdb::Config::Uds(std::path::PathBuf::from(
                "/tmp/riscv-gdb.sock"
            )))
        );
    }

    #[cfg(unix)]
    #[test]
    fn gdb_tcp_and_uds_are_mutually_exclusive() {
        let error = Args::try_parse_from([
            "here",
            "program.elf",
            "-G",
            "--gdb-port",
            "4321",
            "--gdb-uds",
            "/tmp/riscv-gdb.sock",
        ])
        .unwrap_err();

        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }
}
