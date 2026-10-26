mod commands;
mod printer;
mod session;

mod nostd_repl;

#[cfg(feature = "native-cli")]
mod native_repl;
#[cfg(feature = "native-cli")]
mod sync_repl;

use crate::config::arch_config::REGFILE_CNT;
use crate::config::arch_config::WordType;
use crate::isa::riscv::RawInstr;
use crate::isa::riscv::csr_reg::PrivilegeLevel;
use crate::isa::riscv::debugger;
use crate::isa::riscv::mmu::AccessType;
use crate::isa::riscv::{address::Address, decoder::DecodeInstr};
use clap::error::ErrorKind;
use clap::{Parser, Subcommand};
use std::fmt;

pub use printer::Printer;
pub use session::{RvdbCommandResponse, RvdbSession};

pub use nostd_repl::{AsyncREPL, REPLResponse, RvdbChannelRx, RvdbChannelTx, RvdbChannels};

#[cfg(feature = "native-cli")]
pub use native_repl::RustylineREPL;
#[cfg(feature = "native-cli")]
pub use sync_repl::SyncREPL;

const PROMPT: &str = "(rvdb) ";

#[derive(clap::ValueEnum, Debug, Clone, PartialEq, Eq)]
enum ClapAccessType {
    Read,
    Write,
}

impl fmt::Display for ClapAccessType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClapAccessType::Read => f.write_str("read"),
            ClapAccessType::Write => f.write_str("write"),
        }
    }
}

impl From<ClapAccessType> for AccessType {
    fn from(value: ClapAccessType) -> Self {
        match value {
            ClapAccessType::Read => AccessType::Read,
            ClapAccessType::Write => AccessType::Write,
        }
    }
}

fn is_clap_display(error: &clap::Error) -> bool {
    matches!(
        error.kind(),
        ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
    )
}

fn format_clap_error(error: clap::Error) -> String {
    let message = error.to_string();
    message
        .strip_prefix("error: ")
        .unwrap_or(&message)
        .to_string()
}

#[derive(Debug, PartialEq, Eq, Parser)]
#[command(
    multicall = true,
    about = "Inspect and control a running RISC-V guest.",
    after_help = "Addresses are physical unless --virt is specified.\nEnter an empty line to repeat the previous command."
)]
enum RvdbCommand {
    /// Print registers, memory, and other CPU state.
    #[command(visible_alias = "p", subcommand)]
    Print(PrintCmd),

    /// Print an item whenever guest execution stops.
    #[command(visible_alias = "d", subcommand)]
    Display(PrintCmd),

    /// Translate a virtual address using the current CPU and page-table state.
    #[command(visible_aliases = ["t", "trans"])]
    Translate {
        /// Virtual address to translate. Decimal by default; prefix hexadecimal values with 0x.
        #[arg(value_name = "ADDRESS")]
        addr: String,

        /// Access type used for permission checks.
        #[arg(value_enum, value_name = "ACCESS", default_value_t = ClapAccessType::Read)]
        access: ClapAccessType,
    },

    /// Stop printing an item whenever guest execution stops.
    #[command(subcommand)]
    Undisplay(PrintCmd),

    /// Disassemble instructions starting at the current PC.
    #[command(visible_aliases = ["l", "ls"])]
    List,

    /// Show recently executed instructions.
    #[command(visible_alias = "his")]
    History {
        /// Maximum number of history entries to show.
        #[arg(value_name = "COUNT", default_value_t = 20)]
        count: usize,
    },

    /// Record and inspect function calls and returns.
    #[command(name = "ftrace", visible_alias = "ft", alias = "f-trace", subcommand)]
    FTrace(FTraceCmd),

    /// Load debug symbols from an ELF file.
    #[command(visible_aliases = ["symbol", "file"])]
    SymbolFile {
        /// ELF file containing the symbol table.
        #[arg(value_name = "FILE")]
        path: String,
    },

    /// Execute one guest instruction.
    #[command(name = "step", visible_aliases = ["s", "si"])]
    Si,

    /// Resume guest execution with an optional maximum steps.
    #[command(name = "continue", visible_alias = "c")]
    Continue {
        /// Maximum execution steps before returning; omit to run until a breakpoint or halt.
        #[arg(value_name = "STEPS", default_value_t = u64::MAX, hide_default_value = true)]
        steps: u64,
    },

    /// Set a breakpoint, or delete one with --delete.
    #[command(name = "break", visible_alias = "b")]
    Breakpoint {
        /// Delete the breakpoint instead of setting it.
        #[arg(short = 'd', long = "delete")]
        delete: bool,

        /// Guest address or function name. Decimal by default; prefix hexadecimal values with 0x.
        #[arg(value_name = "TARGET")]
        symbol: String,

        /// Treat an address as virtual; addresses are physical by default.
        #[arg(short, long, default_value_t = false)]
        virt: bool,
    },

    /// Show debugger state such as breakpoints and loaded symbols.
    #[command(subcommand)]
    Info(InfoCmd),

    /// Quit rvdb.
    #[command(name = "quit", visible_aliases = ["q", "exit"])]
    Quit,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum PrintCmd {
    /// Print the current program counter.
    Pc,

    /// Print one general-purpose register.
    Reg {
        /// Register name (for example, a0 or x10).
        #[arg(value_name = "REGISTER")]
        reg: String,
    },

    /// Print a range of general-purpose registers.
    Regs {
        /// Index of the first register to print.
        #[arg(long, value_name = "INDEX", default_value_t = 0)]
        start: u8,

        /// Maximum number of registers to print.
        #[arg(short, long, value_name = "COUNT", default_value_t = REGFILE_CNT as u8)]
        len: u8,
    },

    /// Read bytes from guest memory.
    Mem {
        /// Starting address. Decimal by default; prefix hexadecimal values with 0x.
        #[arg(value_name = "ADDRESS")]
        addr: String,

        /// Number of bytes to read.
        #[arg(short, long, value_name = "BYTES", default_value_t = 16)]
        len: u32,

        /// Treat the address as virtual; addresses are physical by default.
        #[arg(short, long, default_value_t = false)]
        virt: bool,
    },

    /// Print a control and status register.
    Csr {
        /// CSR name or address (for example, mstatus or 0x300).
        #[arg(value_name = "CSR")]
        addr: String,
    },

    /// Print one floating-point register.
    #[command(name = "freg", alias = "f-reg")]
    FReg {
        /// Register name, ABI name, or f-register number (for example, fa0 or f10).
        #[arg(value_name = "REGISTER")]
        reg: String,
    },

    /// Print one vector register in several element widths.
    #[command(name = "vreg", alias = "v-reg")]
    VReg {
        /// Vector register name (for example, v0 or v31).
        #[arg(value_name = "REGISTER")]
        reg: String,
    },

    /// Print the current privilege level.
    Priv,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum InfoCmd {
    /// List active breakpoints.
    #[command(visible_aliases = ["b", "bp", "break"])]
    Breakpoints,

    /// List symbols loaded from the current ELF or a symbol file.
    #[command(visible_aliases = ["sym", "symbol"])]
    Symbols,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum FTraceCmd {
    /// Start recording function calls and returns.
    Start,

    /// Stop recording function calls and returns.
    Stop,

    /// Show the most recent function calls and returns.
    Show {
        /// Maximum number of trace entries to show.
        #[arg(value_name = "COUNT", default_value_t = 20)]
        count: usize,
    },

    /// Show trace status and per-function call counts.
    Stat,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PrintObject {
    Pc,
    Reg(u8),
    Regs(u8, u8),
    Mem(u64, u32, bool), // addr, len, is_virt
    CSR(WordType),
    FReg(u8),
    VReg(u8), // index
    Privilege,
}

#[derive(Debug, PartialEq)]
pub(crate) struct DbgInstrLine {
    pub addr: u64,
    pub raw: Option<RawInstr>,
    pub decoded: Option<DecodeInstr>,
    pub symbol: Option<String>,
    pub is_current_pc: bool,
}

#[derive(Debug, PartialEq)]
pub(crate) enum CommandOutput {
    None,
    Exit,

    Pc(WordType),
    Reg {
        name: String,
        val: WordType,
    },
    Regs(Vec<(&'static str, WordType)>),
    FReg {
        name: String,
        f32_val: f32,
        f64_val: f64,
    },
    VReg {
        name: String,
        val: Vec<(String, Vec<WordType>)>,
    },
    Csr {
        name: String,
        val: Option<WordType>,
    },
    Mem {
        addr: Address,
        data: Vec<Option<u8>>,
    },

    Translate {
        virt_addr: WordType,
        phys_addr: u64,
    },

    Privilege(PrivilegeLevel),

    History(Vec<DbgInstrLine>),
    CodeList(Vec<DbgInstrLine>),
    Breakpoints(Vec<debugger::Breakpoint>),
    Symbols(Vec<(String, WordType)>),
    FTraceShow(Vec<debugger::FuncTrace>),
    FTraceStat(debugger::FtraceStatsSnapshot),
    FTraceStatus {
        enabled: bool,
    },

    ContinueDone {
        instr: DbgInstrLine,
        watch_results: Vec<CommandOutput>,
        event: debugger::DebugEvent,
        actual_steps: u64,
    },

    BreakpointSet {
        ok: bool,
        addr: Address,
        symbol: Option<String>,
    },
    BreakpointCleared {
        ok: bool,
        addr: Address,
        symbol: Option<String>,
    },
}
