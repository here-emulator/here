use super::{CommandOutput, DbgInstrLine};
use crate::{
    config::arch_config::{REG_NAME, SignedWordType, WordType},
    isa::riscv::{
        RawInstr,
        address::Address,
        csr_reg::{PrivilegeLevel, csr_macro::CSR_NAME},
        debugger,
        decoder::DecodeInstr,
        instruction::{RVInstrInfo, instr_table::RiscvInstr},
    },
};

#[derive(Clone, Copy)]
enum AnsiColor {
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    BrightBlack,
}

impl AnsiColor {
    const fn sgr_code(self) -> u8 {
        match self {
            AnsiColor::Red => 31,
            AnsiColor::Green => 32,
            AnsiColor::Yellow => 33,
            AnsiColor::Blue => 34,
            AnsiColor::Magenta => 35,
            AnsiColor::Cyan => 36,
            AnsiColor::BrightBlack => 90,
        }
    }
}

#[derive(Clone, Copy)]
struct AnsiStyle {
    foreground: AnsiColor,
}

impl AnsiStyle {
    const fn foreground(color: AnsiColor) -> Self {
        Self { foreground: color }
    }

    fn paint(self, value: &str) -> String {
        let code = self.foreground.sgr_code();
        format!("\x1b[{code}m{value}\x1b[0m")
    }
}

struct OutputPalette {
    ansi_color: bool,
    index: AnsiStyle,
    address: AnsiStyle,
    register: AnsiStyle,
    csr: AnsiStyle,
    instruction: AnsiStyle,
    arrow: AnsiStyle,
    data: AnsiStyle,
    identifier: AnsiStyle,
    invalid: AnsiStyle,
    privilege: AnsiStyle,
}

impl OutputPalette {
    const fn new(ansi_color: bool) -> Self {
        use AnsiColor::*;

        Self {
            ansi_color,
            index: AnsiStyle::foreground(Yellow),
            address: AnsiStyle::foreground(Blue),
            register: AnsiStyle::foreground(Magenta),
            csr: AnsiStyle::foreground(BrightBlack),
            instruction: AnsiStyle::foreground(Green),
            arrow: AnsiStyle::foreground(Cyan),
            data: AnsiStyle::foreground(Yellow),
            identifier: AnsiStyle::foreground(Yellow),
            invalid: AnsiStyle::foreground(Red),
            privilege: AnsiStyle::foreground(BrightBlack),
        }
    }

    fn paint(&self, value: &str, style: AnsiStyle) -> String {
        if self.ansi_color {
            style.paint(value)
        } else {
            value.to_string()
        }
    }

    fn index(&self, index: &str) -> String {
        self.paint(index, self.index)
    }

    fn addr(&self, addr: &str) -> String {
        self.paint(addr, self.address)
    }

    fn reg(&self, reg: &str, padding: usize) -> String {
        self.paint(&format!("{:<width$}", reg, width = padding), self.register)
    }

    fn csr(&self, csr: &str) -> String {
        self.paint(csr, self.csr)
    }

    fn instr(&self, instr: impl AsRef<str>) -> String {
        self.paint(instr.as_ref(), self.instruction)
    }

    fn arrow(&self, ch: &str) -> String {
        self.paint(ch, self.arrow)
    }

    fn data(&self, value: &str) -> String {
        self.paint(value, self.data)
    }

    fn identifier(&self, value: &str) -> String {
        self.paint(value, self.identifier)
    }

    fn invalid(&self, value: &str) -> String {
        self.paint(value, self.invalid)
    }

    fn privilege(&self, value: &str) -> String {
        self.paint(value, self.privilege)
    }
}

pub struct Printer {
    output_palette: OutputPalette,
}

impl Printer {
    pub fn plain() -> Self {
        Self {
            output_palette: OutputPalette::new(false),
        }
    }

    pub fn ansi_color() -> Self {
        Self {
            output_palette: OutputPalette::new(true),
        }
    }

    pub(crate) fn render(&self, output: &CommandOutput) -> String {
        self.render_with_active_palette(output)
    }

    #[cfg(feature = "native-cli")]
    pub(crate) fn print(&self, output: &CommandOutput) {
        print!("{}", self.render(output));
    }

    fn render_with_active_palette(&self, output: &CommandOutput) -> String {
        use std::fmt::Write;

        let mut text = String::new();
        let palette = &self.output_palette;

        match output {
            CommandOutput::None => {}
            CommandOutput::Exit => {}

            CommandOutput::Pc(pc) => {
                writeln!(text, "pc = {}", palette.format_addr(*pc)).unwrap();
            }
            CommandOutput::Reg { name, val } => {
                writeln!(
                    text,
                    "{} = {}",
                    palette.reg(name, 3),
                    palette.format_data(*val)
                )
                .unwrap();
            }
            CommandOutput::Regs(regs) => {
                for (idx, (name, val)) in regs.iter().enumerate() {
                    writeln!(
                        text,
                        "x{:<3} {} = {}",
                        idx,
                        palette.reg(name, 5),
                        palette.format_data(*val)
                    )
                    .unwrap();
                }
            }
            CommandOutput::FReg {
                name,
                f32_val,
                f64_val,
            } => {
                writeln!(
                    text,
                    "{} = {{f32: {}, f64: {}}}",
                    palette.reg(name, 0),
                    f32_val,
                    f64_val,
                )
                .unwrap();
            }
            CommandOutput::VReg { name, val } => {
                writeln!(text, "{} = {{", palette.reg(name, 0)).unwrap();
                val.iter().for_each(|(c, data)| {
                    writeln!(
                        text,
                        "\t{} = {}, ",
                        palette.data(c),
                        palette.data(format!("{:?}", data).as_str())
                    )
                    .unwrap();
                });
                text.push_str("\n}\n");
            }
            CommandOutput::Translate {
                virt_addr,
                phys_addr,
            } => {
                writeln!(
                    text,
                    "{} -> {}",
                    palette.format_addr(*virt_addr),
                    palette.format_addr(*phys_addr)
                )
                .unwrap();
            }
            CommandOutput::Csr { name, val } => {
                if let Some(v) = val {
                    #[cfg(feature = "riscv64")]
                    writeln!(text, "{} = {}", name, palette.format_data_64(*v)).unwrap();
                    #[cfg(feature = "riscv32")]
                    writeln!(text, "{} = {}", name, palette.format_data(*v)).unwrap();
                } else {
                    text.push_str("Illegal CSR.\n");
                }
            }
            CommandOutput::Mem { addr, data } => {
                const BYTE_PER_LINE: u32 = 16;
                let mut curr_addr = *addr;
                let mut i = 0;
                let len = data.len() as u32;

                while i < len {
                    if i % BYTE_PER_LINE == 0 {
                        write!(text, "{}: ", palette.format_address(curr_addr)).unwrap();
                    }

                    if let Some(byte) = data[i as usize] {
                        write!(text, "{:02x} ", byte).unwrap();
                    } else {
                        text.push_str("?? ");
                    }

                    curr_addr = curr_addr + 1;
                    i += 1;
                    if i % BYTE_PER_LINE == 0 {
                        text.push('\n');
                    }
                }
                if len > 0 && len % BYTE_PER_LINE != 0 {
                    text.push('\n');
                }
            }
            CommandOutput::Privilege(privilege) => {
                writeln!(text, "{}", palette.format_privilege(*privilege)).unwrap();
            }

            CommandOutput::History(history) => {
                for (i, line) in history.iter().enumerate() {
                    writeln!(
                        text,
                        "  [{}] {}",
                        palette.format_idx(i),
                        palette.format_instr(line)
                    )
                    .unwrap();
                }
            }
            CommandOutput::CodeList(lines) => {
                for line in lines {
                    if line.is_current_pc {
                        write!(text, "{} ", palette.arrow(">")).unwrap();
                    } else {
                        text.push_str("  ");
                    }

                    writeln!(text, "{}", palette.format_instr_detailed(line)).unwrap();
                }
            }
            CommandOutput::Breakpoints(bps) => {
                for bp in bps {
                    writeln!(
                        text,
                        "{}: {}",
                        palette.format_idx(bp.id),
                        palette.format_address(bp.addr)
                    )
                    .unwrap();
                }
            }
            CommandOutput::Symbols(symbols) => {
                for (name, addr) in symbols {
                    writeln!(
                        text,
                        "{}: {}",
                        palette.format_addr(*addr),
                        palette.identifier(name)
                    )
                    .unwrap();
                }
            }

            CommandOutput::FTraceShow(traces) => {
                for trace in traces {
                    match trace {
                        debugger::FuncTrace::Call { name, addr } => {
                            let name = name.clone().unwrap_or("???".to_string());
                            writeln!(
                                text,
                                "Call   -> [{}@{}]",
                                palette.identifier(&name),
                                palette.format_addr(*addr)
                            )
                            .unwrap();
                        }
                        debugger::FuncTrace::Return { name, addr } => {
                            let name = name.clone().unwrap_or("???".to_string());
                            writeln!(
                                text,
                                "Return <- [{}@{}]",
                                palette.identifier(&name),
                                palette.format_addr(*addr)
                            )
                            .unwrap();
                        }
                    }
                }
            }
            CommandOutput::FTraceStat(stats) => {
                writeln!(
                    text,
                    "ftrace: {}",
                    if stats.enabled { "running" } else { "stopped" }
                )
                .unwrap();
                writeln!(
                    text,
                    "queue: {} / {}",
                    stats.queue_len,
                    debugger::MAX_FTRACE
                )
                .unwrap();
                writeln!(text, "calls: {}", stats.call_count).unwrap();
                writeln!(text, "returns: {}", stats.return_count).unwrap();
                writeln!(text, "unknown calls: {}", stats.unknown_calls).unwrap();
                writeln!(text, "unknown returns: {}", stats.unknown_returns).unwrap();

                if !stats.per_func.is_empty() {
                    text.push_str("function stats:\n");
                    let mut per_func = stats.per_func.clone().into_iter().collect::<Vec<_>>();
                    per_func.sort_by_key(|(_, e)| e.calls + e.returns);
                    for (name, entry) in per_func.into_iter().rev() {
                        writeln!(
                            text,
                            "{} calls={:<5} returns={:<5}",
                            palette.identifier(&format!("{:<32}", name)),
                            entry.calls,
                            entry.returns,
                        )
                        .unwrap();
                    }
                }
            }
            CommandOutput::FTraceStatus { enabled } => {
                writeln!(
                    text,
                    "ftrace {}",
                    if *enabled { "started" } else { "stopped" }
                )
                .unwrap();
            }

            CommandOutput::ContinueDone {
                instr,
                watch_results,
                event,
                actual_steps: steps,
            } => {
                match event {
                    debugger::DebugEvent::StepCompleted => {
                        writeln!(text, "Completed, next: {}", palette.format_instr(instr)).unwrap();
                    }
                    debugger::DebugEvent::BreakpointHit => {
                        writeln!(
                            text,
                            "Breakpoint hit after {} steps: {}",
                            steps,
                            palette.format_instr(instr)
                        )
                        .unwrap();
                    }
                    debugger::DebugEvent::BoardHalted => {
                        if *steps == 0 {
                            text.push_str("Board already halted\n");
                        } else {
                            writeln!(
                                text,
                                "Board halted after {} steps: {}",
                                steps,
                                palette.format_instr(instr)
                            )
                            .unwrap();
                        }
                    }
                }
                for res in watch_results {
                    text.push_str(&self.render_with_active_palette(res));
                }
            }

            CommandOutput::BreakpointSet { ok, addr, symbol } => {
                let addr_text = palette.format_address(*addr).to_string();
                if *ok {
                    if let Some(sym) = symbol {
                        writeln!(text, "Breakpoint set at {} <{}>", sym, addr_text).unwrap();
                    } else {
                        writeln!(text, "Breakpoint set at {}", addr_text).unwrap();
                    }
                } else {
                    writeln!(text, "Breakpoint already exists at {}", addr_text).unwrap();
                }
            }
            CommandOutput::BreakpointCleared { ok, addr, symbol } => {
                let addr_text = palette.format_address(*addr).to_string();
                if *ok {
                    if let Some(sym) = symbol {
                        writeln!(text, "Breakpoint removed at {} <{}>", sym, addr_text).unwrap();
                    } else {
                        writeln!(text, "Breakpoint removed at {}", addr_text).unwrap();
                    }
                } else {
                    writeln!(text, "Breakpoint not found at {}", addr_text).unwrap();
                }
            }
        }

        text
    }
}

impl OutputPalette {
    fn format_idx(&self, idx: usize) -> impl std::fmt::Display {
        self.index(&format!("{:2}", idx)).to_string()
    }

    fn format_addr(&self, word: WordType) -> impl std::fmt::Display {
        self.addr(&format!("0x{:08x}", word)).to_string()
    }

    fn format_address(&self, addr: Address) -> impl std::fmt::Display {
        match addr {
            Address::Phys(addr) => format!("paddr({})", self.format_addr(addr)),
            Address::Virt(addr) => format!("vaddr({})", self.format_addr(addr)),
        }
    }

    fn format_data(&self, data: WordType) -> impl std::fmt::Display {
        self.data(&format!("0x{:08x}", data)).to_string()
    }

    fn format_privilege(&self, privilege: PrivilegeLevel) -> impl std::fmt::Display {
        self.privilege(&format!("{:?}", privilege)).to_string()
    }

    fn format_data_64(&self, data: WordType) -> impl std::fmt::Display {
        self.data(&format!("0x{:016x}", data)).to_string()
    }

    fn format_instr(&self, instr: &DbgInstrLine) -> impl std::fmt::Display {
        if let Some(symbol) = &instr.symbol {
            format!(
                "{}: {} {}",
                self.format_addr(instr.addr),
                self.format_asm(instr.decoded),
                self.identifier(&symbol)
            )
        } else {
            format!(
                "{}: {}",
                self.format_addr(instr.addr),
                self.format_asm(instr.decoded)
            )
        }
    }

    fn format_instr_detailed(&self, instr: &DbgInstrLine) -> impl std::fmt::Display {
        if let Some(symbol) = &instr.symbol {
            format!(
                "{}: {} {} {}",
                self.format_addr(instr.addr),
                self.format_raw(instr.raw),
                self.format_asm(instr.decoded),
                self.identifier(&symbol)
            )
        } else {
            format!(
                "{}: {} {}",
                self.format_addr(instr.addr),
                self.format_raw(instr.raw),
                self.format_asm(instr.decoded)
            )
        }
    }

    fn format_raw(&self, raw: Option<RawInstr>) -> impl std::fmt::Display {
        use crate::isa::InstrLen;
        match raw {
            Some(raw) if raw.len() == 2 => self.data(&format!("0x{:04x}    ", raw.val)).to_string(),
            Some(raw) => self.data(&format!("0x{:08x}", raw.val)).to_string(),
            None => self.invalid("<invalid>").to_string(),
        }
    }

    fn format_imm(&self, imm: WordType) -> String {
        self.data(&(imm as SignedWordType).to_string())
    }

    fn format_csr_name(&self, addr: WordType) -> String {
        match CSR_NAME.get(&addr) {
            Some(name) => self.csr(name),
            None => self.csr(&format!("csr[0x{addr:03x}]")),
        }
    }

    fn format_pseudo_asm(&self, instr: RiscvInstr, info: RVInstrInfo) -> Option<String> {
        use RVInstrInfo as Info;
        use RiscvInstr::*;

        let reg = |index: u8| self.reg(REG_NAME[index as usize], 0);
        let mnemonic = |name: &str| self.instr(name);

        match (instr, info) {
            (
                ADDI,
                Info::I {
                    rd: 0,
                    rs1: 0,
                    imm: 0,
                },
            ) => Some(mnemonic("nop")),
            (ADDI, Info::I { rd, rs1, imm: 0 }) if rd != 0 && rs1 != 0 => {
                Some(format!("{} {}, {}", mnemonic("mv"), reg(rd), reg(rs1),))
            }
            (ADDI, Info::I { rd, rs1: 0, imm }) if rd != 0 => Some(format!(
                "{} {}, {}",
                mnemonic("li"),
                reg(rd),
                self.format_imm(imm),
            )),
            (XORI, Info::I { rd, rs1, imm }) if rd != 0 && imm == WordType::MAX => {
                Some(format!("{} {}, {}", mnemonic("not"), reg(rd), reg(rs1),))
            }
            (ADDIW, Info::I { rd, rs1, imm: 0 }) if rd != 0 => {
                Some(format!("{} {}, {}", mnemonic("sext.w"), reg(rd), reg(rs1),))
            }
            (SLTIU, Info::I { rd, rs1, imm: 1 }) if rd != 0 => {
                Some(format!("{} {}, {}", mnemonic("seqz"), reg(rd), reg(rs1),))
            }
            (SLTU, Info::R { rd, rs1: 0, rs2 }) if rd != 0 => {
                Some(format!("{} {}, {}", mnemonic("snez"), reg(rd), reg(rs2),))
            }
            (SLT, Info::R { rd, rs1, rs2: 0 }) if rd != 0 => {
                Some(format!("{} {}, {}", mnemonic("sltz"), reg(rd), reg(rs1),))
            }
            (SLT, Info::R { rd, rs1: 0, rs2 }) if rd != 0 => {
                Some(format!("{} {}, {}", mnemonic("sgtz"), reg(rd), reg(rs2),))
            }
            (JAL, Info::J { rd: 0, imm }) => {
                Some(format!("{} {}", mnemonic("j"), self.format_imm(imm)))
            }
            (
                JALR,
                Info::I {
                    rd: 0,
                    rs1: 1,
                    imm: 0,
                },
            ) => Some(mnemonic("ret")),
            (JALR, Info::I { rd: 0, rs1, imm: 0 }) => {
                Some(format!("{} {}", mnemonic("jr"), reg(rs1)))
            }
            (CSRRWI | CSRRSI | CSRRCI, Info::I { rd: 0, rs1, imm }) => {
                let name = match instr {
                    CSRRWI => "csrwi",
                    CSRRSI => "csrsi",
                    CSRRCI => "csrci",
                    _ => unreachable!(),
                };
                Some(format!(
                    "{} {}, {}",
                    mnemonic(name),
                    self.format_csr_name(imm),
                    self.data(&rs1.to_string()),
                ))
            }
            (CSRRS, Info::I { rd, rs1: 0, imm }) => Some(format!(
                "{} {}, {}",
                mnemonic("csrr"),
                reg(rd),
                self.format_csr_name(imm),
            )),
            (CSRRW | CSRRS | CSRRC, Info::I { rd: 0, rs1, imm }) => {
                let name = match instr {
                    CSRRW => "csrw",
                    CSRRS => "csrs",
                    CSRRC => "csrc",
                    _ => unreachable!(),
                };
                Some(format!(
                    "{} {}, {}",
                    mnemonic(name),
                    self.format_csr_name(imm),
                    reg(rs1),
                ))
            }
            (SUB | SUBW, Info::R { rs1: 0, rs2, rd }) => Some(format!(
                "{} {}, {}",
                mnemonic(if instr == SUB { "neg" } else { "negw" }),
                reg(rd),
                reg(rs2),
            )),
            (BEQ | BNE | BLT | BGE, Info::B { rs1, rs2: 0, imm }) => {
                let name = match instr {
                    BEQ => "beqz",
                    BNE => "bnez",
                    BLT => "bltz",
                    BGE => "bgez",
                    _ => unreachable!(),
                };
                Some(format!(
                    "{} {}, {}",
                    mnemonic(name),
                    reg(rs1),
                    self.format_imm(imm),
                ))
            }
            (BLT | BGE, Info::B { rs1: 0, rs2, imm }) => Some(format!(
                "{} {}, {}",
                mnemonic(if instr == BLT { "bgtz" } else { "blez" }),
                reg(rs2),
                self.format_imm(imm),
            )),
            _ => None,
        }
    }

    fn format_asm(&self, decode_instr: Option<DecodeInstr>) -> impl std::fmt::Display {
        let Some(DecodeInstr { instr, info, .. }) = decode_instr else {
            return self.invalid("<invalid instruction>");
        };
        if let Some(pseudo) = self.format_pseudo_asm(instr, info) {
            return pseudo;
        }

        use RiscvInstr::*;

        match info {
            RVInstrInfo::I { rd, rs1, imm } => match instr {
                CSRRC | CSRRS | CSRRW => {
                    format!(
                        "{} {}, {}, {}",
                        self.instr(instr.name()),
                        self.reg(REG_NAME[rd as usize], 0),
                        self.format_csr_name(imm),
                        self.reg(REG_NAME[rs1 as usize], 0),
                    )
                }
                CSRRCI | CSRRSI | CSRRWI => {
                    format!(
                        "{} {}, {}, {}",
                        self.instr(instr.name()),
                        self.reg(REG_NAME[rd as usize], 0),
                        self.format_csr_name(imm),
                        self.data(rs1.to_string().as_str()),
                    )
                }
                LB | LBU | LH | LHU | LW | LWU | LD | FLW | FLD | JALR => format!(
                    "{} {}, {}({})",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd as usize], 0),
                    self.format_imm(imm),
                    self.reg(REG_NAME[rs1 as usize], 0),
                ),
                _ => format!(
                    "{} {}, {}, {}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd as usize], 0),
                    self.reg(REG_NAME[rs1 as usize], 0),
                    self.format_imm(imm),
                ),
            },

            RVInstrInfo::R { rs1, rs2, rd } => {
                format!(
                    "{} {}, {}, {}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd as usize], 0),
                    self.reg(REG_NAME[rs1 as usize], 0),
                    self.reg(REG_NAME[rs2 as usize], 0)
                )
            }

            RVInstrInfo::R_rm { rs1, rs2, rd, rm } => {
                format!(
                    "{} {}, {}, {}, rm={}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd as usize], 0),
                    self.reg(REG_NAME[rs1 as usize], 0),
                    self.reg(REG_NAME[rs2 as usize], 0),
                    rm,
                )
            }

            RVInstrInfo::R4_rm {
                rs1,
                rs2,
                rs3,
                rd,
                rm,
            } => {
                format!(
                    "{} {}, {}, {}, {}, rm={}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd as usize], 0),
                    self.reg(REG_NAME[rs1 as usize], 0),
                    self.reg(REG_NAME[rs2 as usize], 0),
                    self.reg(REG_NAME[rs3 as usize], 0),
                    rm
                )
            }

            RVInstrInfo::A {
                rs1,
                rs2,
                rd,
                rl,
                aq,
            } => {
                if instr.name().starts_with("amo") {
                    format!(
                        "{} {}, {}, ({}) rl={}, aq={}",
                        self.instr(instr.name()),
                        self.reg(REG_NAME[rd as usize], 0),
                        self.reg(REG_NAME[rs2 as usize], 0),
                        self.reg(REG_NAME[rs1 as usize], 0),
                        rl,
                        aq,
                    )
                } else if instr.name().starts_with("lr") {
                    format!(
                        "{} {}, ({}) rl={}, aq={}",
                        self.instr(instr.name()),
                        self.reg(REG_NAME[rd as usize], 0),
                        self.reg(REG_NAME[rs1 as usize], 0),
                        rl,
                        aq,
                    )
                } else {
                    format!(
                        "{} {}, {}, ({}) rl={}, aq={}",
                        self.instr(instr.name()),
                        self.reg(REG_NAME[rd as usize], 0),
                        self.reg(REG_NAME[rs2 as usize], 0),
                        self.reg(REG_NAME[rs1 as usize], 0),
                        rl,
                        aq,
                    )
                }
            }

            RVInstrInfo::B { rs1, rs2, imm } => {
                format!(
                    "{} {}, {}, {}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rs1 as usize], 0),
                    self.reg(REG_NAME[rs2 as usize], 0),
                    self.format_imm(imm)
                )
            }

            RVInstrInfo::J { rd, imm } => {
                format!(
                    "{} {}, {}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd as usize], 0),
                    self.format_imm(imm)
                )
            }

            RVInstrInfo::S { rs1, rs2, imm } => {
                format!(
                    "{} {}, {}({})",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rs2 as usize], 0),
                    self.format_imm(imm),
                    self.reg(REG_NAME[rs1 as usize], 0),
                )
            }
            RVInstrInfo::U { rd, imm } => {
                format!(
                    "{} {}, {}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd as usize], 0),
                    self.data((imm >> 12).to_string().as_str())
                )
            }
            RVInstrInfo::V {
                rs1, rs2, rd, vm, ..
            } => {
                let vector_reg_name = |id: u8| -> String { "v".to_string() + &id.to_string() };
                format!(
                    "{} {}, {}, {}, {}",
                    self.instr(instr.name()),
                    self.reg(vector_reg_name(rd).as_str(), 0),
                    self.reg(vector_reg_name(rs1).as_str(), 0),
                    self.reg(vector_reg_name(rs2).as_str(), 0),
                    self.data(if vm { "vm" } else { "" }),
                )
            }

            RVInstrInfo::CR { rd_rs1, rs2 } => {
                format!(
                    "{} {}, {}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd_rs1 as usize], 0),
                    self.reg(REG_NAME[rs2 as usize], 0),
                )
            }

            RVInstrInfo::CI { rd_rs1, imm } => {
                format!(
                    "{} {}, {}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd_rs1 as usize], 0),
                    self.format_imm(imm),
                )
            }

            RVInstrInfo::CSS { rs2, imm } => {
                format!(
                    "{} {}, {}(sp)",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rs2 as usize], 0),
                    self.format_imm(imm),
                )
            }

            RVInstrInfo::CIW { rd, imm } => {
                format!(
                    "{} {}, {}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd as usize], 0),
                    self.format_imm(imm),
                )
            }

            RVInstrInfo::CL { rd, rs1, imm } => {
                format!(
                    "{} {}, {}({})",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd as usize], 0),
                    self.format_imm(imm),
                    self.reg(REG_NAME[rs1 as usize], 0),
                )
            }

            RVInstrInfo::CS { rs1, rs2, imm } => {
                format!(
                    "{} {}, {}({})",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rs2 as usize], 0),
                    self.format_imm(imm),
                    self.reg(REG_NAME[rs1 as usize], 0),
                )
            }

            RVInstrInfo::CA { rd_rs1, rs2 } => {
                format!(
                    "{} {}, {}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd_rs1 as usize], 0),
                    self.reg(REG_NAME[rs2 as usize], 0),
                )
            }

            RVInstrInfo::CB { rd_rs1, imm } => {
                format!(
                    "{} {}, {}",
                    self.instr(instr.name()),
                    self.reg(REG_NAME[rd_rs1 as usize], 0),
                    self.format_imm(imm),
                )
            }

            RVInstrInfo::CJ { target } => {
                format!("{} {}", self.instr(instr.name()), self.format_imm(target),)
            }

            RVInstrInfo::None => {
                format!("{}", self.instr(instr.name()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use RVInstrInfo as Info;
    use RiscvInstr::*;

    fn asm(instr: RiscvInstr, info: Info) -> String {
        Printer::plain()
            .output_palette
            .format_asm(Some(DecodeInstr {
                instr,
                info,
                len: 4,
            }))
            .to_string()
    }

    #[test]
    fn format_asm_uses_conventional_memory_operands() {
        let negative_offset = (-16 as SignedWordType) as WordType;

        assert_eq!(
            asm(
                LW,
                Info::I {
                    rd: 10,
                    rs1: 2,
                    imm: negative_offset,
                },
            ),
            "lw a0, -16(sp)"
        );
        assert_eq!(
            asm(
                SW,
                Info::S {
                    rs1: 2,
                    rs2: 10,
                    imm: 12,
                },
            ),
            "sw a0, 12(sp)"
        );
    }

    #[test]
    fn format_asm_keeps_decoded_branch_and_jump_offsets() {
        assert_eq!(
            asm(
                BEQ,
                Info::B {
                    rs1: 10,
                    rs2: 11,
                    imm: 24,
                },
            ),
            "beq a0, a1, 24"
        );
        assert_eq!(asm(JAL, Info::J { rd: 1, imm: 4096 }), "jal ra, 4096");
    }

    #[test]
    fn format_raw_aligns_compressed_and_regular_instructions() {
        let palette = OutputPalette::new(false);

        assert_eq!(
            palette.format_raw(Some(0x0001.into())).to_string(),
            "0x0001    "
        );
        assert_eq!(
            palette.format_raw(Some(0x0000_0013.into())).to_string(),
            "0x00000013"
        );
    }

    #[test]
    fn instruction_names_use_assembly_spelling() {
        assert_eq!(C_ADDI.name(), "c.addi");
    }

    #[test]
    fn format_asm_recognizes_common_pseudo_instructions() {
        assert_eq!(
            asm(
                ADDI,
                Info::I {
                    rd: 0,
                    rs1: 0,
                    imm: 0,
                },
            ),
            "nop"
        );
        assert_eq!(asm(JAL, Info::J { rd: 0, imm: 16 }), "j 16");
        assert_eq!(
            asm(
                JALR,
                Info::I {
                    rd: 0,
                    rs1: 1,
                    imm: 0,
                },
            ),
            "ret"
        );
        assert_eq!(
            asm(
                JALR,
                Info::I {
                    rd: 0,
                    rs1: 5,
                    imm: 0,
                },
            ),
            "jr t0"
        );
        assert_eq!(
            asm(
                SUB,
                Info::R {
                    rd: 10,
                    rs1: 0,
                    rs2: 11,
                },
            ),
            "neg a0, a1"
        );
        assert_eq!(
            asm(
                SUBW,
                Info::R {
                    rd: 10,
                    rs1: 0,
                    rs2: 11,
                },
            ),
            "negw a0, a1"
        );
        for (instr, info, expected) in [
            (
                ADDI,
                Info::I {
                    rd: 10,
                    rs1: 11,
                    imm: 0,
                },
                "mv a0, a1",
            ),
            (
                ADDI,
                Info::I {
                    rd: 10,
                    rs1: 0,
                    imm: 42,
                },
                "li a0, 42",
            ),
            (
                XORI,
                Info::I {
                    rd: 10,
                    rs1: 11,
                    imm: WordType::MAX,
                },
                "not a0, a1",
            ),
            (
                ADDIW,
                Info::I {
                    rd: 10,
                    rs1: 11,
                    imm: 0,
                },
                "sext.w a0, a1",
            ),
            (
                SLTIU,
                Info::I {
                    rd: 10,
                    rs1: 11,
                    imm: 1,
                },
                "seqz a0, a1",
            ),
            (
                SLTU,
                Info::R {
                    rd: 10,
                    rs1: 0,
                    rs2: 11,
                },
                "snez a0, a1",
            ),
            (
                SLT,
                Info::R {
                    rd: 10,
                    rs1: 11,
                    rs2: 0,
                },
                "sltz a0, a1",
            ),
            (
                SLT,
                Info::R {
                    rd: 10,
                    rs1: 0,
                    rs2: 11,
                },
                "sgtz a0, a1",
            ),
        ] {
            assert_eq!(asm(instr, info), expected);
        }
    }

    #[test]
    fn format_asm_recognizes_csr_pseudo_instructions() {
        let csr = 0xabc;

        for (instr, expected) in [
            (CSRRWI, "csrwi csr[0xabc], 3"),
            (CSRRSI, "csrsi csr[0xabc], 3"),
            (CSRRCI, "csrci csr[0xabc], 3"),
        ] {
            assert_eq!(
                asm(
                    instr,
                    Info::I {
                        rd: 0,
                        rs1: 3,
                        imm: csr,
                    },
                ),
                expected
            );
        }

        assert_eq!(
            asm(
                CSRRS,
                Info::I {
                    rd: 10,
                    rs1: 0,
                    imm: csr,
                },
            ),
            "csrr a0, csr[0xabc]"
        );
        for (instr, expected) in [
            (CSRRW, "csrw csr[0xabc], a0"),
            (CSRRS, "csrs csr[0xabc], a0"),
            (CSRRC, "csrc csr[0xabc], a0"),
        ] {
            assert_eq!(
                asm(
                    instr,
                    Info::I {
                        rd: 0,
                        rs1: 10,
                        imm: csr,
                    },
                ),
                expected
            );
        }
    }

    #[test]
    fn format_asm_recognizes_zero_branch_pseudo_instructions() {
        for (instr, expected) in [
            (BEQ, "beqz a0, 12"),
            (BNE, "bnez a0, 12"),
            (BLT, "bltz a0, 12"),
            (BGE, "bgez a0, 12"),
        ] {
            assert_eq!(
                asm(
                    instr,
                    Info::B {
                        rs1: 10,
                        rs2: 0,
                        imm: 12,
                    },
                ),
                expected
            );
        }
        assert_eq!(
            asm(
                BLT,
                Info::B {
                    rs1: 0,
                    rs2: 10,
                    imm: 12,
                },
            ),
            "bgtz a0, 12"
        );
        assert_eq!(
            asm(
                BGE,
                Info::B {
                    rs1: 0,
                    rs2: 10,
                    imm: 12,
                },
            ),
            "blez a0, 12"
        );
    }
}
