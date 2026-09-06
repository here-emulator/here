use std::{
    collections::{BTreeMap, VecDeque},
    fmt::Debug,
    ops::Add,
};

use crate::{
    board::Board,
    config::arch_config::WordType,
    device::MemError,
    isa::{
        DebugTarget, ISATypes,
        riscv::{
            RawInstr, RiscvTypes,
            csr_reg::{NamedCsrReg, PrivilegeLevel, csr_macro::Mcycle},
            decoder::DecodeInstr,
            executor::{ExecutionHook, RVCPU},
            instruction::{RVInstrInfo, instr_table::RiscvInstr},
            mmu::{AccessType, PageTableError},
        },
    },
    load::SymTab,
    utils::UnsignedInteger,
};

#[derive(Debug, Clone, PartialEq)]
pub enum DebugEvent {
    StepCompleted,
    BreakpointHit,
    BoardHalted,
}

#[derive(thiserror::Error, Debug)]
pub enum DebugError {
    #[error("memory error: {0}")]
    MemoryError(MemError),

    #[error("CSR {0} does not exist")]
    CSRNotExist(WordType),

    #[error("symbol {0} not found in symbol table")]
    SymbolNotFound(String),

    #[error("symbol table not available")]
    NoSymbolTable,
}

impl From<MemError> for DebugError {
    fn from(e: MemError) -> Self {
        DebugError::MemoryError(e)
    }
}

// TODO: `DebugTarget` is not needed to be a trait,
// we can directly implement these methods on `RVCPU` and call them in `Debugger`,
// because no more targets will be added in the future (even if one day we will, extract `DebugTarget` trait at that time is easy)
impl DebugTarget<RiscvTypes> for RVCPU {
    fn read_pc(&self) -> WordType {
        self.pc
    }

    fn write_pc(&mut self, new_pc: WordType) {
        self.pc = new_pc;
    }

    fn read_reg(&self, idx: u8) -> WordType {
        self.reg_file[idx as usize]
    }

    fn write_reg(&mut self, idx: u8, value: WordType) {
        self.reg_file.write(idx, value)
    }

    /// Fetch the instruction at `vaddr`, respecting its length.
    ///
    /// The low 16 bits are read first to learn whether the instruction is a
    /// 2-byte RVC instruction or a full 4-byte one. This avoids a misaligned
    /// 32-bit fetch when a 4-byte instruction sits at a 2-byte boundary (e.g.
    /// right after a compressed instruction).
    fn read_instr(&mut self, vaddr: WordType) -> Result<RawInstr, MemError> {
        let lo = self.read_for_debug_ifetch::<u16>(vaddr)? as u32;
        if lo & 0b11 != 0b11 {
            return Ok(RawInstr::from(lo));
        }
        let hi = self.read_for_debug_ifetch::<u16>(vaddr.wrapping_add(2))? as u32;
        Ok(RawInstr::from(lo | (hi << 16)))
    }

    fn read_instr_directly(&mut self, addr: Address) -> Result<RawInstr, MemError> {
        let lo = self.memory.debug_read::<u16>(addr)? as u32;
        if lo & 0b11 != 0b11 {
            return Ok(RawInstr::from(lo));
        }
        let hi = self.memory.debug_read::<u16>(addr + 2)? as u32;
        Ok(RawInstr::from(lo | (hi << 16)))
    }

    fn read_memory<T: UnsignedInteger>(&mut self, addr: Address) -> Result<T, MemError> {
        self.memory.debug_read::<T>(addr)
    }

    fn write_memory<T: UnsignedInteger>(&mut self, addr: Address, data: T) -> Result<(), MemError> {
        self.memory.debug_write::<T>(addr, data)
    }

    fn get_current_privilege(&self) -> PrivilegeLevel {
        self.csr.privelege_level()
    }

    fn read_float_reg(&self, idx: u8) -> (f32, f64) {
        (self.fpu.load::<f32>(idx), self.fpu.load::<f64>(idx))
    }

    fn read_vector_reg<T>(&self, idx: u8) -> Option<&[T]> {
        match self.vector.read_as_type(idx) {
            Ok(data) => Some(data),
            Err(_) => None,
        }
    }

    /// match input {
    ///     Some() => `Read Write`,
    ///     None => `Read Only`,
    /// }
    fn debug_csr(&mut self, addr: WordType, new_value: Option<WordType>) -> Option<WordType> {
        self.csr.debug(addr, new_value)
    }

    fn decoded_instr(&self, instr: RawInstr) -> Option<DecodeInstr> {
        self.decoder.decode(instr)
    }

    fn debug_vaddr_to_paddr(&mut self, vaddr: WordType) -> Result<u64, PageTableError> {
        self.memory.debug_vaddr_to_paddr(vaddr)
    }

    fn debug_translate(
        &mut self,
        vaddr: WordType,
        access: AccessType,
    ) -> Result<u64, PageTableError> {
        self.translate_for_debug(vaddr, access)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Address {
    Virt(WordType),
    Phys(u64),
}

impl Address {
    pub fn value(&self) -> u64 {
        match self.clone() {
            Address::Virt(vaddr) => vaddr as u64,
            Address::Phys(paddr) => paddr,
        }
    }
}

impl Add<u64> for Address {
    type Output = Address;

    fn add(self, rhs: u64) -> Self::Output {
        match self {
            Address::Phys(addr) => Address::Phys(addr + rhs),
            Address::Virt(addr) => Address::Virt(addr + rhs as WordType),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Breakpoint {
    pub id: usize,
    pub addr: Address,
    // TODO: add symbol_name: Option<String> for better user experience
}

#[derive(Debug, Clone, PartialEq)]
pub enum FuncTrace {
    Call { name: Option<String>, addr: u64 },
    Return { name: Option<String>, addr: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FuncTraceStatEntry {
    pub calls: u64,
    pub returns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtraceStatsSnapshot {
    pub enabled: bool,
    pub queue_len: usize,
    pub call_count: u64,
    pub return_count: u64,
    pub unknown_calls: u64,
    pub unknown_returns: u64,
    pub per_func: Vec<(String, FuncTraceStatEntry)>,
}

#[derive(Debug, Default)]
struct FtraceStats {
    call_count: u64,
    return_count: u64,
    unknown_calls: u64,
    unknown_returns: u64,
    per_func: BTreeMap<String, FuncTraceStatEntry>,
}

impl FtraceStats {
    fn clear(&mut self) {
        *self = Self::default();
    }

    fn record(&mut self, trace: &FuncTrace) {
        match trace {
            FuncTrace::Call { name, .. } => {
                self.call_count += 1;
                if let Some(name) = name {
                    self.per_func.entry(name.clone()).or_default().calls += 1;
                } else {
                    self.unknown_calls += 1;
                }
            }
            FuncTrace::Return { name, .. } => {
                self.return_count += 1;
                if let Some(name) = name {
                    self.per_func.entry(name.clone()).or_default().returns += 1;
                } else {
                    self.unknown_returns += 1;
                }
            }
        }
    }

    fn snapshot(&self, enabled: bool, queue_len: usize) -> FtraceStatsSnapshot {
        FtraceStatsSnapshot {
            enabled,
            queue_len,
            call_count: self.call_count,
            return_count: self.return_count,
            unknown_calls: self.unknown_calls,
            unknown_returns: self.unknown_returns,
            per_func: self
                .per_func
                .iter()
                .map(|(name, entry)| (name.clone(), entry.clone()))
                .collect(),
        }
    }
}

#[derive(Debug)]
struct FtraceState {
    enabled: bool,
    queue: VecDeque<FuncTrace>,
    stats: FtraceStats,
}

pub const MAX_FTRACE: usize = 1024;

impl FtraceState {
    fn new() -> Self {
        Self {
            enabled: false,
            queue: VecDeque::with_capacity(MAX_FTRACE),
            stats: FtraceStats::default(),
        }
    }

    fn start(&mut self) {
        self.enabled = true;
        self.queue.clear();
        self.stats.clear();
    }

    fn stop(&mut self) {
        self.enabled = false;
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn record(&mut self, trace: FuncTrace) {
        if !self.enabled {
            return;
        }

        self.stats.record(&trace);
        self.queue.push_back(trace);
        if self.queue.len() > MAX_FTRACE {
            self.queue.pop_front();
        }
    }

    fn iter(&self) -> impl Iterator<Item = FuncTrace> + '_ {
        self.queue.iter().cloned()
    }

    fn snapshot(&self) -> FtraceStatsSnapshot {
        self.stats.snapshot(self.enabled, self.queue.len())
    }
}

const MAX_HISTORY: usize = 1024;

struct DebugCycle {
    pc: WordType,
    raw_instr: Option<RawInstr>,
    decoded_instr: Option<DecodeInstr>,
}

struct DebuggerExecutionHook<'a> {
    breakpoints: &'a [Breakpoint],
    history: &'a mut VecDeque<(WordType, Option<RawInstr>)>,
    ftrace: &'a mut FtraceState,
    symtab: Option<&'a SymTab>,
}

/// return-address stack action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RasAction {
    None,
    Push,
    Pop,
    PopThenPush,
}

#[inline]
fn is_link(reg: u8) -> bool {
    reg == 1 || reg == 5
}

/// Classify a decoded instruction into a RAS action.
fn ras_action(instr: RiscvInstr, info: RVInstrInfo) -> RasAction {
    // TODO: we haven't provide a function to convert all C instruction to non-C instruction.
    let (rd, rs1) = match (instr, info) {
        (RiscvInstr::JALR, RVInstrInfo::I { rs1, rd, .. }) => (rd, rs1),
        (RiscvInstr::JAL, RVInstrInfo::J { rd, .. }) => (rd, 0),

        // c.jr rs1   => jalr x0, 0(rs1)
        (RiscvInstr::C_JR, RVInstrInfo::CR { rd_rs1: rs1, .. }) => (0, rs1),

        // c.jalr rs1  => jalr x1, 0(rs1)
        (RiscvInstr::C_JALR, RVInstrInfo::CR { rd_rs1: rs1, .. }) => (1, rs1),

        // c.jal offset => jal x1, offset
        (RiscvInstr::C_JAL, _) => (1, 0),

        // c.j offset  => jal x0, offset
        (RiscvInstr::C_J, _) => (0, 0),

        _ => return RasAction::None,
    };

    match (is_link(rd), is_link(rs1)) {
        (false, false) => RasAction::None,
        (false, true) => RasAction::Pop,
        (true, false) => RasAction::Push,
        (true, true) => {
            if rd != rs1 {
                RasAction::PopThenPush
            } else {
                RasAction::Push
            }
        }
    }
}

fn ftrace_for_step(
    symtab: Option<&SymTab>,
    decoded_instr: Option<DecodeInstr>,
    pre_pc: WordType,
    nxt_pc: WordType,
) -> [Option<FuncTrace>; 2] {
    let Some(DecodeInstr { instr, info, .. }) = decoded_instr else {
        return [None, None];
    };

    let leave_sym = symtab.and_then(|table| table.func_name_in_addr_range(pre_pc as u64).cloned());
    let enter_sym = symtab.and_then(|table| table.func_name_by_addr(nxt_pc as u64).cloned());

    match ras_action(instr, info) {
        RasAction::None => [None, None],
        RasAction::Push => [
            Some(FuncTrace::Call {
                name: enter_sym,
                addr: nxt_pc,
            }),
            None,
        ],
        RasAction::Pop => [
            Some(FuncTrace::Return {
                name: leave_sym,
                addr: pre_pc,
            }),
            None,
        ],
        RasAction::PopThenPush => [
            Some(FuncTrace::Return {
                name: leave_sym,
                addr: pre_pc,
            }),
            Some(FuncTrace::Call {
                name: enter_sym,
                addr: nxt_pc,
            }),
        ],
    }
}

fn cpu_on_breakpoint(breakpoints: &[Breakpoint], cpu: &mut RVCPU) -> bool {
    let pc = cpu.read_pc();
    if breakpoints
        .iter()
        .any(|bp| matches!(bp.addr, Address::Virt(addr) if addr == pc))
    {
        return true;
    }

    let Ok(pc_paddr) = cpu.debug_vaddr_to_paddr(pc) else {
        return false;
    };
    breakpoints
        .iter()
        .any(|bp| matches!(bp.addr, Address::Phys(addr) if addr == pc_paddr))
}

impl ExecutionHook for DebuggerExecutionHook<'_> {
    type CycleContext = DebugCycle;

    fn before_step(&mut self, cpu: &mut RVCPU) -> DebugCycle {
        let pc = cpu.read_pc();
        let raw_instr = cpu.read_instr(pc).ok();
        let decoded_instr = raw_instr.and_then(|raw| cpu.decoded_instr(raw));
        DebugCycle {
            pc,
            raw_instr,
            decoded_instr,
        }
    }

    fn on_interrupt_taken(&mut self, pc: WordType) -> DebugCycle {
        DebugCycle {
            pc,
            raw_instr: None,
            decoded_instr: None,
        }
    }

    fn after_step(&mut self, cycle: DebugCycle, cpu: &mut RVCPU) -> bool {
        if self.history.len() == MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back((cycle.pc, cycle.raw_instr));
        if self.ftrace.enabled() {
            for trace in ftrace_for_step(self.symtab, cycle.decoded_instr, cycle.pc, cpu.read_pc())
                .into_iter()
                .flatten()
            {
                self.ftrace.record(trace);
            }
        }

        cpu_on_breakpoint(self.breakpoints, cpu)
    }
}

pub struct Debugger<B: Board> {
    breakpoints: Vec<Breakpoint>,
    board: B,
    history: VecDeque<(WordType, Option<RawInstr>)>,
    ftrace: FtraceState,
    symtab: Option<SymTab>,
}

impl<B: Board> Debugger<B> {
    pub fn board(&self) -> &B {
        &self.board
    }

    pub fn board_mut(&mut self) -> &mut B {
        &mut self.board
    }

    pub fn into_board(self) -> B {
        self.board
    }

    pub fn new(board: B) -> Self {
        let symtab = board.loader().and_then(|loader| loader.get_symbol_table());

        Self {
            breakpoints: Vec::new(),
            board,
            history: VecDeque::with_capacity(MAX_HISTORY),
            ftrace: FtraceState::new(),
            symtab,
        }
    }

    pub fn set_symbol_table(&mut self, symtab: SymTab) {
        self.symtab = Some(symtab);
    }

    /// Get the latest k history.
    pub fn pc_history(&self, k: usize) -> impl Iterator<Item = (WordType, Option<RawInstr>)> {
        self.history.iter().copied().rev().take(k).rev()
    }

    pub fn ftrace_start(&mut self) {
        self.ftrace.start();
    }

    pub fn ftrace_stop(&mut self) {
        self.ftrace.stop();
    }

    pub fn ftrace_enabled(&self) -> bool {
        self.ftrace.enabled()
    }

    pub fn ftrace_show(&self) -> impl Iterator<Item = FuncTrace> + '_ {
        self.ftrace.iter()
    }

    pub fn ftrace_stat(&self) -> FtraceStatsSnapshot {
        self.ftrace.snapshot()
    }

    pub fn breakpoints(&self) -> &Vec<Breakpoint> {
        &self.breakpoints
    }

    pub fn symbol_table(&self) -> Option<&SymTab> {
        self.symtab.as_ref()
    }

    pub fn symbol_by_addr(&self, addr: u64) -> Result<&String, DebugError> {
        let Some(symtab) = self.symbol_table() else {
            return Err(DebugError::NoSymbolTable);
        };
        symtab
            .func_name_by_addr(addr)
            .ok_or(DebugError::SymbolNotFound(format!(
                "Function at address 0x{:08x} not found",
                addr
            )))
    }

    pub fn symbol_in_addr_range(&self, addr: u64) -> Result<&String, DebugError> {
        let Some(symtab) = self.symbol_table() else {
            return Err(DebugError::NoSymbolTable);
        };
        symtab
            .func_name_in_addr_range(addr)
            .ok_or(DebugError::SymbolNotFound(format!(
                "Function at address 0x{:08x} not found",
                addr
            )))
    }

    pub fn addr_by_symbol(&self, func_name: &str) -> Result<u64, DebugError> {
        let Some(symtab) = self.symbol_table() else {
            return Err(DebugError::NoSymbolTable);
        };
        symtab
            .func_addr_by_name(func_name)
            .ok_or(DebugError::SymbolNotFound(func_name.to_string()))
    }

    /// Returns true if a new breakpoint is added, otherwise the breakpoint already exists.
    pub fn set_breakpoint(&mut self, addr: Address) -> Result<bool, DebugError> {
        if let Some(_) = self.breakpoints.iter().find(|bp| bp.addr == addr) {
            return Ok(false);
        }
        let breakpoint = Breakpoint {
            id: self.breakpoints.len(),
            addr,
        };
        self.breakpoints.push(breakpoint);

        Ok(true)
    }

    /// Returns true if any breakpoint is removed.
    pub fn clear_breakpoint(&mut self, addr: Address) -> Result<bool, DebugError> {
        let original_len = self.breakpoints.len();
        self.breakpoints.retain(|bp| bp.addr != addr);
        Ok(self.breakpoints.len() != original_len)
    }

    pub fn on_breakpoint(&mut self) -> bool {
        cpu_on_breakpoint(&self.breakpoints, self.board.cpu_mut())
    }

    pub fn step(&mut self) -> Result<DebugEvent, DebugError> {
        self.continue_until_step(1).map(|(event, _steps)| event)
    }

    /// Continue execution in batches, checking `should_pause` between batches.
    ///
    /// Return `Ok(None)` when `should_pause` requests a pause, otherwise return
    /// the debug event that stopped execution.
    pub(crate) fn continue_with_batch_check(
        &mut self,
        mut should_pause: impl FnMut() -> bool,
    ) -> Result<Option<DebugEvent>, DebugError> {
        loop {
            if self.board.status() == crate::board::BoardStatus::Halt {
                return Ok(Some(DebugEvent::BoardHalted));
            }

            if should_pause() {
                return Ok(None);
            }

            let (event, _) = self.continue_until_step(1024)?;
            if event != DebugEvent::StepCompleted {
                return Ok(Some(event));
            }
        }
    }

    /// Continue running until a breakpoint is hit, `max_steps` steps are executed or the board is halted.
    /// Returns the event that caused the stop and the actual steps executed.
    pub fn continue_until_step(&mut self, max_steps: u64) -> Result<(DebugEvent, u64), DebugError> {
        if max_steps == 0 {
            return Ok((DebugEvent::StepCompleted, 0));
        }
        if self.board.status() == crate::board::BoardStatus::Halt {
            return Ok((DebugEvent::BoardHalted, 0));
        }

        let mut hook = DebuggerExecutionHook {
            breakpoints: &self.breakpoints,
            history: &mut self.history,
            ftrace: &mut self.ftrace,
            symtab: self.symtab.as_ref(),
        };
        let result = self.board.run_cycles_hooked(max_steps, &mut hook);
        let event = if result.hook_stopped {
            DebugEvent::BreakpointHit
        } else if self.board.status() == crate::board::BoardStatus::Halt {
            DebugEvent::BoardHalted
        } else {
            DebugEvent::StepCompleted
        };
        Ok((event, result.cycles))
    }

    /// See [`Self::continue_until_step`].
    pub fn continue_run(&mut self) -> Result<(DebugEvent, u64), DebugError> {
        self.continue_until_step(u64::MAX)
    }

    pub fn next_instr(&mut self) -> Option<RawInstr> {
        let pc = self.board.cpu().read_pc();
        self.board.cpu_mut().read_instr(pc).ok()
    }

    pub fn unify_to_phys_addr(&mut self, addr: Address) -> Option<u64> {
        match addr {
            Address::Phys(paddr) => Some(paddr),
            Address::Virt(vaddr) => self.vaddr_to_paddr(vaddr).ok(),
        }
    }

    // re-export methods from `DebugTarget`

    // TODO: Add checks here.
    pub fn read_reg(&self, idx: u8) -> WordType {
        self.board.cpu().read_reg(idx)
    }

    pub fn write_reg(&mut self, idx: u8, val: WordType) {
        self.board.cpu_mut().write_reg(idx, val)
    }

    pub fn read_pc(&self) -> WordType {
        self.board.cpu().read_pc()
    }

    pub fn write_pc(&mut self, val: WordType) {
        self.board.cpu_mut().write_pc(val)
    }

    pub fn read_float_reg(&self, idx: u8) -> (f32, f64) {
        self.board.cpu().read_float_reg(idx)
    }

    pub fn write_float_reg(&mut self, idx: u8, value: f64) {
        self.board.cpu_mut().fpu.store(idx, value);
    }

    pub fn read_vector_reg<T>(&self, idx: u8) -> Option<&[T]> {
        self.board.cpu().read_vector_reg(idx)
    }

    pub fn read_instr(&mut self, addr: WordType) -> Option<RawInstr> {
        self.board.cpu_mut().read_instr(addr).ok()
    }

    pub fn read_memory<V: UnsignedInteger>(&mut self, addr: Address) -> Result<V, MemError> {
        self.board.cpu_mut().read_memory(addr)
    }

    pub fn write_memory<V: UnsignedInteger>(
        &mut self,
        addr: Address,
        data: V,
    ) -> Result<(), MemError> {
        self.board.cpu_mut().write_memory::<V>(addr, data)
    }

    pub fn read_csr(&mut self, addr: WordType) -> Option<WordType> {
        self.board.cpu_mut().debug_csr(addr, None)
    }

    pub fn write_csr(&mut self, addr: WordType, data: WordType) -> Result<(), DebugError> {
        self.board
            .cpu_mut()
            .debug_csr(addr, Some(data))
            .ok_or(DebugError::CSRNotExist(addr))?;
        Ok(())
    }

    pub fn get_current_privilege(&mut self) -> PrivilegeLevel {
        self.board.cpu_mut().get_current_privilege()
    }

    pub fn set_current_privilege(&mut self, priv_level: PrivilegeLevel) {
        self.board.cpu_mut().csr.set_current_privileged(priv_level);
    }

    pub fn decoded_info(&self, raw: RawInstr) -> Option<<RiscvTypes as ISATypes>::DecodeRst> {
        self.board.cpu().decoded_instr(raw)
    }

    pub fn vaddr_to_paddr(&mut self, vaddr: WordType) -> Result<u64, PageTableError> {
        self.board.cpu_mut().debug_vaddr_to_paddr(vaddr)
    }

    pub fn translate(&mut self, addr: u64, access: AccessType) -> Result<u64, PageTableError> {
        self.board.cpu_mut().debug_translate(addr, access)
    }

    pub fn cycle(&mut self) -> WordType {
        self.board
            .cpu_mut()
            .csr
            .get_by_type_existing::<Mcycle>()
            .data()
    }
}

#[cfg(test)]
mod test {
    use crate::{isa::riscv::cpu_tester::TestCPUBuilder, ram_config::BASE_ADDR};

    use super::*;

    struct TestEmptyBoard {
        cpu: Box<RVCPU>,
    }

    impl TestEmptyBoard {
        fn new(cpu: Box<RVCPU>) -> Self {
            Self { cpu }
        }
    }

    impl Board for TestEmptyBoard {
        fn step_batch_with_hook<H: ExecutionHook>(
            &mut self,
            steps: u64,
            hook: &mut H,
        ) -> crate::isa::riscv::executor::BatchResult {
            self.cpu.step_batch_with_hook(steps, hook)
        }

        fn step_batch(&mut self, steps: u64) -> crate::isa::riscv::executor::BatchResult {
            self.cpu.step_batch(steps);
            crate::isa::riscv::executor::BatchResult {
                cycles: steps,
                hook_stopped: false,
            }
        }

        fn status(&self) -> crate::board::BoardStatus {
            crate::board::BoardStatus::Running
        }

        fn cpu(&self) -> &RVCPU {
            &self.cpu
        }

        fn cpu_mut(&mut self) -> &mut RVCPU {
            &mut self.cpu
        }

        fn loader(&self) -> Option<&crate::load::ELFLoader> {
            None
        }
    }

    fn create_debugger(cpu: Box<RVCPU>) -> Debugger<TestEmptyBoard> {
        Debugger::new(TestEmptyBoard::new(cpu))
    }

    #[test]
    fn test_breakpoint_riscv() {
        // Test that a breakpoint can be hit
        let cpu = TestCPUBuilder::new()
            .program(&[
                0x02520333, // mul x6, x4, x5
                0x02520333, // mul x6, x4, x5
                0x02520333, // mul x6, x4, x5
                0x02520333, // mul x6, x4, x5
                0x02520333, // mul x6, x4, x5
            ])
            .build();

        let mut debugger = create_debugger(cpu);

        debugger
            .set_breakpoint(Address::Phys(BASE_ADDR + 4))
            .unwrap();
        debugger.continue_run().unwrap();

        assert_eq!(debugger.read_pc(), BASE_ADDR + 4);
        assert_eq!(
            debugger
                .read_memory::<u32>(Address::Phys(BASE_ADDR + 4))
                .unwrap(),
            0x02520333
        );

        debugger.step().unwrap();
        assert_eq!(debugger.read_pc(), BASE_ADDR + 8);

        debugger
            .set_breakpoint(Address::Phys(BASE_ADDR + 12))
            .unwrap();

        debugger.continue_until_step(2).unwrap();
        assert_eq!(debugger.read_pc(), BASE_ADDR + 12);
        assert_eq!(
            debugger
                .read_memory::<u32>(Address::Phys(BASE_ADDR + 12))
                .unwrap(),
            0x02520333
        );
    }

    #[test]
    fn test_breakpoint_riscv_on_current() {
        let cpu = TestCPUBuilder::new()
            .program(&[
                0x02520333, // mul x6, x4, x5
                0x02520333, // mul x6, x4, x5
            ])
            .build();

        let mut debugger = create_debugger(cpu);
        debugger.set_breakpoint(Address::Phys(BASE_ADDR)).unwrap();

        debugger.step().unwrap();

        assert_eq!(debugger.read_pc(), BASE_ADDR + 4);
    }

    #[test]
    fn test_ftrace_records_direct_call_and_return() {
        let cpu = TestCPUBuilder::new()
            .program(&[
                0x008000ef, // jal ra, 8
                0x00000013, // nop (the return address)
                0x00008067, // ret
            ])
            .build();

        let mut debugger = create_debugger(cpu);
        debugger.set_symbol_table(SymTab::from(&[
            ("caller".to_string(), BASE_ADDR),
            ("callee".to_string(), BASE_ADDR + 8),
        ]));
        debugger.ftrace_start();

        assert_eq!(debugger.step().unwrap(), DebugEvent::StepCompleted);
        assert_eq!(debugger.read_pc(), BASE_ADDR + 8);
        assert_eq!(
            debugger.ftrace_show().collect::<Vec<_>>(),
            vec![FuncTrace::Call {
                name: Some("callee".to_string()),
                addr: BASE_ADDR + 8,
            }]
        );

        assert_eq!(debugger.step().unwrap(), DebugEvent::StepCompleted);
        assert_eq!(debugger.read_pc(), BASE_ADDR + 4);
        assert_eq!(
            debugger.ftrace_show().collect::<Vec<_>>(),
            vec![
                FuncTrace::Call {
                    name: Some("callee".to_string()),
                    addr: BASE_ADDR + 8,
                },
                FuncTrace::Return {
                    name: Some("callee".to_string()),
                    addr: BASE_ADDR + 8,
                },
            ]
        );

        assert_eq!(
            debugger.ftrace_stat(),
            FtraceStatsSnapshot {
                enabled: true,
                queue_len: 2,
                call_count: 1,
                return_count: 1,
                unknown_calls: 0,
                unknown_returns: 0,
                per_func: vec![(
                    "callee".to_string(),
                    FuncTraceStatEntry {
                        calls: 1,
                        returns: 1,
                    },
                )],
            }
        );
    }

    #[test]
    fn test_ftrace_records_indirect_call_using_link_register() {
        let mut cpu = TestCPUBuilder::new()
            .program(&[
                0x000300e7, // jalr ra, 0(t1)
                0x00000013, // nop (the return address)
                0x00008067, // ret
            ])
            .build();
        cpu.write_reg(6, BASE_ADDR + 8);

        let mut debugger = create_debugger(cpu);
        debugger.set_symbol_table(SymTab::from(&[
            ("caller".to_string(), BASE_ADDR),
            ("callee".to_string(), BASE_ADDR + 8),
        ]));
        debugger.ftrace_start();
        debugger.continue_until_step(2).unwrap();

        assert_eq!(
            debugger.ftrace_show().collect::<Vec<_>>(),
            vec![
                FuncTrace::Call {
                    name: Some("callee".to_string()),
                    addr: BASE_ADDR + 8,
                },
                FuncTrace::Return {
                    name: Some("callee".to_string()),
                    addr: BASE_ADDR + 8,
                },
            ]
        );
    }

    #[test]
    fn test_ftrace_records_compressed_call_and_return() {
        let mut cpu = TestCPUBuilder::new()
            .program(&[
                0x00009402, // c.jalr s0
                0x00000000, // unused instruction at the return address
                0x00008082, // c.jr ra
            ])
            .build();
        cpu.write_reg(8, BASE_ADDR + 8);

        let mut debugger = create_debugger(cpu);
        debugger.set_symbol_table(SymTab::from(&[(
            "compressed_callee".to_string(),
            BASE_ADDR + 8,
        )]));
        debugger.ftrace_start();

        assert_eq!(debugger.step().unwrap(), DebugEvent::StepCompleted);
        assert_eq!(debugger.read_pc(), BASE_ADDR + 8);
        assert_eq!(
            debugger.ftrace_show().collect::<Vec<_>>(),
            vec![FuncTrace::Call {
                name: Some("compressed_callee".to_string()),
                addr: BASE_ADDR + 8,
            }]
        );

        assert_eq!(debugger.step().unwrap(), DebugEvent::StepCompleted);
        assert_eq!(debugger.read_pc(), BASE_ADDR + 2);
        assert_eq!(
            debugger.ftrace_show().collect::<Vec<_>>(),
            vec![
                FuncTrace::Call {
                    name: Some("compressed_callee".to_string()),
                    addr: BASE_ADDR + 8,
                },
                FuncTrace::Return {
                    name: Some("compressed_callee".to_string()),
                    addr: BASE_ADDR + 8,
                },
            ]
        );
    }

    #[test]
    fn test_ftrace_records_pop_then_push_as_return_then_call() {
        let mut cpu = TestCPUBuilder::new()
            .program(&[
                0x008000ef, // jal ra, first
                0x00000013, // nop (the return address)
                0x000280e7, // jalr ra, 0(t0), second
                0x00000013, // nop (the return address)
                0x00008067, // ret
            ])
            .build();
        cpu.write_reg(5, BASE_ADDR + 16);

        let mut debugger = create_debugger(cpu);
        debugger.set_symbol_table(SymTab::from(&[
            ("caller".to_string(), BASE_ADDR),
            ("first".to_string(), BASE_ADDR + 8),
            ("second".to_string(), BASE_ADDR + 16),
        ]));
        debugger.ftrace_start();
        debugger.continue_until_step(3).unwrap();

        assert_eq!(
            debugger.ftrace_show().collect::<Vec<_>>(),
            vec![
                FuncTrace::Call {
                    name: Some("first".to_string()),
                    addr: BASE_ADDR + 8,
                },
                FuncTrace::Return {
                    name: Some("first".to_string()),
                    addr: BASE_ADDR + 8,
                },
                FuncTrace::Call {
                    name: Some("second".to_string()),
                    addr: BASE_ADDR + 16,
                },
                FuncTrace::Return {
                    name: Some("second".to_string()),
                    addr: BASE_ADDR + 16,
                },
            ]
        );
    }

    #[test]
    fn test_ftrace_ignores_unlinked_jumps() {
        let mut cpu = TestCPUBuilder::new()
            .program(&[
                0x00030067, // jalr zero, 0(t1)
                0x00000013, // nop
                0x00000013, // nop
            ])
            .build();
        cpu.write_reg(6, BASE_ADDR + 8);

        let mut debugger = create_debugger(cpu);
        debugger.set_symbol_table(SymTab::from(&[("jump_target".to_string(), BASE_ADDR + 8)]));
        debugger.ftrace_start();
        debugger.step().unwrap();

        assert!(debugger.ftrace_show().collect::<Vec<_>>().is_empty());
        assert_eq!(debugger.ftrace_stat().call_count, 0);
        assert_eq!(debugger.ftrace_stat().return_count, 0);
    }

    #[test]
    fn test_ftrace_classifies_riscv_return_address_stack_actions() {
        assert_eq!(
            ras_action(RiscvInstr::JAL, RVInstrInfo::J { rd: 1, imm: 8 }),
            RasAction::Push
        );
        assert_eq!(
            ras_action(RiscvInstr::JAL, RVInstrInfo::J { rd: 0, imm: 8 }),
            RasAction::None
        );
        assert_eq!(
            ras_action(
                RiscvInstr::JALR,
                RVInstrInfo::I {
                    rs1: 6,
                    rd: 1,
                    imm: 0,
                },
            ),
            RasAction::Push
        );
        assert_eq!(
            ras_action(
                RiscvInstr::JALR,
                RVInstrInfo::I {
                    rs1: 1,
                    rd: 0,
                    imm: 0,
                },
            ),
            RasAction::Pop
        );
        assert_eq!(
            ras_action(
                RiscvInstr::JALR,
                RVInstrInfo::I {
                    rs1: 5,
                    rd: 1,
                    imm: 0,
                },
            ),
            RasAction::PopThenPush
        );
        assert_eq!(
            ras_action(RiscvInstr::C_JR, RVInstrInfo::CR { rd_rs1: 1, rs2: 0 },),
            RasAction::Pop
        );
        assert_eq!(
            ras_action(RiscvInstr::C_JALR, RVInstrInfo::CR { rd_rs1: 5, rs2: 0 },),
            RasAction::PopThenPush
        );
    }

    #[test]
    fn test_ftrace_disabled_until_started_and_restart_clears_state() {
        let cpu = TestCPUBuilder::new()
            .program(&[
                0x008000ef, // jal ra, 8
                0x00000013, // nop
                0x00008067, // ret
            ])
            .build();

        let mut debugger = create_debugger(cpu);
        debugger.set_symbol_table(SymTab::from(&[
            ("caller".to_string(), BASE_ADDR),
            ("callee".to_string(), BASE_ADDR + 8),
        ]));

        debugger.step().unwrap();
        assert!(!debugger.ftrace_enabled());
        assert!(debugger.ftrace_show().collect::<Vec<_>>().is_empty());

        debugger.ftrace_start();
        debugger.step().unwrap();
        assert_eq!(debugger.ftrace_stat().return_count, 1);

        debugger.ftrace_stop();
        let stopped = debugger.ftrace_stat();
        debugger.step().unwrap();
        assert!(!debugger.ftrace_enabled());
        assert_eq!(debugger.ftrace_stat(), stopped);

        debugger.ftrace_start();
        assert!(debugger.ftrace_enabled());
        assert!(debugger.ftrace_show().collect::<Vec<_>>().is_empty());
        assert_eq!(debugger.ftrace_stat().queue_len, 0);
        assert_eq!(debugger.ftrace_stat().call_count, 0);
        assert_eq!(debugger.ftrace_stat().return_count, 0);
    }

    #[test]
    fn test_ftrace_unknown_symbols_are_counted_separately() {
        let cpu = TestCPUBuilder::new()
            .program(&[
                0x008000ef, // jal ra, 8
                0x00000013, // nop
                0x00008067, // ret
            ])
            .build();

        let mut debugger = create_debugger(cpu);
        debugger.ftrace_start();
        debugger.continue_until_step(2).unwrap();

        assert_eq!(
            debugger.ftrace_show().collect::<Vec<_>>(),
            vec![
                FuncTrace::Call {
                    name: None,
                    addr: BASE_ADDR + 8,
                },
                FuncTrace::Return {
                    name: None,
                    addr: BASE_ADDR + 8,
                },
            ]
        );
        assert_eq!(
            debugger.ftrace_stat(),
            FtraceStatsSnapshot {
                enabled: true,
                queue_len: 2,
                call_count: 1,
                return_count: 1,
                unknown_calls: 1,
                unknown_returns: 1,
                per_func: Vec::new(),
            }
        );
    }

    #[test]
    fn test_ftrace_queue_is_bounded_but_stats_are_not() {
        let cpu = TestCPUBuilder::new()
            .program(&[
                0x008000ef, // jal ra, 8
                0xffdff06f, // jal zero, -4
                0x00008067, // ret
            ])
            .build();

        let mut debugger = create_debugger(cpu);
        debugger.set_symbol_table(SymTab::from(&[("callee".to_string(), BASE_ADDR + 8)]));
        debugger.ftrace_start();

        let loops = MAX_FTRACE as u64 / 2 + 1;
        debugger.continue_until_step(loops * 3).unwrap();

        let traces = debugger.ftrace_show().collect::<Vec<_>>();
        assert_eq!(traces.len(), MAX_FTRACE);
        assert_eq!(
            traces.first(),
            Some(&FuncTrace::Call {
                name: Some("callee".to_string()),
                addr: BASE_ADDR + 8,
            })
        );
        assert_eq!(
            traces.last(),
            Some(&FuncTrace::Return {
                name: Some("callee".to_string()),
                addr: BASE_ADDR + 8,
            })
        );

        let stats = debugger.ftrace_stat();
        assert_eq!(stats.queue_len, MAX_FTRACE);
        assert_eq!(stats.call_count, loops);
        assert_eq!(stats.return_count, loops);
        assert_eq!(stats.unknown_calls, 0);
        assert_eq!(stats.unknown_returns, 0);
        assert_eq!(
            stats.per_func,
            vec![(
                "callee".to_string(),
                FuncTraceStatEntry {
                    calls: loops,
                    returns: loops,
                },
            )]
        );
    }
}
