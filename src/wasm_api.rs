use std::{cell::RefCell, panic, rc::Rc};

use wasm_bindgen::prelude::*;

use crate::{
    board::{Board, BoardStatus, virt::VirtBoard},
    config::arch_config::REGFILE_CNT,
    device::uart16550a::UartBytePort,
    isa::DebugTarget,
    load::{ELFLoader, SymTab},
    rvdb::{AsyncREPL, REPLResponse, RvdbChannelTx, RvdbChannels},
};

#[wasm_bindgen(start)]
fn init_on_wasm() {
    wasm_logger::init(wasm_logger::Config::new(log::Level::Info));
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[wasm_bindgen]
pub struct WasmEmulator {
    inner: VirtBoard,
    uart_port: Rc<RefCell<UartBytePort>>,
}

#[wasm_bindgen]
impl WasmEmulator {
    pub fn from_elf_bytes(bytes: &[u8]) -> Result<Self, JsValue> {
        let mut inner = VirtBoard::from_elf(bytes.to_vec())
            .map_err(|e| JsValue::from_str(&format!("ELF load failed: {e}")))?;
        let uart_port = inner
            .take_uart_port()
            .expect("WASM boards use External UART I/O");
        Ok(Self {
            inner,
            uart_port: Rc::new(RefCell::new(uart_port)),
        })
    }

    pub fn from_bin_bytes(bytes: &[u8]) -> Result<Self, JsValue> {
        let mut inner = VirtBoard::from_binary_with(bytes, Default::default())
            .map_err(|e| JsValue::from_str(&format!("binary load failed: {e}")))?;
        let uart_port = inner
            .take_uart_port()
            .expect("WASM boards use External UART I/O");
        Ok(Self {
            inner,
            uart_port: Rc::new(RefCell::new(uart_port)),
        })
    }

    pub async fn into_rvdb(self) -> WasmRvdb {
        WasmRvdb::from_board(self.inner, self.uart_port).await
    }

    pub fn step(&mut self) -> Result<(), JsValue> {
        if self.inner.status() != BoardStatus::Halt {
            self.inner.step();
        }
        Ok(())
    }

    pub fn continue_for_steps(&mut self, max_steps: u64) -> Result<u64, JsValue> {
        Ok(self.inner.run_cycles(max_steps))
    }

    pub fn is_halted(&self) -> bool {
        self.inner.status() == BoardStatus::Halt
    }

    pub fn clock_cycles(&self) -> u64 {
        self.inner.cycles()
    }

    pub fn read_pc(&self) -> u64 {
        self.inner.cpu().read_pc() as u64
    }

    pub fn read_reg(&self, idx: u8) -> u64 {
        self.inner.cpu().read_reg(idx) as u64
    }

    /// Reads all 32 integer registers.
    pub fn read_regs(&self) -> Vec<u64> {
        (0..REGFILE_CNT)
            .map(|idx| self.inner.cpu().read_reg(idx as u8) as u64)
            .collect()
    }

    pub fn push_uart_input(&mut self, input: &[u8]) -> Result<(), JsValue> {
        self.uart_port
            .borrow()
            .push_input(input)
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }

    pub fn take_uart_output(&mut self) -> Result<Vec<u8>, JsValue> {
        Ok(self.uart_port.borrow_mut().take_output())
    }
}

#[wasm_bindgen]
pub struct WasmRvdb {
    channel: RvdbChannelTx,
    rvdb: AsyncREPL<VirtBoard>,
    shared: Rc<RefCell<RvdbSharedState>>,
    uart_port: Rc<RefCell<UartBytePort>>,
}

struct RvdbSharedState {
    pc: u64,
    cycles: u64,
    regs: [u64; REGFILE_CNT],
    halted: bool,
    continue_running: bool,
    pending_symbol_table: Option<SymTab>,
}

impl RvdbSharedState {
    fn new() -> Self {
        Self {
            pc: 0,
            cycles: 0,
            regs: [0; REGFILE_CNT],
            halted: false,
            continue_running: false,
            pending_symbol_table: None,
        }
    }

    fn apply_pending(&mut self, session: &mut crate::rvdb::RvdbSession<VirtBoard>) {
        if let Some(symtab) = self.pending_symbol_table.take() {
            session.set_symbol_table(symtab);
        }
    }

    fn sync_from(&mut self, session: &mut crate::rvdb::RvdbSession<VirtBoard>) {
        self.apply_pending(session);
        let board = session.board();

        self.halted = board.status() == BoardStatus::Halt;
        self.cycles = board.cycles();
        self.pc = board.cpu().read_pc() as u64;
        for idx in 0..REGFILE_CNT {
            self.regs[idx] = board.cpu().read_reg(idx as u8) as u64;
        }
    }
}

#[wasm_bindgen]
pub struct WasmRvdbHandle {
    channel: RvdbChannelTx,
    shared: Rc<RefCell<RvdbSharedState>>,
    uart_port: Rc<RefCell<UartBytePort>>,
}

impl WasmRvdb {
    async fn from_board(board: VirtBoard, uart_port: Rc<RefCell<UartBytePort>>) -> Self {
        let (tx, rx) = RvdbChannels::new();
        let rvdb = AsyncREPL::new(board, rx).await;
        let shared = Rc::new(RefCell::new(RvdbSharedState::new()));

        let mut wasm_rvdb = Self {
            channel: tx,
            rvdb,
            shared,
            uart_port,
        };
        wasm_rvdb.sync_shared_state();
        wasm_rvdb
    }

    fn sync_shared_state(&mut self) {
        self.shared.borrow_mut().sync_from(self.rvdb.session_mut());
    }
}

#[wasm_bindgen]
impl WasmRvdb {
    pub async fn from_elf_bytes(bytes: &[u8]) -> Result<Self, JsValue> {
        Ok(WasmEmulator::from_elf_bytes(bytes)?.into_rvdb().await)
    }

    pub async fn from_bin_bytes(bytes: &[u8]) -> Result<Self, JsValue> {
        Ok(WasmEmulator::from_bin_bytes(bytes)?.into_rvdb().await)
    }

    pub fn handle(&self) -> WasmRvdbHandle {
        WasmRvdbHandle {
            channel: self.channel.clone(),
            shared: self.shared.clone(),
            uart_port: Rc::clone(&self.uart_port),
        }
    }

    pub async fn tick(&mut self) -> Result<REPLResponse, String> {
        self.sync_shared_state();

        let line = self
            .rvdb
            .readline()
            .await
            .map_err(|e| std::format!("{:?}", e))?;

        self.sync_shared_state();

        let shared = self.shared.clone();
        let continue_command = self.rvdb.line_is_continue_command(&line);
        shared.borrow_mut().continue_running = continue_command;
        let response = self
            .rvdb
            .execute_line_with_hook(&line, |session| {
                shared.borrow_mut().sync_from(session);
            })
            .await;

        self.shared.borrow_mut().continue_running = false;
        self.sync_shared_state();

        response
    }

    pub fn into_emulator(mut self) -> WasmEmulator {
        self.shared
            .borrow_mut()
            .apply_pending(self.rvdb.session_mut());

        WasmEmulator {
            inner: self.rvdb.into_board(),
            uart_port: self.uart_port,
        }
    }
}

#[wasm_bindgen]
impl WasmRvdbHandle {
    pub fn push_repl_input(&mut self, input: &[u8]) {
        self.channel.push_input(input);
    }

    pub fn cancel_continue(&mut self) {
        self.channel.cancel_continue();
    }

    pub fn take_repl_output(&mut self) -> Vec<u8> {
        self.channel.take_output()
    }

    pub fn push_uart_input(&mut self, input: &[u8]) -> Result<(), JsValue> {
        self.uart_port
            .borrow()
            .push_input(input)
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }

    pub fn take_uart_output(&mut self) -> Result<Vec<u8>, JsValue> {
        Ok(self.uart_port.borrow_mut().take_output())
    }

    pub fn load_symbol_file(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let symtab = parse_symbol_file(bytes)?;
        self.shared.borrow_mut().pending_symbol_table = Some(symtab);
        Ok(())
    }

    pub fn is_halted(&self) -> bool {
        self.shared.borrow().halted
    }

    pub fn is_continue_running(&self) -> bool {
        self.shared.borrow().continue_running
    }

    pub fn clock_cycles(&self) -> u64 {
        self.shared.borrow().cycles
    }

    pub fn read_pc(&self) -> u64 {
        self.shared.borrow().pc
    }

    pub fn read_reg(&self, idx: u8) -> u64 {
        self.shared.borrow().regs[idx as usize]
    }

    /// Reads all 32 integer registers.
    pub fn read_regs(&self) -> Vec<u64> {
        self.shared.borrow().regs.to_vec()
    }
}

fn parse_symbol_file(bytes: &[u8]) -> Result<SymTab, JsValue> {
    let loader = ELFLoader::try_new(bytes.to_vec())
        .ok_or_else(|| JsValue::from_str("Failed to parse ELF file"))?;
    loader
        .get_symbol_table()
        .ok_or_else(|| JsValue::from_str("No symbol table found in ELF file"))
}
