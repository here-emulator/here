use gdbstub::target;
use gdbstub::target::ext;

use crate::board::Board;

use crate::isa::riscv::debugger::{DebugEvent, Debugger};

mod basic;
mod breakpoints;
mod desc;
mod eventloop;

pub use eventloop::*;

enum ExecMode {
    Continue,
    Step,
}

enum RunEvent {
    IncomingData,
    StopReason(DebugEvent),
}

pub struct GdbDebugger<B: Board> {
    dbg: Debugger<B>,
    exec_mode: ExecMode,
}

impl<B: Board> GdbDebugger<B> {
    pub fn new(board: B) -> Self {
        Self {
            dbg: Debugger::new(board),
            exec_mode: ExecMode::Continue,
        }
    }

    fn run_by_mode_with_batch_check<F: FnMut() -> bool>(&mut self, should_pause: F) -> RunEvent {
        match self.exec_mode {
            ExecMode::Step => RunEvent::StopReason(self.dbg.step().unwrap()),
            ExecMode::Continue => match self.dbg.continue_with_batch_check(should_pause).unwrap() {
                Some(event) => RunEvent::StopReason(event),
                None => RunEvent::IncomingData,
            },
        }
    }
}
