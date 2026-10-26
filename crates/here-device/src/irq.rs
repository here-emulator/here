//! Interrupt-line plumbing between interrupt controllers and the CPU.
//!
//! Devices own the sources (CLINT, PLIC); the CPU implements
//! [`RiscvIRQHandler`]. Keeping these types here avoids devices depending on
//! the board layer.

use here_types::Interrupt;

pub trait RiscvIRQHandler {
    fn handle_irq(&mut self, interrupt: Interrupt, level: bool);
}

pub trait RiscvIRQSource {
    fn set_irq_line(&mut self, line: IRQLine, id: usize);
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
