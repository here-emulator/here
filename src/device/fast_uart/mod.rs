//! TODO: this module is not fully implemented according to the spec.
//! Some features are missing, and some behavior may be incorrect due to limited test coverage.

use std::{cell::RefCell, sync::Arc, u8};

use tokio::sync::mpsc::{
    self, Receiver, Sender, UnboundedReceiver, UnboundedSender,
    error::{TryRecvError, TrySendError},
};

use crate::{
    config::arch_config::WordType,
    device::{
        DeviceTrait, MemError, MemMappedDeviceTrait, PlicDevice,
        config::{UART_BASE, UART_DEFAULT_DIV, UART_IRQ, UART_SIZE},
        plic::PeriphIrqId,
    },
    utils::{clear_bit, read_bit, set_bit},
};

const UART_DATA_LENGTH: u8 = 8;
pub const UART_INPUT_CAPACITY: usize = 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UartIoMode {
    None,
    #[default]
    External,
    #[cfg(feature = "native-cli")]
    Stdio,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum UartIoError {
    #[error("UART host I/O is unavailable in {0:?} mode")]
    Unavailable(UartIoMode),
    #[error("UART input buffer is full after accepting {accepted} bytes")]
    InputFull { accepted: usize },
    #[error("UART input channel is closed after accepting {accepted} bytes")]
    InputClosed { accepted: usize },
}

pub struct UartBytePort {
    pub input_tx: Sender<u8>,
    pub output_rx: UnboundedReceiver<u8>,
}

impl UartBytePort {
    pub fn input_sender(&self) -> Sender<u8> {
        self.input_tx.clone()
    }

    pub fn push_input(&self, bytes: &[u8]) -> Result<(), UartIoError> {
        for (accepted, &byte) in bytes.iter().enumerate() {
            match self.input_tx.try_send(byte) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    return Err(UartIoError::InputFull { accepted });
                }
                Err(TrySendError::Closed(_)) => {
                    return Err(UartIoError::InputClosed { accepted });
                }
            }
        }
        Ok(())
    }

    pub fn take_output(&mut self) -> Vec<u8> {
        let mut output = Vec::new();
        loop {
            match self.output_rx.try_recv() {
                Ok(byte) => output.push(byte),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return output,
            }
        }
    }

    #[cfg(feature = "native-cli")]
    pub(crate) fn spawn_stdout(self, spawner: &crate::task_spawner::TaskSpawner) {
        let Self { mut output_rx, .. } = self;
        spawner.spawn_task(Box::pin(async move {
            use std::io::Write;

            let mut buffer = [0u8; 4096];
            while let Some(byte) = output_rx.recv().await {
                buffer[0] = byte;
                let mut len = 1;
                while len < buffer.len() {
                    match output_rx.try_recv() {
                        Ok(byte) => {
                            buffer[len] = byte;
                            len += 1;
                        }
                        Err(_) => break,
                    }
                }

                let result = (|| -> std::io::Result<()> {
                    let mut stdout = std::io::stdout().lock();
                    stdout.write_all(&buffer[..len])?;
                    stdout.flush()
                })();

                if let Err(error) = result {
                    log::error!("failed to write UART output to stdout: {error}");
                    return;
                }
            }
        }));
    }
}

#[allow(unused)]
mod offset {
    use super::WordType;

    pub const RBR: WordType = 0x00;
    pub const THR: WordType = 0x00;
    pub const IER: WordType = 0x01;
    pub const IIR: WordType = 0x02;
    pub const FCR: WordType = 0x02;
    pub const LCR: WordType = 0x03;
    pub const MCR: WordType = 0x04;
    pub const LSR: WordType = 0x05;
    pub const MSR: WordType = 0x06;
    pub const SCR: WordType = 0x07;
    pub const DLL: WordType = 0x00;
    pub const DLM: WordType = 0x01;
}

#[rustfmt::skip]
#[allow(non_snake_case)]
#[repr(C)]
/// See doc/device/uart.md
struct Uart16550Reg {   //  | LCR   |  Addr |         Description               | Access Type
    RBR:    u8,         //  | 0     | +0x0  | Receiver Buffer Register          |   RO
    THR:    u8,         //  | 0     | +0x0  | Transmitter Holding Register      |   WO
    IER:    u8,         //  | 0     | +0x1  | Interrupt Enable Register         |   RW
    IIR:    u8,         //  | Any   | +0x2  | Interrupt Identification Register |   RO
    FCR:    u8,         //  | Any   | +0x2  | FIFO Control Register             |   WO
    LCR:    u8,         //  | Any   | +0x3  | Line Control Register             |   RW
    MCR:    u8,         //  | Any   | +0x4  | Modem Control Register            |   RW
    LSR:    u8,         //  | Any   | +0x5  | Line Status Register              |   RW
    MSR:    u8,         //  | Any   | +0x6  | Modem Status Register             |   RW
    SCR:    u8,         //  | Any   | +0x7  | Scratch Register                  |   RW
    DLL:    u8,         //  | 1     | +0x0  | Divisor Latch(low)   Register     |   RW
    DLM:    u8,         //  | 1     | +0x1  | Divisor Latch(most)  Register     |   RW
}

impl Uart16550Reg {
    fn new() -> Self {
        Self {
            RBR: 0,
            THR: 0,
            IER: 0,
            IIR: 0,
            FCR: 0,
            LCR: 0x07,
            MCR: 0,
            LSR: 0x60,
            MSR: 0,
            SCR: 0,
            DLL: UART_DEFAULT_DIV as u8,
            DLM: (UART_DEFAULT_DIV >> 8) as u8,
        }
    }

    fn get_divisor(&self) -> u16 {
        (self.DLL as u16) + ((self.DLM as u16) << 8)
    }

    fn get_tx_data(&mut self) -> Option<u8> {
        if read_bit(&self.LSR, 5) {
            None
        } else {
            set_bit(&mut self.LSR, 5);
            Some(self.THR)
        }
    }

    fn write_transmit_empty<const BIT: bool>(&mut self) {
        if BIT {
            set_bit(&mut self.LSR, 6);
        } else {
            clear_bit(&mut self.LSR, 6);
        }
    }

    fn get_stop_bits(&self) -> u8 {
        if (self.LCR & (1 << 2)) != 0 { 2 } else { 1 }
    }
}

#[allow(non_snake_case)]
pub struct FastUart16550 {
    reg: Arc<RefCell<Uart16550Reg>>,
    reg_ptr: [*const u8; 8],
    reg_mut_ptr: [*mut u8; 8],
    reg_lcr_ptr: [*mut u8; 8],

    input_rx: Receiver<u8>,
    output_tx: UnboundedSender<u8>,

    /// THRE event latch for simplified ETBEI behavior.
    /// Cleared when IIR reports THRE as the identified interrupt source.
    thre_pending: bool,
}

impl FastUart16550 {
    pub fn new() -> (Self, UartBytePort) {
        let (input_tx, input_rx) = mpsc::channel(UART_INPUT_CAPACITY);
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let uart = Self::from_channel(input_rx, output_tx);

        (
            uart,
            UartBytePort {
                input_tx,
                output_rx,
            },
        )
    }

    pub fn from_channel(input_rx: Receiver<u8>, output_tx: UnboundedSender<u8>) -> Self {
        let reg = Arc::new(RefCell::new(Uart16550Reg::new()));
        let mut reg_ref = reg.borrow_mut();
        let reg_ptr = [
            (&reg_ref.RBR) as *const u8,
            (&reg_ref.IER) as *const u8,
            (&reg_ref.IIR) as *const u8,
            (&reg_ref.LCR) as *const u8,
            (&reg_ref.MCR) as *const u8,
            (&reg_ref.LSR) as *const u8,
            (&reg_ref.MSR) as *const u8,
            (&reg_ref.SCR) as *const u8,
        ];
        let reg_mut_ptr = [
            (&mut reg_ref.THR) as *mut u8,
            (&mut reg_ref.IER) as *mut u8,
            (&mut reg_ref.FCR) as *mut u8,
            (&mut reg_ref.LCR) as *mut u8,
            (&mut reg_ref.MCR) as *mut u8,
            (&mut reg_ref.LSR) as *mut u8,
            (&mut reg_ref.MSR) as *mut u8,
            (&mut reg_ref.SCR) as *mut u8,
        ];
        let reg_lcr_ptr = [
            (&mut reg_ref.DLL) as *mut u8,
            (&mut reg_ref.DLM) as *mut u8,
            (&mut reg_ref.FCR) as *mut u8,
            (&mut reg_ref.LCR) as *mut u8,
            (&mut reg_ref.MCR) as *mut u8,
            (&mut reg_ref.LSR) as *mut u8,
            (&mut reg_ref.MSR) as *mut u8,
            (&mut reg_ref.SCR) as *mut u8,
        ];

        let thre_pending = true; // THR is initially empty.

        drop(reg_ref);
        Self {
            reg: reg.clone(),
            reg_ptr,
            reg_mut_ptr,
            reg_lcr_ptr,
            input_rx,
            output_tx,
            thre_pending,
        }
    }

    /// Compute a simplified IIR (Interrupt Identification Register) view based on current IER/LSR/FCR state.
    fn compute_iir(&mut self) -> u8 {
        let reg = self.reg.borrow();
        let ier = reg.IER;
        let lsr = reg.LSR;
        let fcr = reg.FCR;

        let mut iir: u8 = 0x01;

        // FIFO enabled status in IIR[7:6]
        if fcr & 0x01 != 0 {
            iir |= 0xC0;
        }

        // Check interrupt conditions in priority order.
        if ier & 0x04 != 0 && lsr & 0x1E != 0 {
            // Receiver Line Status (OE, PE, FE, BI)
            iir = (iir & 0xC0) | 0x06;
        } else if ier & 0x01 != 0 && lsr & 0x01 != 0 {
            // Received Data Available
            iir = (iir & 0xC0) | 0x04;
        } else if ier & 0x02 != 0 && self.thre_pending {
            // Transmitter Holding Register Empty (edge-triggered)
            // Reading IIR with THRE identified clears the pending condition.
            self.thre_pending = false;
            iir = (iir & 0xC0) | 0x02;
        } else if ier & 0x08 != 0 {
            // Modem Status
            iir = (iir & 0xC0) | 0x00;
        }

        log::trace!(
            "[UART] compute_iir: IER={:#04x} LSR={:#04x} thre_pending={} => IIR={:#04x}",
            ier,
            lsr,
            self.thre_pending,
            iir
        );

        iir
    }

    /// Pure evaluation of the UART interrupt state,
    /// returning the UART IRQ id when an enabled source is currently active.
    fn eval_irq(ier: u8, thre_pending: bool, rx_pending: bool) -> Option<PeriphIrqId> {
        let rda = ier & 0x01 != 0 && rx_pending; // Received Data Available
        let thre = ier & 0x02 != 0 && thre_pending; // Transmit Holding Register Empty
        (rda || thre).then_some(UART_IRQ)
    }

    /// Snapshot the current interrupt state.
    #[cfg(test)]
    pub fn irq_status(&mut self) -> Option<PeriphIrqId> {
        self.irq_level().then_some(UART_IRQ)
    }

    fn read_impl<T>(&mut self, inner_addr: WordType) -> Result<T, MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        // check terminal input.
        if !read_bit(&mut self.reg.borrow_mut().LSR, 0) {
            // receive data ready.
            if let Ok(data) = self.input_rx.try_recv() {
                self.write_RBR(data)
            }
        }

        let inner_addr: usize = inner_addr as usize;
        let size = size_of::<T>();
        debug_assert!(inner_addr as usize + size <= 8);

        let mut data: T = 0u8.into();
        if (self.reg.borrow().LCR & (1 << 7)) == (1 << 7) {
            // LCR
            for i in inner_addr..8.min(inner_addr + size) {
                data |= T::from(
                    unsafe { self.reg_lcr_ptr[i].read_volatile() } << (8 * (i - inner_addr)),
                )
            }
        } else {
            // Normal
            for i in inner_addr..8.min(inner_addr + size) {
                if i == 0 {
                    data = self.read_RBR().into(); // RBR must be the first byte.
                } else if i == 2 {
                    // IIR: compute dynamically instead of reading stale value
                    data |= T::from(self.compute_iir() << (8 * (i - inner_addr)));
                } else {
                    data |= T::from(
                        unsafe { self.reg_ptr[i].read_volatile() } << (8 * (i - inner_addr)),
                    );
                }
            }
        }

        Ok(data)
    }

    fn write_impl<T>(&mut self, inner_addr: WordType, data: T) -> Result<(), MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        let inner_addr: usize = inner_addr as usize;
        let size = size_of::<T>();
        assert!(inner_addr as usize + size <= 8);
        let mut data: u64 = data.into();

        if (self.reg.borrow().LCR & (1 << 7)) == (1 << 7) {
            // LCR (Divisor Latch Access)
            for i in inner_addr..8.min(inner_addr + size) {
                unsafe { self.reg_lcr_ptr[i].write_volatile((data & (0xff)) as u8) }
                data >>= 8;
            }
        } else {
            // Normal
            for i in inner_addr..8.min(inner_addr + size) {
                if i == 0 {
                    // Writing to THR: send the byte immediately.
                    let byte = (data & 0xff) as u8;
                    log::trace!(
                        "[UART] THR write: {:#04x} '{}'",
                        byte,
                        if byte.is_ascii_graphic() || byte == b' ' {
                            byte as char
                        } else {
                            '.'
                        }
                    );
                    let _ = self.output_tx.send(byte);
                    // In a real 16550, writing THR clears LSR[5] (THRE) momentarily,
                    // then sets it again when the shift register accepts the byte.
                    // Since fast_uart sends instantly, we just re-arm the THRE event.
                    self.thre_pending = true;
                } else {
                    let old_ier = if i == 1 {
                        Some(self.reg.borrow().IER)
                    } else {
                        None
                    };
                    unsafe { self.reg_mut_ptr[i].write_volatile((data & (0xff)) as u8) };
                    if let Some(old_ier) = old_ier {
                        let new_ier = (data & 0xff) as u8;
                        // Re-arm THRE event when ETBEI (IER bit1) transitions 0->1.
                        if new_ier & 0x02 != 0 && old_ier & 0x02 == 0 {
                            self.thre_pending = true;
                            log::trace!(
                                "[UART] IER write: {:#04x} -> {:#04x}, ETBEI newly set, thre_pending=true",
                                old_ier,
                                new_ier
                            );
                        } else {
                            log::trace!("[UART] IER write: {:#04x} -> {:#04x}", old_ier, new_ier);
                        }
                    }
                }
                data >>= 8;
            }
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    fn read_RBR(&mut self) -> u8 {
        clear_bit(&mut self.reg.borrow_mut().LSR, 0); // receive data ready.
        self.reg.borrow().RBR
    }

    #[allow(non_snake_case)]
    fn write_RBR(&mut self, data: u8) {
        set_bit(&mut self.reg.borrow_mut().LSR, 0); // receive data ready.
        self.reg.borrow_mut().RBR = data
    }
}

impl PlicDevice for FastUart16550 {
    fn irq_level(&mut self) -> bool {
        let reg = self.reg.borrow();
        FastUart16550::eval_irq(
            reg.IER,
            self.thre_pending,
            read_bit(&reg.LSR, 0) || !self.input_rx.is_empty(),
        )
        .is_some()
    }
}

impl DeviceTrait for FastUart16550 {
    dispatch_read_write! { read_impl, write_impl }

    fn sync(&mut self) {}
}

impl MemMappedDeviceTrait for FastUart16550 {
    fn base() -> WordType {
        UART_BASE
    }
    fn size() -> WordType {
        UART_SIZE
    }
}

#[cfg(test)]
mod test {
    use crate::device::config::UART_IRQ;

    use super::*;

    #[test]
    fn output_test() {
        let (mut uart, mut port) = FastUart16550::new();

        uart.write_impl(0, 'a' as u8).unwrap();

        assert_eq!(port.take_output(), vec![b'a']);
        assert!(port.take_output().is_empty());
    }

    #[test]
    fn input_test() {
        let (mut uart, port) = FastUart16550::new();

        port.push_input(b"abcd").unwrap();

        assert_eq!(uart.read_impl::<u8>(5).unwrap() & 1u8, 1);
        assert_eq!(uart.read_impl::<u8>(0).unwrap(), 'a' as u8);
        assert_eq!(uart.read_impl::<u8>(0).unwrap(), 'b' as u8);
        assert_eq!(uart.read_impl::<u8>(0).unwrap(), 'c' as u8);
        assert_eq!(uart.read_impl::<u8>(0).unwrap(), 'd' as u8);
        assert_eq!(uart.read_impl::<u8>(5).unwrap() & 1u8, 0);
    }

    #[test]
    fn input_is_bounded() {
        let (mut uart, port) = FastUart16550::new();
        port.push_input(&vec![b'a'; UART_INPUT_CAPACITY]).unwrap();
        assert_eq!(
            port.push_input(b"b"),
            Err(UartIoError::InputFull { accepted: 0 })
        );

        assert_eq!(uart.read_impl::<u8>(0).unwrap(), b'a');
        port.push_input(b"b").unwrap();
    }

    // Interrupt evaluation is exposed as an absolute status for the PLIC.

    /// Receiving input while RDA (IER bit0) is enabled must raise UART_IRQ.
    #[test]
    fn input_raises_external_interrupt_when_rda_enabled() {
        let (mut uart, port) = FastUart16550::new();
        uart.write_impl::<u8>(1, 0x01).unwrap(); // enable Received Data Available (IER bit0)

        // No interrupt before any input arrives.
        assert_eq!(uart.irq_status(), None);

        port.push_input(b"x").unwrap();

        assert_eq!(uart.irq_status(), Some(UART_IRQ));
    }

    /// Without RDA enabled, incoming bytes are buffered but must not interrupt.
    #[test]
    fn input_without_rda_enabled_does_not_interrupt() {
        let (mut uart, port) = FastUart16550::new();

        port.push_input(b"x").unwrap();

        assert_eq!(uart.irq_status(), None);
    }

    /// RDA stays asserted until every queued byte has been read, then clears.
    /// IIR must identify the source as "Received Data Available" (0x04).
    #[test]
    fn rda_stays_asserted_until_input_fully_drained() {
        let (mut uart, port) = FastUart16550::new();
        uart.write_impl::<u8>(1, 0x01).unwrap(); // enable RDA
        port.push_input(b"ab").unwrap();

        assert_eq!(uart.irq_status(), Some(UART_IRQ));
        assert_eq!(uart.read_impl::<u8>(2).unwrap() & 0x0f, 0x04); // IIR: data available

        assert_eq!(uart.read_impl::<u8>(0).unwrap(), b'a');
        assert_eq!(uart.irq_status(), Some(UART_IRQ)); // 'b' still pending
        assert_eq!(uart.read_impl::<u8>(0).unwrap(), b'b');
        assert_eq!(uart.irq_status(), None); // fully drained
    }

    /// THRE (transmit) interrupts are input-independent: enabling ETBEI raises
    /// UART_IRQ with no bytes received and no `after_receive` call at all.
    /// Reading IIR identifies THRE (0x02) and clears the pending condition.
    #[test]
    fn thre_interrupt_is_input_independent() {
        let (mut uart, _port) = FastUart16550::new();
        uart.write_impl::<u8>(1, 0x02).unwrap(); // enable ETBEI (IER bit1)

        assert_eq!(uart.irq_status(), Some(UART_IRQ));
        assert_eq!(uart.read_impl::<u8>(2).unwrap() & 0x0f, 0x02); // IIR: THR empty
        assert_eq!(uart.irq_status(), None); // cleared, no storm
    }

    #[test]
    fn plic_device_reports_current_uart_irq_level() {
        let (mut uart, port) = FastUart16550::new();

        assert!(!uart.irq_level());

        uart.write_impl::<u8>(1, 0x01).unwrap();
        port.input_sender().try_send(b'x').unwrap();
        assert!(uart.irq_level());

        assert_eq!(uart.read_impl::<u8>(5).unwrap() & 1, 1);
        assert!(uart.irq_level()); // The channel is empty, but RBR still holds the byte.

        assert_eq!(uart.read_impl::<u8>(0).unwrap(), b'x');
        assert!(!uart.irq_level());

        uart.write_impl::<u8>(1, 0x02).unwrap();
        assert!(uart.irq_level());
        assert_eq!(uart.read_impl::<u8>(2).unwrap() & 0x0f, 0x02);
        assert!(!uart.irq_level());
    }
}
