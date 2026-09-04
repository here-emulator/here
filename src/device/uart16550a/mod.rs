pub mod io;
mod reg;

pub use io::{UART_INPUT_CAPACITY, UartBytePort, UartIoError, UartIoMode};

use std::ops::Not;

use tokio::sync::mpsc::{self, Receiver, UnboundedSender};

use crate::{
    config::arch_config::WordType,
    device::{DeviceTrait, PlicDevice, uart16550a::reg::*},
};

pub struct Uart16550A {
    reg: Uart16550RegLayout,
    input_rx: Receiver<u8>,
    output_tx: UnboundedSender<u8>,
    ip: InterruptReasonBitflags,
}

impl Uart16550A {
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
        let reg = Uart16550RegLayout::default();
        Self {
            reg,
            input_rx,
            output_tx,
            ip: InterruptReasonBitflags::THRE,
        }
    }

    /// Compute a simplified IIR (Interrupt Identification Register) view based on current IER/LSR/FCR state.
    fn update_iir(&mut self) {
        let ier = self.reg.interrupt_enable_register;
        let lsr = self.reg.line_status_register;
        let ip = self.ip;

        // IIR[7:6] advertise FIFO support only after FCR[0] has been set.
        // Keep the reset value compatible with a plain 16550 (0x01).
        let fifo_bits = if self.reg.fifo_control_register.fifo_enable() {
            0xc0
        } else {
            0
        };
        let mut iir = InterruptIdentificationRegister::from_bits(fifo_bits | 1);

        // Check interrupt conditions in priority order.
        if ier.receiver_line_status_interrupt()
            && (ip.contains(InterruptReasonBitflags::RECEIVER_LINE_STATUS)
                || lsr.line_status_interrupt())
        {
            iir = iir
                .with_reason(InterruptReason::ReceiverLineStatus as u8)
                .with_pending_free_flag(false);
        } else if ier.received_data_available_interrupt()
            && ip.contains(InterruptReasonBitflags::RDA)
        {
            iir = iir
                .with_reason(InterruptReason::ReceiverDataAvailable as u8)
                .with_pending_free_flag(false);
        } else if ier.transmitter_holding_register_empty_interrupt()
            && ip.contains(InterruptReasonBitflags::THRE)
        {
            iir = iir
                .with_reason(InterruptReason::TransmitterHoldingRegisterEmpty as u8)
                .with_pending_free_flag(false);
        } else if ier.modem_status_interrupt()
            && (ip.contains(InterruptReasonBitflags::MODEM_STATUS)
                || self.reg.modem_status_register.interrupt_pending())
        {
            iir = iir
                .with_reason(InterruptReason::ModemStatus as u8)
                .with_pending_free_flag(false);
        }

        log::trace!(
            "[UART] compute_iir: IER={:?} LSR={:?} ip={:?} => IIR={:?}",
            ier,
            lsr,
            self.ip,
            iir
        );

        self.reg.interrupt_identification_register = iir
    }

    // if iir.reason = thre, clear thre pending bit and update iir.
    fn try_clear_thre_interrupt(&mut self) {
        let reason = self.reg.interrupt_identification_register.reason();
        if reason == InterruptReason::TransmitterHoldingRegisterEmpty as u8 {
            self.ip.remove(InterruptReasonBitflags::THRE);
            self.update_iir();
        }
    }

    fn try_raise_thre_interrupt(&mut self, old_ier: IER, new_ier: IER) {
        if !old_ier.transmitter_holding_register_empty_interrupt()
            && new_ier.transmitter_holding_register_empty_interrupt()
            && self.reg.line_status_register.transmit_holding_empty()
        {
            self.ip.insert(InterruptReasonBitflags::THRE);
        }
    }

    #[inline]
    fn update_rx_status(&mut self) {
        if self.reg.line_status_register.receive_data_available() == false {
            // check fifo
            if let Ok(data) = self.input_rx.try_recv() {
                self.reg.receiver_buffer_register = RBR::from_bits(data);
                self.reg
                    .line_status_register
                    .set_receive_data_available(true);
                self.ip.insert(InterruptReasonBitflags::RDA);
            }
        }
    }
}

impl DeviceTrait for Uart16550A {
    fn read(&mut self, addr: WordType, len: u32) -> Result<u64, super::MemError> {
        // Poll host input before servicing the access.  Otherwise the first
        // RBR read after input arrives returns the previous register value and
        // only makes the newly received byte visible to the following read.
        self.update_rx_status();
        self.update_iir();

        let mut data = 0u64;
        let offset = addr as usize;
        for i in offset..(offset + len as usize) {
            // read receive data buffer
            if self.reg.line_control_register.divisor_latch_access() == false && i == RBR::offset()
            {
                self.reg
                    .line_status_register
                    .set_receive_data_available(false);
                self.ip.remove(InterruptReasonBitflags::RDA);
            }

            data |= (self.reg.read(i) as u64) << (8 * (i - offset));

            if i == LSR::offset() {
                self.reg.line_status_register.clear_error_indicators();
                self.ip
                    .remove(InterruptReasonBitflags::RECEIVER_LINE_STATUS);
            } else if i == MSR::offset() {
                self.reg.modem_status_register.clear_delta_indicators();
                self.ip.remove(InterruptReasonBitflags::MODEM_STATUS);
            }

            if i == IIR::offset() {
                // read iir
                self.try_clear_thre_interrupt();
            }
        }

        self.update_rx_status();
        self.update_iir();
        Ok(data)
    }

    fn write(
        &mut self,
        addr: crate::config::arch_config::WordType,
        len: u32,
        mut data: u64,
    ) -> Result<(), super::MemError> {
        let offset = addr as usize;

        for i in offset..offset + len as usize {
            if i == 0 && self.reg.divisor_latch_enable() == false {
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
                // Since this UART sends instantly, we just re-arm the THRE event.
                self.ip.insert(InterruptReasonBitflags::THRE);
            }
            let old_ier = self.reg.interrupt_enable_register;
            let new_ier = IER::from_bits(data as u8);

            self.reg.write(i, data as u8);

            // FCR remains selected at offset 2 even while DLAB selects DLL/DLM
            // at offsets 0 and 1.
            if i == FCR::offset() {
                let fcr = self.reg.fifo_control_register;
                if fcr.receiver_fifo_reset() {
                    while self.input_rx.try_recv().is_ok() {}
                    self.reg.receiver_buffer_register = RBR::from_bits(0);
                    self.reg
                        .line_status_register
                        .set_receive_data_available(false);
                    self.reg.line_status_register.clear_error_indicators();
                    self.ip.remove(InterruptReasonBitflags::RDA);
                    self.ip
                        .remove(InterruptReasonBitflags::RECEIVER_LINE_STATUS);
                }
                if fcr.transmitter_fifo_reset() {
                    self.reg
                        .line_status_register
                        .set_transmit_holding_empty(true);
                    self.reg.line_status_register.set_transmit_empty(true);
                    self.ip.insert(InterruptReasonBitflags::THRE);
                }
            }

            // if unmask ier.thre bit, and transmit FIFO is empty now, raise a thre interrupt.
            if i == IER::offset() && !self.reg.divisor_latch_enable() {
                self.try_raise_thre_interrupt(old_ier, new_ier);
            }

            data >>= 8;
            self.update_iir();
        }

        Ok(())
    }

    fn sync(&mut self) {}
}

impl PlicDevice for Uart16550A {
    fn irq_level(&mut self) -> bool {
        self.update_rx_status();
        self.update_iir();
        let iir = self.reg.interrupt_identification_register;
        iir.pending_free_flag().not()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        board::virt::config::UART_IRQ,
        device::{DeviceTrait, PlicDevice},
    };

    use super::*;

    #[test]
    fn output_test() {
        let (mut uart, mut port) = Uart16550A::new();

        uart.write_u8(0, b'a').unwrap();

        assert_eq!(port.take_output(), vec![b'a']);
        assert!(port.take_output().is_empty());
    }

    #[test]
    fn input_test() {
        let (mut uart, port) = Uart16550A::new();

        port.push_input(b"abcd").unwrap();

        assert_eq!(uart.read_u8(5).unwrap() & 1, 1);
        assert_eq!(uart.read_u8(0).unwrap(), b'a');
        assert_eq!(uart.read_u8(0).unwrap(), b'b');
        assert_eq!(uart.read_u8(0).unwrap(), b'c');
        assert_eq!(uart.read_u8(0).unwrap(), b'd');
        assert_eq!(uart.read_u8(5).unwrap() & 1, 0);
    }

    #[test]
    fn input_is_bounded() {
        let (mut uart, port) = Uart16550A::new();
        port.push_input(&vec![b'a'; UART_INPUT_CAPACITY]).unwrap();
        assert_eq!(
            port.push_input(b"b"),
            Err(UartIoError::InputFull { accepted: 0 })
        );

        assert_eq!(uart.read_u8(0).unwrap(), b'a');
        port.push_input(b"b").unwrap();
    }

    #[test]
    fn input_raises_external_interrupt_when_rda_enabled() {
        let (mut uart, port) = Uart16550A::new();
        uart.write_u8(1, 0x01).unwrap();

        assert!(!uart.irq_level());
        port.push_input(b"x").unwrap();
        assert!(uart.irq_level());
    }

    #[test]
    fn input_without_rda_enabled_does_not_interrupt() {
        let (mut uart, port) = Uart16550A::new();

        port.push_input(b"x").unwrap();

        assert!(!uart.irq_level());
    }

    #[test]
    fn rda_stays_asserted_until_input_fully_drained() {
        let (mut uart, port) = Uart16550A::new();
        uart.write_u8(1, 0x01).unwrap();
        port.push_input(b"ab").unwrap();

        assert!(uart.irq_level());
        assert_eq!(uart.read_u8(2).unwrap() & 0x0f, 0x04);
        assert_eq!(uart.read_u8(0).unwrap(), b'a');
        assert!(uart.irq_level());
        assert_eq!(uart.read_u8(0).unwrap(), b'b');
        assert!(!uart.irq_level());
    }

    #[test]
    fn thre_interrupt_is_input_independent() {
        let (mut uart, _port) = Uart16550A::new();
        uart.write_u8(1, 0x02).unwrap();

        assert!(uart.irq_level());
        assert_eq!(uart.read_u8(2).unwrap() & 0x0f, 0x02);
        assert!(!uart.irq_level());
    }

    #[test]
    fn plic_device_reports_current_uart_irq_level() {
        let (mut uart, port) = Uart16550A::new();

        assert!(!uart.irq_level());

        uart.write_u8(1, 0x01).unwrap();
        port.input_sender().try_send(b'x').unwrap();
        assert!(uart.irq_level());
        assert_eq!(uart.read_u8(5).unwrap() & 1, 1);
        assert!(uart.irq_level());
        assert_eq!(uart.read_u8(0).unwrap(), b'x');
        assert!(!uart.irq_level());

        uart.write_u8(1, 0x02).unwrap();
        assert_eq!(uart.irq_level(), true);
        assert_eq!(uart.read_u8(2).unwrap() & 0x0f, 0x02);
        assert_eq!(uart.irq_level(), false);

        // Keep the source ID in this test so it also guards the board wiring
        // contract shared with the previous implementation.
        assert_eq!(UART_IRQ, 10);
    }

    #[test]
    fn iir_fifo_status_follows_fcr() {
        let (mut uart, _port) = Uart16550A::new();

        assert_eq!(u8::from(uart.reg.fifo_control_register) & 1, 0);
        assert_eq!(uart.read_u8(2).unwrap() & 0xc0, 0);
        uart.write_u8(2, 0x01).unwrap();
        assert_eq!(uart.read_u8(2).unwrap() & 0xc0, 0xc0);
    }

    #[test]
    fn divisor_latch_access_selects_dll_and_dlm() {
        let (mut uart, _port) = Uart16550A::new();

        uart.write_u8(3, 0x80).unwrap();
        uart.write_u8(0, 0x34).unwrap();
        uart.write_u8(1, 0x12).unwrap();
        assert_eq!(uart.read_u8(0).unwrap(), 0x34);
        assert_eq!(uart.read_u8(1).unwrap(), 0x12);

        uart.write_u8(3, 0x03).unwrap();
        uart.write_u8(0, b'z').unwrap();
        assert_eq!(uart.read_u8(0).unwrap(), 0);
    }

    #[test]
    fn line_status_interrupt_is_reported_and_cleared_by_lsr_read() {
        let (mut uart, _port) = Uart16550A::new();

        uart.write_u8(5, 0x1e).unwrap();
        uart.write_u8(1, 0x04).unwrap();
        assert!(uart.irq_level());
        assert_eq!(uart.read_u8(2).unwrap() & 0x0f, 0x06);
        assert_eq!(uart.read_u8(5).unwrap() & 0x1e, 0x1e);
        assert!(!uart.irq_level());
    }

    #[test]
    fn modem_status_interrupt_is_reported_and_cleared_by_msr_read() {
        let (mut uart, _port) = Uart16550A::new();

        uart.write_u8(6, 0x01).unwrap();
        uart.write_u8(1, 0x08).unwrap();
        assert!(uart.irq_level());
        assert_eq!(uart.read_u8(2).unwrap() & 0x0f, 0x00);
        assert_eq!(uart.read_u8(6).unwrap() & 0x0f, 0x01);
        assert!(!uart.irq_level());
    }

    #[test]
    fn fifo_reset_clears_receiver_and_rearms_transmitter_state() {
        let (mut uart, port) = Uart16550A::new();

        uart.write_u8(1, 0x01).unwrap();
        port.push_input(b"xy").unwrap();
        assert!(uart.irq_level());
        uart.write_u8(2, 0x02).unwrap();
        assert!(!uart.irq_level());
        assert_eq!(uart.read_u8(0).unwrap(), 0);

        uart.write_u8(1, 0x02).unwrap();
        assert!(uart.irq_level());
        assert_eq!(uart.read_u8(2).unwrap() & 0x0f, 0x02);
        assert!(!uart.irq_level());
        uart.write_u8(2, 0x04).unwrap();
        assert!(uart.irq_level());
    }

    #[test]
    fn fcr_remains_accessible_while_dlab_selects_divisors() {
        let (mut uart, _port) = Uart16550A::new();

        uart.write_u8(3, 0x80).unwrap();
        uart.write_u8(2, 0x01).unwrap();
        assert_eq!(uart.read_u8(2).unwrap() & 0xc0, 0xc0);
    }

    #[test]
    fn programming_dlm_does_not_enable_thre_interrupt() {
        let (mut uart, _port) = Uart16550A::new();

        uart.write_u8(3, 0x80).unwrap();
        uart.write_u8(1, 0x02).unwrap();
        uart.write_u8(3, 0x03).unwrap();
        assert!(!uart.irq_level());
    }
}
