use bitfield_struct::bitfield;
use bitflags::bitflags;

use crate::device::config::UART_DEFAULT_DIV;

pub type RBR = ReceiverBufferRegister; // byte 0 (R)
pub type THR = TransmitterHoldingRegister; // byte 0 (W)
pub type IER = InterruptEnableRegister; // byte 1
pub type IIR = InterruptIdentificationRegister; // byte 2 (R)
pub type FCR = FIFOControlRegister; // byte 2 (W)
pub type LCR = LineControlRegister; // byte 3
pub type MCR = ModemControlRegister; // byte 4
pub type LSR = LineStatusRegister; // byte 5
pub type MSR = ModemStatusRegister; // byte 6
pub type SCR = ScratchRegister; // byte 7
pub type DLL = DivisorLatchRegisterL; // byte 0
pub type DLH = DivisorLatchRegisterH; // byte 1

#[bitfield(u8)]
pub struct ReceiverBufferRegister {
    #[bits(8)]
    pub data: u8,
}

impl ReceiverBufferRegister {
    pub const fn offset() -> usize {
        0
    }
}

#[bitfield(u8)]
pub struct TransmitterHoldingRegister {
    #[bits(8)]
    pub data: u8,
}

impl TransmitterHoldingRegister {
    pub const fn offset() -> usize {
        0
    }
}

#[bitfield(u8)]
pub struct InterruptEnableRegister {
    /// Received Data available interrupt
    /// ‘0’ - disabled
    /// ‘1’ - enabled
    #[bits(1, default = false)]
    pub received_data_available_interrupt: bool,

    /// Transmitter Holding Register empty interrupt
    /// ‘0’ - disabled
    /// ‘1’ - enabled
    #[bits(1, default = false)]
    pub transmitter_holding_register_empty_interrupt: bool,

    /// Receiver Line Status Interrupt
    /// ‘0’ - disabled
    /// ‘1’ - enabled
    #[bits(1, default = false)]
    pub receiver_line_status_interrupt: bool,

    /// Modem Status Interrupt
    /// ‘0’ - disabled
    /// ‘1’ - enabled
    #[bits(1, default = false)]
    pub modem_status_interrupt: bool,

    /// Ignored
    #[bits(4, default=0b0000, access = RO)]
    ignored: u8,
}

impl InterruptEnableRegister {
    pub const fn offset() -> usize {
        1
    }
}

#[bitfield(u8)]
pub struct InterruptIdentificationRegister {
    /// indicates that an interrupt is pending when it’s logic ‘0’. When it’s ‘1’ - no interrupt is pending.
    #[bits(1, default = true)]
    pub pending_free_flag: bool,

    /// Possible interrupts reason, ordered by priority:
    /// 0b011: Parity, Overrun or Framing errors or Break Interrupt
    /// 0b010: FIFO trigger level reached
    /// 0b110: There’s at least 1 character in the FIFO but no character
    ///        has been input to the FIFO or read from it for the last 4 Char times.
    /// 0b001: Transmitter Holding Register Empty
    /// 0b000: CTS, DSR, RI or DCD.
    #[bits(3)]
    pub reason: u8,

    /// Ignored
    #[bits(4, default = 0b1100)]
    ignored: u8,
}

impl InterruptIdentificationRegister {
    pub const fn offset() -> usize {
        2
    }
}

#[repr(u8)]
pub enum InterruptReason {
    ModemStatus = 0b000,
    TransmitterHoldingRegisterEmpty = 0b001,
    ReceiverDataAvailable = 0b010,
    ReceiverLineStatus = 0b011,
    TimeoutInDication = 0b110,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct InterruptReasonBitflags: u8 {
        const MODEM_STATUS          = 1 << InterruptReason::ModemStatus as u8;
        const THRE                  = 1 << InterruptReason::TransmitterHoldingRegisterEmpty as u8; // TransmitterHoldingRegisterEmpty
        const RDA                   = 1 << InterruptReason::ReceiverDataAvailable as u8; // ReceiverDataAvailable
        const RECEIVER_LINE_STATUS  = 1 << InterruptReason::ReceiverLineStatus as u8;
        // const TimeoutInDication     = 1 << InterruptReason::TimeoutInDication as u8;
    }
}

#[bitfield(u8)]
pub struct FIFOControlRegister {
    /// Ignored (Used to enable FIFOs in NS16550D). Since this UART only supports FIFO mode,
    /// this bit is ignored.
    #[bits(1, default = false)]
    pub fifo_enable: bool,

    /// Writing a ‘1’ to bit 1 clears the Receiver FIFO and resets its logic. But it doesn’t
    /// clear the shift register, i.e. receiving of the current character continues.
    #[bits(1, default = false)]
    pub receiver_fifo_reset: bool,

    /// Writing a ‘1’ to bit 2 clears the Transmitter FIFO and resets its logic. The shift register is not cleared,
    /// i.e. transmitting of the current character continues.
    #[bits(1, default = false)]
    pub transmitter_fifo_reset: bool,

    /// Ignored
    #[bits(3, default = 0b000)]
    ignored: u8,

    /// Define the Receiver FIFO Interrupt trigger level
    /// ‘00’ - 1 byte
    /// ‘01’ - 4 bytes
    /// ‘10’ - 8 bytes
    /// ‘11’ - 14 bytes
    #[bits(2, default = 0b11)]
    pub receiver_fifo_interrupt_trigger_level: u8,
}

impl FIFOControlRegister {
    pub const fn offset() -> usize {
        2
    }
}

#[bitfield(u8)]
pub struct LineControlRegister {
    /// Select number of bits in each character
    /// ‘00’ - 5 bits
    /// ‘01’ - 6 bits
    /// ‘10’ - 7 bits
    /// ‘11’ - 8 bits
    #[bits(2, default = 0b11)]
    pub word_length: u8,

    /// Specify the number of generated stop bits
    /// ‘0’ - 1 stop bit
    /// ‘1’ - 1.5 stop bits when 5-bit character length selected and
    /// 2 bits otherwise
    /// Note that the receiver always checks the first stop bit only.
    #[bits(1, default = false)]
    pub stop_bit: bool,

    /// Parity Enable
    /// ‘0’ - No parity
    /// ‘1’ - Parity bit is generated on each outgoing character and is checked on each incoming one.
    #[bits(1, default = false)]
    pub parity_enable: bool,

    /// Even Parity select
    /// ‘0’ - Odd number of ‘1’ is transmitted and checked in each word (data and parity combined). In other
    /// words, if the data has an even number of ‘1’ in it, then the parity bit is ‘1’.
    /// ‘1’ - Even number of ‘1’ is transmitted in each word.
    #[bits(1, default = false)]
    pub even_parity: bool,

    /// Stick Parity bit.
    /// ‘0’ - Stick Parity disabled
    /// ‘1’ - If bits 3 and 4 are logic ‘1’, the parity bit is transmitted and checked as logic ‘0’. If bit 3 is ‘1’
    /// and bit 4 is ‘0’ then the parity bit is transmitted and checked as ‘1’.
    #[bits(1, default = false)]
    pub stick_parity: bool,

    /// Break Control bit
    /// ‘1’ - the serial out is forced into logic ‘0’ (break state).
    /// ‘0’ - break is disabled
    #[bits(1, default = false)]
    pub break_control: bool,

    /// Divisor Latch Access bit.
    ///‘1’ - The divisor latches can be accessed
    ///‘0’ - The normal registers are accessed
    #[bits(1, default = false)]
    pub divisor_latch_access: bool,
}

impl LineControlRegister {
    pub const fn offset() -> usize {
        3
    }
}

#[bitfield(u8)]
pub struct ModemControlRegister {
    /// Data Terminal Ready (DTR) signal control
    /// ‘0’ - DTR is ‘1’
    /// ‘1’ - DTR is ‘0’
    #[bits(1, default = false)]
    pub dtr: bool,

    /// Request To Send (RTS) signal control
    /// ‘0’ - RTS is ‘1’
    /// ‘1’ - RTS is ‘0’
    #[bits(1, default = false)]
    pub rts: bool,

    /// Out1. In loopback mode, connected to Ring Indicator (RI) signal input
    #[bits(1, default = false)]
    pub out1: bool,

    /// Out2. In loopback mode, connected to Data Carrier Detect (DCD) input
    #[bits(1, default = false)]
    pub out2: bool,

    /// Loopback mode
    /// ‘0’ - normal operation
    /// ‘1’ - loopback mode. When in loopback mode, the Serial Output Signal (STX_PAD_O) is set to logic
    /// ‘1’. The signal of the transmitter shift register is internally connected to the input of the receiver shift
    /// register.
    /// The following connections are made:
    /// DTR -> DSR
    /// RTS -> CTS
    /// Out1 -> RI
    /// Out2 -> DCD
    #[bits(1, default = false)]
    pub loopback_mode: bool,

    /// Ignored
    #[bits(3, default = 0b000)]
    ignored: u8,
}

impl ModemControlRegister {
    pub const fn offset() -> usize {
        4
    }
}

#[bitfield(u8)]
pub struct LineStatusRegister {
    /// Data Ready (DR) indicator.
    /// ‘0’ - No characters in the FIFO
    /// ‘1’ - At least one character has been received and is in the FIFO
    #[bits(1, default = false)]
    pub receive_data_available: bool,

    /// Overrun Error (OE) indicator
    /// ‘1’ - If the FIFO is full and another character has been received in the receiver shift register. If another
    /// character is starting to arrive, it will overwrite the data in the shift register but the FIFO will remain
    /// intact. The bit is cleared upon reading from the register. Generates Receiver Line Status interrupt.
    /// ‘0’ - No overrun state
    #[bits(1, default = false)]
    pub overrun_error: bool,

    /// Parity Error (PE) indicator
    /// ‘1’ - The character that is currently at the top of the FIFO has been received with parity error. The bit
    /// is cleared upon reading from the register. Generates Receiver Line Status interrupt.
    /// ‘0’ - No parity error in the current character
    #[bits(1, default = false)]
    pub parity_error: bool,

    /// Framing Error (FE) indicator
    /// ‘1’ - The received character at the top of the FIFO did not have a valid stop bit. Of course, generally,
    /// it might be that all the following data is corrupt. The bit is cleared upon reading from the register.
    /// Generates Receiver Line Status interrupt.
    /// ‘0’ - No framing error in the current character
    #[bits(1, default = false)]
    pub framing_error: bool,

    /// Break Interrupt (BI) indicator
    /// ‘1’ - A break condition has been reached in the current character. The break occurs when the line is
    /// held in logic 0 for a time of one character (start bit + data + parity + stop bit). In that case, one zero
    /// character enters the FIFO and the UART waits for a valid start bit to receive next character. The bit is
    /// cleared upon reading from the register. Generates Receiver Line Status interrupt.
    /// ‘0’ - No break condition in the current character
    #[bits(1, default = false)]
    pub break_interrupt: bool,

    /// Transmit FIFO is empty.
    /// ‘1’ - The transmitter FIFO is empty. Generates Transmitter Holding Register Empty interrupt. The
    /// bit is cleared when data is being written to the transmitter FIFO.
    /// ‘0’ - Otherwise
    #[bits(1, default = true)]
    pub transmit_holding_empty: bool,

    /// Transmitter Empty indicator.
    /// ‘1’ - Both the transmitter FIFO and transmitter shift register are empty. The bit is cleared when data
    /// is being written to the transmitter FIFO.
    /// ‘0’ - Otherwise
    #[bits(1, default = true)]
    pub transmit_empty: bool,

    /// ‘1’ - At least one parity error, framing error or break indications have been received and are inside
    /// the FIFO. The bit is cleared upon reading from the register.
    /// ‘0’ - Otherwise.
    #[bits(1, default = false)]
    pub fifo_error: bool,
}

impl LineStatusRegister {
    pub const fn offset() -> usize {
        5
    }

    pub fn line_status_interrupt(&self) -> bool {
        let value: u8 = (*self).into();
        (value & 0b00011110) != 0
    }

    /// Clear the receiver error indicators as required after an LSR read.
    pub fn clear_error_indicators(&mut self) {
        self.set_overrun_error(false);
        self.set_parity_error(false);
        self.set_framing_error(false);
        self.set_break_interrupt(false);
        self.set_fifo_error(false);
    }
}

#[bitfield(u8)]
pub struct ModemStatusRegister {
    #[bits(1, default = false)]
    pub delta_cts: bool,

    #[bits(1, default = false)]
    pub delta_dsr: bool,

    #[bits(1, default = false)]
    pub delta_ri: bool,

    #[bits(1, default = false)]
    pub delta_cd: bool,

    #[bits(1, default = false)]
    pub cts: bool,

    #[bits(1, default = false)]
    pub dsr: bool,

    #[bits(1, default = false)]
    pub ri: bool,

    #[bits(1, default = false)]
    pub cd: bool,
}

impl ModemStatusRegister {
    pub const fn offset() -> usize {
        6
    }

    pub fn interrupt_pending(&self) -> bool {
        let value: u8 = (*self).into();
        value & 0x0f != 0
    }

    /// Clear the modem delta indicators as required after an MSR read.
    pub fn clear_delta_indicators(&mut self) {
        self.set_delta_cts(false);
        self.set_delta_dsr(false);
        self.set_delta_ri(false);
        self.set_delta_cd(false);
    }
}

#[bitfield(u8)]
pub struct ScratchRegister {
    // Scratchpad Register data
    #[bits(8)]
    pub data: u8,
}

impl ScratchRegister {
    pub const fn offset() -> usize {
        7
    }
}

#[bitfield(u8)]
pub struct DivisorLatchRegisterL {
    /// divisor latches low bits
    #[bits(8, default = UART_DEFAULT_DIV as u8)]
    pub data: u8,
}

impl DivisorLatchRegisterL {
    pub const fn offset() -> usize {
        0
    }
}

#[bitfield(u8)]
pub struct DivisorLatchRegisterH {
    /// divisor latches high bits
    #[bits(8, default = (UART_DEFAULT_DIV >> 8) as u8)]
    pub data: u8,
}

impl DivisorLatchRegisterH {
    pub const fn offset() -> usize {
        1
    }
}

#[derive(Default)]
pub struct Uart16550RegLayout {
    pub receiver_buffer_register: ReceiverBufferRegister, // byte 0 (R)
    pub transmitter_holding_register: TransmitterHoldingRegister, // byte 0 (W)
    pub interrupt_enable_register: InterruptEnableRegister, // byte 1
    pub interrupt_identification_register: InterruptIdentificationRegister, // byte 2 (R)
    pub fifo_control_register: FIFOControlRegister,       // byte 2 (W)
    pub line_control_register: LineControlRegister,       // byte 3
    pub modem_control_register: ModemControlRegister,     // byte 4
    pub line_status_register: LineStatusRegister,         // byte 5
    pub modem_status_register: ModemStatusRegister,       // byte 6
    pub scratch_register: ScratchRegister,                // byte 7
    pub divisor_latch_register_l: DivisorLatchRegisterL,  // byte 0
    pub divisor_latch_register_h: DivisorLatchRegisterH,  // byte 1
}

impl Uart16550RegLayout {
    #[inline]
    pub(super) fn divisor_latch_enable(&self) -> bool {
        self.line_control_register.divisor_latch_access()
    }

    pub fn read(&self, offset: usize) -> u8 {
        let divisor = self.divisor_latch_enable();
        match offset {
            0 if !divisor => self.receiver_buffer_register.into(),
            0 if divisor => self.divisor_latch_register_l.into(),
            1 if !divisor => self.interrupt_enable_register.into(),
            1 if divisor => self.divisor_latch_register_h.into(),
            2 => self.interrupt_identification_register.into(),
            3 => self.line_control_register.into(),
            4 => self.modem_control_register.into(),
            5 => self.line_status_register.into(),
            6 => self.modem_status_register.into(),
            7 => self.scratch_register.into(),
            _ => unreachable!(),
        }
    }

    pub fn write(&mut self, offset: usize, value: u8) {
        let divisor = self.divisor_latch_enable();
        match offset {
            0 if !divisor => self.transmitter_holding_register = THR::from_bits(value),
            0 if divisor => self.divisor_latch_register_l = DLL::from_bits(value),
            1 if !divisor => self.interrupt_enable_register = IER::from_bits(value),
            1 if divisor => self.divisor_latch_register_h = DLH::from_bits(value),
            2 => self.fifo_control_register = FCR::from_bits(value),
            3 => self.line_control_register = LCR::from_bits(value),
            4 => self.modem_control_register = MCR::from_bits(value),
            5 => self.line_status_register = LSR::from_bits(value),
            6 => self.modem_status_register = MSR::from_bits(value),
            7 => self.scratch_register = SCR::from_bits(value),
            _ => unreachable!(),
        }
    }

    pub fn get_divisor(&self) -> u16 {
        (self.divisor_latch_register_l.data() as u16)
            + ((self.divisor_latch_register_h.data() as u16) << 8)
    }
}
