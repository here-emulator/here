//! SiFive SPI host controller.
//!
//! The register layout follows the Linux `spi-sifive` driver: `SCKDIV`,
//! `SCKMODE`, `CSID/CSDEF/CSMODE`, frame `FMT`, `TXDATA/RXDATA` FIFO ports,
//! watermarks, flash-format registers, and `IE/IP` interrupt registers. Board
//! code can opt in later by mapping [`SifiveSpiController`] at the desired MMIO base
//! and wiring its optional interrupt source to the PLIC.

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};

use crate::{
    config::arch_config::WordType,
    device::{DeviceTrait, MemError, plic::PeriphIrqId},
};

const SIFIVE_SPI_REGISTER_SIZE: WordType = 0x1000;
const SIFIVE_SPI_DEFAULT_FIFO_DEPTH: usize = 8;
const SIFIVE_SPI_NO_CHIP_SELECT: u32 = u32::MAX;

pub(super) mod sifive_spi_reg_offset {
    use crate::config::arch_config::WordType;

    pub const SCKDIV: WordType = 0x00; // Serial clock divisor
    pub const SCKMODE: WordType = 0x04; // Serial clock mode
    pub const CSID: WordType = 0x10; // Chip select ID
    pub const CSDEF: WordType = 0x14; // Chip select default
    pub const CSMODE: WordType = 0x18; // Chip select mode
    pub const DELAY0: WordType = 0x28; // Delay control 0
    pub const DELAY1: WordType = 0x2c; // Delay control 1
    pub const FMT: WordType = 0x40; // Frame format
    pub const TXDATA: WordType = 0x48; // Tx FIFO Data
    pub const RXDATA: WordType = 0x4c; // Rx FIFO data
    pub const TXMARK: WordType = 0x50; // Tx FIFO watermark
    pub const RXMARK: WordType = 0x54; // Rx FIFO watermark
    pub const FCTRL: WordType = 0x60; // SPI flash interface control*
    pub const FFMT: WordType = 0x64; // SPI flash instruction format*
    pub const IE: WordType = 0x70; // SPI interrupt enable
    pub const IP: WordType = 0x74; // SPI interrupt pending
}

pub(super) mod sckmode {
    pub const PHA: u32 = 1 << 0;
    pub const POL: u32 = 1 << 1;
    pub const MODE_MASK: u32 = PHA | POL;
}

pub(super) mod csmode {
    pub const AUTO: u32 = 0;
    pub const HOLD: u32 = 2;
    pub const OFF: u32 = 3;
    pub const MODE_MASK: u32 = 3;
}

pub(super) mod delay0 {
    pub const CSSCK_MASK: u32 = 0xff;
    pub const SCKCS_MASK: u32 = 0xff << 16;
}

pub(super) mod delay1 {
    pub const INTERCS_MASK: u32 = 0xff;
    pub const INTERXFR_MASK: u32 = 0xff << 16;
}

pub(super) mod fmt {
    pub const PROTO_MASK: u32 = 0x3;
    pub const ENDIAN: u32 = 1 << 2;
    pub const DIR: u32 = 1 << 3;
    pub const LEN_OFFSET: u32 = 16;
    pub const LEN_MASK: u32 = 0xf << LEN_OFFSET;
}

pub(super) mod txdata {
    pub const DATA_MASK: u32 = 0xff;
    pub const FULL: u32 = 1 << 31;
}

pub(super) mod rxdata {
    pub const DATA_MASK: u32 = 0xff;
    pub const EMPTY: u32 = 1 << 31;
}

pub(super) mod ip {
    pub const TXWM: u32 = 1 << 0;
    pub const RXWM: u32 = 1 << 1;
    pub const MASK: u32 = TXWM | RXWM;
}

/// Byte-oriented SPI slave contract used by the controller.
///
/// Each `transfer` call clocks one full-duplex frame. Chip-select edges are
/// reported separately so stateful slaves can commit or abort transactions.
pub(super) trait SpiSlaveDevice {
    fn transfer(&mut self, mosi: u8) -> u8;

    fn set_selected(&mut self, selected: bool);

    fn sync(&mut self) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SifiveSpiAttachError {
    InvalidChipSelect,
    ChipSelectInUse,
}

/// SiFive-compatible SPI host controller.
///
/// The model keeps the SiFive register surface Linux expects, while draining
/// TXDATA immediately into the selected slave and buffering returned bytes in
/// a small RX FIFO.
pub(super) struct SifiveSpiController {
    sckdiv: u32,
    sckmode: u32,
    csid: u32,
    csdef: u32,
    csmode: u32,
    delay0: u32,
    delay1: u32,
    fmt: u32,
    txmark: u32,
    rxmark: u32,
    fctrl: u32,
    ffmt: u32,
    ie: Arc<AtomicU32>,
    ip: Arc<AtomicU32>,
    rx_fifo: VecDeque<u8>,
    irq_id: Option<PeriphIrqId>,
    slaves: Vec<Option<Box<dyn SpiSlaveDevice>>>,
}

impl SifiveSpiController {
    pub fn new(cs_num: usize) -> Self {
        Self::with_irq(cs_num, None)
    }

    pub fn with_irq(cs_num: usize, irq_id: Option<PeriphIrqId>) -> Self {
        let csdef = Self::cs_mask(cs_num);
        Self {
            sckdiv: 0,
            sckmode: 0,
            csid: 0,
            csdef,
            csmode: csmode::AUTO,
            delay0: 1 | (1 << 16),
            delay1: 1,
            fmt: 8 << fmt::LEN_OFFSET,
            txmark: 1,
            rxmark: 0,
            fctrl: 0,
            ffmt: 0,
            ie: Arc::new(AtomicU32::new(0)),
            ip: Arc::new(AtomicU32::new(ip::TXWM)),
            rx_fifo: VecDeque::new(),
            irq_id,
            slaves: (0..cs_num).map(|_| None).collect(),
        }
    }

    pub fn attach_slave<D>(
        &mut self,
        chip_select: usize,
        device: D,
    ) -> Result<(), SifiveSpiAttachError>
    where
        D: SpiSlaveDevice + 'static,
    {
        let selected = self.selected_chip() == Some(chip_select);
        let Some(slot) = self.slaves.get_mut(chip_select) else {
            return Err(SifiveSpiAttachError::InvalidChipSelect);
        };

        if slot.is_some() {
            return Err(SifiveSpiAttachError::ChipSelectInUse);
        }

        *slot = Some(Box::new(device));
        if selected {
            slot.as_mut().unwrap().set_selected(true);
        }
        Ok(())
    }

    pub fn selected_chip(&self) -> Option<usize> {
        (self.csmode == csmode::HOLD)
            .then_some(self.csid as usize)
            .filter(|chip_select| *chip_select < self.slaves.len())
    }

    fn cs_mask(chip_selects: usize) -> u32 {
        if chip_selects >= u32::BITS as usize {
            u32::MAX
        } else {
            (1u32 << chip_selects) - 1
        }
    }

    fn update_selected_chip(&mut self, old_selected: Option<usize>, new_selected: Option<usize>) {
        if old_selected == new_selected {
            return;
        }

        if let Some(old) = old_selected {
            if let Some(slave) = self.slaves[old].as_mut() {
                slave.set_selected(false);
            }
        }

        if let Some(new) = new_selected {
            if let Some(slave) = self.slaves[new].as_mut() {
                slave.set_selected(true);
            }
        }
    }

    fn set_csid(&mut self, value: u32) {
        let old_selected = self.selected_chip();
        self.csid = value.min(self.slaves.len().saturating_sub(1) as u32);
        self.update_selected_chip(old_selected, self.selected_chip());
    }

    fn set_csmode(&mut self, value: u32) {
        let old_selected = self.selected_chip();
        self.csmode = value & csmode::MODE_MASK;
        self.update_selected_chip(old_selected, self.selected_chip());
    }

    fn lsb_first(&self) -> bool {
        self.fmt & fmt::ENDIAN != 0
    }

    fn rx_enabled(&self) -> bool {
        self.fmt & fmt::DIR == 0
    }

    fn tx_full(&self) -> bool {
        false
    }

    fn tx_watermark_pending(&self) -> bool {
        // This model drains TXDATA immediately, so the TX FIFO is always below
        // the watermark observed by Linux `spi-sifive`.
        true
    }

    fn rx_watermark_pending(&self) -> bool {
        self.rx_fifo.len() > self.rxmark as usize
    }

    fn refresh_ip(&mut self) {
        let mut pending = 0;
        if self.tx_watermark_pending() {
            pending |= ip::TXWM;
        }
        if self.rx_watermark_pending() {
            pending |= ip::RXWM;
        }
        self.ip.store(pending, Ordering::Release);
    }

    fn transfer_byte(&mut self, value: u8) {
        let Some(chip_select) = self.selected_chip() else {
            return;
        };

        let lsb_first = self.lsb_first();
        let Some(slave) = self.slaves[chip_select].as_mut() else {
            return;
        };

        let tx = if lsb_first {
            value.reverse_bits()
        } else {
            value
        };
        let rx = slave.transfer(tx);

        if self.rx_enabled() && self.rx_fifo.len() < SIFIVE_SPI_DEFAULT_FIFO_DEPTH {
            self.rx_fifo
                .push_back(if lsb_first { rx.reverse_bits() } else { rx });
        }
    }

    fn read_rxdata(&mut self) -> u32 {
        match self.rx_fifo.pop_front() {
            Some(data) => {
                self.refresh_ip();
                data as u32
            }
            None => rxdata::EMPTY,
        }
    }

    fn read_register(&mut self, addr: WordType) -> Result<u32, MemError> {
        let value = match addr {
            sifive_spi_reg_offset::SCKDIV => self.sckdiv,
            sifive_spi_reg_offset::SCKMODE => self.sckmode,
            sifive_spi_reg_offset::CSID => self.csid,
            sifive_spi_reg_offset::CSDEF => self.csdef,
            sifive_spi_reg_offset::CSMODE => self.csmode,
            sifive_spi_reg_offset::DELAY0 => self.delay0,
            sifive_spi_reg_offset::DELAY1 => self.delay1,
            sifive_spi_reg_offset::FMT => self.fmt,
            sifive_spi_reg_offset::TXDATA => {
                if self.tx_full() {
                    txdata::FULL
                } else {
                    0
                }
            }
            sifive_spi_reg_offset::RXDATA => self.read_rxdata(),
            sifive_spi_reg_offset::TXMARK => self.txmark,
            sifive_spi_reg_offset::RXMARK => self.rxmark,
            sifive_spi_reg_offset::FCTRL => self.fctrl,
            sifive_spi_reg_offset::FFMT => self.ffmt,
            sifive_spi_reg_offset::IE => self.ie.load(Ordering::Acquire),
            sifive_spi_reg_offset::IP => {
                self.refresh_ip();
                self.ip.load(Ordering::Acquire)
            }
            _ => return Err(MemError::LoadFault),
        };

        Ok(value)
    }

    fn write_register(&mut self, addr: WordType, value: u32) -> Result<(), MemError> {
        match addr {
            sifive_spi_reg_offset::SCKDIV => self.sckdiv = value & 0xfff,
            sifive_spi_reg_offset::SCKMODE => self.sckmode = value & sckmode::MODE_MASK,
            sifive_spi_reg_offset::CSID => self.set_csid(value),
            sifive_spi_reg_offset::CSDEF => self.csdef = value & Self::cs_mask(self.slaves.len()),
            sifive_spi_reg_offset::CSMODE => self.set_csmode(value),
            sifive_spi_reg_offset::DELAY0 => {
                self.delay0 = value & (delay0::CSSCK_MASK | delay0::SCKCS_MASK)
            }
            sifive_spi_reg_offset::DELAY1 => {
                self.delay1 = value & (delay1::INTERCS_MASK | delay1::INTERXFR_MASK)
            }
            sifive_spi_reg_offset::FMT => {
                self.fmt = value & (fmt::PROTO_MASK | fmt::ENDIAN | fmt::DIR | fmt::LEN_MASK)
            }
            sifive_spi_reg_offset::TXDATA => {
                self.transfer_byte((value & txdata::DATA_MASK) as u8);
                self.refresh_ip();
            }
            sifive_spi_reg_offset::TXMARK => {
                self.txmark = value.min((SIFIVE_SPI_DEFAULT_FIFO_DEPTH - 1) as u32);
                self.refresh_ip();
            }
            sifive_spi_reg_offset::RXMARK => {
                self.rxmark = value.min((SIFIVE_SPI_DEFAULT_FIFO_DEPTH - 1) as u32);
                self.refresh_ip();
            }
            sifive_spi_reg_offset::FCTRL => self.fctrl = value,
            sifive_spi_reg_offset::FFMT => self.ffmt = value,
            sifive_spi_reg_offset::IE => self.ie.store(value & ip::MASK, Ordering::Release),
            sifive_spi_reg_offset::RXDATA | sifive_spi_reg_offset::IP => {
                return Err(MemError::StoreFault);
            }
            _ => return Err(MemError::StoreFault),
        }

        Ok(())
    }

    fn read_impl<T>(&mut self, addr: WordType) -> Result<T, MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        match size_of::<T>() {
            4 if addr % 4 == 0 => Ok(T::truncate_from(self.read_register(addr)?)),
            _ => Err(MemError::LoadFault),
        }
    }

    fn write_impl<T>(&mut self, addr: WordType, data: T) -> Result<(), MemError>
    where
        T: crate::utils::UnsignedInteger,
    {
        match size_of::<T>() {
            4 if addr % 4 == 0 => self.write_register(addr, data.truncate_to()),
            _ => Err(MemError::StoreFault),
        }
    }
}

impl DeviceTrait for SifiveSpiController {
    dispatch_read_write! { read_impl, write_impl }

    fn sync(&mut self) {
        for slave in self.slaves.iter_mut().flatten() {
            slave.sync();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::device::spi::w25q512::{
        W25Q512JVQ, W25Q512JVQ_JEDEC_ID, W25Q512JVQ_SECTOR_SIZE, command,
    };

    fn test_image_path(name: &str) -> String {
        std::fs::create_dir_all("./tmp").unwrap();
        format!("./tmp/{}", name)
    }

    fn transfer(spi: &mut SifiveSpiController, byte: u8) -> u8 {
        spi.write_impl::<u32>(sifive_spi_reg_offset::TXDATA, byte as u32)
            .unwrap();
        let data = spi.read_impl::<u32>(sifive_spi_reg_offset::RXDATA).unwrap();
        assert_eq!(data & rxdata::EMPTY, 0);
        (data & rxdata::DATA_MASK) as u8
    }

    struct EchoSlave;

    impl SpiSlaveDevice for EchoSlave {
        fn transfer(&mut self, mosi: u8) -> u8 {
            mosi.wrapping_add(1)
        }
        fn set_selected(&mut self, _selected: bool) {}
    }

    #[test]
    fn transfer_to_selected_slave_through_sifive_txdata_rxdata() {
        let mut spi = SifiveSpiController::new(2);
        spi.attach_slave(1, EchoSlave).unwrap();

        spi.write_impl::<u32>(sifive_spi_reg_offset::CSID, 1)
            .unwrap();
        spi.write_impl::<u32>(sifive_spi_reg_offset::CSMODE, csmode::HOLD)
            .unwrap();
        spi.write_impl::<u32>(sifive_spi_reg_offset::TXDATA, 0x41)
            .unwrap();

        assert_eq!(
            spi.read_impl::<u32>(sifive_spi_reg_offset::RXDATA).unwrap() & rxdata::DATA_MASK,
            0x42
        );
        assert_ne!(
            spi.read_impl::<u32>(sifive_spi_reg_offset::RXDATA).unwrap() & rxdata::EMPTY,
            0
        );
    }

    // #[test]
    // fn irq_pending_tracks_rx_watermark() {
    //     let mut spi = SifiveSpiController::with_irq(1, Some(12));
    //     spi.attach_slave(0, EchoSlave).unwrap();

    //     spi.write_impl::<u32>(sifive_spi_reg_offset::CSMODE, csmode::HOLD)
    //         .unwrap();
    //     spi.write_impl::<u32>(sifive_spi_reg_offset::RXMARK, 0)
    //         .unwrap();
    //     spi.write_impl::<u32>(sifive_spi_reg_offset::IE, ip::RXWM)
    //         .unwrap();
    //     spi.write_impl::<u32>(sifive_spi_reg_offset::TXDATA, 0x10)
    //         .unwrap();

    //     let mut poller = spi.get_poll_event().unwrap();
    //     assert_eq!(poller.poll_nonblocking(), PlicIRQState::new(12, true));
    //     assert_eq!(
    //         spi.read_impl::<u32>(sifive_spi_reg_offset::RXDATA).unwrap() & rxdata::DATA_MASK,
    //         0x11
    //     );
    //     assert_eq!(poller.poll_nonblocking(), PlicIRQState::new(12, false));
    // }

    #[test]
    fn connects_controller_to_w25q512jvq_block_slave() {
        let path = test_image_path("spi_sifive_w25q512jvq.img");
        let _ = std::fs::remove_file(&path);
        let flash = W25Q512JVQ::with_capacity(&path, W25Q512JVQ_SECTOR_SIZE).unwrap();
        let mut spi = SifiveSpiController::new(1);

        spi.attach_slave(0, flash).unwrap();
        spi.write_impl::<u32>(sifive_spi_reg_offset::CSMODE, csmode::HOLD)
            .unwrap();

        transfer(&mut spi, command::READ_JEDEC_ID);

        for expected in W25Q512JVQ_JEDEC_ID {
            assert_eq!(transfer(&mut spi, 0), expected);
        }
    }

    #[test]
    fn changing_csid_while_held_deselects_previous_slave() {
        let path0 = test_image_path("spi_sifive_csid_flash0.img");
        let path1 = test_image_path("spi_sifive_csid_flash1.img");
        let _ = std::fs::remove_file(&path0);
        let _ = std::fs::remove_file(&path1);
        let flash0 = W25Q512JVQ::with_capacity(&path0, W25Q512JVQ_SECTOR_SIZE).unwrap();
        let flash1 = W25Q512JVQ::with_capacity(&path1, W25Q512JVQ_SECTOR_SIZE).unwrap();
        let mut spi = SifiveSpiController::new(2);

        spi.attach_slave(0, flash0).unwrap();
        spi.attach_slave(1, flash1).unwrap();
        spi.write_impl::<u32>(sifive_spi_reg_offset::CSID, 0)
            .unwrap();
        spi.write_impl::<u32>(sifive_spi_reg_offset::CSMODE, csmode::HOLD)
            .unwrap();

        transfer(&mut spi, command::WRITE_ENABLE);
        spi.write_impl::<u32>(sifive_spi_reg_offset::CSMODE, csmode::AUTO)
            .unwrap();
        spi.write_impl::<u32>(sifive_spi_reg_offset::CSMODE, csmode::HOLD)
            .unwrap();
        transfer(&mut spi, command::PAGE_PROGRAM);
        for byte in [0, 0, 4, 0xaa] {
            transfer(&mut spi, byte);
        }

        spi.write_impl::<u32>(sifive_spi_reg_offset::CSID, 1)
            .unwrap();

        assert_eq!(std::fs::read(&path0).unwrap()[4], 0xaa);
        assert_eq!(std::fs::read(&path1).unwrap()[4], 0xff);
    }
}
