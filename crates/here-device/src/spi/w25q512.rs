use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};

use super::spi_sifive::SpiSlaveDevice;

pub const W25Q512JVQ_CAPACITY: u64 = 64 * 1024 * 1024;
pub const W25Q512JVQ_SECTOR_SIZE: u64 = 4 * 1024;
pub const W25Q512JVQ_BLOCK_32K_SIZE: u64 = 32 * 1024;
pub const W25Q512JVQ_BLOCK_64K_SIZE: u64 = 64 * 1024;
pub const W25Q512JVQ_PAGE_SIZE: usize = 256;
pub const W25Q512JVQ_JEDEC_ID: [u8; 3] = [0xef, 0x40, 0x20];
pub const ERASED_BYTE: u8 = 0xff;

pub mod command {
    pub const WRITE_ENABLE: u8 = 0x06;
    pub const WRITE_DISABLE: u8 = 0x04;
    pub const READ_STATUS1: u8 = 0x05;
    pub const READ_STATUS2: u8 = 0x35;
    pub const READ_STATUS3: u8 = 0x15;
    pub const WRITE_STATUS1_2: u8 = 0x01;
    pub const WRITE_STATUS2: u8 = 0x31;
    pub const WRITE_STATUS3: u8 = 0x11;
    pub const READ_DATA: u8 = 0x03;
    pub const READ_DATA_4B: u8 = 0x13;
    pub const FAST_READ: u8 = 0x0b;
    pub const FAST_READ_4B: u8 = 0x0c;
    pub const PAGE_PROGRAM: u8 = 0x02;
    pub const PAGE_PROGRAM_4B: u8 = 0x12;
    pub const SECTOR_ERASE_4K: u8 = 0x20;
    pub const SECTOR_ERASE_4K_4B: u8 = 0x21;
    pub const BLOCK_ERASE_32K: u8 = 0x52;
    pub const BLOCK_ERASE_64K: u8 = 0xd8;
    pub const BLOCK_ERASE_64K_4B: u8 = 0xdc;
    pub const CHIP_ERASE1: u8 = 0xc7;
    pub const CHIP_ERASE2: u8 = 0x60;
    pub const READ_JEDEC_ID: u8 = 0x9f;
    pub const READ_SFDP: u8 = 0x5a;
    pub const ENTER_4BYTE_ADDR: u8 = 0xb7;
    pub const EXIT_4BYTE_ADDR: u8 = 0xe9;
    pub const ENABLE_RESET: u8 = 0x66;
    pub const RESET_DEVICE: u8 = 0x99;
}

mod status1 {
    pub const BUSY: u8 = 1 << 0;
    pub const WRITE_ENABLE_LATCH: u8 = 1 << 1;
}

mod status2 {
    pub const QUAD_ENABLE: u8 = 1 << 1;
}

/// Address width requested by the current command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddressKind {
    Current,
    ThreeByte,
    FourByte,
}

impl AddressKind {
    fn len(self, four_byte_address_mode: bool) -> usize {
        match self {
            Self::Current if four_byte_address_mode => 4,
            Self::Current | Self::ThreeByte => 3,
            Self::FourByte => 4,
        }
    }
}

/// Operation to start once the command address bytes have been collected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddressAction {
    Read { dummy_len: usize },
    Program,
    Erase { len: u64 },
    Sfdp,
}

/// In-flight SPI NOR command state.
///
/// Commands are byte-streamed while chip select is asserted. Mutating commands
/// such as page program and status writes are finalized on the deselect edge.
enum Transaction {
    Idle,
    Ignore,
    ReadStatus1,
    ReadStatus2,
    ReadStatus3,
    ReadJedecId {
        index: usize,
    },
    CollectAddress {
        action: AddressAction,
        addr_len: usize,
        buf: Vec<u8>,
    },
    Dummy {
        action: AddressAction,
        address: u64,
        remaining: usize,
    },
    ReadData {
        next_addr: u64,
    },
    ReadSfdp {
        next_addr: u32,
    },
    PageProgram {
        start_addr: u64,
        data: Vec<u8>,
    },
    WriteStatus {
        target: StatusWriteTarget,
        data: Vec<u8>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatusWriteTarget {
    Status1And2,
    Status2,
    Status3,
}

/// Winbond W25Q512JVQ SPI NOR flash backed by a host image file.
///
/// The backing file is grown to the chip capacity and initialized with erased
/// bytes, so Linux can treat it as persistent flash storage through SPI NOR.
pub struct W25Q512JVQ {
    file: File,
    capacity: u64,
    transaction: Transaction,
    status1: u8,
    status2: u8,
    status3: u8,
    four_byte_address_mode: bool,
    reset_enabled: bool,
}

pub type SpiBlockDevice = W25Q512JVQ;

impl W25Q512JVQ {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::with_capacity(path, W25Q512JVQ_CAPACITY)
    }

    pub(super) fn with_capacity<P: AsRef<Path>>(path: P, capacity: u64) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .append(false)
            .create(true)
            .open(path)?;
        Self::ensure_capacity(&mut file, capacity)?;

        Ok(Self {
            file,
            capacity,
            transaction: Transaction::Idle,
            status1: 0,
            status2: status2::QUAD_ENABLE,
            status3: 0,
            four_byte_address_mode: false,
            reset_enabled: false,
        })
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.capacity
    }

    pub fn capacity_sectors(&self) -> u64 {
        self.capacity / W25Q512JVQ_SECTOR_SIZE
    }

    pub fn page_size(&self) -> usize {
        W25Q512JVQ_PAGE_SIZE
    }

    fn ensure_capacity(file: &mut File, capacity: u64) -> io::Result<()> {
        let old_len = file.seek(SeekFrom::End(0))?;
        if old_len >= capacity {
            return Ok(());
        }

        file.set_len(capacity)?;
        Self::fill_file_range(file, old_len, capacity - old_len, ERASED_BYTE)
    }

    fn fill_file_range(file: &mut File, offset: u64, len: u64, value: u8) -> io::Result<()> {
        const CHUNK_LEN: usize = 16 * 1024;

        file.seek(SeekFrom::Start(offset))?;
        let chunk = [value; CHUNK_LEN];
        let mut remaining = len;
        while remaining != 0 {
            let write_len = remaining.min(CHUNK_LEN as u64) as usize;
            file.write_all(&chunk[..write_len])?;
            remaining -= write_len as u64;
        }
        Ok(())
    }

    fn read_byte(&mut self, addr: u64) -> u8 {
        if addr >= self.capacity {
            return ERASED_BYTE;
        }

        let mut byte = [ERASED_BYTE];
        if self
            .file
            .seek(SeekFrom::Start(addr))
            .and_then(|_| self.file.read_exact(&mut byte))
            .is_err()
        {
            return ERASED_BYTE;
        }
        byte[0]
    }

    fn write_nor_bytes(&mut self, addr: u64, data: &[u8]) -> io::Result<()> {
        if addr >= self.capacity || data.is_empty() {
            return Ok(());
        }

        let write_len = data.len().min((self.capacity - addr) as usize);
        let mut old = vec![ERASED_BYTE; write_len];
        self.file.seek(SeekFrom::Start(addr))?;
        self.file.read_exact(&mut old)?;

        for (old_byte, new_byte) in old.iter_mut().zip(data.iter().take(write_len)) {
            *old_byte &= *new_byte;
        }

        self.file.seek(SeekFrom::Start(addr))?;
        self.file.write_all(&old)?;
        Ok(())
    }

    fn erase_range(&mut self, addr: u64, len: u64) -> io::Result<()> {
        if addr >= self.capacity {
            return Ok(());
        }

        let start = addr & !(len - 1);
        let len = len.min(self.capacity - start);
        Self::fill_file_range(&mut self.file, start, len, ERASED_BYTE)
    }

    fn write_enable(&mut self, enabled: bool) {
        if enabled {
            self.status1 |= status1::WRITE_ENABLE_LATCH;
        } else {
            self.status1 &= !status1::WRITE_ENABLE_LATCH;
        }
    }

    fn write_is_enabled(&self) -> bool {
        self.status1 & status1::WRITE_ENABLE_LATCH != 0
    }

    fn reset(&mut self) {
        self.transaction = Transaction::Idle;
        self.status1 &= !status1::BUSY;
        self.write_enable(false);
        self.four_byte_address_mode = false;
        self.reset_enabled = false;
    }

    fn ignore_until_deselect(&mut self) {
        self.transaction = Transaction::Ignore;
    }

    fn start_command(&mut self, opcode: u8) {
        self.transaction = Transaction::Idle;

        match opcode {
            command::WRITE_ENABLE => {
                self.write_enable(true);
                self.ignore_until_deselect();
            }
            command::WRITE_DISABLE => {
                self.write_enable(false);
                self.ignore_until_deselect();
            }
            command::READ_STATUS1 => self.transaction = Transaction::ReadStatus1,
            command::READ_STATUS2 => self.transaction = Transaction::ReadStatus2,
            command::READ_STATUS3 => self.transaction = Transaction::ReadStatus3,
            command::READ_JEDEC_ID => self.transaction = Transaction::ReadJedecId { index: 0 },
            command::READ_DATA => self.collect_read(0, AddressKind::Current),
            command::READ_DATA_4B => self.collect_read(0, AddressKind::FourByte),
            command::FAST_READ => self.collect_read(1, AddressKind::Current),
            command::FAST_READ_4B => self.collect_read(1, AddressKind::FourByte),
            command::PAGE_PROGRAM => self.collect_program(AddressKind::Current),
            command::PAGE_PROGRAM_4B => self.collect_program(AddressKind::FourByte),
            command::SECTOR_ERASE_4K => {
                self.collect_erase(W25Q512JVQ_SECTOR_SIZE, AddressKind::Current)
            }
            command::SECTOR_ERASE_4K_4B => {
                self.collect_erase(W25Q512JVQ_SECTOR_SIZE, AddressKind::FourByte)
            }
            command::BLOCK_ERASE_32K => {
                self.collect_erase(W25Q512JVQ_BLOCK_32K_SIZE, AddressKind::Current)
            }
            command::BLOCK_ERASE_64K => {
                self.collect_erase(W25Q512JVQ_BLOCK_64K_SIZE, AddressKind::Current)
            }
            command::BLOCK_ERASE_64K_4B => {
                self.collect_erase(W25Q512JVQ_BLOCK_64K_SIZE, AddressKind::FourByte)
            }
            command::CHIP_ERASE1 | command::CHIP_ERASE2 => {
                if self.write_is_enabled() {
                    if let Err(error) = self.erase_range(0, self.capacity) {
                        log::error!("w25q512jvq chip erase failed: {}", error);
                    }
                    self.write_enable(false);
                }
                self.ignore_until_deselect();
            }
            command::READ_SFDP => self.collect_address(AddressAction::Sfdp, AddressKind::ThreeByte),
            command::ENTER_4BYTE_ADDR => {
                self.four_byte_address_mode = true;
                self.ignore_until_deselect();
            }
            command::EXIT_4BYTE_ADDR => {
                self.four_byte_address_mode = false;
                self.ignore_until_deselect();
            }
            command::WRITE_STATUS1_2 => {
                self.collect_status_write(StatusWriteTarget::Status1And2, 2);
            }
            command::WRITE_STATUS2 => {
                self.collect_status_write(StatusWriteTarget::Status2, 1);
            }
            command::WRITE_STATUS3 => {
                self.collect_status_write(StatusWriteTarget::Status3, 1);
            }
            command::ENABLE_RESET => {
                self.reset_enabled = true;
                self.ignore_until_deselect();
            }
            command::RESET_DEVICE if self.reset_enabled => {
                self.reset();
                self.ignore_until_deselect();
            }
            _ => self.ignore_until_deselect(),
        }
    }

    fn collect_read(&mut self, dummy_len: usize, kind: AddressKind) {
        self.collect_address(AddressAction::Read { dummy_len }, kind);
    }

    fn collect_program(&mut self, kind: AddressKind) {
        self.collect_write_address(AddressAction::Program, kind);
    }

    fn collect_erase(&mut self, len: u64, kind: AddressKind) {
        self.collect_write_address(AddressAction::Erase { len }, kind);
    }

    fn collect_write_address(&mut self, action: AddressAction, kind: AddressKind) {
        if self.write_is_enabled() {
            self.collect_address(action, kind);
        } else {
            self.ignore_until_deselect();
        }
    }

    fn collect_address(&mut self, action: AddressAction, kind: AddressKind) {
        self.transaction = Transaction::CollectAddress {
            action,
            addr_len: kind.len(self.four_byte_address_mode),
            buf: Vec::new(),
        };
    }

    fn collect_status_write(&mut self, target: StatusWriteTarget, len: usize) {
        if self.write_is_enabled() {
            self.transaction = Transaction::WriteStatus {
                target,
                data: Vec::with_capacity(len),
            };
        } else {
            self.ignore_until_deselect();
        }
    }

    fn address_from_bytes(buf: &[u8]) -> u64 {
        buf.iter().fold(0, |addr, byte| (addr << 8) | *byte as u64)
    }

    fn on_address_complete(&mut self, action: AddressAction, address: u64) {
        match action {
            AddressAction::Read { dummy_len: 0 } => {
                self.transaction = Transaction::ReadData { next_addr: address };
            }
            AddressAction::Read { dummy_len } => {
                self.transaction = Transaction::Dummy {
                    action,
                    address,
                    remaining: dummy_len,
                };
            }
            AddressAction::Program => {
                if self.write_is_enabled() {
                    self.transaction = Transaction::PageProgram {
                        start_addr: address,
                        data: Vec::with_capacity(W25Q512JVQ_PAGE_SIZE),
                    };
                } else {
                    self.ignore_until_deselect();
                }
            }
            AddressAction::Erase { len } => {
                if self.write_is_enabled() {
                    if let Err(error) = self.erase_range(address, len) {
                        log::error!("w25q512jvq erase failed: {}", error);
                    }
                    self.write_enable(false);
                }
                self.ignore_until_deselect();
            }
            AddressAction::Sfdp => {
                self.transaction = Transaction::Dummy {
                    action,
                    address,
                    remaining: 1,
                };
            }
        }
    }

    fn read_sfdp_byte(&self, addr: u32) -> u8 {
        match addr {
            0..=3 => b"SFDP"[addr as usize],
            // SFDP revision 1.6, no parameter headers. Linux can still fall
            // back to the Winbond JEDEC table entry when it needs no-SFDP data.
            4 => 0x06,
            5 => 0x01,
            6 => 0x00,
            7 => 0xff,
            _ => 0xff,
        }
    }

    fn finish_page_program(&mut self, start_addr: u64, data: &[u8]) {
        if data.is_empty() {
            self.write_enable(false);
            return;
        }

        let page_base = start_addr & !((W25Q512JVQ_PAGE_SIZE as u64) - 1);
        let mut page = [ERASED_BYTE; W25Q512JVQ_PAGE_SIZE];
        for (idx, byte) in data.iter().enumerate() {
            let page_offset = (start_addr as usize + idx) & (W25Q512JVQ_PAGE_SIZE - 1);
            page[page_offset] = *byte;
        }

        if let Err(error) = self.write_nor_bytes(page_base, &page) {
            log::error!("w25q512jvq page program failed: {}", error);
        }
        self.write_enable(false);
    }

    fn finish_status_write(&mut self, target: StatusWriteTarget, data: &[u8]) {
        match target {
            StatusWriteTarget::Status1And2 => {
                if let Some(status1) = data.first() {
                    self.status1 = *status1 & !status1::BUSY;
                }
                if let Some(status2) = data.get(1) {
                    self.status2 = *status2;
                }
            }
            StatusWriteTarget::Status2 => {
                if let Some(status2) = data.first() {
                    self.status2 = *status2;
                }
            }
            StatusWriteTarget::Status3 => {
                if let Some(status3) = data.first() {
                    self.status3 = *status3;
                }
            }
        }
        self.write_enable(false);
    }

    fn finish_transaction(&mut self) {
        let transaction = std::mem::replace(&mut self.transaction, Transaction::Idle);
        match transaction {
            Transaction::PageProgram { start_addr, data } => {
                self.finish_page_program(start_addr, &data);
            }
            Transaction::WriteStatus { target, data } => {
                self.finish_status_write(target, &data);
            }
            _ => {}
        }
    }
}

impl SpiSlaveDevice for W25Q512JVQ {
    fn transfer(&mut self, mosi: u8) -> u8 {
        let transaction = std::mem::replace(&mut self.transaction, Transaction::Idle);

        match transaction {
            Transaction::Idle => {
                self.start_command(mosi);
                0
            }
            Transaction::Ignore => {
                self.transaction = Transaction::Ignore;
                0
            }
            Transaction::ReadStatus1 => {
                self.transaction = Transaction::ReadStatus1;
                self.status1
            }
            Transaction::ReadStatus2 => {
                self.transaction = Transaction::ReadStatus2;
                self.status2
            }
            Transaction::ReadStatus3 => {
                self.transaction = Transaction::ReadStatus3;
                self.status3
            }
            Transaction::ReadJedecId { index } => {
                self.transaction = Transaction::ReadJedecId { index: index + 1 };
                W25Q512JVQ_JEDEC_ID[index % W25Q512JVQ_JEDEC_ID.len()]
            }
            Transaction::CollectAddress {
                action,
                addr_len,
                mut buf,
            } => {
                buf.push(mosi);
                if buf.len() == addr_len {
                    let address = Self::address_from_bytes(&buf);
                    self.on_address_complete(action, address);
                } else {
                    self.transaction = Transaction::CollectAddress {
                        action,
                        addr_len,
                        buf,
                    };
                }
                0
            }
            Transaction::Dummy {
                action,
                address,
                remaining,
            } => {
                if remaining > 1 {
                    self.transaction = Transaction::Dummy {
                        action,
                        address,
                        remaining: remaining - 1,
                    };
                } else {
                    match action {
                        AddressAction::Read { .. } => {
                            self.transaction = Transaction::ReadData { next_addr: address };
                        }
                        AddressAction::Sfdp => {
                            self.transaction = Transaction::ReadSfdp {
                                next_addr: address as u32,
                            };
                        }
                        AddressAction::Program | AddressAction::Erase { .. } => {}
                    }
                }
                0
            }
            Transaction::ReadData { next_addr } => {
                self.transaction = Transaction::ReadData {
                    next_addr: next_addr.wrapping_add(1),
                };
                self.read_byte(next_addr)
            }
            Transaction::ReadSfdp { next_addr } => {
                self.transaction = Transaction::ReadSfdp {
                    next_addr: next_addr.wrapping_add(1),
                };
                self.read_sfdp_byte(next_addr)
            }
            Transaction::PageProgram {
                start_addr,
                mut data,
            } => {
                data.push(mosi);
                self.transaction = Transaction::PageProgram { start_addr, data };
                0
            }
            Transaction::WriteStatus { target, mut data } => {
                data.push(mosi);
                self.transaction = Transaction::WriteStatus { target, data };
                0
            }
        }
    }

    fn set_selected(&mut self, selected: bool) {
        if selected {
            self.transaction = Transaction::Idle;
        } else {
            self.finish_transaction();
        }
    }

    fn sync(&mut self) {
        if let Err(error) = self.file.flush() {
            log::error!("w25q512jvq flush failed: {}", error);
        }
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use crate::device::{
        DeviceTrait,
        spi::spi_sifive::{SifiveSpiController, csmode, rxdata, sifive_spi_reg_offset},
    };

    use super::*;

    fn test_image_path(name: &str) -> String {
        format!("./tmp/{}", name)
    }

    fn init_image(path: &str, len: usize) {
        let _ = fs::create_dir_all("./tmp");
        fs::write(path, vec![ERASED_BYTE; len]).unwrap();
    }

    fn build_flash(path: &str) -> W25Q512JVQ {
        W25Q512JVQ::with_capacity(path, W25Q512JVQ_SECTOR_SIZE).unwrap()
    }

    fn build_spi(path: &str) -> SifiveSpiController {
        let flash = build_flash(path);
        let mut spi = SifiveSpiController::new(1);
        spi.attach_slave(0, flash).unwrap();
        spi.write_u32(sifive_spi_reg_offset::CSMODE, csmode::HOLD)
            .unwrap();
        spi
    }

    fn deselect(spi: &mut SifiveSpiController) {
        spi.write_u32(sifive_spi_reg_offset::CSMODE, csmode::AUTO)
            .unwrap();
    }

    fn select(spi: &mut SifiveSpiController) {
        spi.write_u32(sifive_spi_reg_offset::CSMODE, csmode::HOLD)
            .unwrap();
    }

    fn transfer(spi: &mut SifiveSpiController, byte: u8) -> u8 {
        spi.write_u32(sifive_spi_reg_offset::TXDATA, byte as u32)
            .unwrap();
        let data = spi.read_u32(sifive_spi_reg_offset::RXDATA).unwrap();
        assert_eq!(data & rxdata::EMPTY, 0);
        (data & rxdata::DATA_MASK) as u8
    }

    fn read_status1(spi: &mut SifiveSpiController) -> u8 {
        select(spi);
        transfer(spi, command::READ_STATUS1);
        let status = transfer(spi, 0);
        deselect(spi);
        status
    }

    #[test]
    fn reads_jedec_id_for_linux_spi_nor_match() {
        let path = test_image_path("w25q512jvq_jedec.img");
        init_image(&path, 0);
        let mut spi = build_spi(&path);

        transfer(&mut spi, command::READ_JEDEC_ID);

        assert_eq!(transfer(&mut spi, 0), 0xef);
        assert_eq!(transfer(&mut spi, 0), 0x40);
        assert_eq!(transfer(&mut spi, 0), 0x20);
    }

    #[test]
    fn page_program_requires_write_enable_and_obeys_nor_bits() {
        let path = test_image_path("w25q512jvq_program.img");
        init_image(&path, 0);
        let mut spi = build_spi(&path);

        transfer(&mut spi, command::PAGE_PROGRAM);
        for byte in [0, 0, 4, 0xaa] {
            transfer(&mut spi, byte);
        }
        deselect(&mut spi);
        assert_eq!(fs::read(&path).unwrap()[4], ERASED_BYTE);

        select(&mut spi);
        transfer(&mut spi, command::WRITE_ENABLE);
        deselect(&mut spi);
        select(&mut spi);
        transfer(&mut spi, command::PAGE_PROGRAM);
        for byte in [0, 0, 4, 0xaa, 0x0f] {
            transfer(&mut spi, byte);
        }
        deselect(&mut spi);

        let data = fs::read(&path).unwrap();
        assert_eq!(&data[4..6], &[0xaa, 0x0f]);

        select(&mut spi);
        transfer(&mut spi, command::WRITE_ENABLE);
        deselect(&mut spi);
        select(&mut spi);
        transfer(&mut spi, command::PAGE_PROGRAM);
        for byte in [0, 0, 4, 0xff] {
            transfer(&mut spi, byte);
        }
        deselect(&mut spi);

        assert_eq!(fs::read(&path).unwrap()[4], 0xaa);
    }

    #[test]
    fn rejected_program_ignores_remaining_bytes_until_deselect() {
        let path = test_image_path("w25q512jvq_rejected_program.img");
        init_image(&path, 0);
        let mut spi = build_spi(&path);

        transfer(&mut spi, command::PAGE_PROGRAM);
        for byte in [0, 0, 4, command::WRITE_ENABLE] {
            transfer(&mut spi, byte);
        }
        deselect(&mut spi);

        assert_eq!(fs::read(&path).unwrap()[4], ERASED_BYTE);
        assert_eq!(read_status1(&mut spi) & status1::WRITE_ENABLE_LATCH, 0);
    }

    #[test]
    fn write_enable_ignores_extra_bytes_until_deselect() {
        let path = test_image_path("w25q512jvq_write_enable_extra.img");
        init_image(&path, 0);
        let mut spi = build_spi(&path);

        transfer(&mut spi, command::WRITE_ENABLE);
        for byte in [command::PAGE_PROGRAM, 0, 0, 4, 0xaa] {
            transfer(&mut spi, byte);
        }
        deselect(&mut spi);

        assert_eq!(fs::read(&path).unwrap()[4], ERASED_BYTE);
        assert_ne!(read_status1(&mut spi) & status1::WRITE_ENABLE_LATCH, 0);
    }

    #[test]
    fn erase_restores_erased_bytes() {
        let path = test_image_path("w25q512jvq_erase.img");
        init_image(&path, 0);
        let mut spi = build_spi(&path);

        transfer(&mut spi, command::WRITE_ENABLE);
        deselect(&mut spi);
        select(&mut spi);
        transfer(&mut spi, command::PAGE_PROGRAM);
        for byte in [0, 0, 8, 0x00] {
            transfer(&mut spi, byte);
        }
        deselect(&mut spi);
        assert_eq!(fs::read(&path).unwrap()[8], 0);

        select(&mut spi);
        transfer(&mut spi, command::WRITE_ENABLE);
        deselect(&mut spi);
        select(&mut spi);
        transfer(&mut spi, command::SECTOR_ERASE_4K);
        for byte in [0, 0, 8] {
            transfer(&mut spi, byte);
        }
        deselect(&mut spi);

        assert_eq!(fs::read(&path).unwrap()[8], ERASED_BYTE);
    }

    #[test]
    fn supports_4byte_read_opcode() {
        let path = test_image_path("w25q512jvq_read4b.img");
        init_image(&path, 0);
        let mut image = fs::OpenOptions::new().write(true).open(&path).unwrap();
        image.seek(SeekFrom::Start(0x100)).unwrap();
        image.write_all(&[0x11, 0x22]).unwrap();
        drop(image);

        let mut spi = build_spi(&path);
        transfer(&mut spi, command::READ_DATA_4B);
        for byte in [0, 0, 1, 0] {
            transfer(&mut spi, byte);
        }

        assert_eq!(transfer(&mut spi, 0), 0x11);
        assert_eq!(transfer(&mut spi, 0), 0x22);
    }

    #[test]
    fn exposes_minimal_sfdp_header() {
        let path = test_image_path("w25q512jvq_sfdp.img");
        init_image(&path, 0);
        let mut spi = build_spi(&path);

        transfer(&mut spi, command::READ_SFDP);
        for byte in [0, 0, 0, 0] {
            transfer(&mut spi, byte);
        }

        assert_eq!(transfer(&mut spi, 0), b'S');
        assert_eq!(transfer(&mut spi, 0), b'F');
        assert_eq!(transfer(&mut spi, 0), b'D');
        assert_eq!(transfer(&mut spi, 0), b'P');
    }
}
