use crate::riscv::cpu::RVCPU;
use crate::riscv::csr_reg::{NamedCsr, Satp};
use crate::riscv::trap::Exception;
use here_types::WordType;

impl RVCPU {
    pub fn read_csr(&mut self, addr: WordType) -> Result<WordType, Exception> {
        if addr == 0xc01 {
            if let Some(time_addr) = self.time_addr {
                if let Ok(time) = self.memory.read_by_paddr::<u64>(time_addr) {
                    return Ok(time as WordType);
                }
            }
        } else if let Some(data) = self.csr.read_checked(addr, self.privilege_level()) {
            return Ok(data);
        }

        Err(Exception::IllegalInstruction)
    }

    /// Write a CSR and apply CPU-wide side effects.
    pub fn write_csr(&mut self, addr: WordType, data: WordType) -> Result<(), Exception> {
        if !self.csr.write_checked(addr, data, self.privilege_level()) {
            log::warn!("Failed to write CSR {:#x} with data {:#x}", addr, data);
            return Err(Exception::IllegalInstruction);
        }

        // satp.MODE changes take effect immediately, without an SFENCE.VMA.
        if addr == Satp::as_addr() {
            let mode = Satp::get_mode(&self.csr) as u8;
            let ppn = Satp::get_ppn(&self.csr) as u64;
            self.memory.set_mode(mode);
            self.memory.set_root_ppn(ppn);
            self.memory.flush_tlb();
            self.flush_icache();
        }

        Ok(())
    }
}
