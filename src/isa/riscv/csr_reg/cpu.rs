use crate::{
    config::arch_config::WordType,
    isa::riscv::{
        csr_reg::{NamedCsrReg, csr_macro::Satp},
        executor::RVCPU,
        trap::Exception,
    },
};

impl RVCPU {
    pub fn read_csr(&mut self, addr: WordType) -> Result<WordType, Exception> {
        if addr == 0xc01 {
            if let Some(time_addr) = self.time_addr {
                if let Ok(time) = self.memory.read_by_paddr::<u64>(time_addr) {
                    return Ok(time as WordType);
                }
            }
        } else if let Some(data) = self.csr.read_checked(addr) {
            return Ok(data);
        }

        Err(Exception::IllegalInstruction)
    }

    /// Write a CSR and apply CPU-wide side effects.
    pub fn write_csr(&mut self, addr: WordType, data: WordType) -> Result<(), Exception> {
        if !self.csr.write_checked(addr, data) {
            log::warn!("Failed to write CSR {:#x} with data {:#x}", addr, data);
            return Err(Exception::IllegalInstruction);
        }

        // satp.MODE changes take effect immediately, without SFENCE.VMA.
        if addr == Satp::get_index() {
            let satp = self.csr.get_by_type_existing::<Satp>();
            self.memory.set_mode(satp.get_mode() as u8);
            self.memory.set_root_ppn(satp.get_ppn() as u64);
            self.memory.flush_tlb();
            self.flush_icache();
        }

        Ok(())
    }
}
