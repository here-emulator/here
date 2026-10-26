use crate::{
    config::arch_config::WordType,
    device::MemError,
    isa::{
        DebugTarget,
        riscv::{
            RawInstr, RiscvTypes,
            address::Address,
            csr_reg::PrivilegeLevel,
            decoder::DecodeInstr,
            executor::RVCPU,
            mmu::{AccessType, PageTableError},
        },
    },
    utils::UnsignedInteger,
};

// TODO: `DebugTarget` is not needed to be a trait,
// we can directly implement these methods on `RVCPU` and call them in `Debugger`,
// because no more targets will be added in the future (even if one day we will, extract `DebugTarget` trait at that time is easy)
impl DebugTarget<RiscvTypes> for RVCPU {
    fn read_pc(&self) -> WordType {
        self.pc
    }

    fn write_pc(&mut self, new_pc: WordType) {
        self.pc = new_pc;
    }

    fn read_reg(&self, idx: u8) -> WordType {
        self.reg_file[idx as usize]
    }

    fn write_reg(&mut self, idx: u8, value: WordType) {
        self.reg_file.write(idx, value)
    }

    /// Fetch the instruction at `vaddr`, respecting its length.
    ///
    /// The low 16 bits are read first to learn whether the instruction is a
    /// 2-byte RVC instruction or a full 4-byte one. This avoids a misaligned
    /// 32-bit fetch when a 4-byte instruction sits at a 2-byte boundary (e.g.
    /// right after a compressed instruction).
    fn read_instr(&mut self, vaddr: WordType) -> Result<RawInstr, MemError> {
        let lo = self.read_for_debug_ifetch::<u16>(vaddr)? as u32;
        if lo & 0b11 != 0b11 {
            return Ok(RawInstr::from(lo));
        }
        let hi = self.read_for_debug_ifetch::<u16>(vaddr.wrapping_add(2))? as u32;
        Ok(RawInstr::from(lo | (hi << 16)))
    }

    fn read_instr_directly(&mut self, addr: Address) -> Result<RawInstr, MemError> {
        let lo = self.memory.debug_read::<u16>(addr)? as u32;
        if lo & 0b11 != 0b11 {
            return Ok(RawInstr::from(lo));
        }
        let hi = self.memory.debug_read::<u16>(addr + 2)? as u32;
        Ok(RawInstr::from(lo | (hi << 16)))
    }

    fn read_memory<T: UnsignedInteger>(&mut self, addr: Address) -> Result<T, MemError> {
        self.memory.debug_read::<T>(addr)
    }

    fn write_memory<T: UnsignedInteger>(&mut self, addr: Address, data: T) -> Result<(), MemError> {
        self.memory.debug_write::<T>(addr, data)
    }

    fn get_current_privilege(&self) -> PrivilegeLevel {
        self.csr.privelege_level()
    }

    fn read_float_reg(&self, idx: u8) -> (f32, f64) {
        (self.fpu.load::<f32>(idx), self.fpu.load::<f64>(idx))
    }

    fn read_vector_reg<T>(&self, idx: u8) -> Option<&[T]> {
        match self.vector.read_as_type(idx) {
            Ok(data) => Some(data),
            Err(_) => None,
        }
    }

    /// match input {
    ///     Some() => `Read Write`,
    ///     None => `Read Only`,
    /// }
    fn debug_csr(&mut self, addr: WordType, new_value: Option<WordType>) -> Option<WordType> {
        self.csr.debug(addr, new_value)
    }

    fn decoded_instr(&self, instr: RawInstr) -> Option<DecodeInstr> {
        self.decoder.decode(instr)
    }

    fn debug_vaddr_to_paddr(&mut self, vaddr: WordType) -> Result<u64, PageTableError> {
        self.memory.debug_vaddr_to_paddr(vaddr)
    }

    fn debug_translate(
        &mut self,
        vaddr: WordType,
        access: AccessType,
    ) -> Result<u64, PageTableError> {
        self.translate_for_debug(vaddr, access)
    }
}
