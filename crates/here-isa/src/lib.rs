use crate::{
    config::arch_config::WordType,
    device::MemError,
    isa::riscv::{
        address::Address,
        csr_reg::PrivilegeLevel,
        mmu::{AccessType, PageTableError},
    },
    utils::UnsignedInteger,
};

pub mod cache;
pub mod riscv;

mod utils;

pub trait DebugTarget<I: ISATypes> {
    fn read_pc(&self) -> WordType;
    fn write_pc(&mut self, new_pc: WordType);

    fn read_reg(&self, idx: u8) -> WordType;
    fn write_reg(&mut self, idx: u8, value: WordType);
    fn read_float_reg(&self, idx: u8) -> (f32, f64);
    fn read_vector_reg<T>(&self, idx: u8) -> Option<&[T]>;

    fn read_instr(&mut self, addr: WordType) -> Result<I::RawInstr, MemError>;
    fn read_instr_directly(&mut self, addr: Address) -> Result<I::RawInstr, MemError>;

    fn read_memory<T: UnsignedInteger>(&mut self, addr: Address) -> Result<T, MemError>;
    fn write_memory<T: UnsignedInteger>(&mut self, addr: Address, data: T) -> Result<(), MemError>;

    fn debug_vaddr_to_paddr(&mut self, vaddr: WordType) -> Result<u64, PageTableError>;
    fn debug_translate(&mut self, addr: u64, access: AccessType) -> Result<u64, PageTableError>;

    fn get_current_privilege(&self) -> PrivilegeLevel;

    fn debug_csr(&mut self, addr: WordType, new_value: Option<WordType>) -> Option<WordType>;

    fn decoded_instr(&self, instr: I::RawInstr) -> Option<I::DecodeRst>;
}

pub trait DecoderTrait<I: ISATypes> {
    fn from_isa(instrs: Vec<I::ISADesc>) -> Self;
    fn decode(&self, instr: I::RawInstr) -> Option<I::DecodeRst>;
}

pub trait InstrLen {
    fn len(&self) -> WordType;
}

pub trait ISATypes: Sized {
    type RawInstr: Copy + InstrLen;
    type ISADesc;
    type DecodeRst: Clone + Copy;
    type CPU: DebugTarget<Self>;
}
