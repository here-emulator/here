use crate::{
    config::arch_config::WordType,
    isa::{
        ISATypes, InstrLen,
        riscv::{decoder::DecodeInstr, executor::RVCPU, instruction::instr_table::RVInstrDesc},
    },
};

pub mod address;
#[cfg(test)]
pub mod cpu_tester;
pub mod csr_reg;
mod debug_target;
pub mod debugger;
pub mod decoder;
pub mod executor;
pub mod instruction;
pub mod isa_builder;
pub mod mmu;
pub mod trap;
pub mod vector;

#[derive(Debug)]
pub struct RiscvTypes;

impl ISATypes for RiscvTypes {
    type RawInstr = RawInstr;
    type ISADesc = RVInstrDesc;
    type DecodeRst = DecodeInstr;
    type CPU = RVCPU;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawInstr {
    pub val: u32,
}

impl From<u32> for RawInstr {
    fn from(value: u32) -> Self {
        if instr_len(value) == 2 {
            Self {
                val: value & 0xFFFF,
            }
        } else {
            Self { val: value }
        }
    }
}

fn instr_len(instr: u32) -> WordType {
    if instr & 0b11 == 0b11 { 4 } else { 2 }
}

impl InstrLen for RawInstr {
    fn len(&self) -> WordType {
        instr_len(self.val)
    }
}
