use crate::{define_instr_enum, define_riscv_isa, isa::riscv::instruction::InstrFormat};

include!(concat!(env!("OUT_DIR"), "/rvinstr_gen.rs"));
