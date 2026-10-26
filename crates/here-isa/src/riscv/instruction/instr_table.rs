use crate::define_instr_enum;
use crate::define_riscv_isa;
use crate::riscv::instruction::InstrFormat;

include!(concat!(env!("OUT_DIR"), "/rvinstr_gen.rs"));
