use std::fmt::Display;

use crate::InstrLen;
use crate::cache::Cacheable;
use crate::riscv::RawInstr;
use crate::riscv::decoder::compress_decoder::CompressedDecoder;
use crate::riscv::instruction::InstrFormat;
use crate::riscv::instruction::RVInstrInfo;
use crate::riscv::instruction::instr_table::*;
use crate::riscv::isa_builder::DEFAULT_ISA;
use crate::riscv::isa_builder::ISABuilder;
use crate::riscv::isa_builder::IsaParseError;
use here_types::WordType;
use here_types::debug_unreachable;
use here_types::sign_extend;

mod compress_decoder;
mod funct_decoder;
mod mask_decoder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInstr {
    pub instr: RiscvInstr,
    pub info: RVInstrInfo,
    pub len: WordType,
}

impl Cacheable for DecodeInstr {
    const ADDR_SHIFT_BITS: usize = 1;
}

impl Display for DecodeInstr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}, {:?}", self.instr, self.info)
    }
}

pub struct Decoder {
    funct_decoder: funct_decoder::Decoder,
    mask_decoder: mask_decoder::MaskDecoder,
    compress_decoder: compress_decoder::CompressedDecoder,
    /// `misa` extension bitmap of the ISA this decoder was built for.
    extension_bits: WordType,
}

impl Decoder {
    pub fn new() -> Self {
        Self::from_isa_str(DEFAULT_ISA).expect("DEFAULT_ISA must be valid")
    }

    /// Builds a decoder for the ISA described by `isa`,
    /// e.g. `"RV64IMAFDC_Zicsr_Zifencei"`. See [`ISABuilder`].
    pub fn from_isa_str(isa: &str) -> Result<Self, IsaParseError> {
        Ok(Self::from_builder(isa.parse::<ISABuilder>()?))
    }

    /// The `misa` extension bitmap of the ISA this decoder decodes.
    pub fn extension_bits(&self) -> WordType {
        self.extension_bits
    }

    fn from_builder(builder: ISABuilder) -> Self {
        let extension_bits = builder.extension_bits();

        #[allow(unused_mut)]
        let mut isa = builder.build();

        #[cfg(feature = "custom-instr")]
        {
            isa.extend_from_slice(TABLE_RVCUSTOM0);
            isa.extend_from_slice(TABLE_RVCUSTOM1);
        }

        Self::from_isa(isa, extension_bits)
    }
}

fn is_compressed(desc: &RVInstrDesc) -> bool {
    desc.key & 0b11 != 0b11
}

impl Decoder {
    pub fn from_isa(instrs: Vec<RVInstrDesc>, extension_bits: WordType) -> Self {
        let compress =
            CompressedDecoder::from_isa(instrs.iter().filter(|d| is_compressed(d)).cloned());

        let mut funct = funct_decoder::Decoder::new();
        let mut remain = vec![];

        let mut funct_count = 0;
        for desc in instrs.iter().filter(|d| !is_compressed(d)).cloned() {
            if funct.try_insert(desc.clone()) {
                funct_count += 1;
            } else {
                remain.push(desc);
            }
        }

        let mask = mask_decoder::MaskDecoder::from_isa(remain.into_iter());

        log::info!("funct_decoder has {} instructions.", funct_count);

        Self {
            compress_decoder: compress,
            funct_decoder: funct,
            mask_decoder: mask,
            extension_bits: extension_bits,
        }
    }

    pub fn decode(&self, instr: RawInstr) -> Option<DecodeInstr> {
        if instr.len() == 2 {
            self.compress_decoder.decode(instr)
        } else {
            None.or_else(|| self.funct_decoder.decode(instr))
                .or_else(|| self.mask_decoder.decode(instr))
        }
    }
}

/// This function doesn't handle compressed instruction.
fn decode_info(raw_instr: u32, instr: RiscvInstr, fmt: InstrFormat) -> RVInstrInfo {
    // TODO: These codes can be simplified like the later written `decode_compressed_info`

    let rd = ((raw_instr >> 7) & 0b11111) as u8;
    let rs1 = ((raw_instr >> 15) & 0b11111) as u8;
    let rs2 = ((raw_instr >> 20) & 0b11111) as u8;
    let f3 = ((raw_instr >> 12) & 0b111) as u8;

    match fmt {
        InstrFormat::V => {
            let vm = (raw_instr & (1 << 25)) != 0;
            let func6 = ((raw_instr >> 26) & 0x3F) as u8;
            RVInstrInfo::V {
                rs1,
                rs2,
                rd,
                vm,
                func6,
            }
        }
        InstrFormat::R => RVInstrInfo::R { rd, rs1, rs2 },
        InstrFormat::R_rm => RVInstrInfo::R_rm {
            rs1,
            rs2,
            rd,
            rm: f3,
        },
        InstrFormat::R4_rm => RVInstrInfo::R4_rm {
            rd,
            rs1,
            rs2,
            rs3: ((raw_instr >> 27) & 0b11111) as u8,
            rm: f3,
        },
        InstrFormat::A => {
            let aq = ((raw_instr >> 26) & 1) != 0;
            let rl = ((raw_instr >> 25) & 1) != 0;
            RVInstrInfo::A {
                rs1,
                rs2,
                rd,
                aq,
                rl,
            }
        }
        InstrFormat::I => {
            let mut imm = ((raw_instr >> 20) & 0xFFF) as WordType;

            match instr {
                RiscvInstr::SRLI | RiscvInstr::SRAI => {
                    imm &= 0x3F;
                }
                RiscvInstr::SRLIW | RiscvInstr::SRAIW => {
                    imm &= 0x1F;
                }
                _ => {}
            }

            RVInstrInfo::I {
                rd: rd,
                rs1: rs1,
                imm: sign_extend(imm, 12),
            }
        }
        InstrFormat::S => {
            let imm = (((raw_instr >> 25) & 0xFF) << 5) | ((raw_instr >> 7) & 0b11111);
            RVInstrInfo::S {
                rs1: rs1,
                rs2: rs2,
                imm: sign_extend(imm as WordType, 12),
            }
        }
        InstrFormat::U => {
            let imm = (raw_instr >> 12) << 12;
            RVInstrInfo::U {
                rd: rd,
                imm: sign_extend(imm as WordType, 32),
            }
        }
        InstrFormat::B => {
            let imm = (((raw_instr >> 31) & 1) << 12)
                | (((raw_instr >> 7) & 1) << 11)
                | (((raw_instr >> 25) & 0b111111) << 5)
                | (((raw_instr >> 8) & 0b1111) << 1);
            RVInstrInfo::B {
                rs1: rs1,
                rs2: rs2,
                imm: sign_extend(imm as WordType, 13),
            }
        }
        InstrFormat::J => {
            let imm = (((raw_instr >> 31) & 1) << 20)
                | (((raw_instr >> 12) & 0xFF) << 12)
                | (((raw_instr >> 20) & 1) << 11)
                | (((raw_instr >> 21) & 0x3FF) << 1);
            RVInstrInfo::J {
                rd: rd,
                imm: sign_extend(imm as WordType, 21),
            }
        }
        InstrFormat::None => RVInstrInfo::None,

        InstrFormat::CA
        | InstrFormat::CB
        | InstrFormat::CI
        | InstrFormat::CIW
        | InstrFormat::CJ
        | InstrFormat::CL
        | InstrFormat::CR
        | InstrFormat::CS
        | InstrFormat::CSS => debug_unreachable!(),
    }
}

#[cfg(test)]
mod decoder_test;
