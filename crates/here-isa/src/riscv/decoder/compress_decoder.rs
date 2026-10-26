use crate::{
    config::arch_config::WordType,
    debug_unreachable,
    isa::{
        riscv::{
            RawInstr,
            decoder::DecodeInstr,
            instruction::{
                InstrFormat, RVInstrInfo,
                instr_table::{
                    RVInstrDesc,
                    RiscvInstr::{self, *},
                },
            },
        },
        utils::DecodeMask,
    },
    utils::UnsignedInteger,
};

pub(super) struct CompressedDecoder {
    masks: Vec<(DecodeMask, RiscvInstr, InstrFormat)>,
}

macro_rules! extract_field {
    ($raw:expr, $( $dst_high:literal : $dst_low:literal <- $src_low:literal),* $(,)?) => {
        {
            let mut field = 0u32;
            $(
                debug_assert!($dst_low <= $dst_high);
                let val = $raw.extract_range($src_low, $src_low + $dst_high - $dst_low) << $dst_low;
                debug_assert!((val & field) == 0);
                field |= val;
            )*
            field
        }
    }
}

macro_rules! sign_ext_extract {
    ($raw:expr, $($dst_high:literal : $dst_low:literal <- $src_low:literal),* $(,)?) => {{
        let field = extract_field!($raw, $($dst_high : $dst_low <- $src_low),*);
        // The highest destination bit is the immediate's sign bit, so the field
        // width (and thus the bit count to sign-extend from) is that bit + 1.
        let from_bits = *[$($dst_high as u32),*].iter().max().unwrap() + 1;
        crate::utils::sign_extend(field as WordType, from_bits)
    }};
}

macro_rules! zero_ext_extract {
    ($($tokens:tt)*) => {
        extract_field!($($tokens)*) as WordType
    };
}

#[must_use]
fn cvt_short_reg(reg_idx: u8) -> u8 {
    reg_idx + 8
}

fn decode_compressed_info(raw: u32, instr: RiscvInstr, fmt: InstrFormat) -> RVInstrInfo {
    let rs2 = raw.extract_range(2, 6) as u8;
    let rd_rs1 = raw.extract_range(7, 11) as u8;
    let rd_rs2_short = cvt_short_reg(raw.extract_range(2, 4) as u8);
    let rd_rs1_short = cvt_short_reg(raw.extract_range(7, 9) as u8);

    match fmt {
        InstrFormat::CA => RVInstrInfo::CA {
            rd_rs1: rd_rs1_short,
            rs2: rd_rs2_short,
        },
        InstrFormat::CB => {
            let offset = match instr {
                C_SRLI | C_SRAI => sign_ext_extract!(raw, 5:5 <- 12, 4:0 <- 2),
                C_ANDI => sign_ext_extract!(raw, 5:5 <- 12, 4:0 <- 2),
                C_BEQZ | C_BNEZ => {
                    sign_ext_extract!(raw, 8:8 <- 12, 4:3 <- 10, 7:6 <- 5, 2:1 <- 3, 5:5 <- 2)
                }
                _ => debug_unreachable!(),
            };
            RVInstrInfo::CB {
                rd_rs1: rd_rs1_short,
                imm: offset,
            }
        }
        InstrFormat::CI => {
            let imm = match instr {
                C_ADDI16SP => {
                    sign_ext_extract!(raw, 9:9 <-12, 4:4 <-6, 6:6 <- 5, 8:7 <- 3, 5:5 <- 2)
                }
                C_LDSP | C_FLDSP => zero_ext_extract!(raw, 5:5 <- 12, 4:3 <- 5, 8:6 <- 2),
                C_LWSP | C_FLWSP => zero_ext_extract!(raw, 5:5 <- 12, 4:2 <- 4, 7:6 <- 2),
                _ => {
                    let mut imm = sign_ext_extract!(raw, 5:5 <- 12, 4:0 <- 2);
                    if instr == C_LUI {
                        imm <<= 12;
                    }
                    imm
                }
            };
            RVInstrInfo::CI { rd_rs1, imm }
        }
        InstrFormat::CIW => RVInstrInfo::CIW {
            rd: rd_rs2_short,
            imm: zero_ext_extract!(raw, 5:4 <- 11, 9:6 <-7, 2:2 <- 6, 3:3 <- 5),
        },
        InstrFormat::CJ => RVInstrInfo::CJ {
            target: sign_ext_extract!(
                raw,
                11:11 <- 12,
                4:4 <- 11,
                9:8 <- 9,
                10:10 <- 8,
                6:6 <- 7,
                7:7 <- 6,
                3:1 <- 3,
                5:5 <- 2
            ),
        },
        InstrFormat::CL | InstrFormat::CS => {
            let uimm = match instr {
                C_FLD | C_LD | C_FSD | C_SD => {
                    zero_ext_extract!(raw, 5:3 <- 10, 7:6 <- 5)
                }
                C_LW | C_FLW | C_SW | C_FSW => {
                    zero_ext_extract!(raw, 5:3 <- 10, 2:2 <- 6, 6: 6 <- 5)
                }
                _ => debug_unreachable!(),
            };

            if fmt == InstrFormat::CL {
                RVInstrInfo::CL {
                    rd: rd_rs2_short,
                    rs1: rd_rs1_short,
                    imm: uimm,
                }
            } else {
                debug_assert!(fmt == InstrFormat::CS);
                RVInstrInfo::CS {
                    rs2: rd_rs2_short,
                    rs1: rd_rs1_short,
                    imm: uimm,
                }
            }
        }
        InstrFormat::CSS => {
            let uimm = match instr {
                C_FSDSP | C_SDSP => {
                    zero_ext_extract!(raw, 5:3 <-10, 8:6 <- 7)
                }
                C_SWSP | C_FSWSP => {
                    zero_ext_extract!(raw, 5:2 <- 9, 7:6 <- 7)
                }
                _ => debug_unreachable!(),
            };

            RVInstrInfo::CSS {
                rs2: rs2,
                imm: uimm,
            }
        }
        InstrFormat::CR => RVInstrInfo::CR { rd_rs1, rs2 },
        InstrFormat::None => RVInstrInfo::None,
        _ => {
            debug_unreachable!()
        }
    }
}

impl CompressedDecoder {
    pub fn decode(&self, raw: RawInstr) -> Option<DecodeInstr> {
        for (mask, instr, fmt) in self.masks.iter() {
            if mask.matches(raw.val) {
                return Some(DecodeInstr {
                    instr: *instr,
                    info: decode_compressed_info(raw.val, *instr, *fmt),
                    len: 2,
                });
            }
        }

        None
    }

    pub fn from_isa(instrs: impl Iterator<Item = RVInstrDesc>) -> Self {
        let mut masks = vec![];
        for desc in instrs {
            masks.push((
                DecodeMask {
                    mask: desc.mask,
                    key: desc.key,
                },
                desc.instr,
                desc.format,
            ));
        }

        // sort by desending popcount(mask), so that overlapped mask can work fine.
        masks.sort_by_key(|(mask, _, _)| mask.mask.count_zeros());

        log::info!("Compressed decoder loads {} instructions", masks.len());

        Self { masks }
    }
}
