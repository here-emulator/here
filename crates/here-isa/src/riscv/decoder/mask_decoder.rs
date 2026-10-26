use crate::isa::{
    InstrLen,
    riscv::{
        RawInstr,
        decoder::{DecodeInstr, decode_info},
        instruction::{
            InstrFormat,
            instr_table::{RVInstrDesc, RiscvInstr},
        },
    },
    utils::DecodeMask,
};

pub(super) struct MaskDecoder {
    masks: Vec<(DecodeMask, RiscvInstr, InstrFormat)>,
}

impl MaskDecoder {
    pub fn decode(&self, raw: RawInstr) -> Option<DecodeInstr> {
        for (mask, instr, fmt) in self.masks.iter() {
            if mask.matches(raw.val) {
                return Some(DecodeInstr {
                    instr: *instr,
                    info: decode_info(raw.val, *instr, *fmt),
                    len: raw.len(),
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

        log::info!("Mask decoder loads {} instructions", masks.len());

        for (_, instr, _) in masks.iter() {
            log::debug!("[decoder] mask decoder handling {:?}", instr);
        }

        Self { masks }
    }
}
