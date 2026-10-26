use smallvec::SmallVec;

use crate::isa::{
    InstrLen,
    riscv::{
        RawInstr,
        decoder::{DecodeInstr, decode_info},
        instruction::{
            instr_table::{RVInstrDesc, RiscvInstr},
            *,
        },
    },
};

type PartialResult = (RiscvInstr, InstrFormat);

#[derive(Debug, Clone)]
enum PartialDecode {
    // TODO: Remove branch `Unknown` by returning RiscvInstr::Illegal
    Unknown,
    Complete {
        result: PartialResult,
    },
    SparseF7 {
        table: SmallMap<u8, PartialResult>,
    },
    DenseF7 {
        table: Box<[Option<PartialResult>; 128]>,
    },
}

impl Default for PartialDecode {
    fn default() -> Self {
        PartialDecode::Unknown
    }
}

impl PartialDecode {
    #[inline]
    fn try_make_dense(&mut self) {
        let PartialDecode::SparseF7 { table: sparse } = self else {
            return;
        };

        if sparse.len() < MAP_LENGTH {
            return;
        }

        let mut dense = Box::new(std::array::from_fn(|_| None));
        for (f7, result) in sparse.iter() {
            dense[*f7 as usize] = Some(result.clone());
        }

        *self = PartialDecode::DenseF7 { table: dense }
    }
}

const MAP_LENGTH: usize = 8;

#[derive(Debug, Clone)]
struct SmallMap<K, V> {
    data: SmallVec<[(K, V); MAP_LENGTH]>,
}

#[allow(unused)]
impl<K: Ord + Copy, V> SmallMap<K, V> {
    fn new() -> Self {
        SmallMap {
            data: SmallVec::new(),
        }
    }

    fn insert(&mut self, key: K, value: V) {
        self.data.push((key, value));
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.data.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.data.iter()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

pub(super) struct Decoder {
    /// table[opcode][funct3]
    ///
    /// If you don't need funct3, fill all 8 element with the same value.
    decode_table: Box<[[PartialDecode; 1 << 3]; 1 << 7]>,
}

const OPCODE_BITS: u32 = 0b1111111;
const F3_BITS: u32 = 0b111 << 12;
const F7_BITS: u32 = 0b1111111 << 25;

fn get_opcode_f3_f7(instr: u32) -> (u8, u8, u8) {
    let opcode = (instr & 0b1111111) as u8;
    let funct3 = ((instr >> 12) & 0b111) as u8;
    let funct7 = (instr >> 25) as u8;

    (opcode, funct3, funct7)
}

impl Decoder {
    fn insert_f3_f7(&mut self, opcode: u8, f3: u8, f7: u8, result: PartialResult) {
        let slot = &mut self.decode_table[opcode as usize][f3 as usize];

        match slot {
            PartialDecode::Unknown => {
                let mut f7_table = SmallMap::new();
                f7_table.insert(f7, result);
                *slot = PartialDecode::SparseF7 { table: f7_table }
            }
            PartialDecode::Complete { result: prev } => {
                panic!(
                    "[decoder] trying to insert f7 for {:?} but already have {:?}",
                    result, prev
                )
            }
            PartialDecode::SparseF7 { table } => {
                table.insert(f7, result);
                slot.try_make_dense();
            }
            PartialDecode::DenseF7 { table } => {
                table[f7 as usize] = Some(result);
            }
        }
    }

    pub fn new() -> Self {
        Self {
            decode_table: Box::new(std::array::from_fn(|_| {
                std::array::from_fn(|_| PartialDecode::Unknown)
            })),
        }
    }

    /// return `true` when success.
    pub fn try_insert(&mut self, desc: RVInstrDesc) -> bool {
        let RVInstrDesc {
            instr,
            format,
            mask,
            key,
            ..
        } = desc;

        if (mask & !(OPCODE_BITS | F3_BITS | F7_BITS)) != 0 {
            return false;
        }

        assert!((mask & OPCODE_BITS) == OPCODE_BITS);

        let (opcode, f3, f7) = get_opcode_f3_f7(key);
        let result = (instr, format);

        if mask == OPCODE_BITS {
            // only opcode
            for slot in &mut self.decode_table[opcode as usize] {
                *slot = PartialDecode::Complete { result };
            }
            true
        } else if mask == (OPCODE_BITS | F3_BITS) {
            // only opcode + f3
            self.decode_table[opcode as usize][f3 as usize] = PartialDecode::Complete { result };
            true
        } else if mask == (OPCODE_BITS | F3_BITS | F7_BITS) {
            // only opcode + f3 + f7
            self.insert_f3_f7(opcode, f3, f7, result);
            true
        } else if mask == (OPCODE_BITS | F7_BITS) {
            // only opcode + f7
            for f3 in 0..8 {
                // enumerate f3
                self.insert_f3_f7(opcode, f3, f7, result);
            }
            true
        } else if ((mask & F3_BITS) == F3_BITS) && ((mask & F7_BITS).count_ones() >= 5) {
            // mask contains opcode + f3, but doesn't contain the whole f7,
            // it's common in A extension or V extension.
            let rev_mask = ((!(mask & F7_BITS) & F7_BITS) >> 25) as u8;
            let f7_match = ((key & F7_BITS) >> 25) as u8;

            let mut curr = rev_mask;
            loop {
                self.insert_f3_f7(opcode, f3, curr | f7_match, result);

                if curr == 0 {
                    break;
                }

                curr = rev_mask & (curr - 1);
            }

            true
        } else {
            false
        }
    }

    #[inline]
    fn recognize_instr(&self, instr: RawInstr) -> Option<PartialResult> {
        let (opcode, f3, f7) = get_opcode_f3_f7(instr.val);

        match &self.decode_table[opcode as usize][f3 as usize] {
            PartialDecode::Unknown => None,
            PartialDecode::Complete { result } => Some(result.clone()),
            PartialDecode::SparseF7 { table } => table.get(&f7).cloned(),
            PartialDecode::DenseF7 { table } => table[f7 as usize],
        }
    }

    pub fn decode(&self, instr: RawInstr) -> Option<DecodeInstr> {
        let len = instr.len();

        self.recognize_instr(instr)
            .map(|(instr_kind, fmt)| DecodeInstr {
                instr: instr_kind,
                info: decode_info(instr.val, instr_kind, fmt),
                len,
            })
    }
}
