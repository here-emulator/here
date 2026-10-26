//! Assembling the decodable instruction set from RISC-V extensions.
//!
//! [`ISABuilder`] turns a set of [`Extension`]s into the flat
//! `Vec<RVInstrDesc>` the decoders consume. Adding an extension automatically
//! pulls in the extensions it depends on (e.g. `D` implies `F` implies
//! `Zicsr`).
//!
//! Table selection is `XLEN`-aware, so the same code produces an RV32 or RV64
//! instruction set depending on the compiled `WordType`.

use std::str::FromStr;

use crate::{
    config::arch_config::{WordType, XLEN},
    isa::riscv::instruction::instr_table::*,
};

/// Stable default ISA exposed to users. Keep vector out until the V extension
/// support is stable enough to enable by default.
#[cfg(feature = "riscv64")]
pub const DEFAULT_ISA: &str = "RV64GC";

/// Stable default ISA exposed to users. RV32 is not fully maintained, but
/// parser defaults still need to match the selected word size.
#[cfg(feature = "riscv32")]
pub const DEFAULT_ISA: &str = "RV32GC";

/// ISA used by CPU tests that exercise unstable vector support.
#[cfg(feature = "riscv64")]
pub(crate) const TEST_ISA: &str = "RV64GCV";

/// ISA used by CPU tests that exercise unstable vector support.
#[cfg(feature = "riscv32")]
pub(crate) const TEST_ISA: &str = "RV32GCV";

/// A standard RISC-V extension this emulator knows how to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension {
    I,
    M,
    A,
    F,
    D,
    C,
    V,
    Zicsr,
    Zifencei,
}

impl Extension {
    /// Extensions that must be present for `self` to make sense.
    ///
    /// Only direct dependencies are listed; [`ISABuilder::add`] resolves them
    /// transitively.
    fn dependencies(self) -> &'static [Extension] {
        use Extension::*;
        match self {
            D => &[F],
            F => &[Zicsr],
            _ => &[],
        }
    }

    /// Instruction tables contributed by this extension for the current `XLEN`.
    fn tables(self) -> Vec<&'static [RVInstrDesc]> {
        use Extension::*;
        let is_rv64 = XLEN == 64;
        match self {
            I => xlen_tables(TABLE_RV32I, TABLE_RV64I, is_rv64),
            M => xlen_tables(TABLE_RV32M, TABLE_RV64M, is_rv64),
            A => xlen_tables(TABLE_RV32A, TABLE_RV64A, is_rv64),
            F => xlen_tables(TABLE_RV32F, TABLE_RV64F, is_rv64),
            D => xlen_tables(TABLE_RV32D, TABLE_RV64D, is_rv64),
            V => vec![TABLE_RVV],

            // The RV32 and RV64 compressed tables reuse the same encodings for
            // different instructions, so exactly one of them is active.
            C => vec![TABLE_RVC, if is_rv64 { TABLE_RV64C } else { TABLE_RV32C }],
            Zicsr => vec![TABLE_RVZICSR],
            Zifencei => vec![TABLE_RVZIFENCEI],
        }
    }

    fn misa_letter(self) -> Option<char> {
        use Extension::*;
        Some(match self {
            I => 'I',
            M => 'M',
            A => 'A',
            F => 'F',
            D => 'D',
            C => 'C',
            V => 'V',
            Zicsr | Zifencei => return None,
        })
    }
}

/// On RV64 the base set is the RV32 table plus the 64-bit additions;
/// on RV32 only the RV32 table applies.
///
/// Don't use this for special extension like RVC (RV32C and RV64C are mutually exclusive).
fn xlen_tables(
    rv32: &'static [RVInstrDesc],
    rv64: &'static [RVInstrDesc],
    is_rv64: bool,
) -> Vec<&'static [RVInstrDesc]> {
    if is_rv64 {
        vec![rv32, rv64]
    } else {
        vec![rv32]
    }
}

/// `misa` bit for a single-letter extension: bit `letter - 'A'`.
fn misa_bit(letter: char) -> WordType {
    1 << (letter as u8 - b'A')
}

/// Builds the decoder's instruction set from a collection of [`Extension`]s.
///
/// ```ignore
/// // Equivalent ways to describe RV64GC:
/// let a = ISABuilder::new().add(Extension::M).add(Extension::A).add(Extension::D).add(Extension::C);
/// let b: ISABuilder = "RV64IMAFDC_Zicsr_Zifencei".parse().unwrap();
/// ```
#[derive(Debug)]
pub struct ISABuilder {
    /// Selected extensions, deduplicated. Order is not guarenteed.
    extensions: Vec<Extension>,
}

impl ISABuilder {
    /// A builder seeded with ([`Extension::I`]).
    pub fn new() -> Self {
        let mut builder = ISABuilder {
            extensions: Vec::new(),
        };
        builder.insert(Extension::I);
        builder
    }

    /// Adds `ext` and, transitively, every extension it depends on.
    pub fn add(mut self, ext: Extension) -> Self {
        self.insert(ext);
        self
    }

    pub fn has(&self, ext: Extension) -> bool {
        self.extensions.contains(&ext)
    }

    fn insert(&mut self, ext: Extension) {
        if self.has(ext) {
            return;
        }
        for &dep in ext.dependencies() {
            self.insert(dep);
        }
        self.extensions.push(ext);
    }

    /// Flattens the selected extensions into the instruction descriptors the
    /// decoder consumes, resolving combination "glue" tables and appending the
    /// always-on privileged / system instructions.
    pub fn build(&self) -> Vec<RVInstrDesc> {
        let mut instrs: Vec<RVInstrDesc> = Vec::new();

        for &ext in &self.extensions {
            for table in ext.tables() {
                instrs.extend_from_slice(table);
            }
        }

        if self.has(Extension::C) && self.has(Extension::D) {
            instrs.extend_from_slice(TABLE_RVC_D);
        }
        if XLEN == 32 && self.has(Extension::C) && self.has(Extension::F) {
            instrs.extend_from_slice(TABLE_RV32C_F);
        }

        // Mandatory regardless of the ISA string
        instrs.extend_from_slice(TABLE_RVSYSTEM);
        instrs.extend_from_slice(TABLE_RVS);
        instrs.extend_from_slice(TABLE_RVILLEGAL);

        instrs
    }

    pub fn extension_bits(&self) -> WordType {
        let mut bits = misa_bit('S') | misa_bit('U');
        for &ext in &self.extensions {
            if let Some(letter) = ext.misa_letter() {
                bits |= misa_bit(letter);
            }
        }
        bits
    }
}

impl Default for ISABuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Error returned when an ISA string cannot be parsed into an [`ISABuilder`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IsaParseError {
    #[error("ISA string must start with \"RV\"")]
    MissingRvPrefix,
    #[error("ISA string must specify the width, e.g. \"RV32\" or \"RV64\"")]
    MissingXlen,
    #[error("ISA string targets RV{requested} but this build is RV{actual}")]
    XlenMismatch { requested: usize, actual: usize },
    #[error("unknown extension '{0}'")]
    UnknownBaseExtension(char),
    #[error("unknown extension \"{0}\"")]
    UnknownExtension(String),
}

impl FromStr for ISABuilder {
    type Err = IsaParseError;

    /// Parses an ISA string such as `RV64IMAFDC_Zicsr_Zifencei` or `RV64GC`.
    ///
    /// Parsing is case-insensitive.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        let body = lower
            .strip_prefix("rv")
            .ok_or(IsaParseError::MissingRvPrefix)?;

        let digits: String = body.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            return Err(IsaParseError::MissingXlen);
        }
        let requested: usize = digits.parse().map_err(|_| IsaParseError::MissingXlen)?;
        if requested != XLEN {
            return Err(IsaParseError::XlenMismatch {
                requested,
                actual: XLEN,
            });
        }

        let mut builder = ISABuilder::new();

        let mut segments = body[digits.len()..].split('_');

        // single-letter extensions:
        if let Some(single) = segments.next() {
            for ch in single.chars() {
                for &ext in base_extensions(ch)? {
                    builder = builder.add(ext);
                }
            }
        }

        // multi-letter extensions:
        for token in segments {
            if token.is_empty() {
                continue; // tolerate trailing or doubled underscores
            }
            builder = builder.add(multi_letter_extension(token)?);
        }

        Ok(builder)
    }
}

/// Maps a single-letter extension to the extensions it stands for.
fn base_extensions(letter: char) -> Result<&'static [Extension], IsaParseError> {
    use Extension::*;
    Ok(match letter {
        'i' => &[I],
        'm' => &[M],
        'a' => &[A],
        'f' => &[F],
        'd' => &[D],
        'c' => &[C],
        'v' => &[V],
        // The "general" shorthand.
        'g' => &[I, M, A, F, D, Zicsr, Zifencei],
        other => return Err(IsaParseError::UnknownBaseExtension(other)),
    })
}

fn multi_letter_extension(token: &str) -> Result<Extension, IsaParseError> {
    Ok(match token {
        "zicsr" => Extension::Zicsr,
        "zifencei" => Extension::Zifencei,
        other => return Err(IsaParseError::UnknownExtension(other.to_string())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::riscv::instruction::instr_table::RiscvInstr;

    fn has_instr(instrs: &[RVInstrDesc], instr: RiscvInstr) -> bool {
        instrs.iter().any(|d| d.instr == instr)
    }

    /// The width that does *not* match this build, used for negative tests.
    fn wrong_xlen() -> usize {
        if XLEN == 64 { 32 } else { 64 }
    }

    /// Prefix an extension body with the correct `RVxx` for this build.
    fn isa(body: &str) -> String {
        format!("RV{XLEN}{body}")
    }

    /// `misa` bitmap for a string of single-letter extensions.
    fn misa_of(letters: &str) -> WordType {
        letters
            .chars()
            .fold(0, |acc, c| acc | (1 << (c as u8 - b'A')))
    }

    #[test]
    fn base_and_privileged_always_present() {
        let isa = ISABuilder::new().build();
        assert!(has_instr(&isa, RiscvInstr::ADDI)); // base I
        assert!(has_instr(&isa, RiscvInstr::ECALL)); // system
        assert!(has_instr(&isa, RiscvInstr::SRET)); // supervisor
    }

    #[test]
    fn extension_bits_match_misa_bitmap() {
        // Default RVxxGC: A, C, D, F, I, M plus always-on S and U.
        let builder = ISABuilder::new()
            .add(Extension::M)
            .add(Extension::A)
            .add(Extension::D) // pulls in F and Zicsr
            .add(Extension::C);
        assert_eq!(builder.extension_bits(), misa_of("ACDFIMSU"));
    }

    #[test]
    fn zicsr_and_zifencei_have_no_misa_bit() {
        // Base only: I plus the always-on S and U.
        assert_eq!(ISABuilder::new().extension_bits(), misa_of("ISU"));

        let with_z = ISABuilder::new()
            .add(Extension::Zicsr)
            .add(Extension::Zifencei);
        assert_eq!(with_z.extension_bits(), misa_of("ISU"));
    }

    #[test]
    fn d_pulls_in_f_and_zicsr() {
        let builder = ISABuilder::new().add(Extension::D);
        assert!(builder.has(Extension::F));
        assert!(builder.has(Extension::Zicsr));

        let isa = builder.build();
        assert!(has_instr(&isa, RiscvInstr::FADD_S)); // from F
        assert!(has_instr(&isa, RiscvInstr::FADD_D)); // from D
        assert!(has_instr(&isa, RiscvInstr::CSRRW)); // from Zicsr
    }

    #[test]
    fn compressed_double_glue_requires_both_c_and_d() {
        let only_c = ISABuilder::new().add(Extension::C).build();
        assert!(!has_instr(&only_c, RiscvInstr::C_FLD));

        let c_and_d = ISABuilder::new()
            .add(Extension::C)
            .add(Extension::D)
            .build();
        assert!(has_instr(&c_and_d, RiscvInstr::C_FLD));
    }

    #[test]
    fn parses_full_isa_string() {
        let builder: ISABuilder = isa("IMAFDC_Zifencei").parse().unwrap();
        for ext in [
            Extension::I,
            Extension::M,
            Extension::A,
            Extension::F,
            Extension::D,
            Extension::C,
            Extension::Zicsr,
            Extension::Zifencei,
        ] {
            assert!(builder.has(ext), "missing {ext:?}");
        }
    }

    #[test]
    fn g_expands_to_general() {
        let builder: ISABuilder = isa("GC").parse().unwrap();
        for ext in [
            Extension::M,
            Extension::A,
            Extension::F,
            Extension::D,
            Extension::Zicsr,
            Extension::Zifencei,
            Extension::C,
        ] {
            assert!(builder.has(ext), "missing {ext:?}");
        }
    }

    #[test]
    fn parses_vector_extension() {
        let builder: ISABuilder = isa("GCV").parse().unwrap();
        assert!(builder.has(Extension::V));
        assert_eq!(builder.extension_bits(), misa_of("ACDFIMSUV"));
    }

    #[test]
    fn rejects_wrong_xlen() {
        let s = format!("RV{}I", wrong_xlen());
        assert_eq!(
            s.parse::<ISABuilder>().unwrap_err(),
            IsaParseError::XlenMismatch {
                requested: wrong_xlen(),
                actual: XLEN,
            }
        );
    }

    #[test]
    fn rejects_unknown_base_letter() {
        assert_eq!(
            isa("IQ").parse::<ISABuilder>().unwrap_err(),
            IsaParseError::UnknownBaseExtension('q')
        );
    }

    #[test]
    fn rejects_unknown_multi_letter_extension() {
        assert_eq!(
            isa("I_Zfoo").parse::<ISABuilder>().unwrap_err(),
            IsaParseError::UnknownExtension("zfoo".to_string())
        );
    }

    #[test]
    fn rejects_missing_prefix() {
        assert_eq!(
            "64I".parse::<ISABuilder>().unwrap_err(),
            IsaParseError::MissingRvPrefix
        );
    }
}
