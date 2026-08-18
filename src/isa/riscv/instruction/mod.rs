mod exec_atomic_function;
mod exec_compress_function;
mod exec_core;
mod exec_float_function;
mod exec_vector_function;

pub(super) mod exec_function;
pub mod exec_mapping;
pub mod instr_table;

use crate::{
    config::arch_config::WordType,
    isa::riscv::{
        self,
        csr_reg::{
            NamedCsrReg,
            csr_macro::{Minstret, Misa, Mstatus, Vstart},
        },
        executor::RVCPU,
        instruction::exec_function::save_fflags_to_cpu,
        trap::Exception,
        vector::VectorMemException,
    },
};

#[inline]
pub(super) fn check_jump_alignment(cpu: &mut RVCPU, target: WordType) -> Result<(), Exception> {
    if !cpu.csr.get_by_type_existing::<Misa>().c_enabled() && (target & 0x3) != 0 {
        cpu.pending_tval = Some(target);
        return Err(Exception::InstructionMisaligned);
    }
    Ok(())
}
/// A helper function for normal instruction execution.
///
/// It takes a closure `f` that performs the actual instruction logic.
/// If `f` executes successfully, it will increase PC by 4 and increase the Minstret CSR by 1.
///
/// Don't use this for C extension.
#[inline(always)]
pub(super) fn normal_exec<F>(cpu: &mut RVCPU, f: F) -> Result<(), riscv::trap::Exception>
where
    F: FnOnce(&mut RVCPU) -> Result<(), riscv::trap::Exception>,
{
    f(cpu)?;
    cpu.pc = cpu.pc.wrapping_add(4);
    cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
    Ok(())
}

/// A helper function for normal instruction execution in the C extension.
#[inline(always)]
pub(super) fn normal_compress_exec<F>(cpu: &mut RVCPU, f: F) -> Result<(), riscv::trap::Exception>
where
    F: FnOnce(&mut RVCPU) -> Result<(), riscv::trap::Exception>,
{
    f(cpu)?;
    cpu.pc = cpu.pc.wrapping_add(2);
    cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
    Ok(())
}

/// A helper function for normal floating-point instruction execution.
///
/// It first checks if the floating-point unit is enabled by examining the FS field in the Mstatus CSR.
///
/// If the FS field is 0, it returns an illegal instruction exception.
/// Otherwise, it calls [`normal_exec`].
#[inline(always)]
pub(super) fn normal_float_exec<F>(cpu: &mut RVCPU, f: F) -> Result<(), riscv::trap::Exception>
where
    F: FnOnce(&mut RVCPU) -> Result<(), riscv::trap::Exception>,
{
    if cpu.csr.get_by_type_existing::<Mstatus>().get_fs() == 0 {
        return Err(riscv::trap::Exception::IllegalInstruction);
    }

    normal_exec(cpu, f)?;

    save_fflags_to_cpu(cpu);

    Ok(())
}

/// A helper function for normal float instruction execution in the C extension.
#[inline(always)]
pub(super) fn normal_compress_float_exec<F>(
    cpu: &mut RVCPU,
    f: F,
) -> Result<(), riscv::trap::Exception>
where
    F: FnOnce(&mut RVCPU) -> Result<(), riscv::trap::Exception>,
{
    if cpu.csr.get_by_type_existing::<Mstatus>().get_fs() == 0 {
        return Err(riscv::trap::Exception::IllegalInstruction);
    }

    normal_compress_exec(cpu, f)?;

    save_fflags_to_cpu(cpu);

    Ok(())
}

/// A helper function for normal vector instruction execution.
///
/// It first checks if the vector unit is enabled by examining the VS field in the Mstatus CSR.
///
/// If the VS field is 0, it returns an illegal instruction exception.
/// Otherwise, it calls [`normal_exec`].
pub(super) trait VectorExecError {
    fn finish_vector_exec_error(self, cpu: &mut RVCPU) -> Exception;
}

impl VectorExecError for Exception {
    #[inline]
    fn finish_vector_exec_error(self, _cpu: &mut RVCPU) -> Exception {
        self
    }
}

impl VectorExecError for VectorMemException {
    #[inline]
    fn finish_vector_exec_error(self, cpu: &mut RVCPU) -> Exception {
        finish_vector_memory_access(cpu, self)
    }
}

#[inline]
fn finish_vector_memory_access(cpu: &mut RVCPU, err: VectorMemException) -> Exception {
    // Only precise memory faults carry an element index. Other errors are
    // raised as-is and do not pretend to be resumable traps.
    if let Some(index) = err.fault_index() {
        cpu.csr
            .write_directly(Vstart::get_index(), index as WordType)
            .then_some(())
            .unwrap();
    }
    err.exception()
}

#[inline(always)]
pub(super) fn normal_vector_exec<F, E>(cpu: &mut RVCPU, f: F) -> Result<(), riscv::trap::Exception>
where
    F: FnOnce(&mut RVCPU, usize) -> Result<(), E>,
    E: VectorExecError,
{
    if cpu.csr.get_by_type_existing::<Mstatus>().get_vs() == 0 {
        return Err(riscv::trap::Exception::IllegalInstruction);
    }

    let vstart = cpu.csr.get_by_type_existing::<Vstart>().get_vstart() as usize;
    normal_exec(cpu, |cpu| {
        f(cpu, vstart).map_err(|err| err.finish_vector_exec_error(cpu))
    })?;

    // Any successfully retired vector instruction completes all resumable work.
    // Precise memory traps return before this point and preserve their fault index.
    cpu.csr
        .write_directly(Vstart::get_index(), 0)
        .then_some(())
        .unwrap();

    // TODO: updata vector registers status.

    Ok(())
}

/// XXX: The `imm` value has been processed (shifted and sign_extended) for performance.
/// DO NOT process it again.
///
/// `imm` value is shifted by:
///
/// Type B: 1,
/// Type J: 12,
/// Type U: 12,
///
/// `imm` value is sign_extended by:
///
/// Type B: 13,
/// Type I: 12,
/// Type J: 21,
/// Type U: 32,
/// Type S: 12,
///
/// The `imm` value has been masked in the right shift instructions like `SRLI`.
///
/// For detailed implementation, see [`super::decoder::decode_info`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RVInstrInfo {
    None,
    R {
        rs1: u8,
        rs2: u8,
        rd: u8,
    },
    R_rm {
        rs1: u8,
        rs2: u8,
        rd: u8,
        rm: u8,
    },
    R4_rm {
        rs1: u8,
        rs2: u8,
        rs3: u8,
        rd: u8,
        rm: u8,
    },
    I {
        rs1: u8,
        rd: u8,
        imm: WordType,
    },
    S {
        rs1: u8,
        rs2: u8,
        imm: WordType,
    },
    B {
        rs1: u8,
        rs2: u8,
        imm: WordType,
    },
    U {
        rd: u8,
        imm: WordType,
    },
    J {
        rd: u8,
        imm: WordType,
    },
    A {
        rs1: u8,
        rs2: u8,
        rd: u8,
        rl: bool,
        aq: bool,
    },
    V {
        rs1: u8,
        rs2: u8,
        rd: u8,
        vm: bool,
        func6: u8,
    },

    // Compressed
    CR {
        rd_rs1: u8,
        rs2: u8,
    },
    CI {
        rd_rs1: u8,
        imm: WordType,
    },
    CSS {
        rs2: u8,
        imm: WordType,
    },
    CIW {
        rd: u8,
        imm: WordType,
    },
    CL {
        rd: u8,
        rs1: u8,
        imm: WordType,
    },
    CS {
        rs1: u8,
        rs2: u8,
        imm: WordType,
    },
    CA {
        rd_rs1: u8,
        rs2: u8,
    },
    CB {
        rd_rs1: u8,
        imm: WordType,
    },
    CJ {
        target: WordType,
    },
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrFormat {
    None,
    R,
    R_rm,
    R4_rm,
    I,
    S,
    B,
    U,
    J,
    A,
    V,

    CI,
    CIW,
    CA,
    CB,
    CJ,
    CL,
    CR,
    CS,
    CSS,
}

// define a single enum for every instruction
// define tables for each instruction set
#[macro_export]
macro_rules! define_riscv_isa {
    ( $tot_instr_name:ident,
        $( $isa_name:ident, $isa_table_name:ident, {$(
                $name:ident {
                    format: $fmt:expr,
                    mask: $mask:literal,
                    key: $key:literal,
                }),* $(,)?
            }
        ),* $(,)?
    ) => {

        define_instr_enum!($tot_instr_name, $($($name,)*)*);

        impl $tot_instr_name {
            /// Returns the encoded instruction length in bytes.
            #[allow(clippy::len_without_is_empty)]
            pub const fn len(&self) -> $crate::config::arch_config::WordType {
                match self {
                    $(
                        $(
                            $tot_instr_name::$name => {
                                if ($key as u32) & 0b11 == 0b11 { 4 } else { 2 }
                            }
                        )*
                    )*
                }
            }

            pub fn isa_name(&self) -> &'static str {
                match self {
                    $(
                        $(
                            $tot_instr_name::$name => stringify!($isa_name),
                        )*
                    )*
                }
            }
        }

        #[derive(Debug, Clone)]
        pub struct RVInstrDesc {
            pub instr: $tot_instr_name,
            pub format: InstrFormat,
            pub mask: u32,
            pub key: u32,
        }

        $(
            pub const $isa_table_name: &[RVInstrDesc] = &[
                $(
                    RVInstrDesc {
                        instr: $tot_instr_name::$name,
                        format: $fmt,
                        mask: $mask,
                        key: $key,
                    }
                ),*
            ];
        )*
    };
}
