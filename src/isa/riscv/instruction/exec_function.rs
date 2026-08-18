use std::{
    hint::{unlikely, unreachable_unchecked},
    marker::PhantomData,
};

pub(super) use super::exec_float_function::*;
use super::normal_exec;

use crate::{
    config::arch_config::WordType,
    isa::riscv::{
        csr_reg::{NamedCsrReg, csr_macro::Minstret},
        executor::RVCPU,
        instruction::RVInstrInfo,
        trap::Exception,
    },
    utils::{
        TruncateFrom, TruncateToBits, UnsignedInteger, as_signed_i128, from_signed_i128,
        shift_amount, sign_extend_u32, wrapping_add_as_signed,
    },
};

/// ExecTrait will generate operation result to `exec_xxx` function.
/// ExecTrait::exec only do calculate.
/// `exec_xxx` function interact with other mod in CPU.
pub(in super::super) trait ExecTrait<OUT, IN = WordType> {
    fn exec(a: IN, b: IN) -> OUT;
}

pub(in super::super) trait ExecUnaryTrait<OUT, IN = WordType> {
    fn exec(a: IN) -> OUT;
}

pub(in super::super) struct ExecMove<T = WordType> {
    _marker: PhantomData<T>,
}

impl<T> ExecUnaryTrait<Result<T, Exception>, T> for ExecMove<T>
where
    T: Copy,
{
    fn exec(a: T) -> Result<T, Exception> {
        Ok(a)
    }
}

// XXX: Remeber that `imm` has been sign_extended in decoder.

/// Process arithmetic instructions with `rs1`, (`rs2` or `imm`) and `rd` in RV32I/RV64I.
#[inline]
pub(super) fn exec_arith<F>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    F: ExecTrait<Result<WordType, Exception>>,
{
    super::normal_exec(cpu, |cpu| {
        let (rd, rst) = match info {
            RVInstrInfo::R { rs1, rs2, rd } => {
                let (val1, val2) = cpu.reg_file.read(rs1, rs2);
                (rd, F::exec(val1, val2)?)
            }
            RVInstrInfo::I { rs1, rd, imm } => {
                let val1 = cpu.reg_file.read(rs1, 0).0;
                (rd, F::exec(val1, imm)?) // imm has been sign_extended
            }
            _ => unsafe { unreachable_unchecked() },
        };

        cpu.reg_file.write(rd, rst);
        Ok(())
    })
}

pub(super) fn exec_branch<F>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    F: ExecTrait<bool>,
{
    if let RVInstrInfo::B { rs1, rs2, imm } = info {
        let (val1, val2) = cpu.reg_file.read(rs1, rs2);

        if F::exec(val1, val2) {
            let target = cpu.pc.wrapping_add(imm); // imm has been sign_extended

            // Like JAL(R), branch instructions will generate an exception.
            super::check_jump_alignment(cpu, target)?;

            cpu.pc = target;
        } else {
            cpu.pc = cpu.pc.wrapping_add(4);
        }
    } else {
        std::unreachable!();
    }

    cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
    Ok(())
}

pub(super) fn exec_load<T, const EXTEND: bool>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    T: UnsignedInteger,
{
    normal_exec(cpu, |cpu| {
        let RVInstrInfo::I { rs1, rd, imm } = info else {
            std::unreachable!();
        };
        let val = cpu.reg_file.read(rs1, 0).0;
        let addr = wrapping_add_as_signed(val, imm);
        super::exec_core::handle_load::<T, EXTEND>(cpu, rd, addr)
    })
}

pub(super) fn exec_store<T>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    T: UnsignedInteger,
{
    normal_exec(cpu, |cpu| {
        let RVInstrInfo::S { rs1, rs2, imm } = info else {
            std::unreachable!();
        };
        let (val1, val2) = cpu.reg_file.read(rs1, rs2);
        let addr = wrapping_add_as_signed(val1, imm);

        cpu.write(addr, T::truncate_from(val2))
    })
}

pub(super) fn exec_csrw<const UIMM: bool>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    if let RVInstrInfo::I { rs1, rd, imm } = info {
        // For I-type instructions, `imm` has been 12-bit sign-extended in decoder for performance,
        // but in CSRW instruction, `imm` is actually a CSR index, sign-extend is not needed.
        let imm = imm.truncate_to_bits(12);

        // Check write permission before read CSR.
        if cpu.csr.is_write_priv_legal(imm) == false {
            return Err(Exception::IllegalInstruction);
        }

        let new_val = if UIMM {
            rs1 as WordType
        } else {
            cpu.reg_file.read(rs1, rs1).0
        };

        if rd != 0 {
            let value = cpu.read_csr(imm)?;
            cpu.reg_file.write(rd, value);
        }

        cpu.write_csr(imm, new_val)?;

        if imm != Minstret::get_index() {
            cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
        }
    }

    cpu.pc = cpu.pc.wrapping_add(4);

    Ok(())
}

pub(super) fn exec_csr_bit<const SET: bool, const UIMM: bool>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    let RVInstrInfo::I { rs1, rd, imm } = info else {
        unreachable!();
    };

    // See the comments in [`exec_csrw`].
    let imm = imm.truncate_to_bits(12);

    let rhs = if UIMM {
        rs1 as WordType
    } else {
        cpu.reg_file.read(rs1, rs1).0
    };

    if rhs == 0 {
        // Only read CSR, no write permission check needed.
        let value = cpu.read_csr(imm)?;
        cpu.reg_file.write(rd, value);
    } else {
        // Check write permission before read CSR.
        if cpu.csr.is_write_priv_legal(imm) == false {
            return Err(Exception::IllegalInstruction);
        }

        let value = cpu.read_csr(imm)?;
        cpu.reg_file.write(rd, value);

        let data = if SET { value | rhs } else { value & !rhs };
        cpu.write_csr(imm, data)?;

        if imm != Minstret::get_index() {
            cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
        }
    }

    cpu.pc = cpu.pc.wrapping_add(4);

    Ok(())
}

pub(super) fn exec_nop(_info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception> {
    normal_exec(cpu, |_| Ok(()))
}

// =============================================
//                  ExecTrait
// =============================================
// Arith
pub(in super::super) struct ExecAdd<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecAdd<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a.wrapping_add(&b))
    }
}

pub(in super::super) type ExecAddu<T = WordType> = ExecAdd<T>;

pub(in super::super) struct ExecSub<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecSub<T>
where
    T: num_traits::WrappingSub,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a.wrapping_sub(&b))
    }
}

pub(in super::super) type ExecSubu<T = WordType> = ExecSub<T>;

/// Reverse subtraction executor for instructions whose scalar/immediate operand is the minuend.
///
/// `ExecSub` computes `a - b`; this executor computes `b - a` with wrapping semantics.
/// It is used by vector reverse-subtract forms such as `vrsub.vx` and `vrsub.vi`, where the
/// element from `vs2` is passed as `a` and the scalar or immediate value is passed as `b`.
pub(in super::super) struct ExecRevSub<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecRevSub<T>
where
    T: num_traits::WrappingSub,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(b.wrapping_sub(&a))
    }
}

pub(in super::super) struct ExecMulLow<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecMulLow<T>
where
    T: num_traits::WrappingMul,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a.wrapping_mul(&b))
    }
}

pub(in super::super) struct ExecMulHighUnsigned<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecMulHighUnsigned<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        let product: u128 = a.into();
        let rhs: u128 = b.into();
        Ok(T::truncate_from((product * rhs) >> T::BITS))
    }
}

// NOTE: This version is slow and deprecated.

// pub(in super::super) struct ExecMulHighUnsigned {}
// impl ExecTrait<Result<WordType, Exception>> for ExecMulHighUnsigned {
//     fn exec(a: WordType, b: WordType) -> Result<WordType, Exception> {
//         const XLEN: WordType = (size_of::<WordType>() << 3) as WordType;
//         const HALF_XLEN: WordType = XLEN >> 1;
//         const HALF_XLEN_MAX: WordType = WordType::MAX >> (XLEN >> 1);

//         let lhs_hi = a >> HALF_XLEN;
//         let lhs_lo = a & HALF_XLEN_MAX;
//         let rhs_hi = b >> HALF_XLEN;
//         let rhs_lo = b & HALF_XLEN_MAX;

//         // 4个部分
//         let p1 = lhs_hi * rhs_hi; // 高高
//         let p2 = lhs_hi * rhs_lo; // 高低
//         let p3 = lhs_lo * rhs_hi; // 低高
//         let p4 = lhs_lo * rhs_lo; // 低低

//         // 合并高位
//         let mid = (p2 & HALF_XLEN_MAX) + (p3 & HALF_XLEN_MAX) + (p4 >> HALF_XLEN);
//         let high = p1 + (p2 >> HALF_XLEN) + (p3 >> HALF_XLEN) + (mid >> HALF_XLEN);

//         Ok(high)
//     }
// }

pub(in super::super) struct ExecMulHighSigned<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecMulHighSigned<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        let product = as_signed_i128(a) * as_signed_i128(b);
        Ok(from_signed_i128(product >> T::BITS))
    }
}

pub(in super::super) struct ExecMulHighSignedUnsigned<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecMulHighSignedUnsigned<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        let rhs: u128 = b.into();
        let product = as_signed_i128(a) * (rhs as i128);
        Ok(from_signed_i128(product >> T::BITS))
    }
}

pub(in super::super) struct ExecDivSigned<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecDivSigned<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        if unlikely(b == T::from(0u8)) {
            return Ok(T::MAX);
        }

        Ok(from_signed_i128(
            as_signed_i128(a).wrapping_div(as_signed_i128(b)),
        ))
    }
}

pub(in super::super) struct ExecDivUnsigned<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecDivUnsigned<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        if unlikely(b == T::from(0u8)) {
            return Ok(T::MAX);
        }
        Ok(a / b)
    }
}

pub(in super::super) struct ExecRemSigned<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecRemSigned<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        if unlikely(b == T::from(0u8)) {
            return Ok(a);
        }

        Ok(from_signed_i128(
            as_signed_i128(a).wrapping_rem(as_signed_i128(b)),
        ))
    }
}

pub(in super::super) struct ExecRemUnsigned<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecRemUnsigned<T>
where
    T: UnsignedInteger + std::ops::Rem<Output = T>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        if unlikely(b == T::from(0u8)) {
            return Ok(a);
        }
        Ok(a % b)
    }
}

// Arith word
pub(in super::super) struct ExecAddw<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<WordType, Exception>, T> for ExecAddw<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<WordType, Exception> {
        let [a, b]: [u32; 2] = [a.truncate_to(), b.truncate_to()];
        Ok(sign_extend_u32(a.wrapping_add(b)))
    }
}

pub(in super::super) struct ExecSubw<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<WordType, Exception>, T> for ExecSubw<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<WordType, Exception> {
        let [a, b]: [u32; 2] = [a.truncate_to(), b.truncate_to()];
        Ok(sign_extend_u32(a.wrapping_sub(b)))
    }
}

pub(in super::super) struct ExecMulw<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<WordType, Exception>, T> for ExecMulw<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<WordType, Exception> {
        let [a, b]: [u32; 2] = [a.truncate_to(), b.truncate_to()];
        Ok(sign_extend_u32(a.wrapping_mul(b)))
    }
}

pub(in super::super) struct ExecDivw<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<WordType, Exception>, T> for ExecDivw<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<WordType, Exception> {
        let [sa, sb]: [u32; 2] = [a.truncate_to(), b.truncate_to()];
        let [sa, sb] = [sa as i32, sb as i32];
        if unlikely(sb == 0) {
            return Ok(WordType::MAX);
        }

        Ok(sign_extend_u32((sa.wrapping_div(sb)).cast_unsigned()))
    }
}

pub(in super::super) struct ExecRemw<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<WordType, Exception>, T> for ExecRemw<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<WordType, Exception> {
        let [sa, sb]: [u32; 2] = [a.truncate_to(), b.truncate_to()];
        let [sa, sb] = [sa as i32, sb as i32];
        if unlikely(sb == 0) {
            return Ok(sign_extend_u32(sa as u32));
        }

        Ok(sign_extend_u32(sa.wrapping_rem(sb) as u32))
    }
}

pub(in super::super) struct ExecDivuw<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<WordType, Exception>, T> for ExecDivuw<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<WordType, Exception> {
        let [sa, sb]: [u32; 2] = [a.truncate_to(), b.truncate_to()];
        if unlikely(sb == 0) {
            return Ok(WordType::MAX);
        }
        Ok(sign_extend_u32(sa / sb))
    }
}

pub(in super::super) struct ExecRemuw<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<WordType, Exception>, T> for ExecRemuw<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<WordType, Exception> {
        let [sa, sb]: [u32; 2] = [a.truncate_to(), b.truncate_to()];
        if unlikely(sb == 0) {
            return Ok(sign_extend_u32(sa));
        }
        Ok(sign_extend_u32(sa % sb))
    }
}

// Bit
pub(in super::super) struct ExecSLL<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecSLL<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a << shift_amount(b))
    }
}

pub(in super::super) struct ExecSRL<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecSRL<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a >> shift_amount(b))
    }
}

pub(in super::super) struct ExecSRA<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecSRA<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(from_signed_i128(as_signed_i128(a) >> shift_amount(b)))
    }
}

pub(in super::super) struct ExecSLLW<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<WordType, Exception>, T> for ExecSLLW<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<WordType, Exception> {
        let a: u32 = a.truncate_to();
        Ok(sign_extend_u32(a.wrapping_shl((b.into() as u32) & 0x1F)))
    }
}

pub(in super::super) struct ExecSRLW<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<WordType, Exception>, T> for ExecSRLW<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<WordType, Exception> {
        let a: u32 = a.truncate_to();
        Ok(sign_extend_u32(a.wrapping_shr((b.into() as u32) & 0x1F)))
    }
}

pub(in super::super) struct ExecSRAW<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<WordType, Exception>, T> for ExecSRAW<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> Result<WordType, Exception> {
        let a: u32 = a.truncate_to();
        let a = as_signed_i128(a);
        let shamt = (Into::<u64>::into(b) as u32) & 0x1F;
        Ok(sign_extend_u32((a >> shamt) as u32))
    }
}

pub(in super::super) struct ExecZext<TOut = WordType, TIn = WordType> {
    phantom: PhantomData<(TOut, TIn)>,
}

impl<TOut, TIn> ExecUnaryTrait<Result<TOut, Exception>, TIn> for ExecZext<TOut, TIn>
where
    TOut: UnsignedInteger + TruncateFrom<TIn>,
    TIn: UnsignedInteger,
{
    fn exec(a: TIn) -> Result<TOut, Exception> {
        Ok(TOut::truncate_from(a))
    }
}

pub(in super::super) struct ExecSext<TOut = WordType, TIn = WordType> {
    phantom: PhantomData<(TOut, TIn)>,
}

impl<TOut, TIn> ExecUnaryTrait<Result<TOut, Exception>, TIn> for ExecSext<TOut, TIn>
where
    TOut: UnsignedInteger,
    TIn: UnsignedInteger + Into<u128>,
{
    fn exec(a: TIn) -> Result<TOut, Exception> {
        Ok(from_signed_i128(as_signed_i128(a)))
    }
}

pub(in super::super) struct ExecNsrl<TOut = WordType, TIn = WordType> {
    phantom: PhantomData<(TOut, TIn)>,
}

impl<TOut, TIn> ExecTrait<Result<TOut, Exception>, TIn> for ExecNsrl<TOut, TIn>
where
    TOut: UnsignedInteger + TruncateFrom<TIn>,
    TIn: UnsignedInteger,
{
    fn exec(a: TIn, b: TIn) -> Result<TOut, Exception> {
        Ok(TOut::truncate_from(a >> shift_amount(b)))
    }
}

pub(in super::super) struct ExecNsra<TOut = WordType, TIn = WordType> {
    phantom: PhantomData<(TOut, TIn)>,
}

impl<TOut, TIn> ExecTrait<Result<TOut, Exception>, TIn> for ExecNsra<TOut, TIn>
where
    TOut: UnsignedInteger,
    TIn: UnsignedInteger + Into<u128>,
{
    fn exec(a: TIn, b: TIn) -> Result<TOut, Exception> {
        Ok(from_signed_i128(as_signed_i128(a) >> shift_amount(b)))
    }
}

pub(in super::super) struct ExecAnd<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecAnd<T>
where
    T: std::ops::BitAnd<Output = T>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a & b)
    }
}

pub(in super::super) struct ExecNand<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecNand<T>
where
    T: std::ops::BitAnd<Output = T> + std::ops::Not<Output = T>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(!(a & b))
    }
}

pub(in super::super) struct ExecAndn<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecAndn<T>
where
    T: std::ops::BitAnd<Output = T> + std::ops::Not<Output = T>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a & !b)
    }
}

pub(in super::super) struct ExecOr<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecOr<T>
where
    T: std::ops::BitOr<Output = T>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a | b)
    }
}

pub(in super::super) struct ExecNor<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecNor<T>
where
    T: std::ops::BitOr<Output = T> + std::ops::Not<Output = T>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(!(a | b))
    }
}

pub(in super::super) struct ExecOrn<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecOrn<T>
where
    T: std::ops::BitOr<Output = T> + std::ops::Not<Output = T>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a | !b)
    }
}

pub(in super::super) struct ExecXor<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecXor<T>
where
    T: std::ops::BitXor<Output = T>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a ^ b)
    }
}

pub(in super::super) struct ExecXnor<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecXnor<T>
where
    T: std::ops::BitXor<Output = T> + std::ops::Not<Output = T>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(!(a ^ b))
    }
}

// Compare
pub(in super::super) struct ExecMax<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecMax<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        if as_signed_i128(a) > as_signed_i128(b) {
            Ok(a)
        } else {
            Ok(b)
        }
    }
}

pub(in super::super) struct ExecMaxu<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecMaxu<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a.max(b))
    }
}

pub(in super::super) struct ExecMin<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecMin<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        if as_signed_i128(a) < as_signed_i128(b) {
            Ok(a)
        } else {
            Ok(b)
        }
    }
}

pub(in super::super) struct ExecMinu<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecMinu<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a.min(b))
    }
}

pub(in super::super) struct ExecSignedLess<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<bool, T> for ExecSignedLess<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> bool {
        as_signed_i128(a) < as_signed_i128(b)
    }
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecSignedLess<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(T::from((as_signed_i128(a) < as_signed_i128(b)) as u8))
    }
}

pub(in super::super) struct ExecUnsignedLess<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<bool, T> for ExecUnsignedLess<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> bool {
        a < b
    }
}

impl<T> ExecTrait<Result<T, Exception>, T> for ExecUnsignedLess<T>
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(T::from((a < b) as u8))
    }
}

pub(in super::super) struct ExecEqual<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<bool, T> for ExecEqual<T>
where
    T: PartialEq,
{
    fn exec(a: T, b: T) -> bool {
        a == b
    }
}

pub(in super::super) struct ExecNotEqual<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<bool, T> for ExecNotEqual<T>
where
    T: PartialEq,
{
    fn exec(a: T, b: T) -> bool {
        a != b
    }
}

pub(in super::super) struct ExecSignedGreatEqual<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<bool, T> for ExecSignedGreatEqual<T>
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> bool {
        as_signed_i128(a) >= as_signed_i128(b)
    }
}

pub(in super::super) struct ExecUnsignedGreatEqual<T = WordType> {
    phantom: PhantomData<T>,
}

impl<T> ExecTrait<bool, T> for ExecUnsignedGreatEqual<T>
where
    T: PartialOrd,
{
    fn exec(a: T, b: T) -> bool {
        a >= b
    }
}
