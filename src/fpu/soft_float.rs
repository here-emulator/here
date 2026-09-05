use std::hint::{cold_path, unreachable_unchecked};

use rustc_apfloat::{
    Float, FloatConvert, Status, StatusAnd,
    ieee::{Double, Single},
};

use rustc_apfloat::Round as APFloatRound;

use crate::{
    fpu::{Classification, Round},
    utils::{
        BinaryOp, CmpOp, FloatPoint, InFloat, SignedInteger, TruncateFrom, WordTrait, make_mask,
    },
};

impl Into<APFloatRound> for Round {
    fn into(self) -> APFloatRound {
        match self {
            Round::NearestTiesToEven => APFloatRound::NearestTiesToEven,
            Round::TowardPositive => APFloatRound::TowardPositive,
            Round::TowardNegative => APFloatRound::TowardNegative,
            Round::TowardZero => APFloatRound::TowardZero,
            Round::NearestTiesToAway => APFloatRound::NearestTiesToAway,
        }
    }
}

#[derive(Clone, Copy)]
pub enum APFloat {
    Single(Single),
    Double(Double),
}

impl Into<APFloat> for f32 {
    fn into(self) -> APFloat {
        APFloat::Single(Single::from_bits(self.to_bits() as u128))
    }
}

impl Into<APFloat> for f64 {
    fn into(self) -> APFloat {
        APFloat::Double(Double::from_bits(self.to_bits() as u128))
    }
}

// Reinterpret to given type of soft float.
impl From<APFloat> for Single {
    fn from(value: APFloat) -> Self {
        match value {
            APFloat::Single(s) => s,
            APFloat::Double(d) => {
                // "Apart from transfer operations described in the previous paragraph, all other floating-point operations on
                // narrower n-bit operations, n<FLEN, check if the input operands are correctly NaN-boxed, i.e., all upper
                // FLEN-n bits are 1. If so, the n least-significant bits of the input are used as the input value, otherwise the
                // input value is treated as an n-bit canonical NaN."
                if d.to_bits() as u64 & make_mask(32, 63) != make_mask(32, 63) {
                    // Not a valid NaN-boxed f32
                    Single::qnan(None)
                } else {
                    Single::from_bits(d.to_bits() as u128)
                }
            }
        }
    }
}

impl From<APFloat> for Double {
    fn from(value: APFloat) -> Self {
        match value {
            APFloat::Single(s) => {
                let bits = s.to_bits() as u64;
                Double::from_bits((bits | make_mask(32, 63)) as u128)
            }
            APFloat::Double(d) => d,
        }
    }
}

impl Into<APFloat> for Single {
    fn into(self) -> APFloat {
        APFloat::Single(self)
    }
}

impl Into<APFloat> for Double {
    fn into(self) -> APFloat {
        APFloat::Double(self)
    }
}

/// Map a Rust primitive float type to its `rustc_apfloat` representation.
pub trait APFloatOf: Into<APFloat> {
    type Float: Float + Into<APFloat> + From<APFloat> + InFloat;
}

impl APFloatOf for f32 {
    type Float = Single;
}

impl APFloatOf for f64 {
    type Float = Double;
}

impl InFloat for Single {
    type Float = f32;

    fn into_float(self) -> f32 {
        f32::from_bits(self.to_bits() as u32)
    }

    fn from_float(f: Self::Float) -> Self {
        Self::from_bits(f.to_bits() as u128)
    }
}

impl InFloat for Double {
    type Float = f64;

    fn into_float(self) -> f64 {
        f64::from_bits(self.to_bits() as u64)
    }

    fn from_float(f: Self::Float) -> Self {
        Self::from_bits(f.to_bits() as u128)
    }
}

pub trait UnaryOpWithRound<F> {
    fn apply(a: F, round: Round) -> F;
}

pub trait BinaryOpWithRound<F> {
    fn apply(a: F, b: F, round: Round) -> StatusAnd<F>;
}

pub trait TernaryOpWithRound<F> {
    fn apply(a: F, b: F, c: F, round: Round) -> StatusAnd<F>;
}

// Implementation of operations:

macro_rules! define_binary_op {
    ($struct_name:ident, $method_name:ident) => {
        pub struct $struct_name;
        impl<F: Float> BinaryOp<F> for $struct_name {
            fn apply(a: F, b: F) -> F {
                a.$method_name(b)
            }
        }
    };
}

macro_rules! define_binary_op_r {
    ($struct_name:ident, $method_name:ident) => {
        pub struct $struct_name;
        impl<F: Float> BinaryOpWithRound<F> for $struct_name {
            fn apply(a: F, b: F, round: Round) -> StatusAnd<F> {
                a.$method_name(b, round.into())
            }
        }
    };
}

// Arithmetic

define_binary_op_r!(AddOp, add_r);
define_binary_op_r!(SubOp, sub_r);
define_binary_op_r!(MulOp, mul_r);
define_binary_op_r!(DivOp, div_r);

pub struct MulAddOp;
impl<F: Float> TernaryOpWithRound<F> for MulAddOp {
    fn apply(a: F, b: F, c: F, round: Round) -> StatusAnd<F> {
        a.mul_add_r(b, c, round.into())
    }
}

pub struct MulSubOp;
impl<F: Float> TernaryOpWithRound<F> for MulSubOp {
    fn apply(a: F, b: F, c: F, round: Round) -> StatusAnd<F> {
        a.mul_add_r(b, -c, round.into())
    }
}

pub struct NegMulAddOp;
impl<F: Float> TernaryOpWithRound<F> for NegMulAddOp {
    fn apply(a: F, b: F, c: F, round: Round) -> StatusAnd<F> {
        (-a).mul_add_r(b, -c, round.into())
    }
}

pub struct NegMulSubOp;
impl<F: Float> TernaryOpWithRound<F> for NegMulSubOp {
    fn apply(a: F, b: F, c: F, round: Round) -> StatusAnd<F> {
        (-a).mul_add_r(b, c, round.into())
    }
}

// Sign injection

define_binary_op!(SignInjectOp, copy_sign);

pub struct SignInjectNegOp;
impl<F: Float> BinaryOp<F> for SignInjectNegOp {
    fn apply(a: F, b: F) -> F {
        if a.is_negative() == b.is_negative() {
            -a
        } else {
            a
        }
    }
}

pub struct SignInjectXorOp;
impl<F: Float> BinaryOp<F> for SignInjectXorOp {
    fn apply(a: F, b: F) -> F {
        if a.is_negative() != b.is_negative() {
            a.abs().neg()
        } else {
            a.abs()
        }
    }
}

// Classify (RISC-V)
fn classify<F: Float>(f: F) -> Classification {
    if f.is_normal() {
        if f.is_negative() {
            Classification::NormalNegative
        } else {
            Classification::NormalPositive
        }
    } else {
        cold_path();
        if f.is_denormal() {
            if f.is_negative() {
                Classification::SubnormalNegative
            } else {
                Classification::SubnormalPositive
            }
        } else if f.is_zero() {
            if f.is_negative() {
                Classification::NegativeZero
            } else {
                Classification::PositiveZero
            }
        } else if f.is_infinite() {
            if f.is_negative() {
                Classification::NegativeInfinity
            } else {
                Classification::PositiveInfinity
            }
        } else if f.is_nan() {
            if f.is_signaling() {
                Classification::SignalingNaN
            } else {
                Classification::QuietNaN
            }
        } else {
            unsafe {
                unreachable_unchecked();
            }
        }
    }
}

// Compare
pub struct EqOp;
impl<F: Float> CmpOp<F> for EqOp {
    fn apply(a: F, b: F) -> StatusAnd<bool> {
        // `feq` do "quiet comparison", according to RISC-V manual 2025-08-05,
        // which means only sets the invalid op if either input is a signaling NaN.
        if a.is_signaling() || b.is_signaling() {
            Status::INVALID_OP.and(false)
        } else {
            Status::OK.and(a == b)
        }
    }
}

pub struct LtOp;
impl<F: Float> CmpOp<F> for LtOp {
    fn apply(a: F, b: F) -> StatusAnd<bool> {
        // Unlike `feq`, `flt` do "signaling comparison",
        // that means set the invalid operation exception flag if either input is NaN.
        if a.is_nan() || b.is_nan() {
            Status::INVALID_OP.and(false)
        } else {
            Status::OK.and(a < b)
        }
    }
}

pub struct LeOp;
impl<F: Float> CmpOp<F> for LeOp {
    fn apply(a: F, b: F) -> StatusAnd<bool> {
        // Same with `flt`
        if a.is_nan() || b.is_nan() {
            Status::INVALID_OP.and(false)
        } else {
            Status::OK.and(a <= b)
        }
    }
}

pub struct SoftFPU {
    last_status: std::cell::Cell<Status>,
    reg_file: [APFloat; 32],
    pub unify_cnan: bool,
}

impl SoftFPU {
    pub fn from(unify_cnan: bool) -> Self {
        Self {
            last_status: std::cell::Cell::new(Status::OK),
            reg_file: [APFloat::Single(Single::from_bits(0)); 32],
            unify_cnan: unify_cnan,
        }
    }

    pub fn last_status(&self) -> Status {
        self.last_status.get()
    }

    fn save_and_unwrap<T>(&mut self, status_and: StatusAnd<T>) -> T {
        self.last_status.set(status_and.status);
        status_and.value
    }

    pub fn load_raw(&self, index: u8) -> u64 {
        Double::from(self.reg_file[index as usize]).to_bits() as u64
    }

    pub fn load<F: FloatPoint>(&self, index: u8) -> F {
        let f: <F as APFloatOf>::Float = self.reg_file[index as usize].into();
        F::from_bits(F::BitsType::truncate_from(f.to_bits()))
    }

    pub fn store<F: Into<APFloat>>(&mut self, index: u8, value: F) {
        self.reg_file[index as usize] = value.into();
    }

    pub fn store_raw<F: APFloatOf>(&mut self, index: u8, value: u64) {
        self.reg_file[index as usize] = F::Float::from_bits(value as u128).into();
    }

    pub fn cvt_unsigned_and_store<F: FloatPoint>(&mut self, index: u8, value: u128, round: Round) {
        let rst = <F as APFloatOf>::Float::from_u128_r(value, round.into());
        let val = self.save_and_unwrap(rst);
        self.reg_file[index as usize] = val.into();
    }

    pub fn cvt_signed_and_store<F: FloatPoint>(&mut self, index: u8, value: i128, round: Round) {
        let rst = <F as APFloatOf>::Float::from_i128_r(value, round.into());
        let val = self.save_and_unwrap(rst);
        self.reg_file[index as usize] = val.into();
    }

    pub fn get_and_cvt_unsigned<F: APFloatOf, U: WordTrait>(
        &mut self,
        index: u8,
        round: Round,
    ) -> u128 {
        let f: F::Float = self.reg_file[index as usize].into();

        if f.is_nan() {
            return U::MAX.into();
        }

        let mut _is_exact = true;
        let StatusAnd::<_> { status, value } = f.to_u128_r(U::BITS, round.into(), &mut _is_exact);

        self.last_status.set(status);
        value
    }

    pub fn get_and_cvt_signed<F: APFloatOf, U: WordTrait>(
        &mut self,
        index: u8,
        round: Round,
    ) -> i128 {
        let f: F::Float = self.reg_file[index as usize].into();

        if f.is_nan() {
            return U::SignedType::MAX.into();
        }

        let mut _is_exact = true;
        let StatusAnd::<_> { status, value } = f.to_i128_r(U::BITS, round.into(), &mut _is_exact);

        self.last_status.set(status);
        value
    }

    pub fn cvt_float_and_store<F: APFloatOf, T: APFloatOf>(&mut self, rd: u8, rs: u8, round: Round)
    where
        F::Float: FloatConvert<T::Float>,
    {
        let mut _loses_info = true;
        let f: F::Float = self.reg_file[rs as usize].into();
        let mut f: T::Float = self.save_and_unwrap(f.convert_r(round.into(), &mut _loses_info));
        if self.unify_cnan && f.is_nan() {
            f = T::Float::qnan(None);
        }
        self.reg_file[rd as usize] = f.into();
    }

    pub fn classify<T: APFloatOf>(&self, rs: u8) -> Classification {
        let f: T::Float = self.reg_file[rs as usize].into();
        classify(f)
    }

    pub fn compare<Op: CmpOp<T::Float>, T: FloatPoint>(&mut self, rs1: u8, rs2: u8) -> bool {
        let a: T::Float = self.reg_file[rs1 as usize].into();
        let b: T::Float = self.reg_file[rs2 as usize].into();
        self.save_and_unwrap(Op::apply(a, b))
    }

    pub fn min_num<T: FloatPoint>(&mut self, rs1: u8, rs2: u8, rd: u8) {
        let a: T::Float = self.reg_file[rs1 as usize].into();
        let b: T::Float = self.reg_file[rs2 as usize].into();

        if a.is_signaling() || b.is_signaling() {
            self.last_status.set(Status::INVALID_OP);
            if self.unify_cnan && a.is_nan() && b.is_nan() {
                self.reg_file[rd as usize] = <T as APFloatOf>::Float::qnan(None).into();
                return;
            }
            self.reg_file[rd as usize] = a.min(b).into();
            return;
        } else {
            self.last_status.set(Status::OK);
        }

        if self.unify_cnan && a.is_nan() && b.is_nan() {
            self.reg_file[rd as usize] = <T as APFloatOf>::Float::qnan(None).into();
            return;
        }

        // RISC-V requires -0.0 < +0.0 for min/max
        if a.is_zero() && b.is_zero() && a.is_negative() != b.is_negative() {
            self.reg_file[rd as usize] = if a.is_negative() { a.into() } else { b.into() };
            return;
        }

        self.reg_file[rd as usize] = a.min(b).into();
    }

    pub fn max_num<T: FloatPoint>(&mut self, rs1: u8, rs2: u8, rd: u8) {
        let a: T::Float = self.reg_file[rs1 as usize].into();
        let b: T::Float = self.reg_file[rs2 as usize].into();

        if a.is_signaling() || b.is_signaling() {
            self.last_status.set(Status::INVALID_OP);
            if self.unify_cnan && a.is_nan() && b.is_nan() {
                self.reg_file[rd as usize] = <T as APFloatOf>::Float::qnan(None).into();
                return;
            }
            self.reg_file[rd as usize] = a.max(b).into();
            return;
        } else {
            self.last_status.set(Status::OK);
        }

        if self.unify_cnan && a.is_nan() && b.is_nan() {
            self.reg_file[rd as usize] = <T as APFloatOf>::Float::qnan(None).into();
            return;
        }

        // -0.0 < +0.0 for min/max
        if a.is_zero() && b.is_zero() && a.is_negative() != b.is_negative() {
            self.reg_file[rd as usize] = if a.is_negative() { b.into() } else { a.into() };
            return;
        }

        self.reg_file[rd as usize] = a.max(b).into();
    }

    /// APFloat doesn't support sqrt, so current implementation use native float sqrt.
    ///
    /// This may not be fully IEEE 754 compliant.
    pub fn sqrt<T: FloatPoint>(&mut self, rs: u8, rd: u8, _round: Round) {
        let f: T::Float = self.reg_file[rs as usize].into();
        let mut status = Status::OK;

        if f.is_nan() {
            if f.is_signaling() {
                status |= Status::INVALID_OP;
            }
        } else if f.is_negative() && !f.is_zero() {
            status |= Status::INVALID_OP;
        }

        let f_native = f.into_float();
        let res_native = f_native.sqrt();
        let mut res = T::Float::from_float(res_native);

        if status == Status::OK {
            if !res.is_infinite() && !res.is_nan() && !res.is_zero() {
                // Use fused multiply-add to check for inexactness.
                // If sqrt(x) is exact, then res * res - x == 0.
                // We use mul_add(res, res, -f_native) which computes res * res - f_native with only one rounding.
                let zero = <T::Float as InFloat>::Float::from(0.0f32);
                if res_native.mul_add(res_native, -f_native) != zero {
                    status |= Status::INEXACT;
                }
            }
        }

        if self.unify_cnan && res.is_nan() {
            res = <T as APFloatOf>::Float::qnan(None);
        }
        self.last_status.set(status);
        self.reg_file[rd as usize] = res.into();
    }

    /// This function is used for operations like sign injection in RISC-V,
    /// which should not change the payload of NaN.
    pub fn exec_binary_ignore_cnan<Op, T: FloatPoint>(&mut self, rs1: u8, rs2: u8, rd: u8)
    where
        Op: BinaryOp<T::Float>,
    {
        let a: T::Float = self.reg_file[rs1 as usize].into();
        let b: T::Float = self.reg_file[rs2 as usize].into();
        self.reg_file[rd as usize] = Op::apply(a, b).into();
    }

    pub fn exec_binary_r<Op, T: FloatPoint>(&mut self, rs1: u8, rs2: u8, rd: u8, round: Round)
    where
        Op: BinaryOpWithRound<<T as APFloatOf>::Float>,
    {
        let a: T::Float = self.reg_file[rs1 as usize].into();
        let b: T::Float = self.reg_file[rs2 as usize].into();
        let mut res = self.save_and_unwrap(Op::apply(a, b, round));
        if self.unify_cnan && res.is_nan() {
            res = <T as APFloatOf>::Float::qnan(None);
        }
        self.reg_file[rd as usize] = res.into();
    }

    pub fn exec_ternary_r<Op, T: FloatPoint>(
        &mut self,
        rs1: u8,
        rs2: u8,
        rs3: u8,
        rd: u8,
        round: Round,
    ) where
        Op: TernaryOpWithRound<T::Float>,
    {
        let a: T::Float = self.reg_file[rs1 as usize].into();
        let b: T::Float = self.reg_file[rs2 as usize].into();
        let c: T::Float = self.reg_file[rs3 as usize].into();
        let mut res = self.save_and_unwrap(Op::apply(a, b, c, round));
        if self.unify_cnan && res.is_nan() {
            res = <T as APFloatOf>::Float::qnan(None);
        }
        self.reg_file[rd as usize] = res.into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_arith() {
        let mut fpu = SoftFPU::from(true);
        fpu.store::<f32>(1, 2.0f32);
        fpu.store::<f32>(2, 3.0f32);

        // add
        fpu.exec_binary_r::<AddOp, f32>(1, 2, 3, Round::NearestTiesToEven);
        assert_eq!(fpu.load::<f32>(3), 5.0f32);

        // sub
        fpu.exec_binary_r::<SubOp, f32>(2, 1, 6, Round::NearestTiesToEven);
        assert_eq!(fpu.load::<f32>(6), 1.0f32);

        // mul
        fpu.exec_binary_r::<MulOp, f32>(1, 2, 4, Round::NearestTiesToEven);
        assert_eq!(fpu.load::<f32>(4), 6.0f32);

        // div
        fpu.exec_binary_r::<DivOp, f32>(2, 1, 5, Round::NearestTiesToEven);
        assert_eq!(fpu.load::<f32>(5), 1.5f32);
    }

    #[test]
    fn test_ternary_mul_add() {
        let mut fpu = SoftFPU::from(true);
        fpu.store::<f64>(1, 1.5f64);
        fpu.store::<f64>(2, 2.0f64);
        fpu.store::<f64>(3, 0.5f64);

        fpu.exec_ternary_r::<MulAddOp, f64>(1, 2, 3, 4, Round::NearestTiesToEven);
        assert_eq!(fpu.load::<f64>(4), 1.5f64 * 2.0f64 + 0.5f64);
    }

    #[test]
    fn test_classify() {
        let mut fpu = SoftFPU::from(true);

        fpu.store::<f32>(1, 0.0f32);
        assert!(matches!(
            fpu.classify::<f32>(1),
            Classification::PositiveZero
        ));

        fpu.store::<f32>(1, -0.0f32);
        assert!(matches!(
            fpu.classify::<f32>(1),
            Classification::NegativeZero
        ));

        fpu.store::<f32>(2, f32::INFINITY);
        assert!(matches!(
            fpu.classify::<f32>(2),
            Classification::PositiveInfinity
        ));

        fpu.store::<f32>(2, f32::NEG_INFINITY);
        assert!(matches!(
            fpu.classify::<f32>(2),
            Classification::NegativeInfinity
        ));

        fpu.store::<f32>(3, f32::NAN);
        let c = fpu.classify::<f32>(3);
        assert!(matches!(
            c,
            Classification::QuietNaN | Classification::SignalingNaN
        ));
    }

    #[test]
    fn test_subnormal_and_sign() {
        let mut fpu = SoftFPU::from(true);

        // smallest positive subnormal for f32
        fpu.store::<f32>(1, f32::from_bits(0x0000_0001));
        assert!(matches!(
            fpu.classify::<f32>(1),
            Classification::SubnormalPositive
        ));

        // smallest negative subnormal
        fpu.store::<f32>(1, f32::from_bits(0x8000_0001));
        assert!(matches!(
            fpu.classify::<f32>(1),
            Classification::SubnormalNegative
        ));
    }

    #[test]
    fn test_div_inexact_status() {
        use rustc_apfloat::Status;

        let mut fpu = SoftFPU::from(true);
        fpu.store::<f32>(1, 1.0f32);
        fpu.store::<f32>(2, 3.0f32);

        // 1/3 is inexact in binary32
        fpu.exec_binary_r::<DivOp, f32>(1, 2, 3, Round::NearestTiesToEven);
        let r = fpu.load::<f32>(3);

        assert!((r - (1.0f32 / 3.0f32)).abs() < 1e-7);
        assert!(fpu.last_status.get().contains(Status::INEXACT));
    }

    #[test]
    fn test_float_convert() {
        // f32 -> f64 conversion should preserve value for a simple value
        let mut fpu = SoftFPU::from(true);
        fpu.store::<f32>(1, 1.5f32);

        fpu.cvt_float_and_store::<f32, f64>(2, 1, Round::NearestTiesToEven);

        let as_u64 = fpu.load::<f64>(2).to_bits() as u64;
        assert_eq!(as_u64, f64::from(1.5f32).to_bits());
    }

    #[test]
    fn test_float_cmp() {
        let mut fpu = SoftFPU::from(true);

        fpu.store::<f32>(1, 3.0);
        fpu.store::<f32>(2, 3.0);
        assert!(fpu.compare::<EqOp, f32>(1, 2));
        assert!(fpu.last_status() == Status::OK);

        fpu.store::<f32>(1, f32::NAN);
        fpu.store::<f32>(2, 3.0);
        assert!(fpu.compare::<EqOp, f32>(1, 2) == false);
        assert!(fpu.last_status() == Status::OK);

        fpu.store_raw::<f32>(1, Single::snan(None).to_bits() as u64);
        fpu.store::<f32>(2, 3.0);
        assert!(fpu.compare::<EqOp, f32>(1, 2) == false);
        assert!(fpu.last_status() == Status::INVALID_OP);

        fpu.store::<f32>(1, 1.5);
        fpu.store::<f32>(2, 3.0);
        assert!(fpu.compare::<LtOp, f32>(1, 2));
        assert!(fpu.last_status() == Status::OK);

        fpu.store::<f32>(1, f32::NAN);
        fpu.store::<f32>(2, 3.0);
        assert!(fpu.compare::<LtOp, f32>(1, 2) == false);
        assert!(fpu.last_status() == Status::INVALID_OP);

        fpu.store_raw::<f32>(1, Single::snan(None).to_bits() as u64);
        fpu.store::<f32>(2, 3.0);
        assert!(fpu.compare::<LtOp, f32>(1, 2) == false);
        assert!(fpu.last_status() == Status::INVALID_OP);
    }

    #[test]
    fn test_nan_generation() {
        let mut fpu = SoftFPU::from(true);

        fpu.store::<f32>(1, 0.0f32);
        fpu.store::<f32>(2, 0.0f32);

        // 0.0 / 0.0 = NaN
        fpu.exec_binary_r::<DivOp, f32>(1, 2, 3, Round::NearestTiesToEven);
        let r = fpu.load::<f32>(3);
        assert!(r.is_nan());

        // Check for canonical NaN (0x7fc00000)
        assert_eq!(r.to_bits(), 0x7fc00000);
    }

    #[test]
    fn test_nan_propagation() {
        let mut fpu = SoftFPU::from(true);

        // A quiet NaN but not canonical (payload != 0)
        fpu.store::<f32>(1, f32::from_bits(0x7fc00001));
        fpu.store::<f32>(2, 1.0f32);

        // NaN + 1.0 = NaN
        fpu.exec_binary_r::<AddOp, f32>(1, 2, 3, Round::NearestTiesToEven);
        let r = fpu.load::<f32>(3);
        assert!(r.is_nan());
        assert_eq!(r.to_bits(), 0x7fc00000);
    }

    #[test]
    fn test_min_max_nan() {
        let mut fpu = SoftFPU::from(true);

        fpu.store::<f32>(1, f32::from_bits(0x7fc00001)); // QNaN with payload
        fpu.store::<f32>(2, f32::from_bits(0x7fc00002)); // QNaN with payload

        // min(QNaN, QNaN) -> Canonical NaN
        fpu.min_num::<f32>(1, 2, 3);
        let r = fpu.load::<f32>(3);
        assert_eq!(r.to_bits(), 0x7fc00000);

        fpu.store::<f32>(4, 1.0);
        // min(QNaN, 1.0) -> 1.0
        fpu.min_num::<f32>(1, 4, 5);
        let r = fpu.load::<f32>(5);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn test_sign_injection_nan() {
        let mut fpu = SoftFPU::from(true);

        // Source is a NaN with payload
        let payload = 0x7fc00001;
        fpu.store::<f32>(1, f32::from_bits(payload));
        fpu.store::<f32>(2, -1.0f32); // Negative sign

        // FSGNJ (copy sign from rs2 to rs1)
        // Should preserve payload of rs1, but take sign of rs2
        fpu.exec_binary_ignore_cnan::<SignInjectOp, f32>(1, 2, 3);
        let r = fpu.load::<f32>(3);

        assert!(r.is_nan());
        // Sign bit is MSB (bit 31).
        // Original payload: 0x7fc00001 (positive)
        // New value should be: 0xffc00001 (negative)
        assert_eq!(r.to_bits(), 0xffc00001);
    }

    #[test]
    fn test_nan_boxing() {
        let mut fpu = SoftFPU::from(true);

        // 1. Test Boxing: f32 -> f64
        let val_f32 = 1.2345f32;
        fpu.store::<f32>(1, val_f32);

        // When reading as f64, it should be NaN boxed
        let val_f64 = fpu.load::<f64>(1);
        let val_u64 = val_f64.to_bits();

        // Check upper 32 bits are all 1s
        assert_eq!((val_u64 >> 32), 0xFFFFFFFF);

        // Check lower 32 bits match the f32 value
        assert_eq!((val_u64 & 0xFFFFFFFF) as u32, val_f32.to_bits());

        // 2. Test Unboxing: f64 (boxed) -> f32
        // Manually create a boxed value
        let boxed_val = 0xFFFFFFFF00000000u64 | (val_f32.to_bits() as u64);
        fpu.store::<f64>(2, f64::from_bits(boxed_val));

        // Read back as f32
        let res_f32 = fpu.load::<f32>(2);
        assert_eq!(res_f32.to_bits(), val_f32.to_bits());

        // 3. Test Unboxing with invalid box
        // If upper bits are not all 1s, it should be Canonical NaN.
        let not_boxed = 0x00000000_12345678u64;
        fpu.store::<f64>(3, f64::from_bits(not_boxed));
        assert_eq!(fpu.load_raw(3), 0x12345678);
    }

    #[test]
    fn test_sqrt() {
        let mut fpu = SoftFPU::from(true);

        // sqrt(4.0) = 2.0 (Exact)
        fpu.store::<f32>(1, 4.0);
        fpu.sqrt::<f32>(1, 2, Round::NearestTiesToEven);
        assert_eq!(fpu.load::<f32>(2), 2.0);
        assert_eq!(fpu.last_status(), Status::OK);

        // sqrt(2.0) (Inexact)
        fpu.store::<f32>(1, 2.0);
        fpu.sqrt::<f32>(1, 2, Round::NearestTiesToEven);
        let r = fpu.load::<f32>(2);
        assert!((r - 2.0f32.sqrt()).abs() < 1e-7);
        assert!(fpu.last_status().contains(Status::INEXACT));

        // sqrt(-1.0) (Invalid)
        fpu.store::<f32>(1, -1.0);
        fpu.sqrt::<f32>(1, 2, Round::NearestTiesToEven);
        assert!(fpu.load::<f32>(2).is_nan());
        assert!(fpu.last_status().contains(Status::INVALID_OP));

        // sqrt(-0.0) = -0.0
        fpu.store::<f32>(1, -0.0);
        fpu.sqrt::<f32>(1, 2, Round::NearestTiesToEven);
        let r = fpu.load::<f32>(2);
        assert_eq!(r, -0.0);
        assert!(r.is_sign_negative());
        assert_eq!(fpu.last_status(), Status::OK);

        // sqrt(3.14159265) (Inexact, but tricky in f32)
        fpu.store::<f32>(1, 3.14159265);
        fpu.sqrt::<f32>(1, 2, Round::NearestTiesToEven);
        assert!(fpu.last_status().contains(Status::INEXACT));
    }

    #[test]
    fn test_sqrt_exact_f64() {
        let mut fpu = SoftFPU::from(true);
        fpu.store::<f64>(1, 10000.0);
        fpu.sqrt::<f64>(1, 2, Round::NearestTiesToEven);
        assert_eq!(fpu.load::<f64>(2), 100.0);
        assert_eq!(fpu.last_status(), Status::OK);
    }

    #[test]
    fn test_min_max_signed_zero() {
        let mut fpu = SoftFPU::from(true);

        // fmin(-0.0, 0.0) -> -0.0
        fpu.store::<f32>(1, -0.0);
        fpu.store::<f32>(2, 0.0);
        fpu.min_num::<f32>(1, 2, 3);
        let r = fpu.load::<f32>(3);
        assert_eq!(r, -0.0);
        assert!(r.is_sign_negative());

        // fmin(0.0, -0.0) -> -0.0
        fpu.min_num::<f32>(2, 1, 4);
        let r = fpu.load::<f32>(4);
        assert_eq!(r, -0.0);
        assert!(r.is_sign_negative());

        // fmax(-0.0, 0.0) -> 0.0
        fpu.max_num::<f32>(1, 2, 5);
        let r = fpu.load::<f32>(5);
        assert_eq!(r, 0.0);
        assert!(r.is_sign_positive());

        // fmax(0.0, -0.0) -> 0.0
        fpu.max_num::<f32>(2, 1, 6);
        let r = fpu.load::<f32>(6);
        assert_eq!(r, 0.0);
        assert!(r.is_sign_positive());
    }
}
