#[cfg(test)]
use crate::isa::riscv::vector::{Vector, tester::TestOpParameter};
use crate::{
    config::arch_config::WordType,
    isa::riscv::{
        instruction::exec_function::{
            ExecAdd, ExecAddu, ExecAnd, ExecAndn, ExecDivSigned, ExecDivUnsigned, ExecEqual,
            ExecMax, ExecMaxu, ExecMin, ExecMinu, ExecMove, ExecMulHighSigned,
            ExecMulHighSignedUnsigned, ExecMulHighUnsigned, ExecMulLow, ExecNand, ExecNor,
            ExecNotEqual, ExecOr, ExecOrn, ExecRemSigned, ExecRemUnsigned, ExecRevSub, ExecSLL,
            ExecSRA, ExecSRL, ExecSext, ExecSub, ExecSubu, ExecTrait, ExecUnaryTrait,
            ExecUnsignedLess, ExecXnor, ExecXor, ExecZext,
        },
        trap::Exception,
        vector::{
            VLEN_BYTE, VecOpMask,
            types::{VGFRef, VGFRefMut},
        },
    },
    utils::{TruncateFrom, UnsignedInteger, as_signed_i128},
};

trait ExecTernaryTrait<OUT, IN = WordType> {
    fn exec(a: IN, b: IN, c: bool) -> OUT;
}

trait WideningIntegerBinaryExec<T> {
    fn exec(a: T, b: T) -> Result<T, Exception>;
}

trait IntegerMultiplyAddExec<T> {
    fn exec(vs2: T, vs1: T, vd: T) -> Result<T, Exception>;
}

trait BitBinaryExec {
    fn exec(vs2: bool, vs1: bool) -> Result<bool, Exception>;
}

trait WideningIntegerMultiplyAddExec<Src1, Src2, Dst> {
    fn exec(vs2: Src2, vs1: Src1, vd: Dst) -> Result<Dst, Exception>;
}

trait NarrowingIntegerShiftExec<Src, Shift, Dst> {
    fn exec(src: Src, shift: Shift) -> Dst;
}

struct ExecAdc;
struct ExecSbc;
struct ExecAdcCarryNoCarry;
struct ExecSbcBorrowNoBorrow;
struct ExecAdcCarry;
struct ExecSbcBorrow;
struct ExecWideningAdd;
struct ExecWideningSub;
struct ExecWideningMul;
struct ExecMacc;
struct ExecNmsac;
struct ExecMadd;
struct ExecNmsub;
struct ExecWmacc;
struct ExecNsrl;
struct ExecNsra;

// res<bit> = a<bit> & b<bit>
impl BitBinaryExec for ExecAnd<bool> {
    fn exec(vs2: bool, vs1: bool) -> Result<bool, Exception> {
        <ExecAnd<bool> as ExecTrait<Result<bool, Exception>, bool>>::exec(vs2, vs1)
    }
}

// res<bit> = !(a<bit> & b<bit>)
impl BitBinaryExec for ExecNand<bool> {
    fn exec(vs2: bool, vs1: bool) -> Result<bool, Exception> {
        <ExecNand<bool> as ExecTrait<Result<bool, Exception>, bool>>::exec(vs2, vs1)
    }
}

// res<bit> = a<bit> & !b<bit>
impl BitBinaryExec for ExecAndn<bool> {
    fn exec(vs2: bool, vs1: bool) -> Result<bool, Exception> {
        <ExecAndn<bool> as ExecTrait<Result<bool, Exception>, bool>>::exec(vs2, vs1)
    }
}

// res<bit> = a<bit> ^ b<bit>
impl BitBinaryExec for ExecXor<bool> {
    fn exec(vs2: bool, vs1: bool) -> Result<bool, Exception> {
        <ExecXor<bool> as ExecTrait<Result<bool, Exception>, bool>>::exec(vs2, vs1)
    }
}

// res<bit> = a<bit> | b<bit>
impl BitBinaryExec for ExecOr<bool> {
    fn exec(vs2: bool, vs1: bool) -> Result<bool, Exception> {
        <ExecOr<bool> as ExecTrait<Result<bool, Exception>, bool>>::exec(vs2, vs1)
    }
}

// res<bit> = !(a<bit> | b<bit>)
impl BitBinaryExec for ExecNor<bool> {
    fn exec(vs2: bool, vs1: bool) -> Result<bool, Exception> {
        <ExecNor<bool> as ExecTrait<Result<bool, Exception>, bool>>::exec(vs2, vs1)
    }
}

// res<bit> = a<bit> | !b<bit>
impl BitBinaryExec for ExecOrn<bool> {
    fn exec(vs2: bool, vs1: bool) -> Result<bool, Exception> {
        <ExecOrn<bool> as ExecTrait<Result<bool, Exception>, bool>>::exec(vs2, vs1)
    }
}

// res<bit> = !(a<bit> ^ b<bit>)
impl BitBinaryExec for ExecXnor<bool> {
    fn exec(vs2: bool, vs1: bool) -> Result<bool, Exception> {
        <ExecXnor<bool> as ExecTrait<Result<bool, Exception>, bool>>::exec(vs2, vs1)
    }
}

// res<2*sew> = widen(a) + widen(b)
impl<T> WideningIntegerBinaryExec<T> for ExecWideningAdd
where
    T: num_traits::WrappingAdd,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a.wrapping_add(&b))
    }
}

// res<2*sew> = widen(a) - widen(b)
impl<T> WideningIntegerBinaryExec<T> for ExecWideningSub
where
    T: num_traits::WrappingSub,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a.wrapping_sub(&b))
    }
}

// res<2*sew> = widen(a) * widen(b)
impl<T> WideningIntegerBinaryExec<T> for ExecWideningMul
where
    T: num_traits::WrappingMul,
{
    fn exec(a: T, b: T) -> Result<T, Exception> {
        Ok(a.wrapping_mul(&b))
    }
}

// res<sew> = vs1<sew> * vs2<sew> + vd<sew>
impl<T> IntegerMultiplyAddExec<T> for ExecMacc
where
    T: num_traits::WrappingAdd + num_traits::WrappingMul,
{
    fn exec(vs2: T, vs1: T, vd: T) -> Result<T, Exception> {
        Ok(vs1.wrapping_mul(&vs2).wrapping_add(&vd))
    }
}

// res<sew> = -(vs1<sew> * vs2<sew>) + vd<sew>
impl<T> IntegerMultiplyAddExec<T> for ExecNmsac
where
    T: num_traits::WrappingAdd + num_traits::WrappingMul + num_traits::WrappingNeg,
{
    fn exec(vs2: T, vs1: T, vd: T) -> Result<T, Exception> {
        Ok(vs1.wrapping_mul(&vs2).wrapping_neg().wrapping_add(&vd))
    }
}

// res<sew> = vs1<sew> * vd<sew> + vs2<sew>
impl<T> IntegerMultiplyAddExec<T> for ExecMadd
where
    T: num_traits::WrappingAdd + num_traits::WrappingMul,
{
    fn exec(vs2: T, vs1: T, vd: T) -> Result<T, Exception> {
        Ok(vs1.wrapping_mul(&vd).wrapping_add(&vs2))
    }
}

// res<sew> = -(vs1<sew> * vd<sew>) + vs2<sew>
impl<T> IntegerMultiplyAddExec<T> for ExecNmsub
where
    T: num_traits::WrappingAdd + num_traits::WrappingMul + num_traits::WrappingNeg,
{
    fn exec(vs2: T, vs1: T, vd: T) -> Result<T, Exception> {
        Ok(vs1.wrapping_mul(&vd).wrapping_neg().wrapping_add(&vs2))
    }
}

// res<2*sew> = widen(vs1<sew>) * widen(vs2<sew>) + vd<2*sew>
impl<Src1, Src2, Dst> WideningIntegerMultiplyAddExec<Src1, Src2, Dst> for ExecWmacc
where
    Src1: Copy,
    Src2: Copy,
    Dst: Copy + From<Src1> + From<Src2> + num_traits::WrappingAdd + num_traits::WrappingMul,
{
    fn exec(vs2: Src2, vs1: Src1, vd: Dst) -> Result<Dst, Exception> {
        Ok(Dst::from(vs1)
            .wrapping_mul(&Dst::from(vs2))
            .wrapping_add(&vd))
    }
}

impl<Src, Shift, Dst> NarrowingIntegerShiftExec<Src, Shift, Dst> for ExecNsrl
where
    Src: UnsignedInteger,
    Shift: UnsignedInteger,
    Dst: UnsignedInteger + crate::utils::TruncateFrom<Src>,
{
    fn exec(src: Src, shift: Shift) -> Dst {
        Dst::truncate_from(src >> crate::utils::shift_amount(shift))
    }
}

impl<Src, Shift, Dst> NarrowingIntegerShiftExec<Src, Shift, Dst> for ExecNsra
where
    Src: UnsignedInteger + Into<u128>,
    Shift: UnsignedInteger,
    Dst: UnsignedInteger,
{
    fn exec(src: Src, shift: Shift) -> Dst {
        crate::utils::from_signed_i128(
            crate::utils::as_signed_i128(src) >> crate::utils::shift_amount(shift),
        )
    }
}

// res<sew> = a<sew> + b<sew> + c<bit>
impl<T> ExecTernaryTrait<Result<T, Exception>, T> for ExecAdc
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T, c: bool) -> Result<T, Exception> {
        let a: u128 = a.into();
        let b: u128 = b.into();
        Ok(T::truncate_from(a + b + c as u128))
    }
}

// res<sew> = a<sew> - b<sew> - c<bit>
impl<T> ExecTernaryTrait<Result<T, Exception>, T> for ExecSbc
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T, c: bool) -> Result<T, Exception> {
        let a: u128 = a.into();
        let b: u128 = b.into();
        Ok(T::truncate_from(a.wrapping_sub(b + c as u128)))
    }
}

// v0<bit> = carry_out(a<sew> + b<sew> + c<bit>)
impl<T> ExecTernaryTrait<Result<bool, Exception>, T> for ExecAdcCarry
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T, c: bool) -> Result<bool, Exception> {
        let a: u128 = a.into();
        let b: u128 = b.into();
        Ok((a + b + c as u128) >> T::BITS != 0)
    }
}

// v0<bit> = carry_out(a<sew> + b<sew>)
impl<T> ExecTrait<bool, T> for ExecAdcCarryNoCarry
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> bool {
        let a: u128 = a.into();
        let b: u128 = b.into();
        (a + b) >> T::BITS != 0
    }
}

// v0<bit> = borrow_out(a<sew> - b<sew>)
impl<T> ExecTrait<bool, T> for ExecSbcBorrowNoBorrow
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> bool {
        let a: u128 = a.into();
        let b: u128 = b.into();
        a < b
    }
}

// v0<bit> = borrow_out(a<sew> - b<sew> - c<bit>)
impl<T> ExecTernaryTrait<Result<bool, Exception>, T> for ExecSbcBorrow
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T, c: bool) -> Result<bool, Exception> {
        let a: u128 = a.into();
        let b: u128 = b.into();
        Ok(a < b + c as u128)
    }
}

struct ExecUnsignedLessEqual;
struct ExecSignedLess;
struct ExecSignedLessEqual;
struct ExecUnsignedGreater;
struct ExecSignedGreater;
struct ExecUnsignedGreaterX;
struct ExecSignedGreaterX;

// v0<bit> = a<sew> <= b<sew>
impl<T> ExecTrait<bool, T> for ExecUnsignedLessEqual
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> bool {
        a <= b
    }
}

// v0<bit> = signed(a<sew>) < signed(b<sew>)
impl<T> ExecTrait<bool, T> for ExecSignedLess
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> bool {
        as_signed_i128(a) < as_signed_i128(b)
    }
}

// v0<bit> = signed(a<sew>) <= signed(b<sew>)
impl<T> ExecTrait<bool, T> for ExecSignedLessEqual
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> bool {
        as_signed_i128(a) <= as_signed_i128(b)
    }
}

// v0<bit> = a<sew> > b<sew>
impl<T> ExecTrait<bool, T> for ExecUnsignedGreater
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> bool {
        a > b
    }
}

// v0<bit> = signed(a<sew>) > signed(b<sew>)
impl<T> ExecTrait<bool, T> for ExecSignedGreater
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> bool {
        as_signed_i128(a) > as_signed_i128(b)
    }
}

// v0<bit> = b<sew> > a<sew>
impl<T> ExecTrait<bool, T> for ExecUnsignedGreaterX
where
    T: UnsignedInteger,
{
    fn exec(a: T, b: T) -> bool {
        b > a
    }
}

// v0<bit> = signed(b<sew>) > signed(a<sew>)
impl<T> ExecTrait<bool, T> for ExecSignedGreaterX
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(a: T, b: T) -> bool {
        as_signed_i128(b) > as_signed_i128(a)
    }
}

#[inline(always)]
fn read_mask_bit(mask: &[u8], index: usize) -> bool {
    (mask[index / 8] & (1 << (index % 8))) != 0
}

#[inline(always)]
fn write_mask_bit(mask: &mut [u8], index: usize, value: bool) {
    let byte = &mut mask[index / 8];
    let bit = 1 << (index % 8);
    if value {
        *byte |= bit;
    } else {
        *byte &= !bit;
    }
}

#[macro_export]
macro_rules! dispatch_integer_sew {
    ($sew:expr, |$ty:ident| $body:block) => {
        match $sew {
            1 => {
                type $ty = u8;
                $body
            }
            2 => {
                type $ty = u16;
                $body
            }
            4 => {
                type $ty = u32;
                $body
            }
            8 => {
                type $ty = u64;
                $body
            }
            _ => unreachable!(),
        }
    };
}

// OPIVV
pub(in crate::isa::riscv) trait VectorOpIntegerVV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_vv::<Self>(param.vs1(), param.vs2(), param.vd(), param.enable_mask(), 0)
    }
}

// OPIVV mask-register logical operations. Mask bits are packed into one vector
// register, so these operations treat the register as bit storage rather than
// using the current SEW/LMUL data grouping.
pub(in crate::isa::riscv) trait VectorOpBitVV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_bit_vv::<Self>(param.vs1(), param.vs2(), param.vd(), param.enable_mask(), 0)
    }
}

// OPIVX
pub(in crate::isa::riscv) trait VectorOpIntegerVX {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_vx::<Self>(param.x1(), param.vs2(), param.vd(), param.enable_mask(), 0)
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerVVV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        old_vd: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_vvv::<Self>(
            param.vs1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerVXV {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        old_vd: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_vxv::<Self>(param.x1(), param.vs2(), param.vd(), param.enable_mask(), 0)
    }
}

pub(in crate::isa::riscv) trait VectorOpWideningIntegerVV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_widening_integer_vv::<Self>(
            param.vs1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpWideningIntegerVVV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        old_vd: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_widening_integer_vvv::<Self>(
            param.vs1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpWideningIntegerVXV {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        old_vd: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_widening_integer_vxv::<Self>(
            param.x1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpWideningIntegerVX {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_widening_integer_vx::<Self>(
            param.x1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpWideningIntegerWV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_widening_integer_wv::<Self>(
            param.vs1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpWideningIntegerWX {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_widening_integer_wx::<Self>(
            param.x1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerNarrowingWV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_narrowing_wv::<Self>(
            param.vs1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerNarrowingVX {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_narrowing_vx::<Self>(
            param.x1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

// OPIV(Unary)
pub(in crate::isa::riscv) trait VectorOpIntegerV {
    fn exec(vs2: &VGFRef, vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_v::<Self>(
            param.vs2(),
            param.vd(),
            param.src_eew(),
            param.dst_eew(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerVVM {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        v0: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_vvm::<Self>(
            param.vs1(),
            param.vs2(),
            param.v0(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerVXM {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        v0: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_vxm::<Self>(
            param.x1(),
            param.vs2(),
            param.v0(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerMaskVV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_mask_vv::<Self>(
            param.vs1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerMaskVX {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_mask_vx::<Self>(
            param.x1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerMaskVVM {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        v0: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_mask_vvm::<Self>(
            param.vs1(),
            param.vs2(),
            param.v0(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerMaskVXM {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        v0: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_mask_vxm::<Self>(
            param.x1(),
            param.vs2(),
            param.v0(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpMaskToX {
    fn exec(vs2: &VGFRef, mask: &VecOpMask) -> Result<WordType, Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<WordType, Exception>
    where
        Self: Sized,
    {
        vector.exec_mask_to_x::<Self>(param.vs2(), param.enable_mask(), 0)
    }
}

pub(in crate::isa::riscv) trait VectorOpMaskUnary {
    fn exec(vs2: &VGFRef, vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_mask_unary::<Self>(param.vs2(), param.vd(), param.enable_mask(), 0)
    }
}

pub(in crate::isa::riscv) trait VectorOpMaskToVector {
    fn exec(vs2: &VGFRef, vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_mask_to_vector::<Self>(param.vs2(), param.vd(), param.enable_mask(), 0)
    }
}

pub(in crate::isa::riscv) trait VectorOpIndex {
    fn exec(vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_index::<Self>(param.vd(), param.enable_mask(), 0)
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerGatherVV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_gather_vv::<Self>(
            param.vs1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpIntegerGatherEI16VV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_integer_gather_ei16_vv::<Self>(
            param.vs1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpCompress {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        vl: usize,
        tail_mask: &VecOpMask,
    ) -> Result<(), Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<(), Exception>
    where
        Self: Sized,
    {
        vector.exec_compress::<Self>(param.vs1(), param.vs2(), param.vd(), 0)
    }
}

macro_rules! impl_vector_op_integer_vv_binary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew && vs1.sew == vd.sew);
                let sew = vd.sew;

                dispatch_integer_sew!(sew, |T| {
                    let vs1 = vs1.as_slice::<T>();
                    let vs2 = vs2.as_slice::<T>();
                    for (index, element) in vd.iter_mut().enumerate() {
                        mask.element_load(
                            element,
                            $exec_ty::<T>::exec(vs2[index], vs1[index])?,
                            index,
                        );
                    }
                });

                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_bit_vv_binary {
    ($op_ty:ty, $exec_ty:ty) => {
        impl VectorOpBitVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                debug_assert!(vs1.sew == size_of::<u32>() as u8);
                debug_assert!(vs2.sew == size_of::<u32>() as u8);
                debug_assert!(vd.sew == size_of::<u32>() as u8);

                // The architectural mask register is a bit array. `u32` is only
                // a local chunking type here: it gives aligned 32-bit batches
                // while the inner loop still applies mask/tail/vstart per bit.
                let vs1 = vs1.as_slice::<u32>();
                let vs2 = vs2.as_slice::<u32>();
                let mut result = vd.as_slice::<u32>().to_vec();
                debug_assert!(vs1.len() == vs2.len() && vs2.len() == result.len());

                let bits_per_chunk = u32::BITS as usize;
                let bit_capacity = result.len() * bits_per_chunk;
                let bit_start = mask.first_pending_index().min(bit_capacity);
                let bit_end = mask.write_end(bit_capacity);
                for index in bit_start..bit_end {
                    let chunk_index = index / bits_per_chunk;
                    let bit = 1u32 << (index % bits_per_chunk);
                    let vs1_bit = (vs1[chunk_index] & bit) != 0;
                    let vs2_bit = (vs2[chunk_index] & bit) != 0;
                    let value = <$exec_ty as BitBinaryExec>::exec(vs2_bit, vs1_bit)?;

                    let Some(value) = mask.mask_value(value, index) else {
                        continue;
                    };
                    if value {
                        result[chunk_index] |= bit;
                    } else {
                        result[chunk_index] &= !bit;
                    }
                }
                vd.as_mut_slice::<u32>().copy_from_slice(&result);

                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_integer_vx_binary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerVX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs2.sew == vd.sew);
                let sew = vd.sew;

                dispatch_integer_sew!(sew, |T| {
                    let scalar = x1 as T;
                    let vs2 = vs2.as_slice::<T>();
                    for (index, element) in vd.iter_mut().enumerate() {
                        mask.element_load(element, $exec_ty::<T>::exec(vs2[index], scalar)?, index);
                    }
                });

                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_integer_vvv_ternary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerVVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                old_vd: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew && vs1.sew == old_vd.sew && old_vd.sew == vd.sew);
                dispatch_integer_sew!(vd.sew, |T| {
                    let vs1 = vs1.as_slice::<T>();
                    let vs2 = vs2.as_slice::<T>();
                    let old_vd = old_vd.as_slice::<T>();
                    for (index, element) in vd.iter_mut().enumerate() {
                        mask.element_load(
                            element,
                            $exec_ty::exec(vs2[index], vs1[index], old_vd[index])?,
                            index,
                        );
                    }
                });
                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_integer_vxv_ternary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerVXV for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                old_vd: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs2.sew == old_vd.sew && old_vd.sew == vd.sew);
                dispatch_integer_sew!(vd.sew, |T| {
                    let scalar = x1 as T;
                    let vs2 = vs2.as_slice::<T>();
                    let old_vd = old_vd.as_slice::<T>();
                    for (index, element) in vd.iter_mut().enumerate() {
                        mask.element_load(
                            element,
                            $exec_ty::exec(vs2[index], scalar, old_vd[index])?,
                            index,
                        );
                    }
                });
                Ok(())
            }
        }
    };
}

fn vector_op_widening_vv_binary<Src1, Src2, Dst, Exec>(
    vs1: &VGFRef,
    vs2: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    Src1: Copy,
    Src2: Copy,
    Dst: Copy + Default + From<Src1> + From<Src2>,
    Exec: WideningIntegerBinaryExec<Dst>,
{
    let vs1 = vs1.as_slice::<Src1>();
    let vs2 = vs2.as_slice::<Src2>();
    for (index, element) in vd.iter_mut().enumerate() {
        mask.element_load(
            element,
            Exec::exec(Dst::from(vs2[index]), Dst::from(vs1[index]))?,
            index,
        );
    }
    Ok(())
}

fn vector_op_widening_vvv_ternary<Src1, Src2, Dst, Exec>(
    vs1: &VGFRef,
    vs2: &VGFRef,
    old_vd: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    Src1: Copy,
    Src2: Copy,
    Dst: Copy + Default,
    Exec: WideningIntegerMultiplyAddExec<Src1, Src2, Dst>,
{
    let vs1 = vs1.as_slice::<Src1>();
    let vs2 = vs2.as_slice::<Src2>();
    let old_vd = old_vd.as_slice::<Dst>();
    for (index, element) in vd.iter_mut().enumerate() {
        mask.element_load(
            element,
            Exec::exec(vs2[index], vs1[index], old_vd[index])?,
            index,
        );
    }
    Ok(())
}

fn vector_op_widening_vxv_ternary<Scalar, Src2, Dst, Exec>(
    scalar: Scalar,
    vs2: &VGFRef,
    old_vd: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    Scalar: Copy,
    Src2: Copy,
    Dst: Copy + Default,
    Exec: WideningIntegerMultiplyAddExec<Scalar, Src2, Dst>,
{
    let vs2 = vs2.as_slice::<Src2>();
    let old_vd = old_vd.as_slice::<Dst>();
    for (index, element) in vd.iter_mut().enumerate() {
        mask.element_load(
            element,
            Exec::exec(vs2[index], scalar, old_vd[index])?,
            index,
        );
    }
    Ok(())
}

fn vector_op_widening_vx_binary<Scalar, Src2, Dst, Exec>(
    scalar: Scalar,
    vs2: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    Scalar: Copy,
    Src2: Copy,
    Dst: Copy + Default + From<Scalar> + From<Src2>,
    Exec: WideningIntegerBinaryExec<Dst>,
{
    let scalar = Dst::from(scalar);
    let vs2 = vs2.as_slice::<Src2>();
    for (index, element) in vd.iter_mut().enumerate() {
        mask.element_load(element, Exec::exec(Dst::from(vs2[index]), scalar)?, index);
    }
    Ok(())
}

fn vector_op_widening_wv_binary<Src1, Dst, Exec>(
    vs1: &VGFRef,
    vs2: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    Src1: Copy,
    Dst: Copy + Default + From<Src1>,
    Exec: WideningIntegerBinaryExec<Dst>,
{
    let vs1 = vs1.as_slice::<Src1>();
    let vs2 = vs2.as_slice::<Dst>();
    for (index, element) in vd.iter_mut().enumerate() {
        mask.element_load(
            element,
            Exec::exec(vs2[index], Dst::from(vs1[index]))?,
            index,
        );
    }
    Ok(())
}

fn vector_op_widening_wx_binary<Scalar, Dst, Exec>(
    scalar: Scalar,
    vs2: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    Scalar: Copy,
    Dst: Copy + Default + From<Scalar>,
    Exec: WideningIntegerBinaryExec<Dst>,
{
    let scalar = Dst::from(scalar);
    let vs2 = vs2.as_slice::<Dst>();
    for (index, element) in vd.iter_mut().enumerate() {
        mask.element_load(element, Exec::exec(vs2[index], scalar)?, index);
    }
    Ok(())
}

// (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]]
fn vector_gather_value<T>(vs2: &[T], vs1_as_index: u128) -> T
where
    T: Copy + Default,
{
    // vs2 size is VLMAX(VLEN * LMUL / SEW). If vs2.get() return `None`, means vs1[i] >= VLMAX
    vs2.get(vs1_as_index as usize).copied().unwrap_or_default()
}

fn vector_op_gather_vv<Idx, T>(
    vs1: &VGFRef,
    vs2: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    Idx: Copy + Into<u128>,
    T: Copy + Default,
{
    let vs1 = vs1.as_slice::<Idx>();
    let vs2 = vs2.as_slice::<T>();
    for (index, element) in vd.iter_mut().enumerate() {
        mask.element_load(element, vector_gather_value(vs2, vs1[index].into()), index);
    }
    Ok(())
}

fn vector_op_gather_vx<T>(
    x1: WordType,
    vs2: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    T: Copy + Default,
{
    let vs2 = vs2.as_slice::<T>();
    let value = vector_gather_value(vs2, x1 as u128);
    for (index, element) in vd.iter_mut().enumerate() {
        mask.element_load(element, value, index);
    }
    Ok(())
}

fn vector_op_slideup<T>(
    offset: usize,
    vs2: &VGFRef,
    _old_vd: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    T: Copy + Default,
{
    let vs2 = vs2.as_slice::<T>();
    for (index, element) in vd.iter_mut().enumerate() {
        if index < offset {
            continue;
        }
        let value = vs2.get(index - offset).copied().unwrap_or_default();
        mask.element_load(element, value, index);
    }
    Ok(())
}

fn vector_op_slidedown<T>(
    offset: usize,
    vs2: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    T: Copy + Default,
{
    let vs2 = vs2.as_slice::<T>();
    for (index, element) in vd.iter_mut().enumerate() {
        let value = vs2.get(index + offset).copied().unwrap_or_default();
        mask.element_load(element, value, index);
    }
    Ok(())
}

fn vector_op_slide1up<T>(
    x1: WordType,
    vs2: &VGFRef,
    _old_vd: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    T: Copy + Default + TruncateFrom<WordType>,
{
    let vs2 = vs2.as_slice::<T>();
    for (index, element) in vd.iter_mut().enumerate() {
        let value = if index == 0 {
            T::truncate_from(x1)
        } else {
            vs2.get(index - 1).copied().unwrap_or_default()
        };
        mask.element_load(element, value, index);
    }
    Ok(())
}

fn vector_op_slide1down<T>(
    x1: WordType,
    vs2: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    T: Copy + Default + TruncateFrom<WordType>,
{
    let vs2 = vs2.as_slice::<T>();
    for (index, element) in vd.iter_mut().enumerate() {
        let value = if mask.is_body_index(index) && !mask.is_body_index(index + 1) {
            // is the end element
            T::truncate_from(x1)
        } else {
            vs2.get(index + 1).copied().unwrap_or_default()
        };
        mask.element_load(element, value, index);
    }
    Ok(())
}

fn vector_op_merge_vvm<T>(
    vs1: &VGFRef,
    vs2: &VGFRef,
    v0: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    T: Copy + Default,
{
    let merge_mask = v0.as_slice::<u8>();
    let vs1 = vs1.as_slice::<T>();
    let vs2 = vs2.as_slice::<T>();
    let len = vd.as_slice::<T>().len();
    debug_assert_eq!(vs1.len(), len);
    debug_assert_eq!(vs2.len(), len);
    for (index, element) in vd.iter_mut().enumerate() {
        let value = if read_mask_bit(merge_mask, index) {
            vs1[index]
        } else {
            vs2[index]
        };
        mask.element_load(element, value, index);
    }
    Ok(())
}

fn vector_op_merge_vxm<T>(
    scalar: T,
    vs2: &VGFRef,
    v0: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
) -> Result<(), Exception>
where
    T: Copy + Default,
{
    let merge_mask = v0.as_slice::<u8>();
    let vs2 = vs2.as_slice::<T>();
    let len = vd.as_slice::<T>().len();
    debug_assert_eq!(vs2.len(), len);
    for (index, element) in vd.iter_mut().enumerate() {
        let value = if read_mask_bit(merge_mask, index) {
            scalar
        } else {
            vs2[index]
        };
        mask.element_load(element, value, index);
    }
    Ok(())
}

fn vector_op_mask_popcount(vs2: &VGFRef, mask: &VecOpMask) -> WordType {
    let source_mask = vs2.as_slice::<u8>();
    let mut count = 0;
    for index in 0..VLEN_BYTE * 8 {
        if mask.should_access(index) && read_mask_bit(source_mask, index) {
            count += 1;
        }
    }
    count
}

fn vector_op_mask_first(vs2: &VGFRef, mask: &VecOpMask) -> WordType {
    let source_mask = vs2.as_slice::<u8>();
    for index in 0..VLEN_BYTE * 8 {
        if mask.should_access(index) && read_mask_bit(source_mask, index) {
            return index as WordType;
        }
    }
    WordType::MAX
}

enum MaskBeforeFirstMode {
    Before,
    Including,
    Only,
}

fn vector_op_mask_before_first(
    vs2: &VGFRef,
    vd: &mut VGFRefMut,
    mask: &VecOpMask,
    mode: MaskBeforeFirstMode,
) -> Result<(), Exception> {
    let source_mask = vs2.as_slice::<u8>();
    let dest_mask = vd.as_mut_slice::<u8>();
    let bit_capacity = dest_mask.len() * u8::BITS as usize;
    let mut seen_first = (0..mask.first_pending_index().min(bit_capacity))
        .any(|index| mask.should_access_ignoring_start(index) && read_mask_bit(source_mask, index));

    for index in mask.first_pending_index().min(bit_capacity)..mask.write_end(bit_capacity) {
        let active_source_bit =
            mask.should_access_ignoring_start(index) && read_mask_bit(source_mask, index);
        let value = match mode {
            MaskBeforeFirstMode::Before => !seen_first && !active_source_bit,
            MaskBeforeFirstMode::Including => !seen_first,
            MaskBeforeFirstMode::Only => !seen_first && active_source_bit,
        };
        mask.mask_bit_load(dest_mask, index, value);
        if active_source_bit {
            seen_first = true;
        }
    }

    Ok(())
}

fn vector_op_iota<T>(vs2: &VGFRef, vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception>
where
    T: UnsignedInteger + Default + TruncateFrom<u128>,
{
    let source_mask = vs2.as_slice::<u8>();
    let mut count = 0u128;
    for (index, element) in vd.iter_mut().enumerate() {
        mask.element_load(element, T::truncate_from(count), index);
        if mask.is_body_index(index) && read_mask_bit(source_mask, index) {
            count += 1;
        }
    }
    Ok(())
}

fn vector_op_index<T>(vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception>
where
    T: UnsignedInteger + Default + TruncateFrom<u128>,
{
    for (index, element) in vd.iter_mut().enumerate() {
        mask.element_load(element, T::truncate_from(index as u128), index);
    }
    Ok(())
}

fn vector_op_compress<T>(
    vs1: &VGFRef,
    vs2: &VGFRef,
    vd: &mut VGFRefMut,
    vl: usize,
    tail_mask: &VecOpMask,
) -> Result<(), Exception>
where
    T: Copy + Default,
{
    let select_mask = vs1.as_slice::<u8>();
    let vs2 = vs2.as_slice::<T>();
    let mut dest_index = 0;
    let mut dest_iter = vd.iter_mut();

    for source_index in 0..vl {
        if read_mask_bit(select_mask, source_index) {
            let element = dest_iter.next().unwrap();
            tail_mask.element_load(element, vs2[source_index], dest_index);
            dest_index += 1;
        }
    }

    for (index, element) in dest_iter.enumerate() {
        let index = index + dest_index;
        tail_mask.element_load_default::<T>(element, index);
    }

    Ok(())
}

// =================================================
//         Widening Integer Operations
// =================================================
macro_rules! dispatch_widening_integer_sew {
    ($src_sew:expr, |$i_src:ident, $i_dst:ident, $u_src:ident, $u_dst:ident| $body:block) => {
        match $src_sew {
            1 => {
                type $i_src = i8;
                type $i_dst = i16;
                type $u_src = u8;
                type $u_dst = u16;
                $body
            }
            2 => {
                type $i_src = i16;
                type $i_dst = i32;
                type $u_src = u16;
                type $u_dst = u32;
                $body
            }
            4 => {
                type $i_src = i32;
                type $i_dst = i64;
                type $u_src = u32;
                type $u_dst = u64;
                $body
            }
            _ => unreachable!(),
        }
    };
}

macro_rules! dispatch_narrowing_integer_sew {
    ($dst_sew:expr, |$src_ty:ident, $dst_ty:ident| $body:block) => {
        match $dst_sew {
            1 => {
                type $src_ty = u16;
                type $dst_ty = u8;
                $body
            }
            2 => {
                type $src_ty = u32;
                type $dst_ty = u16;
                $body
            }
            4 => {
                type $src_ty = u64;
                type $dst_ty = u32;
                $body
            }
            _ => unreachable!(),
        }
    };
}

macro_rules! impl_vector_op_widening_interger_vv_binary {
    ($op_ty:ty, $exec_ty:ident, signed) => {
        impl VectorOpWideningIntegerVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew && vd.sew == vs2.sew * 2);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vv_binary::<ISrc, ISrc, IDst, $exec_ty>(vs1, vs2, vd, mask)
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, unsigned) => {
        impl VectorOpWideningIntegerVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew && vd.sew == vs2.sew * 2);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vv_binary::<USrc, USrc, UDst, $exec_ty>(vs1, vs2, vd, mask)
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, signed_unsigned) => {
        impl VectorOpWideningIntegerVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew && vd.sew == vs2.sew * 2);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vv_binary::<USrc, ISrc, IDst, $exec_ty>(vs1, vs2, vd, mask)
                })
            }
        }
    };
}

macro_rules! impl_vector_op_widening_interger_vx_binary {
    ($op_ty:ty, $exec_ty:ident, signed) => {
        impl VectorOpWideningIntegerVX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vd.sew == vs2.sew * 2);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vx_binary::<ISrc, ISrc, IDst, $exec_ty>(
                        x1 as ISrc, vs2, vd, mask,
                    )
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, unsigned) => {
        impl VectorOpWideningIntegerVX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vd.sew == vs2.sew * 2);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vx_binary::<USrc, USrc, UDst, $exec_ty>(
                        x1 as USrc, vs2, vd, mask,
                    )
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, signed_unsigned) => {
        impl VectorOpWideningIntegerVX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vd.sew == vs2.sew * 2);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vx_binary::<USrc, ISrc, IDst, $exec_ty>(
                        x1 as USrc, vs2, vd, mask,
                    )
                })
            }
        }
    };
}

macro_rules! impl_vector_op_widening_integer_vvv_ternary {
    ($op_ty:ty, $exec_ty:ident, signed) => {
        impl VectorOpWideningIntegerVVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                old_vd: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew && vd.sew == vs2.sew * 2 && old_vd.sew == vd.sew);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vvv_ternary::<ISrc, ISrc, IDst, $exec_ty>(
                        vs1, vs2, old_vd, vd, mask,
                    )
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, unsigned) => {
        impl VectorOpWideningIntegerVVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                old_vd: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew && vd.sew == vs2.sew * 2 && old_vd.sew == vd.sew);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vvv_ternary::<USrc, USrc, UDst, $exec_ty>(
                        vs1, vs2, old_vd, vd, mask,
                    )
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, signed_unsigned) => {
        impl VectorOpWideningIntegerVVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                old_vd: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew && vd.sew == vs2.sew * 2 && old_vd.sew == vd.sew);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vvv_ternary::<USrc, ISrc, IDst, $exec_ty>(
                        vs1, vs2, old_vd, vd, mask,
                    )
                })
            }
        }
    };
}

macro_rules! impl_vector_op_widening_integer_vxv_ternary {
    ($op_ty:ty, $exec_ty:ident, signed) => {
        impl VectorOpWideningIntegerVXV for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                old_vd: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vd.sew == vs2.sew * 2 && old_vd.sew == vd.sew);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vxv_ternary::<ISrc, ISrc, IDst, $exec_ty>(
                        x1 as ISrc, vs2, old_vd, vd, mask,
                    )
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, unsigned) => {
        impl VectorOpWideningIntegerVXV for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                old_vd: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vd.sew == vs2.sew * 2 && old_vd.sew == vd.sew);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vxv_ternary::<USrc, USrc, UDst, $exec_ty>(
                        x1 as USrc, vs2, old_vd, vd, mask,
                    )
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, signed_unsigned) => {
        impl VectorOpWideningIntegerVXV for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                old_vd: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vd.sew == vs2.sew * 2 && old_vd.sew == vd.sew);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vxv_ternary::<USrc, ISrc, IDst, $exec_ty>(
                        x1 as USrc, vs2, old_vd, vd, mask,
                    )
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, unsigned_signed) => {
        impl VectorOpWideningIntegerVXV for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                old_vd: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vd.sew == vs2.sew * 2 && old_vd.sew == vd.sew);
                dispatch_widening_integer_sew!(vs2.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_vxv_ternary::<ISrc, USrc, IDst, $exec_ty>(
                        x1 as ISrc, vs2, old_vd, vd, mask,
                    )
                })
            }
        }
    };
}

macro_rules! impl_vector_op_widening_interger_wv_binary {
    ($op_ty:ty, $exec_ty:ident, signed) => {
        impl VectorOpWideningIntegerWV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew * 2 == vs2.sew && vs2.sew == vd.sew);
                dispatch_widening_integer_sew!(vs1.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_wv_binary::<ISrc, IDst, $exec_ty>(vs1, vs2, vd, mask)
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, unsigned) => {
        impl VectorOpWideningIntegerWV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew * 2 == vs2.sew && vs2.sew == vd.sew);
                dispatch_widening_integer_sew!(vs1.sew, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_wv_binary::<USrc, UDst, $exec_ty>(vs1, vs2, vd, mask)
                })
            }
        }
    };
}

macro_rules! impl_vector_op_widening_interger_wx_binary {
    ($op_ty:ty, $exec_ty:ident, signed) => {
        impl VectorOpWideningIntegerWX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs2.sew == vd.sew);
                dispatch_widening_integer_sew!(vd.sew / 2, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_wx_binary::<ISrc, IDst, $exec_ty>(x1 as ISrc, vs2, vd, mask)
                })
            }
        }
    };
    ($op_ty:ty, $exec_ty:ident, unsigned) => {
        impl VectorOpWideningIntegerWX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs2.sew == vd.sew);
                dispatch_widening_integer_sew!(vd.sew / 2, |ISrc, IDst, USrc, UDst| {
                    vector_op_widening_wx_binary::<USrc, UDst, $exec_ty>(x1 as USrc, vs2, vd, mask)
                })
            }
        }
    };
}

macro_rules! impl_vector_op_integer_narrowing_wv_binary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerNarrowingWV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vd.sew && vs2.sew == vd.sew * 2);
                dispatch_narrowing_integer_sew!(vd.sew, |TSrc, TDst| {
                    let vs1 = vs1.as_slice::<TDst>();
                    let vs2 = vs2.as_slice::<TSrc>();
                    let len = vd.as_slice::<TDst>().len();
                    debug_assert_eq!(vs1.len(), len);
                    debug_assert!(vs2.len() <= len);
                    for (index, element) in vd.iter_mut().take(vs2.len()).enumerate() {
                        mask.element_load(
                            element,
                            <$exec_ty as NarrowingIntegerShiftExec<TSrc, TDst, TDst>>::exec(
                                vs2[index], vs1[index],
                            ),
                            index,
                        );
                    }
                });
                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_integer_narrowing_vx_binary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerNarrowingVX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs2.sew == vd.sew * 2);
                dispatch_narrowing_integer_sew!(vd.sew, |TSrc, TDst| {
                    let vs2 = vs2.as_slice::<TSrc>();
                    let shift = TDst::truncate_from(x1);
                    let len = vd.as_slice::<TDst>().len();
                    debug_assert!(vs2.len() <= len);
                    for (index, element) in vd.iter_mut().take(vs2.len()).enumerate() {
                        mask.element_load(
                            element,
                            <$exec_ty as NarrowingIntegerShiftExec<TSrc, TDst, TDst>>::exec(
                                vs2[index], shift,
                            ),
                            index,
                        );
                    }
                });
                Ok(())
            }
        }
    };
}

/// Generate a `VectorOpIntegerV` implementation for unary extension operations (e.g., `vzext.vf2`, `vsext.vf4`).
///
/// This macro dispatches on `(src_sew, dst_sew)` pairs to call the appropriate type-specific executor,
/// converting source vector elements (`$src_ty`) to wider destination elements (`$dst_ty`).
///
/// # Parameters
/// - `$op_ty`: The marker type for the vector operation (e.g., `VectorOpZextVf2`).
/// - `$exec_ty`: The executor struct implementing `ExecUnaryTrait` (e.g., `ExecZext`, `ExecSext`).
/// - `$factor`: The extension factor (`dst_sew / src_sew`, e.g., `2` for vf2).
/// - `[...]`: A list of `(src_sew, dst_sew, src_ty, dst_ty)` tuples defining valid type pairs.
macro_rules! impl_vector_op_integer_v_unary_ext {
    ($op_ty:ty, $exec_ty:ident, $factor:literal, [$(($src_sew:literal, $dst_sew:literal, $src_ty:ty, $dst_ty:ty)),+ $(,)?]) => {
        impl VectorOpIntegerV for $op_ty {
            fn exec(vs2: &VGFRef, vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception> {
                let src_sew = vs2.sew;
                let dst_sew = vd.sew;
                assert!(dst_sew == src_sew * $factor);

                match (src_sew, dst_sew) {
                    $(
                        ($src_sew, $dst_sew) => {
                            let vs2 = vs2.as_slice::<$src_ty>();
                            for (index, element) in vd.iter_mut().enumerate() {
                                mask.element_load(
                                    element,
                                    $exec_ty::<$dst_ty, $src_ty>::exec(vs2[index])?,
                                    index,
                                );
                            }
                        }
                    )+
                    _ => unreachable!(),
                }

                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_integer_vvm_binary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerVVM for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                v0: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew && vs1.sew == vd.sew);
                let carry = v0.as_slice::<u8>();
                dispatch_integer_sew!(vd.sew, |T| {
                    let vs1 = vs1.as_slice::<T>();
                    let vs2 = vs2.as_slice::<T>();
                    for (index, element) in vd.iter_mut().enumerate() {
                        mask.element_load(
                            element,
                            $exec_ty::exec(vs2[index], vs1[index], read_mask_bit(carry, index))?,
                            index,
                        );
                    }
                });
                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_integer_vxm_binary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerVXM for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                v0: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs2.sew == vd.sew);
                let carry = v0.as_slice::<u8>();
                dispatch_integer_sew!(vd.sew, |T| {
                    let scalar = x1 as T;
                    let vs2 = vs2.as_slice::<T>();
                    for (index, element) in vd.iter_mut().enumerate() {
                        mask.element_load(
                            element,
                            $exec_ty::exec(vs2[index], scalar, read_mask_bit(carry, index))?,
                            index,
                        );
                    }
                });
                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_integer_mask_vv_binary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerMaskVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                op_mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew);
                let mask = vd.as_mut_slice::<u8>();
                dispatch_integer_sew!(vs2.sew, |T| {
                    let vs1 = vs1.as_slice::<T>();
                    let vs2 = vs2.as_slice::<T>();
                    for index in 0..vs2.len() {
                        op_mask.mask_bit_load(mask, index, $exec_ty::exec(vs2[index], vs1[index]));
                    }
                });
                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_integer_mask_vx_binary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerMaskVX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                op_mask: &VecOpMask,
            ) -> Result<(), Exception> {
                let mask = vd.as_mut_slice::<u8>();
                dispatch_integer_sew!(vs2.sew, |T| {
                    let scalar = x1 as T;
                    let vs2 = vs2.as_slice::<T>();
                    for index in 0..vs2.len() {
                        op_mask.mask_bit_load(mask, index, $exec_ty::exec(vs2[index], scalar));
                    }
                });
                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_integer_mask_vvm_binary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerMaskVVM for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                v0: &VGFRef,
                vd: &mut VGFRefMut,
                op_mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew);
                let carry = v0.as_slice::<u8>();
                let mask = vd.as_mut_slice::<u8>();
                dispatch_integer_sew!(vs2.sew, |T| {
                    let vs1 = vs1.as_slice::<T>();
                    let vs2 = vs2.as_slice::<T>();
                    for index in 0..vs2.len() {
                        op_mask.mask_bit_load(
                            mask,
                            index,
                            $exec_ty::exec(vs2[index], vs1[index], read_mask_bit(carry, index))?,
                        );
                    }
                });
                Ok(())
            }
        }
    };
}

macro_rules! impl_vector_op_integer_mask_vxm_binary {
    ($op_ty:ty, $exec_ty:ident) => {
        impl VectorOpIntegerMaskVXM for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                v0: &VGFRef,
                vd: &mut VGFRefMut,
                op_mask: &VecOpMask,
            ) -> Result<(), Exception> {
                let carry = v0.as_slice::<u8>();
                let mask = vd.as_mut_slice::<u8>();
                dispatch_integer_sew!(vs2.sew, |T| {
                    let scalar = x1 as T;
                    let vs2 = vs2.as_slice::<T>();
                    for index in 0..vs2.len() {
                        op_mask.mask_bit_load(
                            mask,
                            index,
                            $exec_ty::exec(vs2[index], scalar, read_mask_bit(carry, index))?,
                        );
                    }
                });
                Ok(())
            }
        }
    };
}

pub(in crate::isa::riscv) struct VectorOpAdd;
pub(in crate::isa::riscv) struct VectorOpAddu;
pub(in crate::isa::riscv) struct VectorOpSub;
pub(in crate::isa::riscv) struct VectorOpSubu;
pub(in crate::isa::riscv) struct VectorOpRevSub;
pub(in crate::isa::riscv) struct VectorOpAnd;
pub(in crate::isa::riscv) struct VectorOpNand;
pub(in crate::isa::riscv) struct VectorOpAndn;
pub(in crate::isa::riscv) struct VectorOpOr;
pub(in crate::isa::riscv) struct VectorOpNor;
pub(in crate::isa::riscv) struct VectorOpOrn;
pub(in crate::isa::riscv) struct VectorOpXor;
pub(in crate::isa::riscv) struct VectorOpXnor;
pub(in crate::isa::riscv) struct VectorOpSll;
pub(in crate::isa::riscv) struct VectorOpSrl;
pub(in crate::isa::riscv) struct VectorOpSra;
pub(in crate::isa::riscv) struct VectorOpAdc;
pub(in crate::isa::riscv) struct VectorOpSbc;
pub(in crate::isa::riscv) struct VectorOpMadc;
pub(in crate::isa::riscv) struct VectorOpMsbc;
pub(in crate::isa::riscv) struct VectorOpMseq;
pub(in crate::isa::riscv) struct VectorOpMsne;
pub(in crate::isa::riscv) struct VectorOpMsltu;
pub(in crate::isa::riscv) struct VectorOpMslt;
pub(in crate::isa::riscv) struct VectorOpMsleu;
pub(in crate::isa::riscv) struct VectorOpMsle;
pub(in crate::isa::riscv) struct VectorOpMsgtu;
pub(in crate::isa::riscv) struct VectorOpMsgt;
pub(in crate::isa::riscv) struct VectorOpMax;
pub(in crate::isa::riscv) struct VectorOpMaxu;
pub(in crate::isa::riscv) struct VectorOpMin;
pub(in crate::isa::riscv) struct VectorOpMinu;
pub(in crate::isa::riscv) struct VectorOpMul;
pub(in crate::isa::riscv) struct VectorOpMulh;
pub(in crate::isa::riscv) struct VectorOpMulhu;
pub(in crate::isa::riscv) struct VectorOpMulhsu;
pub(in crate::isa::riscv) struct VectorOpMacc;
pub(in crate::isa::riscv) struct VectorOpNmsac;
pub(in crate::isa::riscv) struct VectorOpMadd;
pub(in crate::isa::riscv) struct VectorOpNmsub;
pub(in crate::isa::riscv) struct VectorOpWadd;
pub(in crate::isa::riscv) struct VectorOpWaddu;
pub(in crate::isa::riscv) struct VectorOpWsub;
pub(in crate::isa::riscv) struct VectorOpWsubu;
pub(in crate::isa::riscv) struct VectorOpWmul;
pub(in crate::isa::riscv) struct VectorOpWmulu;
pub(in crate::isa::riscv) struct VectorOpWmulsu;
pub(in crate::isa::riscv) struct VectorOpWmacc;
pub(in crate::isa::riscv) struct VectorOpNsrl;
pub(in crate::isa::riscv) struct VectorOpNsra;
pub(in crate::isa::riscv) struct VectorOpWmaccu;
pub(in crate::isa::riscv) struct VectorOpWmaccsu;
pub(in crate::isa::riscv) struct VectorOpWmaccus;
pub(in crate::isa::riscv) struct VectorOpDiv;
pub(in crate::isa::riscv) struct VectorOpDivu;
pub(in crate::isa::riscv) struct VectorOpRem;
pub(in crate::isa::riscv) struct VectorOpRemu;
pub(in crate::isa::riscv) struct VectorOpZextVf2;
pub(in crate::isa::riscv) struct VectorOpZextVf4;
pub(in crate::isa::riscv) struct VectorOpZextVf8;
pub(in crate::isa::riscv) struct VectorOpSextVf2;
pub(in crate::isa::riscv) struct VectorOpSextVf4;
pub(in crate::isa::riscv) struct VectorOpSextVf8;
pub(in crate::isa::riscv) struct VectorOpRGatherVV;
pub(in crate::isa::riscv) struct VectorOpRGatherVX;
pub(in crate::isa::riscv) struct VectorOpRGatherVI;
pub(in crate::isa::riscv) struct VectorOpRGatherEI16VV;
pub(in crate::isa::riscv) struct VectorOpSlideUp;
pub(in crate::isa::riscv) struct VectorOpSlideDown;
pub(in crate::isa::riscv) struct VectorOpSlide1Up;
pub(in crate::isa::riscv) struct VectorOpSlide1Down;
pub(in crate::isa::riscv) struct VectorOpMerge;
pub(in crate::isa::riscv) struct VectorOpCpopM;
pub(in crate::isa::riscv) struct VectorOpFirstM;
pub(in crate::isa::riscv) struct VectorOpMsbfM;
pub(in crate::isa::riscv) struct VectorOpMsifM;
pub(in crate::isa::riscv) struct VectorOpMsofM;
pub(in crate::isa::riscv) struct VectorOpIotaM;
pub(in crate::isa::riscv) struct VectorOpIdV;
pub(in crate::isa::riscv) struct VectorOpCompressVm;

impl_vector_op_integer_vv_binary!(VectorOpAdd, ExecAdd);
impl_vector_op_integer_vv_binary!(VectorOpAddu, ExecAddu);
impl_vector_op_integer_vv_binary!(VectorOpSub, ExecSub);
impl_vector_op_integer_vv_binary!(VectorOpSubu, ExecSubu);
impl_vector_op_integer_vv_binary!(VectorOpAnd, ExecAnd);
impl_vector_op_integer_vv_binary!(VectorOpOr, ExecOr);
impl_vector_op_integer_vv_binary!(VectorOpXor, ExecXor);
impl_vector_op_integer_vv_binary!(VectorOpSll, ExecSLL);
impl_vector_op_integer_vv_binary!(VectorOpSrl, ExecSRL);
impl_vector_op_integer_vv_binary!(VectorOpSra, ExecSRA);
impl_vector_op_integer_vv_binary!(VectorOpMax, ExecMax);
impl_vector_op_integer_vv_binary!(VectorOpMaxu, ExecMaxu);
impl_vector_op_integer_vv_binary!(VectorOpMin, ExecMin);
impl_vector_op_integer_vv_binary!(VectorOpMinu, ExecMinu);
impl_vector_op_integer_vv_binary!(VectorOpMul, ExecMulLow);
impl_vector_op_integer_vv_binary!(VectorOpMulh, ExecMulHighSigned);
impl_vector_op_integer_vv_binary!(VectorOpMulhu, ExecMulHighUnsigned);
impl_vector_op_integer_vv_binary!(VectorOpMulhsu, ExecMulHighSignedUnsigned);
impl_vector_op_integer_vv_binary!(VectorOpDiv, ExecDivSigned);
impl_vector_op_integer_vv_binary!(VectorOpDivu, ExecDivUnsigned);
impl_vector_op_integer_vv_binary!(VectorOpRem, ExecRemSigned);
impl_vector_op_integer_vv_binary!(VectorOpRemu, ExecRemUnsigned);
impl_vector_op_bit_vv_binary!(VectorOpAnd, ExecAnd<bool>);
impl_vector_op_bit_vv_binary!(VectorOpNand, ExecNand<bool>);
impl_vector_op_bit_vv_binary!(VectorOpAndn, ExecAndn<bool>);
impl_vector_op_bit_vv_binary!(VectorOpXor, ExecXor<bool>);
impl_vector_op_bit_vv_binary!(VectorOpOr, ExecOr<bool>);
impl_vector_op_bit_vv_binary!(VectorOpNor, ExecNor<bool>);
impl_vector_op_bit_vv_binary!(VectorOpOrn, ExecOrn<bool>);
impl_vector_op_bit_vv_binary!(VectorOpXnor, ExecXnor<bool>);
impl_vector_op_integer_vvv_ternary!(VectorOpMacc, ExecMacc);
impl_vector_op_integer_vvv_ternary!(VectorOpNmsac, ExecNmsac);
impl_vector_op_integer_vvv_ternary!(VectorOpMadd, ExecMadd);
impl_vector_op_integer_vvv_ternary!(VectorOpNmsub, ExecNmsub);

impl_vector_op_integer_vx_binary!(VectorOpAdd, ExecAdd);
impl_vector_op_integer_vx_binary!(VectorOpAddu, ExecAddu);
impl_vector_op_integer_vx_binary!(VectorOpSub, ExecSub);
impl_vector_op_integer_vx_binary!(VectorOpSubu, ExecSubu);
impl_vector_op_integer_vx_binary!(VectorOpRevSub, ExecRevSub);
impl_vector_op_integer_vx_binary!(VectorOpAnd, ExecAnd);
impl_vector_op_integer_vx_binary!(VectorOpOr, ExecOr);
impl_vector_op_integer_vx_binary!(VectorOpXor, ExecXor);
impl_vector_op_integer_vx_binary!(VectorOpSll, ExecSLL);
impl_vector_op_integer_vx_binary!(VectorOpSrl, ExecSRL);
impl_vector_op_integer_vx_binary!(VectorOpSra, ExecSRA);
impl_vector_op_integer_vx_binary!(VectorOpMax, ExecMax);
impl_vector_op_integer_vx_binary!(VectorOpMaxu, ExecMaxu);
impl_vector_op_integer_vx_binary!(VectorOpMin, ExecMin);
impl_vector_op_integer_vx_binary!(VectorOpMinu, ExecMinu);
impl_vector_op_integer_vx_binary!(VectorOpMul, ExecMulLow);
impl_vector_op_integer_vx_binary!(VectorOpMulh, ExecMulHighSigned);
impl_vector_op_integer_vx_binary!(VectorOpMulhu, ExecMulHighUnsigned);
impl_vector_op_integer_vx_binary!(VectorOpMulhsu, ExecMulHighSignedUnsigned);
impl_vector_op_integer_vx_binary!(VectorOpDiv, ExecDivSigned);
impl_vector_op_integer_vx_binary!(VectorOpDivu, ExecDivUnsigned);
impl_vector_op_integer_vx_binary!(VectorOpRem, ExecRemSigned);
impl_vector_op_integer_vx_binary!(VectorOpRemu, ExecRemUnsigned);
impl_vector_op_integer_vxv_ternary!(VectorOpMacc, ExecMacc);
impl_vector_op_integer_vxv_ternary!(VectorOpNmsac, ExecNmsac);
impl_vector_op_integer_vxv_ternary!(VectorOpMadd, ExecMadd);
impl_vector_op_integer_vxv_ternary!(VectorOpNmsub, ExecNmsub);

impl<T> VectorOpIntegerV for ExecMove<T>
where
    T: Copy + Default,
{
    fn exec(vs2: &VGFRef, vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception> {
        assert!(vs2.sew == size_of::<T>() as u8 && vd.sew == size_of::<T>() as u8);
        let vs2 = vs2.as_slice::<T>();
        for (index, element) in vd.iter_mut().enumerate() {
            mask.element_load(
                element,
                <ExecMove<T> as ExecUnaryTrait<Result<T, Exception>, T>>::exec(vs2[index])?,
                index,
            );
        }

        Ok(())
    }
}

impl_vector_op_widening_interger_vv_binary!(VectorOpWadd, ExecWideningAdd, signed);
impl_vector_op_widening_interger_vv_binary!(VectorOpWaddu, ExecWideningAdd, unsigned);
impl_vector_op_widening_interger_vv_binary!(VectorOpWsub, ExecWideningSub, signed);
impl_vector_op_widening_interger_vv_binary!(VectorOpWsubu, ExecWideningSub, unsigned);
impl_vector_op_widening_interger_vv_binary!(VectorOpWmul, ExecWideningMul, signed);
impl_vector_op_widening_interger_vv_binary!(VectorOpWmulu, ExecWideningMul, unsigned);
impl_vector_op_widening_interger_vv_binary!(VectorOpWmulsu, ExecWideningMul, signed_unsigned);

impl_vector_op_widening_interger_vx_binary!(VectorOpWadd, ExecWideningAdd, signed);
impl_vector_op_widening_interger_vx_binary!(VectorOpWaddu, ExecWideningAdd, unsigned);
impl_vector_op_widening_interger_vx_binary!(VectorOpWsub, ExecWideningSub, signed);
impl_vector_op_widening_interger_vx_binary!(VectorOpWsubu, ExecWideningSub, unsigned);
impl_vector_op_widening_interger_vx_binary!(VectorOpWmul, ExecWideningMul, signed);
impl_vector_op_widening_interger_vx_binary!(VectorOpWmulu, ExecWideningMul, unsigned);
impl_vector_op_widening_interger_vx_binary!(VectorOpWmulsu, ExecWideningMul, signed_unsigned);

impl_vector_op_widening_integer_vvv_ternary!(VectorOpWmacc, ExecWmacc, signed);
impl_vector_op_widening_integer_vvv_ternary!(VectorOpWmaccu, ExecWmacc, unsigned);
impl_vector_op_widening_integer_vvv_ternary!(VectorOpWmaccsu, ExecWmacc, signed_unsigned);

impl_vector_op_widening_integer_vxv_ternary!(VectorOpWmacc, ExecWmacc, signed);
impl_vector_op_widening_integer_vxv_ternary!(VectorOpWmaccu, ExecWmacc, unsigned);
impl_vector_op_widening_integer_vxv_ternary!(VectorOpWmaccsu, ExecWmacc, signed_unsigned);
impl_vector_op_widening_integer_vxv_ternary!(VectorOpWmaccus, ExecWmacc, unsigned_signed);

impl_vector_op_widening_interger_wv_binary!(VectorOpWadd, ExecWideningAdd, signed);
impl_vector_op_widening_interger_wv_binary!(VectorOpWaddu, ExecWideningAdd, unsigned);
impl_vector_op_widening_interger_wv_binary!(VectorOpWsub, ExecWideningSub, signed);
impl_vector_op_widening_interger_wv_binary!(VectorOpWsubu, ExecWideningSub, unsigned);

impl_vector_op_widening_interger_wx_binary!(VectorOpWadd, ExecWideningAdd, signed);
impl_vector_op_widening_interger_wx_binary!(VectorOpWaddu, ExecWideningAdd, unsigned);
impl_vector_op_widening_interger_wx_binary!(VectorOpWsub, ExecWideningSub, signed);
impl_vector_op_widening_interger_wx_binary!(VectorOpWsubu, ExecWideningSub, unsigned);

impl_vector_op_integer_narrowing_wv_binary!(VectorOpNsrl, ExecNsrl);
impl_vector_op_integer_narrowing_wv_binary!(VectorOpNsra, ExecNsra);
impl_vector_op_integer_narrowing_vx_binary!(VectorOpNsrl, ExecNsrl);
impl_vector_op_integer_narrowing_vx_binary!(VectorOpNsra, ExecNsra);

impl_vector_op_integer_vvm_binary!(VectorOpAdc, ExecAdc);
impl_vector_op_integer_vxm_binary!(VectorOpAdc, ExecAdc);
impl_vector_op_integer_vvm_binary!(VectorOpSbc, ExecSbc);
impl_vector_op_integer_vxm_binary!(VectorOpSbc, ExecSbc);

impl_vector_op_integer_mask_vv_binary!(VectorOpMadc, ExecAdcCarryNoCarry);
impl_vector_op_integer_mask_vx_binary!(VectorOpMadc, ExecAdcCarryNoCarry);
impl_vector_op_integer_mask_vvm_binary!(VectorOpMadc, ExecAdcCarry);
impl_vector_op_integer_mask_vxm_binary!(VectorOpMadc, ExecAdcCarry);
impl_vector_op_integer_mask_vv_binary!(VectorOpMsbc, ExecSbcBorrowNoBorrow);
impl_vector_op_integer_mask_vx_binary!(VectorOpMsbc, ExecSbcBorrowNoBorrow);
impl_vector_op_integer_mask_vvm_binary!(VectorOpMsbc, ExecSbcBorrow);
impl_vector_op_integer_mask_vxm_binary!(VectorOpMsbc, ExecSbcBorrow);

impl_vector_op_integer_mask_vv_binary!(VectorOpMseq, ExecEqual);
impl_vector_op_integer_mask_vx_binary!(VectorOpMseq, ExecEqual);
impl_vector_op_integer_mask_vv_binary!(VectorOpMsne, ExecNotEqual);
impl_vector_op_integer_mask_vx_binary!(VectorOpMsne, ExecNotEqual);
impl_vector_op_integer_mask_vv_binary!(VectorOpMsltu, ExecUnsignedLess);
impl_vector_op_integer_mask_vx_binary!(VectorOpMsltu, ExecUnsignedLess);
impl_vector_op_integer_mask_vv_binary!(VectorOpMslt, ExecSignedLess);
impl_vector_op_integer_mask_vx_binary!(VectorOpMslt, ExecSignedLess);
impl_vector_op_integer_mask_vv_binary!(VectorOpMsleu, ExecUnsignedLessEqual);
impl_vector_op_integer_mask_vx_binary!(VectorOpMsleu, ExecUnsignedLessEqual);
impl_vector_op_integer_mask_vv_binary!(VectorOpMsle, ExecSignedLessEqual);
impl_vector_op_integer_mask_vx_binary!(VectorOpMsle, ExecSignedLessEqual);
impl_vector_op_integer_mask_vx_binary!(VectorOpMsgtu, ExecUnsignedGreaterX);
impl_vector_op_integer_mask_vx_binary!(VectorOpMsgt, ExecSignedGreaterX);

impl_vector_op_integer_v_unary_ext!(
    VectorOpZextVf2,
    ExecZext,
    2,
    [(1, 2, u8, u16), (2, 4, u16, u32), (4, 8, u32, u64)] // (src_sew, dst_sew, src_ty, dst_ty), extend from src to dst.
);
impl_vector_op_integer_v_unary_ext!(
    VectorOpZextVf4,
    ExecZext,
    4,
    [(1, 4, u8, u32), (2, 8, u16, u64)]
);
impl_vector_op_integer_v_unary_ext!(VectorOpZextVf8, ExecZext, 8, [(1, 8, u8, u64)]);
impl_vector_op_integer_v_unary_ext!(
    VectorOpSextVf2,
    ExecSext,
    2,
    [(1, 2, u8, u16), (2, 4, u16, u32), (4, 8, u32, u64)]
);
impl_vector_op_integer_v_unary_ext!(
    VectorOpSextVf4,
    ExecSext,
    4,
    [(1, 4, u8, u32), (2, 8, u16, u64)]
);
impl_vector_op_integer_v_unary_ext!(VectorOpSextVf8, ExecSext, 8, [(1, 8, u8, u64)]);

macro_rules! impl_vector_op_integer_gather_vv {
    ($op_ty:ty) => {
        impl VectorOpIntegerGatherVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == vs2.sew && vs2.sew == vd.sew);
                dispatch_integer_sew!(vd.sew, |T| {
                    vector_op_gather_vv::<T, T>(vs1, vs2, vd, mask)
                })
            }
        }
    };
}

macro_rules! impl_vector_op_integer_gather_vx {
    ($op_ty:ty) => {
        impl VectorOpIntegerVX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs2.sew == vd.sew);
                dispatch_integer_sew!(vd.sew, |T| { vector_op_gather_vx::<T>(x1, vs2, vd, mask) })
            }
        }
    };
}

macro_rules! impl_vector_op_integer_gather_vi {
    ($op_ty:ty) => {
        impl VectorOpIntegerVX for $op_ty {
            fn exec(
                imm: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs2.sew == vd.sew);
                dispatch_integer_sew!(vd.sew, |T| { vector_op_gather_vx::<T>(imm, vs2, vd, mask) })
            }
        }
    };
}

macro_rules! impl_vector_op_integer_gather_ei16_vv {
    ($op_ty:ty) => {
        impl VectorOpIntegerGatherEI16VV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
            ) -> Result<(), Exception> {
                assert!(vs1.sew == 2 && vs2.sew == vd.sew);
                dispatch_integer_sew!(vd.sew, |T| {
                    vector_op_gather_vv::<u16, T>(vs1, vs2, vd, mask)
                })
            }
        }
    };
}

impl_vector_op_integer_gather_vv!(VectorOpRGatherVV);
impl_vector_op_integer_gather_vx!(VectorOpRGatherVX);
impl_vector_op_integer_gather_ei16_vv!(VectorOpRGatherEI16VV);
impl_vector_op_integer_gather_vi!(VectorOpRGatherVI);

impl VectorOpIntegerVVM for VectorOpMerge {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        v0: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception> {
        assert!(vs1.sew == vs2.sew && vs2.sew == vd.sew);
        dispatch_integer_sew!(vd.sew, |T| {
            vector_op_merge_vvm::<T>(vs1, vs2, v0, vd, mask)
        })
    }
}

impl VectorOpIntegerVXM for VectorOpMerge {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        v0: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception> {
        assert!(vs2.sew == vd.sew);
        dispatch_integer_sew!(vd.sew, |T| {
            vector_op_merge_vxm::<T>(x1 as T, vs2, v0, vd, mask)
        })
    }
}

impl VectorOpMaskToX for VectorOpCpopM {
    fn exec(vs2: &VGFRef, mask: &VecOpMask) -> Result<WordType, Exception> {
        Ok(vector_op_mask_popcount(vs2, mask))
    }
}

impl VectorOpMaskToX for VectorOpFirstM {
    fn exec(vs2: &VGFRef, mask: &VecOpMask) -> Result<WordType, Exception> {
        Ok(vector_op_mask_first(vs2, mask))
    }
}

impl VectorOpMaskUnary for VectorOpMsbfM {
    fn exec(vs2: &VGFRef, vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception> {
        vector_op_mask_before_first(vs2, vd, mask, MaskBeforeFirstMode::Before)
    }
}

impl VectorOpMaskUnary for VectorOpMsifM {
    fn exec(vs2: &VGFRef, vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception> {
        vector_op_mask_before_first(vs2, vd, mask, MaskBeforeFirstMode::Including)
    }
}

impl VectorOpMaskUnary for VectorOpMsofM {
    fn exec(vs2: &VGFRef, vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception> {
        vector_op_mask_before_first(vs2, vd, mask, MaskBeforeFirstMode::Only)
    }
}

impl VectorOpMaskToVector for VectorOpIotaM {
    fn exec(vs2: &VGFRef, vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception> {
        dispatch_integer_sew!(vd.sew, |T| { vector_op_iota::<T>(vs2, vd, mask) })
    }
}

impl VectorOpIndex for VectorOpIdV {
    fn exec(vd: &mut VGFRefMut, mask: &VecOpMask) -> Result<(), Exception> {
        dispatch_integer_sew!(vd.sew, |T| { vector_op_index::<T>(vd, mask) })
    }
}

impl VectorOpCompress for VectorOpCompressVm {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        vl: usize,
        tail_mask: &VecOpMask,
    ) -> Result<(), Exception> {
        assert!(vs1.sew == 1 && vs2.sew == vd.sew);
        dispatch_integer_sew!(vd.sew, |T| {
            vector_op_compress::<T>(vs1, vs2, vd, vl, tail_mask)
        })
    }
}

impl VectorOpIntegerVXV for VectorOpSlideUp {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        old_vd: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception> {
        assert!(vs2.sew == old_vd.sew && old_vd.sew == vd.sew);
        dispatch_integer_sew!(vd.sew, |T| {
            vector_op_slideup::<T>(x1 as usize, vs2, old_vd, vd, mask)
        })
    }
}

impl VectorOpIntegerVX for VectorOpSlideDown {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception> {
        assert!(vs2.sew == vd.sew);
        dispatch_integer_sew!(vd.sew, |T| {
            vector_op_slidedown::<T>(x1 as usize, vs2, vd, mask)
        })
    }
}

impl VectorOpIntegerVXV for VectorOpSlide1Up {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        old_vd: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception> {
        assert!(vs2.sew == old_vd.sew && old_vd.sew == vd.sew);
        dispatch_integer_sew!(vd.sew, |T| {
            vector_op_slide1up::<T>(x1, vs2, old_vd, vd, mask)
        })
    }
}

impl VectorOpIntegerVX for VectorOpSlide1Down {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
    ) -> Result<(), Exception> {
        assert!(vs2.sew == vd.sew);
        dispatch_integer_sew!(vd.sew, |T| { vector_op_slide1down::<T>(x1, vs2, vd, mask) })
    }
}

#[cfg(test)]
#[path = "integer_test.rs"]
mod test;
