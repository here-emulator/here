#[cfg(test)]
use crate::isa::riscv::vector::tester::TestOpParameter;
use crate::{
    config::arch_config::WordType,
    isa::riscv::{
        trap::Exception,
        vector::{
            VecOpMask, Vector,
            arithmetic::{narrowing_source_lmul, vector_register_group_overlaps},
            types::{FixedPointRoundingMode, VGFRef, VGFRefMut, Vsew},
        },
    },
    utils::{TruncateFrom, UnsignedInteger, as_signed_i128, from_signed_i128, shift_amount},
};

trait FixedPointBinaryExec<T> {
    fn exec(vs2: T, vs1: T, round: FixedPointRoundingMode) -> (T, bool);
}

trait FixedPointNarrowingShiftExec<Src, Shift, Dst> {
    fn exec(src: Src, shift: Shift, round: FixedPointRoundingMode) -> (Dst, bool);
}

pub(in crate::isa::riscv) trait VectorOpFixedPointVV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
        round: FixedPointRoundingMode,
    ) -> Result<bool, Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<bool, Exception>
    where
        Self: Sized,
    {
        let round = vector.fixed_point_rounding_mode();
        vector.exec_fixed_point_vv::<Self>(
            param.vs1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            round,
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpFixedPointVX {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
        round: FixedPointRoundingMode,
    ) -> Result<bool, Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<bool, Exception>
    where
        Self: Sized,
    {
        let round = vector.fixed_point_rounding_mode();
        vector.exec_fixed_point_vx::<Self>(
            param.x1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            round,
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpFixedPointNarrowingWV {
    fn exec(
        vs1: &VGFRef,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
        round: FixedPointRoundingMode,
    ) -> Result<bool, Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<bool, Exception>
    where
        Self: Sized,
    {
        let round = vector.fixed_point_rounding_mode();
        vector.exec_fixed_point_narrowing_wv::<Self>(
            param.vs1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            round,
            0,
        )
    }
}

pub(in crate::isa::riscv) trait VectorOpFixedPointNarrowingVX {
    fn exec(
        x1: WordType,
        vs2: &VGFRef,
        vd: &mut VGFRefMut,
        mask: &VecOpMask,
        round: FixedPointRoundingMode,
    ) -> Result<bool, Exception>;

    #[cfg(test)]
    fn test(vector: &mut Vector, param: TestOpParameter) -> Result<bool, Exception>
    where
        Self: Sized,
    {
        let round = vector.fixed_point_rounding_mode();
        vector.exec_fixed_point_narrowing_vx::<Self>(
            param.x1(),
            param.vs2(),
            param.vd(),
            param.enable_mask(),
            round,
            0,
        )
    }
}

impl Vector {
    #[inline]
    pub(in crate::isa::riscv) fn fixed_point_rounding_mode(&self) -> FixedPointRoundingMode {
        self.config.fixed_point_rounding_mode
    }

    #[inline]
    pub(in crate::isa::riscv) fn fixed_point_accrued_saturation_flag(&self) -> bool {
        self.config.fixed_point_accrued_saturation_flag
    }

    #[inline]
    pub(in crate::isa::riscv) fn set_fixed_point_rounding_mode(
        &mut self,
        round: FixedPointRoundingMode,
    ) {
        self.config.fixed_point_rounding_mode = round;
    }

    #[inline]
    pub(in crate::isa::riscv) fn clear_fixed_point_accrued_saturation_flag(&mut self) {
        self.config.fixed_point_accrued_saturation_flag = false;
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_fixed_point_vv<Op>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        round: FixedPointRoundingMode,
        vstart: usize,
    ) -> Result<bool, Exception>
    where
        Op: VectorOpFixedPointVV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let vs1_data = self.vector_regfile.get_ref(lmul, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs1_ref = VGFRef::new(&vs1_data, sew, lmul, 1);
        let vs2_ref = VGFRef::new(&vs2_data, sew, lmul, 1);
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, sew, lmul, 1);
        self.config.fixed_point_rounding_mode = round;
        let saturated = Op::exec(&vs1_ref, &vs2_ref, &mut vd_ref, &mask, round)?;
        if saturated {
            self.config.fixed_point_accrued_saturation_flag = true;
        }
        Ok(saturated)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_fixed_point_vx<Op>(
        &mut self,
        x1: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        round: FixedPointRoundingMode,
        vstart: usize,
    ) -> Result<bool, Exception>
    where
        Op: VectorOpFixedPointVX,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, sew, lmul, 1);
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, sew, lmul, 1);
        self.config.fixed_point_rounding_mode = round;
        let saturated = Op::exec(x1, &vs2_ref, &mut vd_ref, &mask, round)?;
        if saturated {
            self.config.fixed_point_accrued_saturation_flag = true;
        }
        Ok(saturated)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_fixed_point_narrowing_wv<Op>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        round: FixedPointRoundingMode,
        vstart: usize,
    ) -> Result<bool, Exception>
    where
        Op: VectorOpFixedPointNarrowingWV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (dst_sew, src_sew) = (vsew.into_byte_width(), vsew.into_byte_width() * 2);
        let Some(src_eew) = Vsew::from_byte_width(src_sew) else {
            return Err(Exception::IllegalInstruction);
        };
        let (lmul, src_lmul) = (vlmul.get_lmul(), narrowing_source_lmul(vlmul)?);
        if src_lmul > 8 || vector_register_group_overlaps(vd, lmul, vs2, src_lmul) {
            return Err(Exception::IllegalInstruction);
        }

        let vs1_data = self.vector_regfile.get_ref(lmul, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(src_lmul, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs1_ref = VGFRef::new(&vs1_data, dst_sew, lmul, 1);
        let vs2_ref = VGFRef::new(&vs2_data, src_eew.into_byte_width(), src_lmul, 1);
        let mut vd_ref =
            VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, dst_sew, lmul, 1);
        self.config.fixed_point_rounding_mode = round;
        let saturated = Op::exec(&vs1_ref, &vs2_ref, &mut vd_ref, &mask, round)?;
        if saturated {
            self.config.fixed_point_accrued_saturation_flag = true;
        }
        Ok(saturated)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_fixed_point_narrowing_vx<Op>(
        &mut self,
        x1: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        round: FixedPointRoundingMode,
        vstart: usize,
    ) -> Result<bool, Exception>
    where
        Op: VectorOpFixedPointNarrowingVX,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (dst_sew, src_sew) = (vsew.into_byte_width(), vsew.into_byte_width() * 2);
        let Some(src_eew) = Vsew::from_byte_width(src_sew) else {
            return Err(Exception::IllegalInstruction);
        };
        let (lmul, src_lmul) = (vlmul.get_lmul(), narrowing_source_lmul(vlmul)?);
        if src_lmul > 8 || vector_register_group_overlaps(vd, lmul, vs2, src_lmul) {
            return Err(Exception::IllegalInstruction);
        }

        let vs2_data = self.vector_regfile.get_ref(src_lmul, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, src_eew.into_byte_width(), src_lmul, 1);
        let mut vd_ref =
            VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, dst_sew, lmul, 1);
        self.config.fixed_point_rounding_mode = round;
        let saturated = Op::exec(x1, &vs2_ref, &mut vd_ref, &mask, round)?;
        if saturated {
            self.config.fixed_point_accrued_saturation_flag = true;
        }
        Ok(saturated)
    }
}

fn round_increment(value_bits: u128, shift: u32, round: FixedPointRoundingMode) -> i128 {
    if shift == 0 {
        return 0;
    }

    let bit = |index: u32| ((value_bits >> index) & 1) != 0;
    let discarded_mask = (1u128 << shift) - 1;
    let discarded = value_bits & discarded_mask;
    let half = bit(shift - 1);

    let increment = match round {
        FixedPointRoundingMode::RoundToNearestUp => half,
        FixedPointRoundingMode::RoundToNearestEven => {
            let below_half = if shift == 1 {
                false
            } else {
                (value_bits & ((1u128 << (shift - 1)) - 1)) != 0
            };
            half && (below_half || bit(shift))
        }
        FixedPointRoundingMode::RoundDown => false,
        FixedPointRoundingMode::RoundToOdd => !bit(shift) && discarded != 0,
    };

    increment as i128
}

fn round_unsigned(value: u128, shift: u32, round: FixedPointRoundingMode) -> u128 {
    if shift == 0 {
        value
    } else {
        (value >> shift) + round_increment(value, shift, round) as u128
    }
}

fn round_signed(value: i128, shift: u32, round: FixedPointRoundingMode) -> i128 {
    if shift == 0 {
        value
    } else {
        (value >> shift) + round_increment(value as u128, shift, round)
    }
}

fn unsigned_max(bits: u32) -> u128 {
    (1u128 << bits) - 1
}

fn signed_min(bits: u32) -> i128 {
    -(1i128 << (bits - 1))
}

fn signed_max(bits: u32) -> i128 {
    (1i128 << (bits - 1)) - 1
}

fn saturate_unsigned<T>(value: u128) -> (T, bool)
where
    T: UnsignedInteger,
{
    let max = unsigned_max(T::BITS as u32);
    if value > max {
        (T::MAX, true)
    } else {
        (T::truncate_from(value), false)
    }
}

fn saturate_signed<T>(value: i128) -> (T, bool)
where
    T: UnsignedInteger,
{
    let min = signed_min(T::BITS as u32);
    let max = signed_max(T::BITS as u32);
    if value > max {
        (from_signed_i128(max), true)
    } else if value < min {
        (from_signed_i128(min), true)
    } else {
        (from_signed_i128(value), false)
    }
}

struct ExecSaturatingAddUnsigned;
struct ExecSaturatingAddSigned;
struct ExecSaturatingSubUnsigned;
struct ExecSaturatingSubSigned;
struct ExecAveragingAddUnsigned;
struct ExecAveragingAddSigned;
struct ExecAveragingSubUnsigned;
struct ExecAveragingSubSigned;
struct ExecSaturatingFractionalMul;
struct ExecScalingShiftRightLogical;
struct ExecScalingShiftRightArithmetic;
struct ExecNarrowingClipUnsigned;
struct ExecNarrowingClipSigned;

impl<T> FixedPointBinaryExec<T> for ExecSaturatingAddUnsigned
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, _round: FixedPointRoundingMode) -> (T, bool) {
        let vs2: u128 = vs2.into();
        let vs1: u128 = vs1.into();
        saturate_unsigned(vs2 + vs1)
    }
}

impl<T> FixedPointBinaryExec<T> for ExecSaturatingAddSigned
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, _round: FixedPointRoundingMode) -> (T, bool) {
        saturate_signed(as_signed_i128(vs2) + as_signed_i128(vs1))
    }
}

impl<T> FixedPointBinaryExec<T> for ExecSaturatingSubUnsigned
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, _round: FixedPointRoundingMode) -> (T, bool) {
        let vs2: u128 = vs2.into();
        let vs1: u128 = vs1.into();
        if vs2 < vs1 {
            (T::MIN, true)
        } else {
            (T::truncate_from(vs2 - vs1), false)
        }
    }
}

impl<T> FixedPointBinaryExec<T> for ExecSaturatingSubSigned
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, _round: FixedPointRoundingMode) -> (T, bool) {
        saturate_signed(as_signed_i128(vs2) - as_signed_i128(vs1))
    }
}

impl<T> FixedPointBinaryExec<T> for ExecAveragingAddUnsigned
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, round: FixedPointRoundingMode) -> (T, bool) {
        let vs2: u128 = vs2.into();
        let vs1: u128 = vs1.into();
        (T::truncate_from(round_unsigned(vs2 + vs1, 1, round)), false)
    }
}

impl<T> FixedPointBinaryExec<T> for ExecAveragingAddSigned
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, round: FixedPointRoundingMode) -> (T, bool) {
        (
            from_signed_i128(round_signed(
                as_signed_i128(vs2) + as_signed_i128(vs1),
                1,
                round,
            )),
            false,
        )
    }
}

impl<T> FixedPointBinaryExec<T> for ExecAveragingSubUnsigned
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, round: FixedPointRoundingMode) -> (T, bool) {
        let vs2: u128 = vs2.into();
        let vs1: u128 = vs1.into();
        (
            from_signed_i128(round_signed(vs2 as i128 - vs1 as i128, 1, round)),
            false,
        )
    }
}

impl<T> FixedPointBinaryExec<T> for ExecAveragingSubSigned
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, round: FixedPointRoundingMode) -> (T, bool) {
        (
            from_signed_i128(round_signed(
                as_signed_i128(vs2) - as_signed_i128(vs1),
                1,
                round,
            )),
            false,
        )
    }
}

impl<T> FixedPointBinaryExec<T> for ExecSaturatingFractionalMul
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, round: FixedPointRoundingMode) -> (T, bool) {
        let product = as_signed_i128(vs2) * as_signed_i128(vs1);
        saturate_signed(round_signed(product, T::BITS as u32 - 1, round))
    }
}

impl<T> FixedPointBinaryExec<T> for ExecScalingShiftRightLogical
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, round: FixedPointRoundingMode) -> (T, bool) {
        let shift = shift_amount(vs1);
        let vs2: u128 = vs2.into();
        (T::truncate_from(round_unsigned(vs2, shift, round)), false)
    }
}

impl<T> FixedPointBinaryExec<T> for ExecScalingShiftRightArithmetic
where
    T: UnsignedInteger + Into<u128>,
{
    fn exec(vs2: T, vs1: T, round: FixedPointRoundingMode) -> (T, bool) {
        let shift = shift_amount(vs1);
        (
            from_signed_i128(round_signed(as_signed_i128(vs2), shift, round)),
            false,
        )
    }
}

impl<Src, Shift, Dst> FixedPointNarrowingShiftExec<Src, Shift, Dst> for ExecNarrowingClipUnsigned
where
    Src: UnsignedInteger + Into<u128>,
    Shift: UnsignedInteger,
    Dst: UnsignedInteger,
{
    fn exec(src: Src, shift: Shift, round: FixedPointRoundingMode) -> (Dst, bool) {
        let shift = Into::<u64>::into(shift) as u32 & (Src::BITS as u32 - 1);
        let src: u128 = src.into();
        saturate_unsigned(round_unsigned(src, shift, round))
    }
}

impl<Src, Shift, Dst> FixedPointNarrowingShiftExec<Src, Shift, Dst> for ExecNarrowingClipSigned
where
    Src: UnsignedInteger + Into<u128>,
    Shift: UnsignedInteger,
    Dst: UnsignedInteger,
{
    fn exec(src: Src, shift: Shift, round: FixedPointRoundingMode) -> (Dst, bool) {
        let shift = Into::<u64>::into(shift) as u32 & (Src::BITS as u32 - 1);
        saturate_signed(round_signed(as_signed_i128(src), shift, round))
    }
}

macro_rules! dispatch_fixed_point_sew {
    ($sew:expr, |$ty:ident| $body:expr) => {
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

macro_rules! dispatch_fixed_point_narrowing_sew {
    ($dst_sew:expr, |$src_ty:ident, $dst_ty:ident| $body:expr) => {
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

macro_rules! impl_fixed_point_vv_binary {
    ($op_ty:ty, $exec_ty:ty) => {
        impl VectorOpFixedPointVV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
                round: FixedPointRoundingMode,
            ) -> Result<bool, Exception> {
                assert!(vs1.sew == vs2.sew && vs1.sew == vd.sew);
                let mut saturated = false;
                dispatch_fixed_point_sew!(vd.sew, |T| {
                    let vs1 = vs1.as_slice::<T>();
                    let vs2 = vs2.as_slice::<T>();
                    for (index, element) in vd.iter_mut().enumerate() {
                        let (value, element_saturated) =
                            <$exec_ty as FixedPointBinaryExec<T>>::exec(
                                vs2[index], vs1[index], round,
                            );
                        saturated |= element_saturated;
                        mask.element_load(element, value, index);
                    }
                });
                Ok(saturated)
            }
        }
    };
}

macro_rules! impl_fixed_point_vx_binary {
    ($op_ty:ty, $exec_ty:ty) => {
        impl VectorOpFixedPointVX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
                round: FixedPointRoundingMode,
            ) -> Result<bool, Exception> {
                assert!(vs2.sew == vd.sew);
                let mut saturated = false;
                dispatch_fixed_point_sew!(vd.sew, |T| {
                    let scalar = T::truncate_from(x1);
                    let vs2 = vs2.as_slice::<T>();
                    for (index, element) in vd.iter_mut().enumerate() {
                        let (value, element_saturated) =
                            <$exec_ty as FixedPointBinaryExec<T>>::exec(vs2[index], scalar, round);
                        saturated |= element_saturated;
                        mask.element_load(element, value, index);
                    }
                });
                Ok(saturated)
            }
        }
    };
}

macro_rules! impl_fixed_point_binary {
    ($op_ty:ty, $exec_ty:ty) => {
        impl_fixed_point_vv_binary!($op_ty, $exec_ty);
        impl_fixed_point_vx_binary!($op_ty, $exec_ty);
    };
}

macro_rules! impl_fixed_point_narrowing_wv_binary {
    ($op_ty:ty, $exec_ty:ty) => {
        impl VectorOpFixedPointNarrowingWV for $op_ty {
            fn exec(
                vs1: &VGFRef,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
                round: FixedPointRoundingMode,
            ) -> Result<bool, Exception> {
                assert!(vs1.sew == vd.sew && vs2.sew == vd.sew * 2);
                let mut saturated = false;
                dispatch_fixed_point_narrowing_sew!(vd.sew, |Src, Dst| {
                    let vs1 = vs1.as_slice::<Dst>();
                    let vs2 = vs2.as_slice::<Src>();
                    for (index, element) in vd.iter_mut().take(vs2.len()).enumerate() {
                        let (value, element_saturated) =
                            <$exec_ty as FixedPointNarrowingShiftExec<Src, Dst, Dst>>::exec(
                                vs2[index], vs1[index], round,
                            );
                        saturated |= element_saturated;
                        mask.element_load(element, value, index);
                    }
                });
                Ok(saturated)
            }
        }
    };
}

macro_rules! impl_fixed_point_narrowing_vx_binary {
    ($op_ty:ty, $exec_ty:ty) => {
        impl VectorOpFixedPointNarrowingVX for $op_ty {
            fn exec(
                x1: WordType,
                vs2: &VGFRef,
                vd: &mut VGFRefMut,
                mask: &VecOpMask,
                round: FixedPointRoundingMode,
            ) -> Result<bool, Exception> {
                assert!(vs2.sew == vd.sew * 2);
                let mut saturated = false;
                dispatch_fixed_point_narrowing_sew!(vd.sew, |Src, Dst| {
                    let shift = Dst::truncate_from(x1);
                    let vs2 = vs2.as_slice::<Src>();
                    for (index, element) in vd.iter_mut().take(vs2.len()).enumerate() {
                        let (value, element_saturated) =
                            <$exec_ty as FixedPointNarrowingShiftExec<Src, Dst, Dst>>::exec(
                                vs2[index], shift, round,
                            );
                        saturated |= element_saturated;
                        mask.element_load(element, value, index);
                    }
                });
                Ok(saturated)
            }
        }
    };
}

macro_rules! impl_fixed_point_narrowing_binary {
    ($op_ty:ty, $exec_ty:ty) => {
        impl_fixed_point_narrowing_wv_binary!($op_ty, $exec_ty);
        impl_fixed_point_narrowing_vx_binary!($op_ty, $exec_ty);
    };
}

pub(in crate::isa::riscv) struct VectorOpSaddu;
pub(in crate::isa::riscv) struct VectorOpSadd;
pub(in crate::isa::riscv) struct VectorOpSsubu;
pub(in crate::isa::riscv) struct VectorOpSsub;
pub(in crate::isa::riscv) struct VectorOpAaddu;
pub(in crate::isa::riscv) struct VectorOpAadd;
pub(in crate::isa::riscv) struct VectorOpAsubu;
pub(in crate::isa::riscv) struct VectorOpAsub;
pub(in crate::isa::riscv) struct VectorOpSmul;
pub(in crate::isa::riscv) struct VectorOpSsrl;
pub(in crate::isa::riscv) struct VectorOpSsra;
pub(in crate::isa::riscv) struct VectorOpNclipu;
pub(in crate::isa::riscv) struct VectorOpNclip;

impl_fixed_point_binary!(VectorOpSaddu, ExecSaturatingAddUnsigned);
impl_fixed_point_binary!(VectorOpSadd, ExecSaturatingAddSigned);
impl_fixed_point_binary!(VectorOpSsubu, ExecSaturatingSubUnsigned);
impl_fixed_point_binary!(VectorOpSsub, ExecSaturatingSubSigned);
impl_fixed_point_binary!(VectorOpAaddu, ExecAveragingAddUnsigned);
impl_fixed_point_binary!(VectorOpAadd, ExecAveragingAddSigned);
impl_fixed_point_binary!(VectorOpAsubu, ExecAveragingSubUnsigned);
impl_fixed_point_binary!(VectorOpAsub, ExecAveragingSubSigned);
impl_fixed_point_binary!(VectorOpSmul, ExecSaturatingFractionalMul);
impl_fixed_point_binary!(VectorOpSsrl, ExecScalingShiftRightLogical);
impl_fixed_point_binary!(VectorOpSsra, ExecScalingShiftRightArithmetic);
impl_fixed_point_narrowing_binary!(VectorOpNclipu, ExecNarrowingClipUnsigned);
impl_fixed_point_narrowing_binary!(VectorOpNclip, ExecNarrowingClipSigned);

#[cfg(test)]
#[path = "fix_point_test.rs"]
mod test;
