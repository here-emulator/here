use super::*;
use crate::{
    config::arch_config::WordType,
    utils::{UnsignedInteger, make_mask},
};

pub(super) type WriteValidator = fn(WordType, &CsrContext) -> CsrWriteOp;

/// Condition trait for [`validate_with_cond`] functions.
trait ValidateCond {
    fn check(value: WordType, ctx: &CsrContext) -> bool;
}

struct RangeCond<const MIN: WordType, const MAX: WordType> {}
impl<const MIN: WordType, const MAX: WordType> ValidateCond for RangeCond<MIN, MAX> {
    fn check(value: WordType, _ctx: &CsrContext) -> bool {
        return MIN <= value && value <= MAX;
    }
}

#[allow(unused)]
macro_rules! make_enum_cond {
    ($name:ident, $($valid:expr),+ $(,)?) => {
    struct $name {}
    impl ValidateCond for $name {
            fn check(value: WordType, _ctx: &CsrContext) -> bool {
                const VALID_VALUES: &[WordType] = &[$($valid),+];
                VALID_VALUES.contains(&value)
            }
        }
    }
}

#[inline]
/// Make a validator that only write when the condition is satisfied, and do nothing otherwise.
fn validate_with_cond<const L: usize, const R: usize, C: ValidateCond>(
    value: WordType,
    ctx: &CsrContext,
) -> CsrWriteOp {
    let mask = make_mask(L, R);
    let extracted = value.extract_range(L as u32, R as u32);
    if C::check(extracted, ctx) {
        CsrWriteOp { mask }
    } else {
        CsrWriteOp { mask: 0 }
    }
}

#[inline]
pub(super) fn validate_range<
    const L: usize,
    const R: usize,
    const MIN: WordType,
    const MAX: WordType,
>(
    value: WordType,
    ctx: &CsrContext,
) -> CsrWriteOp {
    validate_with_cond::<L, R, RangeCond<MIN, MAX>>(value, ctx)
}

#[inline]
pub(super) fn validate_write_any<const L: usize, const R: usize>(
    _value: WordType,
    _ctx: &CsrContext,
) -> CsrWriteOp {
    CsrWriteOp::new(make_mask(L, R))
}

/// Make a validator that only write the bits with masks.
#[inline]
pub(super) fn validate_mask<const L: usize, const R: usize, const MASK: WordType>(
    _value: WordType,
    _ctx: &CsrContext,
) -> CsrWriteOp {
    CsrWriteOp::new(make_mask(L, R) & MASK)
}

macro_rules! combine_validators {
    ($value:expr, $ctx:expr, $($validator:expr),* $(,)?) => {
        {
            #[allow(unused_mut)]
            let mut combined = CsrWriteOp { mask: 0 };
            $(
                let result = $validator($value, $ctx);
                combined = combined.merge(&result);
            )*
            combined
        }
    };
}

#[inline]
pub(super) fn validate_readonly(_value: WordType, _ctx: &CsrContext) -> CsrWriteOp {
    CsrWriteOp { mask: 0 }
}
