use std::{
    fmt::{Debug, Display},
    ops::*,
};

use rustc_apfloat::StatusAnd;

use crate::{fpu::soft_float::APFloatOf, utils::UnsignedInteger};

pub trait InBits<U> {
    fn from_bits(x: U) -> Self;
    fn to_bits(self) -> U;
}

pub trait FloatPoint:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + PartialEq
    + PartialOrd
    + Copy
    + From<f32>
    + Into<f64>
    + Debug
    + Display
    + InBits<Self::BitsType>
    + APFloatOf
{
    type BitsType: UnsignedInteger;

    fn sqrt(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl InBits<u32> for f32 {
    #[inline]
    fn from_bits(x: u32) -> Self {
        f32::from_bits(x)
    }

    #[inline]
    fn to_bits(self) -> u32 {
        self.to_bits()
    }
}
impl InBits<u64> for f64 {
    #[inline]
    fn from_bits(x: u64) -> Self {
        f64::from_bits(x)
    }

    #[inline]
    fn to_bits(self) -> u64 {
        self.to_bits()
    }
}

impl FloatPoint for f32 {
    type BitsType = u32;

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
}
impl FloatPoint for f64 {
    type BitsType = u64;

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
}

pub trait InFloat {
    type Float: FloatPoint;

    fn into_float(self) -> Self::Float;

    fn from_float(f: Self::Float) -> Self;
}

impl InFloat for u32 {
    type Float = f32;

    fn into_float(self) -> f32 {
        self as f32
    }

    fn from_float(f: Self::Float) -> Self {
        f as u32
    }
}

impl InFloat for u64 {
    type Float = f64;

    fn into_float(self) -> f64 {
        self as f64
    }

    fn from_float(f: Self::Float) -> Self {
        f as u64
    }
}

pub type FloatOf<T> = <T as InFloat>::Float;

pub trait CmpOp<F> {
    fn apply(a: F, b: F) -> StatusAnd<bool>;
}

pub trait BinaryOp<F> {
    fn apply(a: F, b: F) -> F;
}
