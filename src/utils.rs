use std::{
    fmt::{Debug, Display},
    ops::*,
};

use rustc_apfloat::StatusAnd;

use crate::{
    config::arch_config::{SignedWordType, WordType, XLEN},
    fpu::soft_float::APFloatOf,
};

#[macro_export]
macro_rules! debug_unreachable {
    () => {
        if cfg!(debug_assertions) {
            panic!()
        } else {
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }
    };
}

pub struct BiMap<K, V> {
    pub forward: std::collections::BTreeMap<K, V>,
    pub backward: std::collections::BTreeMap<V, K>,
}

impl<K: Clone + Ord, V: Clone + Ord> BiMap<K, V> {
    pub fn new() -> BiMap<K, V> {
        BiMap {
            forward: std::collections::BTreeMap::new(),
            backward: std::collections::BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.forward.insert(key.clone(), value.clone());
        self.backward.insert(value, key);
    }

    pub fn get_by_left(&self, key: &K) -> Option<&V> {
        self.forward.get(key)
    }

    pub fn get_by_right(&self, value: &V) -> Option<&K> {
        self.backward.get(value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.forward.iter()
    }
}

fn rand_unique<T, F>(rd: F, cnt: usize) -> Vec<T>
where
    T: Copy + Eq + std::hash::Hash + std::fmt::Debug,
    F: Fn() -> T,
{
    let mut set = std::collections::HashSet::new();
    let mut result = Vec::with_capacity(cnt);

    while result.len() < cnt {
        let val = rd();
        if set.insert(val) {
            result.push(val);
        }
    }

    result
}

#[inline(always)]
pub(crate) fn shift_amount<T>(value: T) -> u32
where
    T: UnsignedInteger,
{
    let value: u64 = value.into();
    (value as u32) & ((T::BITS as u32) - 1)
}

#[inline(always)]
pub(crate) fn as_signed_i128<T>(value: T) -> i128
where
    T: UnsignedInteger + Into<u128>,
{
    let value: u128 = value.into();
    let bits = T::BITS as u32;
    let sign_mask = 1u128 << (bits - 1);

    if value & sign_mask == 0 {
        value as i128
    } else {
        (value as i128) - ((1u128 << bits) as i128)
    }
}

#[inline(always)]
pub(crate) fn from_signed_i128<T>(value: i128) -> T
where
    T: UnsignedInteger,
{
    T::truncate_from(value as u128)
}

#[inline(always)]
pub fn sign_extend(value: WordType, from_bits: u32) -> WordType {
    let sign_bit = XLEN as u32 - from_bits;
    ((value << sign_bit) as SignedWordType >> sign_bit) as WordType
}

#[inline]
pub fn sign_extend_u32(value: u32) -> u64 {
    sign_extend(value as WordType, 32)
}

#[inline]
pub fn is_sign_extended(value: WordType, bits: u32) -> bool {
    value == sign_extend(value, bits)
}

#[inline]
pub fn wrapping_add_as_signed(lhs: WordType, rhs: WordType) -> WordType {
    lhs.cast_signed()
        .wrapping_add(rhs.cast_signed())
        .cast_unsigned()
}

/// get the negative of given number of [`WordType`] in 2's complement.
#[inline]
pub fn negative_of(value: WordType) -> WordType {
    (!value).wrapping_add(1)
}

#[macro_export]
macro_rules! concat_bits {
    // End case (only one argument)
    ($t:ty; $last:expr) => {
        ($last as u64)
    };

    // Recursive concatenation (left shift * number of bits)
    ($t:ty; $head:expr, $($tail:expr),+) => {
        (($head as u64) << (8 * $crate::count_args!($($tail),+))) |
        concat_bits!($t; $($tail),+)
    };
}

#[macro_export]
macro_rules! count_args {
    ($($x:expr),*) => {
        <[()]>::len(&[$(count_args![@sub $x]),*])
    };
    (@sub $_:expr) => { () };
}

// ==================================
// read raw ptr ans check align.
// ==================================
const ALIGN_ILST: [WordType; 9] = [
    0x00,
    0x00,
    0x01,
    WordType::MAX,
    0x03,
    WordType::MAX,
    WordType::MAX,
    WordType::MAX,
    0x07,
];

pub unsafe fn read_raw_ptr<T>(addr: *const u8) -> Option<T> {
    let size_of_t: usize = size_of::<T>();
    if (addr as WordType) & ALIGN_ILST[size_of_t] != 0 {
        return None;
    }

    let ptr = addr as *const T;
    Some(unsafe { ptr.read_volatile() })
}

pub unsafe fn write_raw_ptr<T>(addr: *mut u8, data: T) -> Option<()> {
    let size_of_t: usize = size_of::<T>();
    if (addr as WordType) & ALIGN_ILST[size_of_t] != 0 {
        return None;
    }

    let ptr = addr as *mut T;
    unsafe { ptr.write_volatile(data) }
    Some(())
}

pub fn check_align<T>(addr: WordType) -> bool {
    let size_of_t: usize = size_of::<T>();
    (addr as WordType) & ALIGN_ILST[size_of_t] == 0
}

// ========================================
//  gen_name_list ["a1", "a2", "a3", ... ]
// ========================================

/// # Examples
/// ```
/// assert_eq!(gen_name_list("a"; 0, 5), ["a0", "a1", "a2", "a3", "a4", "a5"])
/// ```
#[macro_export]
macro_rules! gen_name_list {
    ($base:literal; $begin: literal, $end: literal) => {
        seq_macro::seq!(N in $begin..= $end {
            [ #(concat!($base, stringify!(N)),) *]
        })
    }
}

pub const fn concat_arrays<const SIZE1: usize, const SIZE2: usize>(
    arr1: [&'static str; SIZE1],
    arr2: [&'static str; SIZE2],
) -> [&'static str; SIZE1 + SIZE2] {
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    let mut ret: [&'static str; SIZE1 + SIZE2] = [""; SIZE1 + SIZE2];
    while i < SIZE1 {
        ret[k] = arr1[i];
        i += 1;
        k += 1;
    }
    while j < SIZE2 {
        ret[k] = arr2[j];
        j += 1;
        k += 1;
    }
    ret
}

/// # NOTE
/// make sure you have use [`crate::utils::gen_name_list`] in the same namespace.
/// # Examples
/// ```
/// assert_eq!(gen_reg_name_list("s"; 0, 2; "t"; 1, 3), ["s0", "s1", "s2", "t1", "t2", "t3"])
/// ```
#[macro_export]
macro_rules! gen_reg_name_list {
    ($base:literal, $begin: literal, $end: literal; $($rest:tt)*) => {
        crate::utils::concat_arrays(gen_name_list!($base; $begin, $end), gen_reg_name_list!($($rest)*))
    };

    ($base:literal, $begin: literal, $end: literal) => {
        gen_name_list!($base; $begin, $end)
    };

    ($name:literal; $($rest:tt)*) => {
        crate::utils::concat_arrays([$name], gen_reg_name_list!($($rest)*))
    };

    ($name:literal) => {
        [$name]
    };
}

pub fn concat_to_u64(high: u32, low: u32) -> u64 {
    (high as u64).wrapping_shl(32) | (low as u64)
}

pub trait TruncateFrom<T>: Sized {
    fn truncate_from(value: T) -> Self;
}

pub trait TruncateTo<T>: Sized {
    fn truncate_to(self) -> T;
}

impl<T, U> TruncateTo<U> for T
where
    U: TruncateFrom<T>,
{
    #[inline]
    fn truncate_to(self) -> U {
        U::truncate_from(self)
    }
}

pub trait TruncateToBits<T>: Sized {
    fn truncate_to_bits(self, bits: u32) -> Self;
}

macro_rules! impl_truncate_from {
    ($from:ty, $to:ty) => {
        impl TruncateFrom<$from> for $to {
            fn truncate_from(val: $from) -> Self {
                val as $to
            }
        }
    };
}

impl_truncate_from!(u8, u8);
impl_truncate_from!(u8, u16);
impl_truncate_from!(u8, u32);
impl_truncate_from!(u8, u64);
impl_truncate_from!(u8, u128);

impl_truncate_from!(u16, u8);
impl_truncate_from!(u16, u16);
impl_truncate_from!(u16, u32);
impl_truncate_from!(u16, u64);
impl_truncate_from!(u16, u128);

impl_truncate_from!(u32, u8);
impl_truncate_from!(u32, u16);
impl_truncate_from!(u32, u32);
impl_truncate_from!(u32, u64);
impl_truncate_from!(u32, u128);

impl_truncate_from!(u64, u8);
impl_truncate_from!(u64, u16);
impl_truncate_from!(u64, u32);
impl_truncate_from!(u64, u64);
impl_truncate_from!(u64, u128);

impl_truncate_from!(u128, u8);
impl_truncate_from!(u128, u16);
impl_truncate_from!(u128, u32);
impl_truncate_from!(u128, u64);
impl_truncate_from!(u128, u128);

macro_rules! impl_truncate_to_bits {
    ($T:ty) => {
        impl TruncateToBits<$T> for $T {
            fn truncate_to_bits(self, bits: u32) -> Self {
                if bits >= 64 {
                    self
                } else {
                    ((self as u64) & ((1u64.wrapping_shl(bits)) - 1)) as Self
                }
            }
        }
    };
}

impl_truncate_to_bits!(u8);
impl_truncate_to_bits!(u16);
impl_truncate_to_bits!(u32);
impl_truncate_to_bits!(u64);

pub trait UnsignedInteger:
    Copy
    + Sized
    + From<u8>
    + Into<u64>
    + Default
    + std::fmt::LowerHex
    + std::fmt::Display
    // 算术运算符
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    // 位运算符
    + BitAnd<Output = Self>
    + BitAndAssign
    + BitOr<Output = Self>
    + BitOrAssign
    + BitXor<Output = Self>
    + BitXorAssign
    + Not<Output = Self>
    // 位移操作（右/左移可能和 u32、usize 混用）
    + Shl<u32, Output = Self>
    + ShlAssign<u32>
    + Shr<u32, Output = Self>
    + ShrAssign<u32>
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Debug
    + Display
    + TruncateFrom<u8>
    + TruncateFrom<u16>
    + TruncateFrom<u32>
    + TruncateFrom<u64>
    + TruncateFrom<u128>
    + TruncateTo<u8>
    + TruncateTo<u16>
    + TruncateTo<u32>
    + TruncateTo<u64>
    + TruncateTo<u128>
    + num_traits::WrappingAdd
    + num_traits::WrappingSub
    + num_traits::WrappingMul
{
    const MAX: Self;
    const MIN: Self;

    const BITS: usize;

    type AtomicType;
    type DoubleWidenType;

    fn mask_range(self, l: u32, r: u32) -> Self {
        debug_assert!(l <= r && (r as usize) < Self::BITS);
        self & make_mask(l as usize, r as usize).truncate_to()
    }

    fn extract_range(self, l: u32, r: u32) -> Self {
        debug_assert!(l <= r && (r as usize) < Self::BITS);
        (self >> l) & (BIT_ONES_ARRAY[(r - l + 1) as usize]).truncate_to()
    }
}

impl UnsignedInteger for u8 {
    const MAX: u8 = u8::MAX;
    const MIN: u8 = u8::MIN;
    const BITS: usize = 8;
    type AtomicType = std::sync::atomic::AtomicU8;
    type DoubleWidenType = u16;
}
impl UnsignedInteger for u16 {
    const MAX: u16 = u16::MAX;
    const MIN: u16 = u16::MIN;
    const BITS: usize = 16;
    type AtomicType = std::sync::atomic::AtomicU16;
    type DoubleWidenType = u32;
}
impl UnsignedInteger for u32 {
    const MAX: u32 = u32::MAX;
    const MIN: u32 = u32::MIN;
    const BITS: usize = 32;
    type AtomicType = std::sync::atomic::AtomicU32;
    type DoubleWidenType = u64;
}
impl UnsignedInteger for u64 {
    const MAX: u64 = u64::MAX;
    const MIN: u64 = u64::MIN;
    const BITS: usize = 64;
    type AtomicType = std::sync::atomic::AtomicU64;
    type DoubleWidenType = u128;
}

pub trait SignedInteger:
    Copy
    + Sized
    + From<i8>
    + Into<i64>
    // 算术运算符
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    // 位运算符
    + BitAnd<Output = Self>
    + BitAndAssign
    + BitOr<Output = Self>
    + BitOrAssign
    + BitXor<Output = Self>
    + BitXorAssign
    + Not<Output = Self>
    // 位移操作
    + Shl<u32, Output = Self>
    + ShlAssign<u32>
    + Shr<u32, Output = Self>
    + ShrAssign<u32>
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Debug
    + Display
{
    const MAX: Self;
    const MIN: Self;

    const BITS: usize;

    type DoubleWidenType;
}

impl SignedInteger for i8 {
    const MAX: i8 = i8::MAX;
    const MIN: i8 = i8::MIN;
    const BITS: usize = 8;
    type DoubleWidenType = i16;
}

impl SignedInteger for i16 {
    const MAX: i16 = i16::MAX;
    const MIN: i16 = i16::MIN;
    const BITS: usize = 16;
    type DoubleWidenType = i32;
}

impl SignedInteger for i32 {
    const MAX: i32 = i32::MAX;
    const MIN: i32 = i32::MIN;
    const BITS: usize = 32;
    type DoubleWidenType = i64;
}

impl SignedInteger for i64 {
    const MAX: i64 = i64::MAX;
    const MIN: i64 = i64::MIN;
    const BITS: usize = 64;
    type DoubleWidenType = i128;
}

pub trait WordTrait: UnsignedInteger + Into<u128> {
    type SignedType: SignedInteger + Into<i128>;
    fn sign_extend_to_wordtype(self) -> WordType;
}

impl WordTrait for u32 {
    type SignedType = i32;

    #[cfg(feature = "riscv32")]
    fn sign_extend_to_wordtype(self) -> WordType {
        self
    }

    #[cfg(feature = "riscv64")]
    fn sign_extend_to_wordtype(self) -> WordType {
        sign_extend_u32(self)
    }
}

#[cfg(feature = "riscv64")]
impl WordTrait for u64 {
    type SignedType = i64;

    fn sign_extend_to_wordtype(self) -> WordType {
        self
    }
}

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

#[allow(unused)]
pub fn set_bit<T>(data: &mut T, idx: u32)
where
    T: BitOrAssign + From<u8> + Shl<u32, Output = T> + Copy,
{
    *data |= T::from(1u8) << idx;
}

#[allow(unused)]
pub fn clear_bit<T>(data: &mut T, idx: u32)
where
    T: BitAndAssign + From<u8> + Shl<u32, Output = T> + Copy + Not<Output = T>,
{
    *data &= !(T::from(1u8) << idx);
}

#[allow(unused)]
pub fn read_bit<T>(data: &T, idx: u32) -> bool
where
    T: BitAnd<Output = T> + From<u8> + Shl<u32, Output = T> + Copy + Not<Output = T> + Eq,
{
    (*data & (T::from(1u8) << idx)) != T::from(0u8)
}

const fn gen_ones_array<const LEN: usize>() -> [WordType; LEN + 1] {
    let mut arr: [WordType; LEN + 1] = [0; LEN + 1];
    let mut i = 0;
    while i < LEN {
        arr[i + 1] = arr[i] | ((1 as WordType) << i);
        i += 1;
    }

    arr
}

pub const BIT_ONES_ARRAY: [WordType; XLEN + 1] = gen_ones_array::<XLEN>();

#[inline]
pub(crate) const fn make_mask(left: usize, right: usize) -> WordType {
    let width = right - left + 1;
    BIT_ONES_ARRAY[width] << left
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sign_extend() {
        assert_eq!(sign_extend(0x123, 12), 0x123);
        assert_eq!(sign_extend(0x7FF, 12), 0x7FF);
        assert_eq!(sign_extend(0xFFF, 12), !0 as WordType);
        assert_eq!(sign_extend(0xF0F, 12), (!0 - 0xF0) as WordType);
    }

    #[test]
    fn test_negative_of() {
        assert_eq!(negative_of(0 as WordType), 0 as WordType);
        assert_eq!(negative_of(1 as WordType), (!0) as WordType);
        assert_eq!(negative_of(2 as WordType), (!0 - 1) as WordType);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(u8::truncate_from(0x1234u32), 0x34u8);
        assert_eq!(0x1234567812345678u64.truncate_to_bits(32), 0x12345678u64);
        assert_eq!(
            0x1234567812345678u64.truncate_to_bits(64),
            0x1234567812345678u64
        );
    }
}
