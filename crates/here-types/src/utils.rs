use std::fmt::Debug;
use std::fmt::Display;
use std::ops::*;

use crate::config::arch_config::SignedWordType;
use crate::config::arch_config::WordType;
use crate::config::arch_config::XLEN;

// TODO: Too many things here. Do some cleanup.

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

#[inline]
pub fn shift_amount<T>(value: T) -> u32
where
    T: UnsignedInteger,
{
    let value: u64 = value.into();
    (value as u32) & ((T::BITS as u32) - 1)
}

#[inline]
pub fn as_signed_i128<T>(value: T) -> i128
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

#[inline]
pub fn from_signed_i128<T>(value: i128) -> T
where
    T: UnsignedInteger,
{
    T::truncate_from(value as u128)
}

#[inline]
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

pub const BIT_ONES_ARRAY: [WordType; XLEN + 1] = {
    let mut arr = [0; XLEN + 1];
    let mut i = 0;
    while i < XLEN {
        arr[i + 1] = arr[i] | ((1 as WordType) << i);
        i += 1;
    }

    arr
};

#[inline]
pub const fn make_mask(left: usize, right: usize) -> WordType {
    let width = right - left + 1;
    BIT_ONES_ARRAY[width] << left
}

#[macro_export]
/// Concatenates bit fields into a `u64` from high to low bits.
///
/// ```
/// # use here_types::pack_bits;
/// assert_eq!(
///     pack_bits! {
///         0b101, 3;
///         0b11, 4;
///         0, 3;
///     },
///     0b101_0011_000
/// );
/// ```
macro_rules! pack_bits {
    ($($value:expr, $bits:expr);* $(;)?) => {{
        #[cfg(debug_assertions)]
        {
            let mut total = 0usize;

            $(
                {
                    let v: u128 = $value as u128;
                    let b: usize = $bits as usize;
                    debug_assert!(v < (1u128 << b));

                    total += b;
                }
            )*

            debug_assert!(total <= 64 as usize);
        }

        pack_bits!(@build $($value, $bits;)*)
    }};

    (@build) => {
        0u64
    };

    (@build $value:expr, $bits:expr; $($rest:tt)*) => {{
        (($value as u64) << pack_bits!(@width $($rest)*))
            | pack_bits!(@build $($rest)*)
    }};

    (@width) => {
        0usize
    };

    (@width $value:expr, $bits:expr; $($rest:tt)*) => {
        ($bits as usize) + pack_bits!(@width $($rest)*)
    };
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
