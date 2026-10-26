use std::slice::{from_raw_parts, from_raw_parts_mut};

use crate::isa::riscv::vector::VLEN_BYTE;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FixedPointRoundingMode {
    RoundToNearestUp = 0x00,
    RoundToNearestEven = 0x01,
    RoundDown = 0x02,
    RoundToOdd = 0x03,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Vlmul {
    M1 = 0,
    M2 = 1,
    M4 = 2,
    M8 = 3,
    Mf8 = 5,
    Mf4 = 6,
    Mf2 = 7,
}

impl From<u8> for Vlmul {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::M1,
            1 => Self::M2,
            2 => Self::M4,
            3 => Self::M8,
            5 => Self::Mf8,
            6 => Self::Mf4,
            7 => Self::Mf2,
            _ => panic!(),
        }
    }
}

impl Vlmul {
    pub(crate) fn get_lmul(self) -> u8 {
        match self {
            Self::M1 => 1,
            Self::M2 => 2,
            Self::M4 => 4,
            Self::M8 => 8,
            Self::Mf8 => 1,
            Self::Mf4 => 1,
            Self::Mf2 => 1,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Vsew {
    E8,
    E16,
    E32,
    E64,
}

impl From<u8> for Vsew {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::E8,
            1 => Self::E16,
            2 => Self::E32,
            3 => Self::E64,
            _ => panic!(),
        }
    }
}

impl Vsew {
    pub(crate) fn from_byte_width(byte_width: u8) -> Option<Self> {
        match byte_width {
            1 => Some(Self::E8),
            2 => Some(Self::E16),
            4 => Some(Self::E32),
            8 => Some(Self::E64),
            _ => None,
        }
    }

    pub(crate) fn into_byte_width(self) -> u8 {
        match self {
            Self::E8 => 1,
            Self::E16 => 2,
            Self::E32 => 4,
            Self::E64 => 8,
        }
    }
}

pub(super) struct NFIELDS {
    nf: u8,
}

impl NFIELDS {
    pub(super) fn new(nf: u8) -> Self {
        Self { nf }
    }

    pub(super) fn encode(self) -> u8 {
        self.nf + 1
    }
}

pub(crate) struct RVVElemTy {
    ptr: *const u8,
    size: u8,
}

impl RVVElemTy {
    pub(crate) fn get<T>(&self) -> T {
        debug_assert!(self.size as usize == size_of::<T>());
        unsafe { (self.ptr as *const T).read() }
    }
}

pub(crate) struct RVVElemMutTy {
    ptr: *mut u8,
    size: u8,
}

impl RVVElemMutTy {
    pub(crate) fn set<T>(&self, val: T) {
        debug_assert!(self.size as usize == size_of::<T>());
        unsafe {
            (self.ptr as *mut T).write(val);
        }
    }

    pub(crate) fn get<T>(&self) -> T {
        debug_assert!(self.size as usize == size_of::<T>());
        unsafe { (self.ptr as *const T).read() }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct VectorConfig {
    pub(crate) vlmul: Vlmul,
    pub(crate) vsew: Vsew,
    pub(crate) tail_agnostic: bool,
    pub(crate) mask_agnostic: bool,
    pub(crate) fixed_point_accrued_saturation_flag: bool,
    pub(crate) fixed_point_rounding_mode: FixedPointRoundingMode,
    pub(crate) vl: u16, // [0, 10240 * 8 / 8]
}

impl VectorConfig {
    pub(crate) fn new() -> Self {
        Self {
            vlmul: Vlmul::M1,
            vsew: Vsew::E8,
            tail_agnostic: false,
            mask_agnostic: false,
            fixed_point_accrued_saturation_flag: false,
            fixed_point_rounding_mode: FixedPointRoundingMode::RoundToNearestUp,
            vl: 0,
        }
    }
}

pub(crate) struct VGFRef<'a> {
    value: &'a [u8],
    pub(super) sew: u8,
    lmul: u8,
    seg: u8,
}

impl<'a> VGFRef<'a> {
    pub(crate) fn new(val: &'a [u8], sew: u8, lmul: u8, seg: u8) -> Self {
        Self {
            value: val,
            lmul,
            sew,
            seg,
        }
    }

    pub(crate) fn get<T: Sized>(&self, index: usize) -> T
    where
        T: Clone,
    {
        self.as_slice::<T>()[index].clone()
    }

    pub(crate) fn as_slice<T: Sized>(&self) -> &'a [T] {
        assert!(self.sew as usize == size_of::<T>());
        unsafe {
            let p = self.value.as_ptr() as *const T;
            from_raw_parts(p, self.value.len() / self.sew as usize)
        }
    }

    pub(crate) fn iter(&'a self) -> VGFRefIterator<'a> {
        VGFRefIterator::new(self.value, self.sew, self.lmul, self.seg)
    }
}

pub(crate) struct VGFRefIterator<'a> {
    current_seg_index: usize,
    current_inner_index: usize,
    lmul: u8,
    sew: u8,
    seg: u8,
    value: &'a [u8],
    seg_length: usize,
}

impl<'a> VGFRefIterator<'a> {
    fn new(value: &'a [u8], sew: u8, lmul: u8, seg: u8) -> Self {
        Self {
            current_seg_index: 0,
            current_inner_index: 0,
            lmul,
            sew,
            seg,
            value,
            seg_length: lmul as usize * VLEN_BYTE,
        }
    }
}

impl<'a> Iterator for VGFRefIterator<'a> {
    type Item = RVVElemTy;
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.current_inner_index + self.current_seg_index * self.seg_length;
        let current_ptr = unsafe { self.value.as_ptr().add(index) };
        let is_last_seg = self.current_seg_index + 1 == self.seg as usize;
        let valid = self.current_inner_index < self.seg_length;
        if valid {
            if is_last_seg {
                self.current_seg_index = 0;
                self.current_inner_index += self.sew as usize;
            } else {
                self.current_seg_index += 1;
            }
            Some(RVVElemTy {
                ptr: current_ptr,
                size: self.sew,
            })
        } else {
            None
        }
    }
}

pub(crate) struct VGFRefMut<'a> {
    value: &'a mut [u8],
    pub(super) sew: u8,
    lmul: u8,
    seg: u8,
}

impl<'a> VGFRefMut<'a> {
    pub(crate) fn new(val: &'a mut [u8], sew: u8, lmul: u8, seg: u8) -> Self {
        assert!(val.len() % lmul as usize == 0);
        Self {
            value: val,
            sew,
            lmul,
            seg,
        }
    }

    pub(crate) fn get<T: Sized>(&self, index: usize) -> T
    where
        T: Clone,
    {
        self.as_slice::<T>()[index].clone()
    }

    pub(crate) fn as_slice<T: Sized>(&self) -> &[T] {
        assert!(self.sew as usize == size_of::<T>());
        unsafe {
            let p = self.value.as_ptr() as *const T;
            from_raw_parts(p, self.value.len() / self.sew as usize)
        }
    }

    pub(crate) fn set<T: Sized>(&mut self, index: usize, value: T)
    where
        T: Clone,
    {
        self.as_mut_slice::<T>()[index] = value
    }

    pub(crate) fn as_mut_slice<T: Sized>(&mut self) -> &mut [T] {
        assert!(self.sew as usize == size_of::<T>());
        unsafe {
            let p = self.value.as_mut_ptr() as *mut T;
            from_raw_parts_mut(p, self.value.len() / self.sew as usize)
        }
    }

    unsafe fn get_raw_mut(&mut self, index: usize) -> &'a mut u8 {
        unsafe {
            let p = self.value.as_mut_ptr();
            p.add(index).as_mut_unchecked()
        }
    }

    pub(crate) fn iter_mut(&mut self) -> VGFRefIteratorMut<'_> {
        VGFRefIteratorMut::new(self.lmul, self.sew, self.seg, self.value)
    }
}

pub(crate) struct VGFRefIteratorMut<'a> {
    current_seg_index: usize,
    current_inner_index: usize,
    lmul: u8,
    sew: u8,
    seg: u8,
    value: &'a mut [u8],
    seg_length: usize,
}

impl<'a> VGFRefIteratorMut<'a> {
    fn new(lmul: u8, sew: u8, seg: u8, value: &'a mut [u8]) -> Self {
        Self {
            current_seg_index: 0,
            current_inner_index: 0,
            lmul,
            sew,
            seg,
            value,
            seg_length: lmul as usize * VLEN_BYTE,
        }
    }
}

impl<'a> Iterator for VGFRefIteratorMut<'a> {
    type Item = RVVElemMutTy;
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.current_inner_index + self.current_seg_index * self.seg_length;
        let current_ptr = unsafe { self.value.as_mut_ptr().add(index) };
        let is_last_seg = self.current_seg_index + 1 == self.seg as usize;
        let valid = self.current_inner_index < self.seg_length;
        if valid {
            if is_last_seg {
                self.current_seg_index = 0;
                self.current_inner_index += self.sew as usize;
            } else {
                self.current_seg_index += 1;
            }
            Some(RVVElemMutTy {
                ptr: current_ptr,
                size: self.sew,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn vector_type_test() {
        let value: Vec<u8> = (0..(2 * VLEN_BYTE))
            .map(|i| if i % 2 == 0 { 0 } else { i as u8 })
            .collect();
        let v = VGFRef::new(value.as_slice(), Vsew::E16.into_byte_width(), 1, 1);
        assert_eq!(v.get::<u16>(3), (3 * 2 + 1) << 8);

        let mut value_mut: Vec<u8> = (0..(4 * VLEN_BYTE))
            .map(|i| if i % 2 == 0 { 0 } else { i as u8 })
            .collect();
        let mut v = VGFRefMut::new(
            value_mut.as_mut_slice(),
            Vsew::E16.into_byte_width(),
            Vlmul::M2.get_lmul(),
            1,
        );
        assert_eq!(v.get::<u16>(3), (3 * 2 + 1) << 8);
        v.set::<u16>(3, 3);
        assert_eq!(v.get::<u16>(3), 3);

        v.iter_mut()
            .enumerate()
            .for_each(|(i, val)| val.set::<u16>(i as u16 * 4));
        v.iter_mut()
            .enumerate()
            .for_each(|(i, val)| assert_eq!(val.get::<u16>(), i as u16 * 4));
    }

    #[test]
    #[should_panic(expected = "assertion failed: self.sew as usize == size_of::<T>()")]
    fn vector_type_test_unequal_sew() {
        let value: Vec<u8> = (0..128).map(|i| if i % 2 == 0 { 0 } else { i }).collect();
        let v = VGFRef::new(value.as_slice(), Vsew::E16.into_byte_width(), 1, 1);
        assert_eq!(v.get::<u32>(3), (3 * size_of::<u32>() as u32 + 1) << 8)
    }
}
