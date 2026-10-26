#![cfg(test)]

use std::{cell::UnsafeCell, fmt::Debug, rc::Rc};

use crate::{
    config::arch_config::WordType, device::mmio::MemoryMapIO, ram::Ram, utils::UnsignedInteger,
};

use super::{
    Vector,
    types::{Vlmul, Vsew},
};

pub(super) struct VectorBuilder {
    vector: Vector,
    mmio: MemoryMapIO,
}

impl VectorBuilder {
    pub(super) fn new() -> Self {
        let ram = Rc::new(UnsafeCell::new(Ram::new()));
        let mmio = MemoryMapIO::from_ram(ram);
        Self {
            vector: Vector::new(),
            mmio,
        }
    }

    pub(super) fn config(
        mut self,
        vlmul: Vlmul,
        vsew: Vsew,
        mask_agnostic: bool,
        tail_agnostic: bool,
        vl: u16,
    ) -> Self {
        self.vector
            .set_config((vlmul, vsew, tail_agnostic, mask_agnostic, vl));
        self
    }

    pub(super) fn reg<T>(mut self, lmul: u8, idx: u8, value: &[T]) -> Self {
        self.vector.write_as_type(lmul, idx, value);
        self
    }

    pub(super) fn mem<T>(mut self, addr: WordType, value: T) -> Self
    where
        T: UnsignedInteger,
    {
        self.mmio.write_by_type(addr, value).unwrap();
        self
    }

    pub(super) fn mem_range<It, F, T>(mut self, indexes: It, mut f: F) -> Self
    where
        It: Iterator<Item = usize>,
        F: FnMut(usize) -> (WordType, T),
        T: UnsignedInteger,
    {
        for i in indexes {
            let (addr, value) = f(i);
            self.mmio.write_by_type(addr, value).unwrap();
        }
        self
    }

    pub(super) fn build(self) -> (Vector, MemoryMapIO) {
        (self.vector, self.mmio)
    }
}

pub(super) struct VectorChecker<'a> {
    pub(super) vector: &'a mut Vector,
    pub(super) mmio: &'a mut MemoryMapIO,
}

impl<'a> VectorChecker<'a> {
    pub(super) fn new(vector: &'a mut Vector, mmio: &'a mut MemoryMapIO) -> Self {
        Self { vector, mmio }
    }

    pub(super) fn reg<T>(self, lmul: u8, idx: u8, value: &[T]) -> Self
    where
        T: Eq + Debug,
    {
        let reg_val = self
            .vector
            .vector_regfile
            .read_as_type::<T>(lmul, idx)
            .unwrap();
        assert_eq!(reg_val.len(), value.len());
        for i in 0..reg_val.len() {
            assert_eq!(
                reg_val[i], value[i],
                "Vector register #{idx} [{i}] incorrect"
            );
        }
        self
    }

    pub(super) fn mem<T>(self, addr: WordType, value: T) -> Self
    where
        T: UnsignedInteger + Debug,
    {
        let got = self.mmio.read_by_type::<T>(addr).unwrap();
        assert_eq!(got, value, "Memory value incorrect at pos {}", addr);
        self
    }

    pub(super) fn customized<F>(self, f: F) -> Self
    where
        F: FnOnce(Self) -> Self,
    {
        f(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::isa::riscv) struct TestOpParameter {
    x1: Option<WordType>,
    vs1: Option<u8>,
    vs2: Option<u8>,
    v0: u8,
    vd: Option<u8>,
    src_eew: Option<Vsew>,
    dst_eew: Option<Vsew>,
    enable_mask: bool,
}

impl Default for TestOpParameter {
    fn default() -> Self {
        Self {
            x1: None,
            vs1: None,
            vs2: None,
            v0: 0,
            vd: None,
            src_eew: None,
            dst_eew: None,
            enable_mask: false,
        }
    }
}

impl TestOpParameter {
    fn from_parts(x1: Option<WordType>, vs1: Option<u8>, vs2: Option<u8>, vd: u8) -> Self {
        Self {
            x1,
            vs1,
            vs2,
            vd: Some(vd),
            ..Default::default()
        }
    }

    pub(super) fn new_vv(vs1: u8, vs2: u8, vd: u8) -> Self {
        Self::from_parts(None, Some(vs1), Some(vs2), vd)
    }

    pub(super) fn new_vx(x1: WordType, vs2: u8, vd: u8) -> Self {
        Self::from_parts(Some(x1), None, Some(vs2), vd)
    }

    pub(super) fn new_v(vs2: u8, vd: u8, src_eew: Vsew, dst_eew: Vsew) -> Self {
        Self {
            vs2: Some(vs2),
            vd: Some(vd),
            src_eew: Some(src_eew),
            dst_eew: Some(dst_eew),
            ..Default::default()
        }
    }

    pub(super) fn new_vvm(vs1: u8, vs2: u8, vd: u8) -> Self {
        Self::from_parts(None, Some(vs1), Some(vs2), vd)
    }

    pub(super) fn new_vxm(x1: WordType, vs2: u8, vd: u8) -> Self {
        Self::from_parts(Some(x1), None, Some(vs2), vd)
    }

    pub(super) fn new_mask_vv(vs1: u8, vs2: u8, vd: u8) -> Self {
        Self::from_parts(None, Some(vs1), Some(vs2), vd)
    }

    pub(super) fn new_mask_vx(x1: WordType, vs2: u8, vd: u8) -> Self {
        Self::from_parts(Some(x1), None, Some(vs2), vd)
    }

    pub(super) fn new_mask_vvm(vs1: u8, vs2: u8, vd: u8) -> Self {
        Self::from_parts(None, Some(vs1), Some(vs2), vd)
    }

    pub(super) fn new_mask_vxm(x1: WordType, vs2: u8, vd: u8) -> Self {
        Self::from_parts(Some(x1), None, Some(vs2), vd)
    }

    pub(super) fn x1(&self) -> WordType {
        self.x1.expect("x1 is required by this vector op test")
    }

    pub(super) fn vs1(&self) -> u8 {
        self.vs1.expect("vs1 is required by this vector op test")
    }

    pub(super) fn vs2(&self) -> u8 {
        self.vs2.expect("vs2 is required by this vector op test")
    }

    pub(super) fn v0(&self) -> u8 {
        self.v0
    }

    pub(super) fn vd(&self) -> u8 {
        self.vd.expect("vd is required by this vector op test")
    }

    pub(super) fn src_eew(&self) -> Vsew {
        self.src_eew
            .expect("src_eew is required by this vector op test")
    }

    pub(super) fn dst_eew(&self) -> Vsew {
        self.dst_eew
            .expect("dst_eew is required by this vector op test")
    }

    pub(super) fn enable_mask(&self) -> bool {
        self.enable_mask
    }

    pub(super) fn with_v0(mut self, v0: u8) -> Self {
        self.v0 = v0;
        self
    }

    pub(super) fn with_enable_mask(mut self, enable_mask: bool) -> Self {
        self.enable_mask = enable_mask;
        self
    }
}
