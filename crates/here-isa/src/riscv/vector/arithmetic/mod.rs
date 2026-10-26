use crate::{
    config::arch_config::WordType,
    isa::riscv::{
        instruction::exec_function::ExecUnaryTrait,
        trap::Exception,
        vector::{
            VLEN_BYTE, VecOpMask, Vector,
            types::{VGFRef, VGFRefMut, Vlmul, Vsew},
        },
    },
    utils::sign_extend,
};

pub(in crate::isa::riscv) mod integer_impl;
pub(in crate::isa::riscv) use integer_impl::*;

pub(in crate::isa::riscv) mod fix_point_impl;
pub(in crate::isa::riscv) use fix_point_impl::*;

#[inline]
fn vector_register_group_overlaps(lhs: u8, lhs_lmul: u8, rhs: u8, rhs_lmul: u8) -> bool {
    lhs < rhs.saturating_add(rhs_lmul) && rhs < lhs.saturating_add(lhs_lmul)
}

fn lmul_ratio(vlmul: Vlmul) -> (u8, u8) {
    match vlmul {
        Vlmul::M1 => (1, 1),
        Vlmul::M2 => (2, 1),
        Vlmul::M4 => (4, 1),
        Vlmul::M8 => (8, 1),
        Vlmul::Mf2 => (1, 2),
        Vlmul::Mf4 => (1, 4),
        Vlmul::Mf8 => (1, 8),
    }
}

fn physical_lmul_from_ratio(numerator: u8, denominator: u8) -> Result<u8, Exception> {
    let physical_lmul = numerator.div_ceil(denominator).max(1);
    if physical_lmul > 8 {
        Err(Exception::IllegalInstruction)
    } else {
        Ok(physical_lmul)
    }
}

fn gather_ei16_index_lmul(vlmul: Vlmul, sew: u8) -> Result<u8, Exception> {
    let (lmul_num, lmul_den) = lmul_ratio(vlmul);
    physical_lmul_from_ratio(lmul_num * Vsew::E16.into_byte_width(), lmul_den * sew)
}

fn narrowing_source_lmul(vlmul: Vlmul) -> Result<u8, Exception> {
    let (lmul_num, lmul_den) = lmul_ratio(vlmul);
    physical_lmul_from_ratio(lmul_num * 2, lmul_den)
}

impl Vector {
    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_vv<OpIVV>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVV: VectorOpIntegerVV,
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
        OpIVV::exec(&vs1_ref, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_gather_vv<OpIVV>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVV: VectorOpIntegerGatherVV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        if vector_register_group_overlaps(vd, lmul, vs1, lmul)
            || vector_register_group_overlaps(vd, lmul, vs2, lmul)
        {
            return Err(Exception::IllegalInstruction);
        }
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
        OpIVV::exec(&vs1_ref, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_gather_ei16_vv<OpIVV>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVV: VectorOpIntegerGatherEI16VV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let index_lmul = gather_ei16_index_lmul(vlmul, sew)?;
        if vector_register_group_overlaps(vd, lmul, vs1, index_lmul)
            || vector_register_group_overlaps(vd, lmul, vs2, lmul)
        {
            return Err(Exception::IllegalInstruction);
        }
        let vs1_data = self.vector_regfile.get_ref(index_lmul, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs1_ref = VGFRef::new(&vs1_data, Vsew::E16.into_byte_width(), index_lmul, 1);
        let vs2_ref = VGFRef::new(&vs2_data, sew, lmul, 1);
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, sew, lmul, 1);
        OpIVV::exec(&vs1_ref, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_compress<Op>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        Op: VectorOpCompress,
    {
        if vstart != 0 {
            return Err(Exception::IllegalInstruction);
        }

        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        if vector_register_group_overlaps(vd, lmul, vs2, lmul)
            || vector_register_group_overlaps(vd, lmul, vs1, 1)
        {
            return Err(Exception::IllegalInstruction);
        }

        let vs1_data = self.vector_regfile.get_ref(1, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let selected_count = (0..self.config.vl as usize)
            .filter(|index| (vs1_data[index / 8] & (1 << (index % 8))) != 0)
            .count() as u16;
        let tail_mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            selected_count,
            false,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            0,
        );
        let vs1_ref = VGFRef::new(&vs1_data, Vsew::E8.into_byte_width(), 1, 1);
        let vs2_ref = VGFRef::new(&vs2_data, sew, lmul, 1);
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, sew, lmul, 1);
        Op::exec(
            &vs1_ref,
            &vs2_ref,
            &mut vd_ref,
            self.config.vl as usize,
            &tail_mask,
        )
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_bit_vv<OpIVV>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVV: VectorOpBitVV,
    {
        // Mask-register logical instructions always operate on one architectural
        // mask register. They ignore the current SEW/LMUL grouping, so use one
        // register and `u32` chunks only as an internal packed-bit transport.
        let sew = size_of::<u32>() as u8;
        let vs1_data = self.vector_regfile.get_ref(1, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(1, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs1_ref = VGFRef::new(&vs1_data, sew, 1, 1);
        let vs2_ref = VGFRef::new(&vs2_data, sew, 1, 1);
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(1, vd, 1)?, sew, 1, 1);
        OpIVV::exec(&vs1_ref, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_vx<OpIVX>(
        &mut self,
        x1: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVX: VectorOpIntegerVX,
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
        OpIVX::exec(x1, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_gather_vx<OpIVX>(
        &mut self,
        x1: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVX: VectorOpIntegerVX,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        if vector_register_group_overlaps(vd, lmul, vs2, lmul) {
            return Err(Exception::IllegalInstruction);
        }
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
        OpIVX::exec(x1, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_vvv<OpIVV>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVV: VectorOpIntegerVVV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let vs1_data = self.vector_regfile.get_ref(lmul, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let vd_data = self.vector_regfile.get_ref(lmul, 1, vd)?.to_vec();
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
        let old_vd_ref = VGFRef::new(&vd_data, sew, lmul, 1);
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, sew, lmul, 1);
        OpIVV::exec(&vs1_ref, &vs2_ref, &old_vd_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_vxv<OpIVX>(
        &mut self,
        x1: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVX: VectorOpIntegerVXV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let vd_data = self.vector_regfile.get_ref(lmul, 1, vd)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, sew, lmul, 1);
        let old_vd_ref = VGFRef::new(&vd_data, sew, lmul, 1);
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, sew, lmul, 1);
        OpIVX::exec(x1, &vs2_ref, &old_vd_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_slideup<OpIVX>(
        &mut self,
        offset: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVX: VectorOpIntegerVXV,
    {
        let lmul = self.config.vlmul.get_lmul();
        if vd < vs2.saturating_add(lmul) && vs2 < vd.saturating_add(lmul) {
            return Err(Exception::IllegalInstruction);
        }
        self.exec_integer_vxv::<OpIVX>(offset, vs2, vd, enable_mask, vstart)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_slidedown<OpIVX>(
        &mut self,
        offset: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVX: VectorOpIntegerVX,
    {
        let lmul = self.config.vlmul.get_lmul();
        if vector_register_group_overlaps(vd, lmul, vs2, lmul) {
            return Err(Exception::IllegalInstruction);
        }
        let (sew, lmul) = (self.config.vsew.into_byte_width(), lmul);
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
        OpIVX::exec(offset, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_widening_integer_vv<OpIVV>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVV: VectorOpWideningIntegerVV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (src_sew, src_lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let Some(dst_eew) = Vsew::from_byte_width(src_sew * 2) else {
            return Err(Exception::IllegalInstruction);
        };
        let dst_lmul = src_lmul * 2;
        if dst_lmul > 8 {
            return Err(Exception::IllegalInstruction);
        }

        let vs1_data = self.vector_regfile.get_ref(src_lmul, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(src_lmul, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs1_ref = VGFRef::new(&vs1_data, src_sew, src_lmul, 1);
        let vs2_ref = VGFRef::new(&vs2_data, src_sew, src_lmul, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut(dst_lmul, vd, 1)?,
            dst_eew.into_byte_width(),
            dst_lmul,
            1,
        );
        OpIVV::exec(&vs1_ref, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_widening_integer_vvv<OpIVV>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVV: VectorOpWideningIntegerVVV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (src_sew, src_lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let Some(dst_eew) = Vsew::from_byte_width(src_sew * 2) else {
            return Err(Exception::IllegalInstruction);
        };
        let dst_lmul = src_lmul * 2;
        if dst_lmul > 8 {
            return Err(Exception::IllegalInstruction);
        }

        let vs1_data = self.vector_regfile.get_ref(src_lmul, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(src_lmul, 1, vs2)?.to_vec();
        let vd_data = self.vector_regfile.get_ref(dst_lmul, 1, vd)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs1_ref = VGFRef::new(&vs1_data, src_sew, src_lmul, 1);
        let vs2_ref = VGFRef::new(&vs2_data, src_sew, src_lmul, 1);
        let old_vd_ref = VGFRef::new(&vd_data, dst_eew.into_byte_width(), dst_lmul, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut(dst_lmul, vd, 1)?,
            dst_eew.into_byte_width(),
            dst_lmul,
            1,
        );
        OpIVV::exec(&vs1_ref, &vs2_ref, &old_vd_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_widening_integer_vxv<OpIVX>(
        &mut self,
        x1: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVX: VectorOpWideningIntegerVXV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (src_sew, src_lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let Some(dst_eew) = Vsew::from_byte_width(src_sew * 2) else {
            return Err(Exception::IllegalInstruction);
        };
        let dst_lmul = src_lmul * 2;
        if dst_lmul > 8 {
            return Err(Exception::IllegalInstruction);
        }

        let vs2_data = self.vector_regfile.get_ref(src_lmul, 1, vs2)?.to_vec();
        let vd_data = self.vector_regfile.get_ref(dst_lmul, 1, vd)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, src_sew, src_lmul, 1);
        let old_vd_ref = VGFRef::new(&vd_data, dst_eew.into_byte_width(), dst_lmul, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut(dst_lmul, vd, 1)?,
            dst_eew.into_byte_width(),
            dst_lmul,
            1,
        );
        OpIVX::exec(x1, &vs2_ref, &old_vd_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_widening_integer_vx<OpIVX>(
        &mut self,
        x1: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVX: VectorOpWideningIntegerVX,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (src_sew, src_lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let Some(dst_eew) = Vsew::from_byte_width(src_sew * 2) else {
            return Err(Exception::IllegalInstruction);
        };
        let dst_lmul = src_lmul * 2;
        if dst_lmul > 8 {
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
        let vs2_ref = VGFRef::new(&vs2_data, src_sew, src_lmul, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut(dst_lmul, vd, 1)?,
            dst_eew.into_byte_width(),
            dst_lmul,
            1,
        );
        OpIVX::exec(x1, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_widening_integer_wv<OpIVV>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVV: VectorOpWideningIntegerWV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (src_sew, src_lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let Some(dst_eew) = Vsew::from_byte_width(src_sew * 2) else {
            return Err(Exception::IllegalInstruction);
        };
        let dst_lmul = src_lmul * 2;
        if dst_lmul > 8 {
            return Err(Exception::IllegalInstruction);
        }

        let vs1_data = self.vector_regfile.get_ref(src_lmul, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(dst_lmul, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs1_ref = VGFRef::new(&vs1_data, src_sew, src_lmul, 1);
        let vs2_ref = VGFRef::new(&vs2_data, dst_eew.into_byte_width(), dst_lmul, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut(dst_lmul, vd, 1)?,
            dst_eew.into_byte_width(),
            dst_lmul,
            1,
        );
        OpIVV::exec(&vs1_ref, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_widening_integer_wx<OpIVX>(
        &mut self,
        x1: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVX: VectorOpWideningIntegerWX,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (src_sew, src_lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let Some(dst_eew) = Vsew::from_byte_width(src_sew * 2) else {
            return Err(Exception::IllegalInstruction);
        };
        let dst_lmul = src_lmul * 2;
        if dst_lmul > 8 {
            return Err(Exception::IllegalInstruction);
        }

        let vs2_data = self.vector_regfile.get_ref(dst_lmul, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, dst_eew.into_byte_width(), dst_lmul, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut(dst_lmul, vd, 1)?,
            dst_eew.into_byte_width(),
            dst_lmul,
            1,
        );
        OpIVX::exec(x1, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_narrowing_wv<OpIVV>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVV: VectorOpIntegerNarrowingWV,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (dst_sew, src_sew) = (vsew.into_byte_width(), vsew.into_byte_width() * 2);
        let Some(src_eew) = Vsew::from_byte_width(src_sew) else {
            return Err(Exception::IllegalInstruction);
        };
        let lmul = vlmul.get_lmul();
        let src_lmul = narrowing_source_lmul(vlmul)?;
        if vector_register_group_overlaps(vd, lmul, vs2, src_lmul) {
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
        OpIVV::exec(&vs1_ref, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_narrowing_vx<OpIVX>(
        &mut self,
        x1: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVX: VectorOpIntegerNarrowingVX,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (dst_sew, src_sew) = (vsew.into_byte_width(), vsew.into_byte_width() * 2);
        let Some(src_eew) = Vsew::from_byte_width(src_sew) else {
            return Err(Exception::IllegalInstruction);
        };
        let lmul = vlmul.get_lmul();
        let src_lmul = narrowing_source_lmul(vlmul)?;
        if vector_register_group_overlaps(vd, lmul, vs2, src_lmul) {
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
        OpIVX::exec(x1, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_v<OpIV>(
        &mut self,
        vs2: u8,
        vd: u8,
        src_eew: Vsew,
        dst_eew: Vsew,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIV: VectorOpIntegerV,
    {
        let lmul = self.config.vlmul.get_lmul();
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, src_eew.into_byte_width(), lmul, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut(lmul, vd, 1)?,
            dst_eew.into_byte_width(),
            lmul,
            1,
        );
        OpIV::exec(&vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_v_ext<OpIV, const FACTOR: u8>(
        &mut self,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIV: VectorOpIntegerV,
    {
        let dst_eew = self.config.vsew;
        let Some(src_eew) = Vsew::from_byte_width(dst_eew.into_byte_width() / FACTOR) else {
            return Err(Exception::IllegalInstruction);
        };
        self.exec_integer_v::<OpIV>(vs2, vd, src_eew, dst_eew, enable_mask, vstart)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_scalar_move<OpIV, T>(
        &mut self,
        src: T,
        vd: u8,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIV: ExecUnaryTrait<Result<T, Exception>, T>,
        T: Copy + Default,
    {
        let lmul = self.config.vlmul.get_lmul();
        let Some(eew) = Vsew::from_byte_width(size_of::<T>() as u8) else {
            return Err(Exception::IllegalInstruction);
        };
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            false,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut(lmul, vd, 1)?,
            eew.into_byte_width(),
            lmul,
            1,
        );
        for (index, element) in vd_ref.iter_mut().enumerate() {
            mask.element_load(element, OpIV::exec(src)?, index);
        }

        Ok(())
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_move_scalar_to_element(
        &mut self,
        src: WordType,
        vd: u8,
        vstart: usize,
    ) -> Result<(), Exception> {
        if vstart != 0 || self.config.vl == 0 {
            return Ok(());
        }

        let sew = self.config.vsew.into_byte_width();
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(1, vd, 1)?, sew, 1, 1);
        match self.config.vsew {
            Vsew::E8 => vd_ref.set(0, src as u8),
            Vsew::E16 => vd_ref.set(0, src as u16),
            Vsew::E32 => vd_ref.set(0, src as u32),
            Vsew::E64 => vd_ref.set(0, src as u64),
        }
        Ok(())
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_move_element_to_scalar(
        &self,
        vs2: u8,
        vstart: usize,
    ) -> Result<Option<WordType>, Exception> {
        if vstart != 0 || self.config.vl == 0 {
            return Ok(None);
        }

        let sew = self.config.vsew.into_byte_width();
        let vs2_ref = VGFRef::new(self.vector_regfile.get_ref(1, 1, vs2)?, sew, 1, 1);
        let value = match self.config.vsew {
            Vsew::E8 => sign_extend(vs2_ref.get::<u8>(0) as WordType, 8),
            Vsew::E16 => sign_extend(vs2_ref.get::<u16>(0) as WordType, 16),
            Vsew::E32 => sign_extend(vs2_ref.get::<u32>(0) as WordType, 32),
            Vsew::E64 => vs2_ref.get::<u64>(0) as WordType,
        };
        Ok(Some(value))
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_whole_register_move<OpIV>(
        &mut self,
        vs2: u8,
        vd: u8,
        lmul: u8,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIV: VectorOpIntegerV,
    {
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            (VLEN_BYTE * lmul as usize / size_of::<u64>()) as u16,
            false,
            false,
            false,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, Vsew::E64.into_byte_width(), lmul, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut(lmul, vd, 1)?,
            Vsew::E64.into_byte_width(),
            lmul,
            1,
        );
        OpIV::exec(&vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_vvm<OpIVVM>(
        &mut self,
        vs1: u8,
        vs2: u8,
        v0: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVVM: VectorOpIntegerVVM,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let vs1_data = self.vector_regfile.get_ref(lmul, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let v0_data = self.vector_regfile.get_ref(1, 1, v0)?.to_vec();
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
        let v0_ref = VGFRef::new(&v0_data, Vsew::E8.into_byte_width(), 1, 1);
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, sew, lmul, 1);
        OpIVVM::exec(&vs1_ref, &vs2_ref, &v0_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_vxm<OpIVXM>(
        &mut self,
        x1: WordType,
        vs2: u8,
        v0: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIVXM: VectorOpIntegerVXM,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let v0_data = self.vector_regfile.get_ref(1, 1, v0)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, sew, lmul, 1);
        let v0_ref = VGFRef::new(&v0_data, Vsew::E8.into_byte_width(), 1, 1);
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, sew, lmul, 1);
        OpIVXM::exec(x1, &vs2_ref, &v0_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_mask_vv<OpIMVV>(
        &mut self,
        vs1: u8,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIMVV: VectorOpIntegerMaskVV,
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
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut::<u8>(1, vd, 1)?,
            Vsew::E8.into_byte_width(),
            1,
            1,
        );
        OpIMVV::exec(&vs1_ref, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_mask_vx<OpIMVX>(
        &mut self,
        x1: WordType,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIMVX: VectorOpIntegerMaskVX,
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
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut::<u8>(1, vd, 1)?,
            Vsew::E8.into_byte_width(),
            1,
            1,
        );
        OpIMVX::exec(x1, &vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_mask_vvm<OpIMVVM>(
        &mut self,
        vs1: u8,
        vs2: u8,
        v0: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIMVVM: VectorOpIntegerMaskVVM,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let vs1_data = self.vector_regfile.get_ref(lmul, 1, vs1)?.to_vec();
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let v0_data = self.vector_regfile.get_ref(1, 1, v0)?.to_vec();
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
        let v0_ref = VGFRef::new(&v0_data, Vsew::E8.into_byte_width(), 1, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut::<u8>(1, vd, 1)?,
            Vsew::E8.into_byte_width(),
            1,
            1,
        );
        OpIMVVM::exec(&vs1_ref, &vs2_ref, &v0_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_integer_mask_vxm<OpIMVXM>(
        &mut self,
        x1: WordType,
        vs2: u8,
        v0: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        OpIMVXM: VectorOpIntegerMaskVXM,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let vs2_data = self.vector_regfile.get_ref(lmul, 1, vs2)?.to_vec();
        let v0_data = self.vector_regfile.get_ref(1, 1, v0)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, sew, lmul, 1);
        let v0_ref = VGFRef::new(&v0_data, Vsew::E8.into_byte_width(), 1, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut::<u8>(1, vd, 1)?,
            Vsew::E8.into_byte_width(),
            1,
            1,
        );
        OpIMVXM::exec(x1, &vs2_ref, &v0_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_mask_to_x<Op>(
        &mut self,
        vs2: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<WordType, Exception>
    where
        Op: VectorOpMaskToX,
    {
        let vs2_data = self.vector_regfile.get_ref(1, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, Vsew::E8.into_byte_width(), 1, 1);
        Op::exec(&vs2_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_mask_unary<Op>(
        &mut self,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        Op: VectorOpMaskUnary,
    {
        if vd == vs2 {
            return Err(Exception::IllegalInstruction);
        }

        let vs2_data = self.vector_regfile.get_ref(1, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, Vsew::E8.into_byte_width(), 1, 1);
        let mut vd_ref = VGFRefMut::new(
            self.vector_regfile.get_mut::<u8>(1, vd, 1)?,
            Vsew::E8.into_byte_width(),
            1,
            1,
        );
        Op::exec(&vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_mask_to_vector<Op>(
        &mut self,
        vs2: u8,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        Op: VectorOpMaskToVector,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        if vector_register_group_overlaps(vd, lmul, vs2, 1) {
            return Err(Exception::IllegalInstruction);
        }

        let vs2_data = self.vector_regfile.get_ref(1, 1, vs2)?.to_vec();
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let vs2_ref = VGFRef::new(&vs2_data, Vsew::E8.into_byte_width(), 1, 1);
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, sew, lmul, 1);
        Op::exec(&vs2_ref, &mut vd_ref, &mask)
    }

    #[inline]
    pub(in crate::isa::riscv) fn exec_index<Op>(
        &mut self,
        vd: u8,
        enable_mask: bool,
        vstart: usize,
    ) -> Result<(), Exception>
    where
        Op: VectorOpIndex,
    {
        let (vlmul, vsew) = (self.config.vlmul, self.config.vsew);
        let (sew, lmul) = (vsew.into_byte_width(), vlmul.get_lmul());
        let mask = VecOpMask::new_with_start(
            &self.vector_regfile,
            self.config.vl,
            enable_mask,
            self.config.mask_agnostic,
            self.config.tail_agnostic,
            vstart,
        );
        let mut vd_ref = VGFRefMut::new(self.vector_regfile.get_mut(lmul, vd, 1)?, sew, lmul, 1);
        Op::exec(&mut vd_ref, &mask)
    }
}
