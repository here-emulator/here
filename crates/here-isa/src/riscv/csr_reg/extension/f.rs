use here_types::{BIT_ONES_ARRAY, XLEN};

use crate::riscv::{
    csr_reg::{
        extension::registe_csr,
        regfile::{
            CsrDataType, CsrIndex, CsrItem, CsrRegfile, CsrValidator, NamedCsr, SubExtCsrRegfile,
        },
    },
    isa_builder::Extension,
};

#[rustfmt::skip]
#[derive(Default)]
pub struct RVFExtCsrRegfile {
    fflags:         CsrItem,
    frm:            CsrItem,
}

impl SubExtCsrRegfile for RVFExtCsrRegfile {
    fn register(regfile: &mut CsrRegfile) {
        Self::macro_register(regfile);
        Fcsr::register(regfile);
    }
}

pub struct Fcsr {}

impl CsrValidator<Fcsr> for Fcsr {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        let fflags = regfile.rvf_csr.fflags.mask & (new_value & 0b11111);
        let frm = regfile.rvf_csr.frm.mask & ((new_value >> 5) & 0b111);
        (frm << 5) | fflags
    }
}

impl NamedCsr for Fcsr {
    const INDEX: CsrIndex = CsrIndex::Direct(0x003);
    const EXTENSION: Extension = Extension::F;

    fn register(regfile: &mut CsrRegfile) {
        registe_csr(regfile, Self::INDEX, (Self::read, Self::write));
    }

    #[inline(always)]
    fn read(regfile: &mut CsrRegfile) -> CsrDataType {
        let fflags = Fflags::read(regfile) & 0b11111;
        let frm = Frm::read(regfile) & 0b111;
        (frm << 5) | fflags
    }

    #[inline(always)]
    fn write(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        Fflags::write(regfile, new_value & 0b11111);
        Frm::write(regfile, (new_value >> 5) & 0b111);
    }

    #[inline(always)]
    fn write_directly(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        Fflags::write_directly(regfile, new_value & 0b11111);
        Frm::write_directly(regfile, (new_value >> 5) & 0b111);
    }
}

gen_csr_bitfield!(
    Fcsr, rvf_csr, fflags, [
        0, 5, fflags;
        0, 1, nx;
        1, 1, uf;
        2, 1, of;
        3, 1, dz;
        4, 1, nv;
        5, 3, rm;
    ]
);

gen_csr_regfile! {
    RVFExtCsrRegfile {
        Fflags, Extension::F, rvf_csr, fflags, CsrIndex::Direct(0x001), [
            0, 5, fflags;
            0, 1, nx;
            1, 1, uf;
            2, 1, of;
            3, 1, dz;
            4, 1, nv;
        ];

        Frm, Extension::F, rvf_csr, frm, CsrIndex::Direct(0x002), [
            0, 3, frm;
        ];
    }
}
