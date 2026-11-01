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
pub struct RVVExtCsrRegfile {
    vstart:         CsrItem,
    vxsat:          CsrItem,
    vxrm:           CsrItem,
    vl:             CsrItem,
    vtype:          CsrItem,
    vlenb:          CsrItem,
}

impl SubExtCsrRegfile for RVVExtCsrRegfile {
    fn register(regfile: &mut CsrRegfile) {
        Self::macro_register(regfile);
        Vcsr::register(regfile);
    }
}

pub struct Vcsr {}

impl CsrValidator<Vcsr> for Vcsr {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        let vxsat = regfile.rvv_csr.vxsat.mask & (new_value & 0b1);
        let vxrm = regfile.rvv_csr.vxrm.mask & ((new_value >> 1) & 0b11);
        (vxrm << 1) | vxsat
    }
}

impl NamedCsr for Vcsr {
    const INDEX: CsrIndex = CsrIndex::Direct(0x00F);
    const EXTENSION: Extension = Extension::V;

    fn register(regfile: &mut CsrRegfile) {
        registe_csr(regfile, Self::INDEX, (Self::read, Self::write));
    }

    #[inline(always)]
    fn read(regfile: &mut CsrRegfile) -> CsrDataType {
        let vxsat = Vxsat::read(regfile) & 0b1;
        let vxrm = Vxrm::read(regfile) & 0b11;
        (vxrm << 1) | vxsat
    }

    #[inline(always)]
    fn write(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        Vxsat::write(regfile, new_value & 0b1);
        Vxrm::write(regfile, (new_value >> 1) & 0b11);
    }

    #[inline(always)]
    fn write_directly(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        Vxsat::write_directly(regfile, new_value & 0b1);
        Vxrm::write_directly(regfile, (new_value >> 1) & 0b11);
    }
}

gen_csr_bitfield!(
    Vcsr, rvv_csr, vxsat, [
        0, 1, vxsat; // Fixed-point accrued saturation flag
        1, 2, vxrm;  // Fixed-point rounding mode
    ]
);

gen_csr_regfile! {
    RVVExtCsrRegfile {
        Vstart, Extension::V, rvv_csr, vstart, CsrIndex::Direct(0x008), [
            0, XLEN, vstart;
        ];

        Vxsat, Extension::V, rvv_csr, vxsat, CsrIndex::Direct(0x009), [
            0, 1, vxsat;
        ];

        Vxrm, Extension::V, rvv_csr, vxrm, CsrIndex::Direct(0x00A), [
            0, 2, vxrm;
        ];

        Vl, Extension::V, rvv_csr, vl, CsrIndex::Direct(0xC20), [
            0, XLEN, vl;
        ];

        Vtype, Extension::V, rvv_csr, vtype, CsrIndex::Direct(0xC21), [
            0, 3, vlmul;
            3, 3, vsew;
            6, 1, vta;
            7, 1, vma;
            -1, 1, vill;
        ];

        Vlenb, Extension::V, rvv_csr, vlenb, CsrIndex::Direct(0xC22), [
            0, XLEN, vlenb;
        ];
    }
}
