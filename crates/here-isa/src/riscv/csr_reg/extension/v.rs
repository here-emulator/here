use here_types::{BIT_ONES_ARRAY, XLEN};

use crate::riscv::{
    csr_reg::{
        extension::registe_csr,
        regfile::{
            CsrDataType, CsrIndex, CsrItem, CsrRegfile, CsrValidator, NamedCsr, SubExtCsrRegfile,
        },
    },
    vector::VLEN_BYTE,
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
        registe_csr(regfile, Self::INDEX, (Self::read, Self::write), None);
    }

    #[inline(always)]
    fn read(regfile: &CsrRegfile) -> CsrDataType {
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
            0, 3, vlmul, RO;
            3, 3, vsew, RO;
            6, 1, vta, RO;
            7, 1, vma, RO;
            -1, 1, vill, RO;
        ];

        Vlenb, Extension::V, rvv_csr, vlenb, CsrIndex::Direct(0xC22), [
            0, XLEN, vlenb;
        ];
    }
}

impl Vtype {
    /// If new vtype is supported return Some(vl). Otherwise return None.
    pub fn vsetvl(regfile: &mut CsrRegfile, vtype: CsrDataType) -> Option<CsrDataType> {
        let new_vlmul = vtype & 0b111;
        let new_vsew = (vtype >> 3) & 0b111;

        let lmul_legal = new_vlmul != 0b100;
        let vsew_legal = new_vsew < 0b100;
        let illegal = !(lmul_legal & vsew_legal);
        Self::write_directly(regfile, vtype | ((illegal as CsrDataType) << (XLEN - 1)));

        if illegal {
            None
        } else {
            let part_vl = (VLEN_BYTE as CsrDataType) >> new_vsew;
            let vl = match new_vlmul {
                0b000 => part_vl,      // 1
                0b001 => part_vl << 1, // 2
                0b010 => part_vl << 2, // 4
                0b011 => part_vl << 3, // 8
                0b111 => part_vl >> 1, // 1 / 2
                0b110 => part_vl >> 2, // 1 / 4
                0b101 => part_vl >> 4, // 1 / 8
                _ => 0,
            };
            Some(vl)
        }
    }
}
