use here_types::{BIT_ONES_ARRAY, XLEN};

use crate::riscv::{
    csr_reg::{
        extension::registe_csr,
        regfile::{CsrDataType, CsrItem, CsrRegfile, CsrValidator, NamedCsr, SubExtCsrRegfile},
    },
    isa_builder::Extension,
};

#[rustfmt::skip]
#[derive(Default)]
pub struct RVBasicCsrRegfile {
    xstatus:        CsrItem,
    misa:           CsrItem,
    medeleg:        CsrItem,
    mideleg:        CsrItem,
    xie:            CsrItem,
    mtvec:          CsrItem,
    mcountinhibit:  CsrItem,
    mscratch:       CsrItem,
    mepc:           CsrItem,
    mcause:         CsrItem,
    mtval:          CsrItem,
    xip:            CsrItem,
    mcycle:         CsrItem,
    minstret:       CsrItem,
    mvendorid:      CsrItem,
    marchid:        CsrItem,
    mimpid:         CsrItem,
    mhartid:        CsrItem,
}

impl SubExtCsrRegfile for RVBasicCsrRegfile {
    fn register(regfile: &mut CsrRegfile) {
        Mstatus::register(regfile);
        Misa::register(regfile);
    }
}

pub struct Mstatus {}

impl Mstatus {
    fn validate_custom(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        todo!()
    }
}

impl CsrValidator<Mstatus> for Mstatus {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        let value = regfile.basic_csr.xstatus.mask & new_value;
        Self::validate_custom(regfile, value)
    }
}

impl NamedCsr for Mstatus {
    const INDEX: CsrIndex = CsrIndex::Direct(0x300);
    const EXTENSION: Extension = Extension::Zicsr;

    fn register(regfile: &mut CsrRegfile) {
        registe_csr(regfile, Self::INDEX, (Self::read, Self::write));
    }

    fn read(regfile: &mut CsrRegfile) -> u64 {
        regfile.basic_csr.xstatus.data
    }

    fn write(regfile: &mut CsrRegfile, new_value: u64) {
        let validated_value = Self::validate(regfile, new_value);
        regfile.basic_csr.xstatus.data = validated_value;
    }

    fn write_directly(regfile: &mut CsrRegfile, new_value: u64) {
        regfile.basic_csr.xstatus.data = new_value;
    }
}

impl Mstatus {
    // sie
    #[allow(non_upper_case_globals)]
    const xstatus_sie_offset: CsrDataType = 1;
    #[allow(non_upper_case_globals)]
    const xstatus_sie_mask: CsrDataType = BIT_ONES_ARRAY[1] << xstatus_sie_offset;

    pub fn get_sie(regfile: &mut CsrRegfile) -> CsrDataType {
        (regfile.basic_csr.xstatus.data & xstatus_sie_mask) >> xstatus_sie_offset
    }

    pub fn set_sie(regfile: &mut CsrRegfile, new_sie: CsrDataType) {
        regfile.basic_csr.xstatus.data &= !xstatus_sie_mask;
        let new_sie = (new_sie << xstatus_sie_offset) & xstatus_sie_mask;
        regfile.basic_csr.xstatus.data |= new_sie;
    }

    pub fn set_sie_readonly(regfile: &mut CsrRegfile) {
        regfile.basic_csr.xstatus.mask &= !xstatus_sie_mask;
    }

    // mie
    #[allow(non_upper_case_globals)]
    const xstatus_mie_offset: CsrDataType = 3;
    #[allow(non_upper_case_globals)]
    const xstatus_mie_mask: CsrDataType = BIT_ONES_ARRAY[1] << xstatus_mie_offset;

    pub fn get_mie(regfile: &mut CsrRegfile) -> CsrDataType {
        (regfile.basic_csr.xstatus.data & xstatus_mie_mask) >> xstatus_mie_offset
    }

    pub fn set_mie(regfile: &mut CsrRegfile, new_mie: CsrDataType) {
        regfile.basic_csr.xstatus.data &= !xstatus_mie_mask;
        let new_mie = (new_mie << xstatus_mie_offset) & xstatus_mie_mask;
        regfile.basic_csr.xstatus.data |= new_mie;
    }

    pub fn set_mie_readonly(regfile: &mut CsrRegfile) {
        regfile.basic_csr.xstatus.mask &= !xstatus_mie_mask;
    }
}

pub struct Misa {}

impl CsrValidator<Misa> for Misa {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        regfile.basic_csr.misa.mask & new_value
    }
}

impl NamedCsr for Misa {
    const INDEX: CsrIndex = CsrIndex::Direct(0x301);
    const EXTENSION: Extension = Extension::Zicsr;

    fn register(regfile: &mut CsrRegfile) {
        registe_csr(regfile, Self::INDEX, (Self::read, Self::write));
    }

    fn read(regfile: &mut CsrRegfile) -> u64 {
        regfile.basic_csr.misa.data
    }

    fn write(regfile: &mut CsrRegfile, new_value: u64) {
        let validated_value = Self::validate(regfile, new_value);
        regfile.basic_csr.misa.data = validated_value;
    }

    fn write_directly(regfile: &mut CsrRegfile, new_value: u64) {
        regfile.basic_csr.misa.data = new_value;
    }
}

impl Misa {
    #[allow(non_upper_case_globals)]
    const misa_extension_offset: usize = 0;
    #[allow(non_upper_case_globals)]
    const misa_extension_mask: CsrDataType = BIT_ONES_ARRAY[26] << misa_extension_offset;

    pub fn get_extension(regfile: &mut CsrRegfile) -> CsrDataType {
        (regfile.basic_csr.misa.data & misa_extension_mask) >> misa_extension_offset
    }

    pub fn set_extension(regfile: &mut CsrRegfile, new_extension: CsrDataType) {
        regfile.basic_csr.misa.data &= !misa_extension_mask;
        let new_extension = (new_extension << misa_extension_offset) & misa_extension_mask;
        regfile.basic_csr.misa.data |= new_extension;
    }

    pub fn set_extension_readonly(regfile: &mut CsrRegfile) {
        regfile.basic_csr.misa.mask &= !misa_extension_mask;
    }

    #[allow(non_upper_case_globals)]
    const misa_mxl_offset: usize = XLEN - 2;
    #[allow(non_upper_case_globals)]
    const misa_mxl_mask: CsrDataType = BIT_ONES_ARRAY[2] << misa_mxl_offset;

    pub fn get_mxl(regfile: &mut CsrRegfile) -> CsrDataType {
        (regfile.basic_csr.misa.data & misa_mxl_mask) >> misa_mxl_offset
    }

    pub fn set_mxl(regfile: &mut CsrRegfile, new_mxl: CsrDataType) {
        regfile.basic_csr.misa.data &= !misa_mxl_mask;
        let new_mxl = (new_mie << misa_mxl_offset) & misa_mxl_mask;
        regfile.basic_csr.misa.data |= new_mxl;
    }

    pub fn set_mxl_readonly(regfile: &mut CsrRegfile) {
        regfile.basic_csr.misa.mask &= !misa_mxl_mask;
    }
}

// gen_csr_regfile! {
//     RVBasicCsrRegfile {
//        Mstatus, Extension::Zicsr, basic_csr, xstatus, 0x300, @validate_custom [
//             1,  1, sie;
//             3,  1, mie;
//             5,  1, spie;
//             6,  1, ube;
//             7,  1, mpie;
//             8,  1, spp;
//             9,  2, vs;
//             11, 2, mpp;
//             13, 2, fs;
//             15, 2, xs, validate_readonly;
//             17, 1, mprv;
//             18, 1, sum;
//             19, 1, mxr;
//             20, 1, tvm;
//             21, 1, tw;
//             22, 1, tsr;
//             23, 1, spelp;
//             24, 1, sdt;
//             32, 2, uxl, validate_readonly;  // TODO: We don't support changing XLEN yet.
//             34, 2, sxl, validate_readonly;
//             36, 1, sbe;
//             37, 1, mbe;
//             38, 1, gva;
//             39, 1, mpv;
//             40, 1, wpri;
//             41, 1, mpelp;
//             42, 1, mdt;
//             -1, 1, sd;
//         ];

//         Misa, Extension::Zicsr, basic_csr, misa, 0x301, [
//             0, 25, extension;  // TODO: We don't support changing extension yet.
//             -2, 2, mxl;
//         ];

//         // ...
//     }
// }
