use std::collections::HashSet;

use here_types::{BIT_ONES_ARRAY, XLEN};
use log::error;

use crate::riscv::{
    csr_reg::regfile::{
        CsrDataType, CsrIndex, CsrRegfile, CsrValidator, NamedCsr, SubExtCsrRegfile,
    },
    isa_builder::Extension,
};

#[rustfmt::skip]
#[derive(Default)]
pub struct RVBasicCsrRegfile {
    xstatus:        CsrDataType,
    misa:           CsrDataType,
    medeleg:        CsrDataType,
    mideleg:        CsrDataType,
    xie:            CsrDataType,
    mtvec:          CsrDataType,
    mcountinhibit:  CsrDataType,
    mscratch:       CsrDataType,
    mepc:           CsrDataType,
    mcause:         CsrDataType,
    mtval:          CsrDataType,
    xip:            CsrDataType,
    mcycle:         CsrDataType,
    minstret:       CsrDataType,
    mvendorid:      CsrDataType,
    marchid:        CsrDataType,
    mimpid:         CsrDataType,
    mhartid:        CsrDataType,
}

impl SubExtCsrRegfile for RVBasicCsrRegfile {
    fn init(regfile: &mut CsrRegfile, extensions: &HashSet<Extension>) {
        MStatus::init(regfile, extensions);
    }

    fn register(regfile: &mut CsrRegfile) {
        MStatus::register(regfile);
        Misa::register(regfile);
    }
}

pub struct MStatus {}

impl CsrValidator<MStatus> for MStatus {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        todo!()
    }
}

impl NamedCsr for MStatus {
    const EXTENSION: Extension = Extension::Zicsr;
    const INDEX: CsrIndex = CsrIndex::Direct(0x300);

    fn register(regfile: &mut CsrRegfile) {
        if let CsrIndex::Direct(adrr) = Self::INDEX {
            regfile
                .map
                .insert(adrr, (Self::read, Self::write))
                .unwrap_or_else(|| {
                    let mes = format!(
                        "attempted to register two CSRs with the same address{}",
                        addr
                    );
                    error!("{}", mes);
                    panic!("{}", mes)
                });
        }
    }

    fn read(regfile: &mut CsrRegfile) -> u64 {
        regfile.basic_csr.xstatus
    }

    fn write(regfile: &mut CsrRegfile, new_value: u64) {
        let validated_value = Self::validate(new_value);
        regfile.basic_csr.xstatus = validated_value;
    }

    fn write_directly(regfile: &mut CsrRegfile, new_value: u64) {
        regfile.basic_csr.xstatus = new_value;
    }
}

impl MStatus {
    fn get_sie(regfile: &mut CsrRegfile) -> CsrDataType {
        (regfile.basic_csr.xstatus >> 1) & BIT_ONES_ARRAY[1];
    }

    fn get_mie(regfile: &mut CsrRegfile) -> CsrDataType {
        (regfile.basic_csr.xstatus >> 3) & BIT_ONES_ARRAY[1];
    }

    // ...

    fn set_sie(regfile: &mut CsrRegfile, new_sie: CsrDataType) {
        regfile.basic_csr.xstatus &= !(BIT_ONES_ARRAY[1] << 1);
        let new_sie = new_sie & BIT_ONES_ARRAY[1];
        regfile.basic_csr.xstatus |= new_sie << 1;
    }

    fn set_mie(regfile: &mut CsrRegfile, new_mie: CsrDataType) {
        regfile.basic_csr.xstatus &= !(BIT_ONES_ARRAY[1] << 1);
        let new_mie = new_mie & BIT_ONES_ARRAY[1];
        regfile.basic_csr.xstatus |= new_mie << 1;
    }
}

pub struct Misa {}

impl CsrValidator<Misa> for Misa {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        todo!()
    }
}

impl NamedCsr for Misa {
    const EXTENSION: Extension = Extension::Zicsr;
    const INDEX: CsrIndex = CsrIndex::Direct(0x301);

    fn register(regfile: &mut CsrRegfile) {
        if let CsrIndex::Direct(adrr) = Self::INDEX {
            regfile
                .map
                .insert(adrr, (Self::read, Self::write))
                .unwrap_or_else(|| {
                    let mes = format!(
                        "attempted to register two CSRs with the same address{}",
                        addr
                    );
                    error!("{}", mes);
                    panic!("{}", mes)
                });
        }
    }

    fn read(regfile: &mut CsrRegfile) -> u64 {
        regfile.basic_csr.misa
    }

    fn write(regfile: &mut CsrRegfile, new_value: u64) {
        let validated_value = Self::validate(new_value);
        regfile.basic_csr.misa = validated_value;
    }

    fn write_directly(regfile: &mut CsrRegfile, new_value: u64) {
        regfile.basic_csr.misa = new_value;
    }
}

impl Misa {
    fn get_extension(regfile: &mut CsrRegfile) -> CsrDataType {
        (regfile.basic_csr.misa >> 0) & BIT_ONES_ARRAY[26];
    }

    fn get_mxl(regfile: &mut CsrRegfile) -> CsrDataType {
        (regfile.basic_csr.misa >> (XLEN - 2)) & BIT_ONES_ARRAY[2];
    }

    // ...

    fn set_extension(regfile: &mut CsrRegfile, new_sie: CsrDataType) {
        regfile.basic_csr.misa &= !(BIT_ONES_ARRAY[26] << 0);
        let new_sie = new_sie & BIT_ONES_ARRAY[26];
        regfile.basic_csr.misa |= new_sie << 0;
    }

    fn set_mxl(regfile: &mut CsrRegfile, new_mie: CsrDataType) {
        regfile.basic_csr.misa &= !(BIT_ONES_ARRAY[2] << (XLEN - 2));
        let new_mie = new_mie & BIT_ONES_ARRAY[2];
        regfile.basic_csr.misa |= new_mie << (XLEN - 2);
    }
}
