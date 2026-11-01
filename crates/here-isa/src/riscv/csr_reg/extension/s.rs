use here_types::{BIT_ONES_ARRAY, XLEN};

use crate::riscv::{
    csr_reg::{
        extension::{
            basic::{Mie, Mip, Mstatus},
            registe_csr,
        },
        regfile::{
            CsrDataType, CsrIndex, CsrItem, CsrRegfile, CsrValidator, NamedCsr, SubExtCsrRegfile,
        },
    },
    isa_builder::Extension,
};

#[rustfmt::skip]
#[derive(Default)]
pub struct RVSExtCsrRegfile {
    // S
    // @shadow sstatus:     CsrItem,
    // @shadow sie:         CsrItem,
    stvec:          CsrItem,
    scounteren:     CsrItem,
    sscratch:       CsrItem,
    sepc:           CsrItem,
    scause:         CsrItem,
    stval:          CsrItem,
    // @shadow sip:         CsrItem,
    satp:           CsrItem,
}

impl SubExtCsrRegfile for RVSExtCsrRegfile {
    fn register(regfile: &mut CsrRegfile) {
        Self::macro_register(regfile);
        Sstatus::register(regfile);
        Sie::register(regfile);
        Sip::register(regfile);
    }
}

#[derive(Debug)]
pub struct Sstatus {}

impl CsrValidator<Sstatus> for Sstatus {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        regfile.basic_csr.xstatus.mask & new_value
    }
}

impl NamedCsr for Sstatus {
    const INDEX: CsrIndex = CsrIndex::Direct(0x100);
    const EXTENSION: Extension = Extension::Zicsr;

    fn register(regfile: &mut CsrRegfile) {
        registe_csr(regfile, Self {}, Self::INDEX, (Self::read, Self::write), None);
    }

    #[inline(always)]
    fn read(regfile: &CsrRegfile) -> CsrDataType {
        Mstatus::read(regfile) & Self::SHADOW_MASK
    }

    #[inline(always)]
    fn write(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        let old = Mstatus::read(regfile);
        let merged =
            (old & !Self::DEFAULT_WRITABLE_MASK) | (new_value & Self::DEFAULT_WRITABLE_MASK);
        Mstatus::write(regfile, merged);
    }

    #[inline(always)]
    fn write_directly(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        let old = Mstatus::read(regfile);
        let merged = (old & !Self::SHADOW_MASK) | (new_value & Self::SHADOW_MASK);
        Mstatus::write_directly(regfile, merged);
    }
}

gen_csr_bitfield!(
    Sstatus, basic_csr, xstatus, [
        1,  1, sie;
        5,  1, spie;
        6,  1, ube;
        8,  1, spp;
        9,  2, vs;
        13, 2, fs;
        15, 2, xs;
        18, 1, sum;
        19, 1, mxr;
        23, 1, spelp;
        24, 1, sdt;
        32, 2, uxl, RO; // TODO: We don't support changing XLEN yet.
        -1, 1, sd;
    ]
);

#[derive(Debug)]
pub struct Sie {}

impl CsrValidator<Sie> for Sie {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        regfile.basic_csr.xie.mask & new_value
    }
}

impl NamedCsr for Sie {
    const INDEX: CsrIndex = CsrIndex::Direct(0x104);
    const EXTENSION: Extension = Extension::Zicsr;

    fn register(regfile: &mut CsrRegfile) {
        registe_csr(regfile, Self {}, Self::INDEX, (Self::read, Self::write), None);
    }

    #[inline(always)]
    fn read(regfile: &CsrRegfile) -> CsrDataType {
        Mie::read(regfile) & Self::SHADOW_MASK
    }

    #[inline(always)]
    fn write(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        let old = Mie::read(regfile);
        let merged =
            (old & !Self::DEFAULT_WRITABLE_MASK) | (new_value & Self::DEFAULT_WRITABLE_MASK);
        Mie::write(regfile, merged);
    }

    #[inline(always)]
    fn write_directly(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        let old = Mie::read(regfile);
        let merged = (old & !Self::SHADOW_MASK) | (new_value & Self::SHADOW_MASK);
        Mie::write_directly(regfile, merged);
    }
}

gen_csr_bitfield!(
    Sie, basic_csr, xie, [
        0,  1, usie; // User Software Interrupt Enable
        1,  1, ssie;
        4,  1, utie; // User Time     Interrupt Enable
        5,  1, stie;
        8,  1, ueie; // User External Interrupt Enable
        9,  1, seie;
    ]
);

#[derive(Debug)]
pub struct Sip {}

impl CsrValidator<Sip> for Sip {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        regfile.basic_csr.xip.mask & new_value
    }
}

impl NamedCsr for Sip {
    const INDEX: CsrIndex = CsrIndex::Direct(0x144);
    const EXTENSION: Extension = Extension::Zicsr;

    fn register(regfile: &mut CsrRegfile) {
        registe_csr(regfile, Self {}, Self::INDEX, (Self::read, Self::write), None);
    }

    #[inline(always)]
    fn read(regfile: &CsrRegfile) -> CsrDataType {
        Mip::read(regfile) & Self::SHADOW_MASK
    }

    #[inline(always)]
    fn write(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        let old = Mip::read(regfile);
        let merged =
            (old & !Self::DEFAULT_WRITABLE_MASK) | (new_value & Self::DEFAULT_WRITABLE_MASK);
        Mip::write(regfile, merged);
    }

    #[inline(always)]
    fn write_directly(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        let old = Mip::read(regfile);
        let merged = (old & !Self::SHADOW_MASK) | (new_value & Self::SHADOW_MASK);
        Mip::write_directly(regfile, merged);
    }
}

gen_csr_bitfield!(
    Sip, basic_csr, xip, [
        0,  1, usip; // User Software Interrupt Pending.
        1,  1, ssip;
        // 2,  1, hsip;
        4,  1, utip; // User Time     Interrupt Pending.
        5,  1, stip;
        // 6,  1, htip;
        8,  1, ueip; // User External Interrupt Pending.
        9,  1, seip;
        // 10, 1, heip;
        // 0, XLEN, mip;
    ]
);

gen_csr_regfile! {
    RVSExtCsrRegfile {
        Stvec, Extension::Zicsr, rvs_csr, stvec, CsrIndex::Direct(0x105), [
            0, 2, mode;
            2, XLEN - 2, base;
        ];

        Scounteren, Extension::Zicsr, rvs_csr, scounteren, CsrIndex::Direct(0x106), [
            0, XLEN, scounteren, RO;
        ];

        Sscratch, Extension::Zicsr, rvs_csr, sscratch, CsrIndex::Direct(0x140), [
            0, XLEN, scratch;
        ];

        Sepc, Extension::Zicsr, rvs_csr, sepc, CsrIndex::Direct(0x141), [
            0, XLEN, sepc;
        ];

        Scause, Extension::Zicsr, rvs_csr, scause, CsrIndex::Direct(0x142), [
            0, XLEN - 1, cause;
            -1, 1, interrupt;
        ];

        Stval, Extension::Zicsr, rvs_csr, stval, CsrIndex::Direct(0x143), [
            0, XLEN, stval;
        ];

        // TODO: riscv-32 support.
        Satp, Extension::Zicsr, rvs_csr, satp, CsrIndex::Direct(0x180), [
            0, 44, ppn;
            44, 16, asid;
            60, 4, mode;  // TODO: Validate mode.
        ];
    }
}
