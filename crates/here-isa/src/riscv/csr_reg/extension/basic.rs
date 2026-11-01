use here_types::{BIT_ONES_ARRAY, XLEN};

use crate::riscv::{
    csr_reg::{
        extension::registe_csr,
        regfile::{CsrDataType, CsrIndex, CsrItem, CsrRegfile, CsrValidator, NamedCsr, SubExtCsrRegfile},
    },
    isa_builder::Extension,
};

#[rustfmt::skip]
pub struct RVBasicCsrRegfile {
    pub(super) xstatus:        CsrItem,
    pub(super) misa:           CsrItem,
    pub(super) medeleg:        CsrItem,
    pub(super) mideleg:        CsrItem,
    pub(super) xie:            CsrItem,
    pub(super) mtvec:          CsrItem,
    pub(super) mcountinhibit:  CsrItem,
    pub(super) mscratch:       CsrItem,
    pub(super) mepc:           CsrItem,
    pub(super) mcause:         CsrItem,
    pub(super) mtval:          CsrItem,
    pub(super) xip:            CsrItem,
    pub(super) mcycle:         CsrItem,
    pub(super) minstret:       CsrItem,
    pub(super) mvendorid:      CsrItem,
    pub(super) marchid:        CsrItem,
    pub(super) mimpid:         CsrItem,
    pub(super) mhartid:        CsrItem,
}

impl SubExtCsrRegfile for RVBasicCsrRegfile {
    fn register(regfile: &mut CsrRegfile) {
        Self::macro_register(regfile);
        Cycle::register(regfile);
        Instret::register(regfile);
    }
}

impl Mstatus {
    fn validate_custom(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        todo!()
    }
}

gen_csr_regfile! {
    RVBasicCsrRegfile {
        Mstatus, Extension::Zicsr, basic_csr, xstatus, CsrIndex::Direct(0x300), @validate_custom [
            1,  1, sie;
            3,  1, mie;
            5,  1, spie;
            6,  1, ube;
            7,  1, mpie;
            8,  1, spp;
            9,  2, vs;
            11, 2, mpp;
            13, 2, fs;
            15, 2, xs, RO;
            17, 1, mprv;
            18, 1, sum;
            19, 1, mxr;
            20, 1, tvm;
            21, 1, tw;
            22, 1, tsr;
            23, 1, spelp;
            24, 1, sdt;
            32, 2, uxl, RO;  // TODO: We don't support changing XLEN yet.
            34, 2, sxl, RO;
            36, 1, sbe;
            37, 1, mbe;
            38, 1, gva;
            39, 1, mpv;
            40, 1, wpri;
            41, 1, mpelp;
            42, 1, mdt;
            -1, 1, sd;
        ];

        Misa, Extension::Zicsr, basic_csr, misa, CsrIndex::Direct(0x301), [
            0, 26, extension, RO;  // TODO: We don't support changing extension yet.
            -2, 2, mxl, RO;
        ];

        Medeleg, Extension::Zicsr, basic_csr, medeleg, CsrIndex::Direct(0x302), [
            0, 1, instruction_misaligned;
            1, 1, instruction_fault;
            2, 1, illegal_instruction;
            3, 1, breakpoint;
            4, 1, load_misaligned;
            5, 1, load_fault;
            6, 1, store_misaligned;
            7, 1, store_fault;
            8, 1, user_env_call;
            9, 1, supervisor_env_call;
            // 10, 1, hypervisor_env_call;
            11, 1, machine_env_call;
            12, 1, instruction_page_fault;
            13, 1, load_page_fault;
            15, 1, store_page_fault;
        ];

        // see mip.
        Mideleg, Extension::Zicsr, basic_csr, mideleg, CsrIndex::Direct(0x303), [
            1, 1, ssip; // Delegate Supervisor Software Interrupt.
            5, 1, stip; // Delegate Supervisor Time     Interrupt.
            9, 1, seip; // Delegate Supervisor External Interrupt.
        ];

        Mie, Extension::Zicsr, basic_csr, xie, CsrIndex::Direct(0x304), [
            0,  1, usie;  // User Software                  Interrupt Enable
            1,  1, ssie;
            2,  1, vssie; // Virtual Supervisor Software    Interrupt Enable
            3,  1, msie;

            4,  1, utie;  // User Time                      Interrupt Enable
            5,  1, stie;
            6,  1, vstie; // Virtual Supervisor Time        Interrupt Enable
            7,  1, mtie;

            8,  1, ueie;  // User External                  Interrupt Enable
            9,  1, seie;
            10, 1, vseie; // Virtual Supervisor External    Interrupt Enable
            11, 1, meie;
            12, 1, sgeie; // Supervisor Guest External Interrupt Enable
            13, 1, hgeie; // Hypervisor Guest External Interrupt Enable
        ];

        Mtvec, Extension::Zicsr, basic_csr, mtvec, CsrIndex::Direct(0x305), [
            0, 2, mode;
            2, XLEN - 2, base;
        ];

        Mcountinhibit, Extension::Zicsr, basic_csr, mcountinhibit, CsrIndex::Direct(0x320), [
            0, XLEN, mcountinhibit, RO;
        ];

        Mscratch, Extension::Zicsr, basic_csr, mscratch, CsrIndex::Direct(0x340), [
            0, XLEN, scratch;
        ];

        Mepc, Extension::Zicsr, basic_csr, mepc, CsrIndex::Direct(0x341), [
            0, XLEN, mepc;
        ];

        Mcause, Extension::Zicsr, basic_csr, mcause, CsrIndex::Direct(0x342), [
            0, XLEN - 1, cause;
            -1, 1, interrupt_flag;
        ];

        Mtval, Extension::Zicsr, basic_csr, mtval, CsrIndex::Direct(0x343), [
            0, XLEN, mtval;
        ];

        Mip, Extension::Zicsr, basic_csr, xip, CsrIndex::Direct(0x344), [
            0,  1, usip; // User Software Interrupt Pending.
            1,  1, ssip;
            2,  1, vssip;
            3,  1, msip;

            4,  1, utip; // User Time     Interrupt Pending.
            5,  1, stip;
            6,  1, vstip;
            7,  1, mtip;

            8,  1, ueip; // User External Interrupt Pending.
            9,  1, seip;
            10, 1, vseip;
            11, 1, meip;

            12, 1, sgeip;
            13, 1, hgeip;
        ];

        Mcycle, Extension::Zicsr, basic_csr, mcycle, CsrIndex::Direct(0xB00), [
            0, XLEN, mcycle;
        ];

        Minstret, Extension::Zicsr, basic_csr, minstret, CsrIndex::Direct(0xB02), [
            0, XLEN, minstret;
        ];

        Mvendorid, Extension::Zicsr, basic_csr, mvendorid, CsrIndex::Direct(0xF11), [
            0, XLEN, mvendorid, RO;
        ];

        Marchid, Extension::Zicsr, basic_csr, marchid, CsrIndex::Direct(0xF12), [
            0, XLEN, marchid, RO;
        ];

        Mimpid, Extension::Zicsr, basic_csr, mimpid, CsrIndex::Direct(0xF13), [
            0, XLEN, mimpid, RO;
        ];

        Mhartid, Extension::Zicsr, basic_csr, mhartid, CsrIndex::Direct(0xF14), [
            0, XLEN, mhartid, RO;
        ];
    }
}

pub struct Cycle {}

impl CsrValidator<Cycle> for Cycle {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        regfile.basic_csr.mcycle.mask & new_value
    }
}

impl NamedCsr for Cycle {
    const INDEX: CsrIndex = CsrIndex::Direct(0xC00);
    const EXTENSION: Extension = Extension::Zicsr;

    fn register(regfile: &mut CsrRegfile) {
        registe_csr(regfile, Self::INDEX, (Self::read, Self::write), None);
    }

    #[inline(always)]
    fn read(regfile: &mut CsrRegfile) -> CsrDataType {
        Mcycle::read(regfile) & Self::SHADOW_MASK
    }

    #[inline(always)]
    fn write(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        let old = Mcycle::read(regfile);
        let merged = (old & !Self::SHADOW_MASK) | (new_value & Self::SHADOW_MASK);
        Mcycle::write(regfile, merged);
    }

    #[inline(always)]
    fn write_directly(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        let old = Mcycle::read(regfile);
        let merged = (old & !Self::SHADOW_MASK) | (new_value & Self::SHADOW_MASK);
        Mcycle::write_directly(regfile, merged);
    }
}

gen_csr_bitfield!(
    Cycle, basic_csr, mcycle, [
        0, XLEN, cycle, RO;
    ]
);

pub struct Instret {}

impl CsrValidator<Instret> for Instret {
    fn validate(regfile: &mut CsrRegfile, new_value: CsrDataType) -> CsrDataType {
        regfile.basic_csr.minstret.mask & new_value
    }
}

impl NamedCsr for Instret {
    const INDEX: CsrIndex = CsrIndex::Direct(0xC02);
    const EXTENSION: Extension = Extension::Zicsr;

    fn register(regfile: &mut CsrRegfile) {
        registe_csr(regfile, Self::INDEX, (Self::read, Self::write), None);
    }

    #[inline(always)]
    fn read(regfile: &mut CsrRegfile) -> CsrDataType {
        Minstret::read(regfile) & Self::SHADOW_MASK
    }

    #[inline(always)]
    fn write(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        let old = Minstret::read(regfile);
        let merged = (old & !Self::SHADOW_MASK) | (new_value & Self::SHADOW_MASK);
        Minstret::write(regfile, merged);
    }

    #[inline(always)]
    fn write_directly(regfile: &mut CsrRegfile, new_value: CsrDataType) {
        let old = Minstret::read(regfile);
        let merged = (old & !Self::SHADOW_MASK) | (new_value & Self::SHADOW_MASK);
        Minstret::write_directly(regfile, merged);
    }
}

gen_csr_bitfield!(
    Instret, basic_csr, minstret, [
        0, XLEN, instret, RO;
    ]
);
