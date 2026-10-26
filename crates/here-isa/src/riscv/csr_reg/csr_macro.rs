use phf::phf_map;

use super::read_validator::*;
use super::write_validator::*;
use super::*;
use crate::config::arch_config::{SignedWordType, WordType, XLEN};
use crate::utils::BIT_ONES_ARRAY;

/// Generator a single csr register.
macro_rules! gen_csr_reg {
    (
        $name:ident, $addr:expr,
        [ $( $bit:expr, $len:expr, $fname:ident ),*  $(,)? ]
    ) => {
        /// A struct representing a CSR register.
        ///
        /// XXX: All shadow CSR's validator won't work with this API. Use `CsrReg::write` directly instead.
        pub struct $name {
            reg: *mut CsrReg,
            ctx: *mut CsrContext,
        }

        impl NamedCsrReg for $name {
            fn new(reg: *mut CsrReg, ctx: *mut CsrContext) -> Self {
                Self { reg, ctx }
            }

            fn data(&self) -> WordType {
                unsafe { (*self.reg).value() }
            }

            fn set_data(&mut self, val: WordType) {
                let reg = unsafe { &mut *self.reg };
                reg.write(val, unsafe {&*self.ctx});
            }

            fn set_data_directly(&mut self, val: WordType) {
                let reg = unsafe { &mut *self.reg };
                reg.write_directly(val);
            }

            #[inline]
            fn get_index() -> WordType {
                $addr
            }
        }

        impl $name {
            $(
                #[allow(non_upper_case_globals)]
                pub const ${concat($fname, _start)}: usize = if ($bit >= 0) {
                    ($bit as i32) as usize
                }
                else {
                    ((XLEN as SignedWordType) + $bit) as usize
                };

                #[allow(non_upper_case_globals)]
                pub const ${concat($fname, _end)}: usize = $name::${concat($fname, _start)} + ($len as usize) - 1;

                #[inline]
                pub fn ${concat(get_, $fname)}(&self) -> WordType {
                    const LOW_BIT: WordType = if ($bit >= 0) {
                        ($bit as SignedWordType).abs() as WordType
                    }
                    else {
                        ((XLEN as SignedWordType) + $bit) as WordType
                    };

                    ((unsafe { self.reg.read_volatile().value() })
                    & (BIT_ONES_ARRAY[$len]) << LOW_BIT) >> LOW_BIT
                }

                #[inline]
                pub fn ${concat(set_, $fname)}(&self, val: WordType) {
                    assert!(val <= BIT_ONES_ARRAY[$len]);
                    const LOW_BIT: WordType = if ($bit >= 0) {
                        ($bit as SignedWordType).abs() as WordType
                    }
                    else {
                        ((XLEN as SignedWordType) + $bit) as WordType
                    };

                    let reg = unsafe { &mut *self.reg };

                    let write_op = CsrWriteOp {mask: (BIT_ONES_ARRAY[$len]) << LOW_BIT};
                    reg.write(write_op.get_new_value(reg.value(), val << LOW_BIT), unsafe {&*self.ctx});
                }

                #[inline]
                pub fn ${concat(set_, $fname, _directly)}(&self, val: WordType) {
                    assert!(val <= BIT_ONES_ARRAY[$len]);
                    const LOW_BIT: WordType = if ($bit >= 0) {
                        ($bit as SignedWordType).abs() as WordType
                    }
                    else {
                        ((XLEN as SignedWordType) + $bit) as WordType
                    };

                    let reg = unsafe { &mut *self.reg };

                    let write_op = CsrWriteOp {mask: (BIT_ONES_ARRAY[$len]) << LOW_BIT};
                    reg.write_directly(write_op.get_new_value(reg.value(), val << LOW_BIT));
                }
            )*
        }
    };
}

macro_rules! gen_csr_address_hashmap {
    ($(($name: literal, $addr: expr)),* $(,)? ) => {
        pub const CSR_ADDRESS: phf::Map<&'static str, WordType> = phf_map! {
            $(
                $name => $addr
            ),*
        };

        pub const CSR_NAME: phf::Map<WordType, &'static str> = phf_map! {
            $(
                $addr => $name
            ),*
        };
    };
}

/// Generates a CSR RegFile. Supports an optional validator for each field, uses [`validate_write_any`] by default.
/// If you don't specify a CSR field, it will be readonly by default.
///
/// NOTE: If you have overlapping fields, the bits will be allowed to write if any validator allows it,
/// so don't forget to make overlapping fields readonly when necessary.
///
/// TODO: Maybe it's better to use `validate_readonly` by default,
/// because for all WARL field, we should ignore writing when we don't support.
macro_rules! gen_csr_regfile {
    (
        $( $name:ident, $name_str: literal, $addr:expr, $default:expr,
            $(@shadow $shadow_of:ident, )?
            [ $( $bit:expr, $len:expr, $fname:ident $(, $validator:expr)?);* $(;)? ]
        );* $(;)?
    ) => {
        // Generate each CSR struct.
        $(
            gen_csr_reg!($name, $addr, [ $( $bit, $len, $fname ),* ]);
        )*

        // Generate validators for each CSR.
        $(
            fn ${concat(_validate_, $name_str)}(value: WordType, ctx: &CsrContext) -> CsrWriteOp {
                combine_validators!{ value, ctx
                    $(,
                        gen_csr_regfile!(@choose_validator $name, $fname $(, $validator)?)
                    )*
                }
            }
        )*

        pub(crate) const CSR_REG_TABLE: &[(WordType, WordType, WriteValidator)] = &[
            $(
                ($addr, $default, ${concat(_validate_, $name_str)})
            ),*
        ];

        gen_csr_address_hashmap!($(($name_str, $addr)),*);

        /// Resolve the shadow CSR address to its base CSR address.
        pub(super) fn resolve_shadow_addr(addr: WordType) -> Option<ReadValidator> {
            match addr {
                $(
                    $addr => gen_csr_regfile!(@resolve_shadow $(@shadow $shadow_of ,)? $name, $($fname),*),
                )*
                _ => None,
            }
        }
    };

    // These branches are used to find the base CSR address by the shadow CSR address.
    (@resolve_shadow @shadow $shadow_of:ident, $name: ident, $($fname: ident),* ) => {
        Some(combine_shadow_read_ops!(
            $shadow_of::get_index(),
            $(
                $name::${concat($fname, _start)},
                $name::${concat($fname, _end)}
            ),*
        ))
    };
    (@resolve_shadow $name: ident, $($_: expr),*) => { None };

    // These branches are used to choose validator.
    (@choose_validator $name:ident, $fname:ident, $v:expr) => { $v };

    // By default, use `validate_write_any`.
    (@choose_validator $name:ident, $fname:ident) => {
        validate_write_any::<{$name::${concat($fname, _start)}}, {$name::${concat($fname, _end)}}>
    };

}

// Ensure you have read the comments above the macro definition.
gen_csr_regfile! {
    // ==================================
    //            U-Mode CSR
    // ==================================
    Fcsr, "fcsr", 0x003u64, 0x00, [
        0, 5, fflags;
        0, 1, nx;
        1, 1, uf;
        2, 1, of;
        3, 1, dz;
        4, 1, nv;

        // rounding mode
        5, 3, rm;
    ];

    Vstart, "vstart", 0x008u64, 0x00, [
        0, XLEN, vstart;
    ];

    // Vxsat, "vxsat", 0x009u64, 0x00, [
    //     0, 1, least_significant; // Fixed-Point Saturate Flag
    // ];

    // Vxrm, "vxrm", 0x00au64, 0x00, [
    //     0, 2, vxrm;
    // ];

    Vcsr, "vcsr", 0x00fu64, 0x00, [
        0, 1, vxsat; // Fixed-point accrued saturation flag
        1, 2, vxrm; // Fixed-point rounding mode
    ];

    Cycle, "cycle", 0xC00u64, 0x00, @shadow Mcycle, [
        0, XLEN, cycle, validate_readonly;
    ];

    Instret, "instret", 0xC02u64, 0x00, @shadow Minstret, [
        0, XLEN, instret, validate_readonly;
    ];

    Vl, "vlen", 0xc20u64, 0x00, [
        0, XLEN, vl;
    ];

    Vtype, "vtype", 0xc21u64, 0x00, [
        0, 3, vlmul, validate_readonly;
        3, 3, vsew, validate_readonly;
        6, 1, vta, validate_readonly;
        7, 1, vma, validate_readonly;
        -1, 1, vill, validate_readonly;
    ];

    Vlenb, "vlenb", 0xc22u64, 0x00, [
        0, XLEN, vlenb;
    ];

    // ==================================
    //            S-Mode CSR
    // ==================================
    Sstatus, "sstatus", 0x100u64, 0x00, @shadow Mstatus, [
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
        32, 2, uxl, validate_readonly;  // TODO: We don't support changing XLEN yet.
        -1, 1, sd;
    ];

    Sie, "sie", 0x104u64, 0x00, @shadow Mie, [
        0,  1, usie; // User Software Interrupt Enable
        1,  1, ssie;
        4,  1, utie; // User Time     Interrupt Enable
        5,  1, stie;
        8,  1, ueie; // User External Interrupt Enable
        9,  1, seie;
    ];

    Stvec, "stvec", 0x105u64, 0x00, [
        0, 2, mode;
        2, XLEN - 2, base;
    ];

    Sscratch, "sscratch", 0x140u64, 0x00, [
        0, XLEN, scratch;
    ];

    Sepc, "sepc", 0x141u64, 0x00, [
        0, XLEN, sepc;
    ];

    Scause, "scause", 0x142u64, 0x00, [
        0, XLEN - 1, cause;
        -1, 1, interrupt;
    ];

    Stval, "stval", 0x143u64, 0x00, [
        0, XLEN, stval;
    ];

    Sip, "sip", 0x144u64, 0x00, @shadow Mip, [
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
    ];

    // TODO: riscv-32 support.
    Satp, "satp", 0x180u64, 0x00, [
        0, 44, ppn;
        44, 16, asid;
        60, 4, mode;  // TODO: Validate mode.
    ];

    // ==================================
    //            M-Mode CSR
    // ==================================
    Mstatus, "mstatus", 0x300u64, 0x00, [
        1,  1, sie;
        3,  1, mie;
        5,  1, spie;
        6,  1, ube;
        7,  1, mpie;
        8,  1, spp;
        9,  2, vs;
        11, 2, mpp;
        13, 2, fs;
        15, 2, xs, validate_readonly;
        17, 1, mprv;
        18, 1, sum;
        19, 1, mxr;
        20, 1, tvm;
        21, 1, tw;
        22, 1, tsr;
        23, 1, spelp;
        24, 1, sdt;
        32, 2, uxl, validate_readonly;  // TODO: We don't support changing XLEN yet.
        34, 2, sxl, validate_readonly;
        36, 1, sbe;
        37, 1, mbe;
        38, 1, gva;
        39, 1, mpv;
        40, 1, wpri;
        41, 1, mpelp;
        42, 1, mdt;
        -1, 1, sd;
    ];

    Misa, "misa", 0x301u64, 0x00, [
        0, 25, extension, validate_readonly;  // TODO: We don't support changing extension yet.
        -2, 2, mxl, validate_readonly;
    ];

    Medeleg, "medeleg", 0x302u64, 0x00, [
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
    Mideleg, "mideleg", 0x303u64, 0x00, [
        1, 1, ssip; // Delegate Supervisor Software Interrupt.
        5, 1, stip; // Delegate Supervisor Time     Interrupt.
        9, 1, seip; // Delegate Supervisor External Interrupt.
    ];

    Mie, "mie", 0x304u64, 0x00, [
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

    Mtvec, "mtvec", 0x305u64, 0x00, [
        0, 2, mode, validate_range::<0, 2, 0, 1>;
        2, XLEN - 2, base;
    ];

    Mcountinhibit, "mcountinhibit", 0x320u64, 0x00, [
        0, XLEN, mcountinhibit, validate_readonly;
    ];

    Mscratch, "mscratch", 0x340u64, 0x00, [
        0, XLEN, scratch;
    ];

    Mepc, "mepc", 0x341u64, 0x00, [
        0, XLEN, mepc;
    ];

    Mcause, "mcause", 0x342u64, 0x00, [
        0, XLEN - 1, cause;
        -1, 1, interrupt_flag;
    ];

    Mtval, "mtval", 0x343u64, 0x00, [
        0, XLEN, mtval;
    ];

    Mip, "mip", 0x344u64, 0x00, [
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

    Mcycle, "mcycle", 0xB00u64, 0x00, [
        0, XLEN, mcycle;
    ];

    Minstret, "minstret", 0xB02u64, 0x00, [
        0, XLEN, minstret;
    ];

    Mvendorid, "mvendorid", 0xF11u64, 0x00, [
        0, XLEN, mvendorid, validate_readonly;
    ];

    Marchid, "marchid", 0xF12u64, 0x00, [
        0, XLEN, marchid, validate_readonly;
    ];

    Mimpid, "mimpid", 0xF13u64, 0x00, [
        0, XLEN, mimpid, validate_readonly;
    ];

    Mhartid, "mhartid", 0xF14u64, 0x00, [
        0, XLEN, mhartid, validate_readonly;
    ];
}
