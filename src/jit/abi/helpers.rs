//! Runtime helpers called from JIT-generated code.
//!
//! Helper calls are emitted by the backend, so their ABI is determined by the
//! backend's calling convention rather than by the host's C ABI.
//! For example, the x64 backend always uses the SysV64 ABI.
//!
//! Some constraints are imposed on helpers:
//! - at most 6 integer/pointer arguments.
//! - integer/pointer arguments and return values only.
//!
//! Helpers can trap (via [`JitContext::raise`]), which should be an extremely cold path.

use crate::{config::arch_config::WordType, isa::riscv::executor::RVCPU, jit::runtime::JitContext};

// TODO: consider rewriting with proc macro

macro_rules! define_load_helper {
    ($abi:literal, $helper:ident, $ty:ty) => {
        unsafe extern $abi fn $helper(
            cpu: *mut RVCPU,
            addr: WordType,
            context: usize,
            guest_pc: WordType,
            consumed_cycles: u64,
        ) -> WordType {
            let cpu = unsafe { &mut *cpu };
            match cpu.read::<$ty>(addr) {
                Ok(loaded) => loaded as WordType,
                Err(exception) => unsafe {
                    JitContext::raise(
                        context as *mut JitContext,
                        guest_pc,
                        consumed_cycles,
                        exception,
                        addr,
                    )
                },
            }
        }
    };
}

macro_rules! define_signed_load_helper {
    ($abi:literal, $helper:ident, $ty:ty, $signed_ty:ty) => {
        unsafe extern $abi fn $helper(
            cpu: *mut RVCPU,
            addr: WordType,
            context: usize,
            guest_pc: WordType,
            consumed_cycles: u64,
        ) -> WordType {
            let cpu = unsafe { &mut *cpu };
            match cpu.read::<$ty>(addr) {
                Ok(loaded) => loaded as $signed_ty as i64 as WordType,
                Err(exception) => unsafe {
                    JitContext::raise(
                        context as *mut JitContext,
                        guest_pc,
                        consumed_cycles,
                        exception,
                        addr,
                    )
                },
            }
        }
    };
}

macro_rules! define_store_helper {
    ($abi:literal, $helper:ident, $ty:ty) => {
        unsafe extern $abi fn $helper(
            cpu: *mut RVCPU,
            addr: WordType,
            value: WordType,
            context: usize,
            guest_pc: WordType,
            consumed_cycles: u64,
        ) {
            let cpu = unsafe { &mut *cpu };
            if let Err(exception) = cpu.write::<$ty>(addr, value as $ty) {
                unsafe {
                    JitContext::raise(
                        context as *mut JitContext,
                        guest_pc,
                        consumed_cycles,
                        exception,
                        addr,
                    )
                }
            }
        }
    };
}

macro_rules! define_helpers {
    ($abi:literal) => {
        define_load_helper!($abi, helper_load_u8, u8);
        define_load_helper!($abi, helper_load_u16, u16);
        define_load_helper!($abi, helper_load_u32, u32);
        define_load_helper!($abi, helper_load_u64, u64);

        define_signed_load_helper!($abi, helper_load_i8, u8, i8);
        define_signed_load_helper!($abi, helper_load_i16, u16, i16);
        define_signed_load_helper!($abi, helper_load_i32, u32, i32);

        define_store_helper!($abi, helper_store_u8, u8);
        define_store_helper!($abi, helper_store_u16, u16);
        define_store_helper!($abi, helper_store_u32, u32);
        define_store_helper!($abi, helper_store_u64, u64);
    };
}

cfg_select! {
    target_arch = "x86_64" => {
        define_helpers!("sysv64");
    }

    _ => {
        define_helpers!("C");
    }
}

#[derive(Clone, Copy)]
pub enum LoadKind {
    SignedByte,
    UnsignedByte,
    SignedHalf,
    UnsignedHalf,
    SignedWord,
    UnsignedWord,
    DoubleWord,
}

impl LoadKind {
    pub fn helper(self) -> usize {
        let helper = match self {
            Self::SignedByte => helper_load_i8 as *const (),
            Self::UnsignedByte => helper_load_u8 as *const (),
            Self::SignedHalf => helper_load_i16 as *const (),
            Self::UnsignedHalf => helper_load_u16 as *const (),
            Self::SignedWord => helper_load_i32 as *const (),
            Self::UnsignedWord => helper_load_u32 as *const (),
            Self::DoubleWord => helper_load_u64 as *const (),
        };
        helper as usize
    }
}

#[derive(Clone, Copy)]
pub enum StoreKind {
    Byte,
    Half,
    Word,
    DoubleWord,
}

impl StoreKind {
    pub fn helper(self) -> usize {
        let helper = match self {
            Self::Byte => helper_store_u8 as *const (),
            Self::Half => helper_store_u16 as *const (),
            Self::Word => helper_store_u32 as *const (),
            Self::DoubleWord => helper_store_u64 as *const (),
        };
        helper as usize
    }
}
