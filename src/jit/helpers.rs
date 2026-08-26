use crate::{
    config::arch_config::WordType,
    isa::riscv::{executor::RVCPU, trap::Exception},
    jit::runtime::JitContext,
};

unsafe extern "C" fn helper_check_instruction_alignment(
    target: WordType,
    context: *mut JitContext,
    guest_pc: WordType,
    consumed_cycles: u64,
) -> WordType {
    if (target & 0x3) == 0 {
        target
    } else {
        unsafe {
            JitContext::raise(
                context,
                guest_pc,
                consumed_cycles,
                Exception::InstructionMisaligned,
                target,
            )
        }
    }
}

macro_rules! define_load_helper {
    ($helper:ident, $ty:ty) => {
        unsafe extern "C" fn $helper(
            cpu: *mut RVCPU,
            addr: WordType,
            context: *mut JitContext,
            guest_pc: WordType,
            consumed_cycles: u64,
        ) -> u64 {
            let cpu = unsafe { &mut *cpu };
            match cpu.read::<$ty>(addr) {
                Ok(loaded) => loaded as u64,
                Err(exception) => unsafe {
                    JitContext::raise(context, guest_pc, consumed_cycles, exception, addr)
                },
            }
        }
    };
}

macro_rules! define_signed_load_helper {
    ($helper:ident, $ty:ty, $signed_ty:ty) => {
        unsafe extern "C" fn $helper(
            cpu: *mut RVCPU,
            addr: WordType,
            context: *mut JitContext,
            guest_pc: WordType,
            consumed_cycles: u64,
        ) -> u64 {
            let cpu = unsafe { &mut *cpu };
            match cpu.read::<$ty>(addr) {
                Ok(loaded) => loaded as $signed_ty as i64 as u64,
                Err(exception) => unsafe {
                    JitContext::raise(context, guest_pc, consumed_cycles, exception, addr)
                },
            }
        }
    };
}

macro_rules! define_store_helper {
    ($helper:ident, $ty:ty) => {
        unsafe extern "C" fn $helper(
            cpu: *mut RVCPU,
            addr: WordType,
            value: WordType,
            context: *mut JitContext,
            guest_pc: WordType,
            consumed_cycles: u64,
        ) {
            let cpu = unsafe { &mut *cpu };
            if let Err(exception) = cpu.write::<$ty>(addr, value as $ty) {
                unsafe { JitContext::raise(context, guest_pc, consumed_cycles, exception, addr) }
            }
        }
    };
}

define_load_helper!(helper_load_u8, u8);
define_load_helper!(helper_load_u16, u16);
define_load_helper!(helper_load_u32, u32);
define_load_helper!(helper_load_u64, u64);

define_signed_load_helper!(helper_load_i8, u8, i8);
define_signed_load_helper!(helper_load_i16, u16, i16);
define_signed_load_helper!(helper_load_i32, u32, i32);

define_store_helper!(helper_store_u8, u8);
define_store_helper!(helper_store_u16, u16);
define_store_helper!(helper_store_u32, u32);
define_store_helper!(helper_store_u64, u64);

#[derive(Clone, Copy)]
pub(super) enum LoadKind {
    SignedByte,
    UnsignedByte,
    SignedHalf,
    UnsignedHalf,
    SignedWord,
    UnsignedWord,
    DoubleWord,
}

impl LoadKind {
    // TODO: return usize
    pub(super) fn helper(self) -> u64 {
        let helper = match self {
            Self::SignedByte => helper_load_i8 as *const (),
            Self::UnsignedByte => helper_load_u8 as *const (),
            Self::SignedHalf => helper_load_i16 as *const (),
            Self::UnsignedHalf => helper_load_u16 as *const (),
            Self::SignedWord => helper_load_i32 as *const (),
            Self::UnsignedWord => helper_load_u32 as *const (),
            Self::DoubleWord => helper_load_u64 as *const (),
        };
        helper as u64
    }
}

#[derive(Clone, Copy)]
pub(super) enum StoreKind {
    Byte,
    Half,
    Word,
    DoubleWord,
}

impl StoreKind {
    // TODO: why not return usize or *const u8
    pub(super) fn helper(self) -> u64 {
        let helper = match self {
            Self::Byte => helper_store_u8 as *const (),
            Self::Half => helper_store_u16 as *const (),
            Self::Word => helper_store_u32 as *const (),
            Self::DoubleWord => helper_store_u64 as *const (),
        };
        helper as u64
    }
}

pub(super) fn check_instruction_alignment_helper() -> u64 {
    helper_check_instruction_alignment as *const () as u64
}
