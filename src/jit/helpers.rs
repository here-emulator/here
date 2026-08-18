use crate::{
    config::arch_config::WordType,
    isa::riscv::{executor::RVCPU, trap::Exception},
    jit::jit_function::JitContext,
};

unsafe extern "C" fn helper_check_instruction_alignment(
    target: WordType,
    context: *mut JitContext,
    guest_pc: WordType,
    icount: u64,
) -> WordType {
    if (target & 0x3) == 0 {
        target
    } else {
        unsafe {
            JitContext::raise(
                context,
                guest_pc,
                icount,
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
            icount: u64,
        ) -> u64 {
            let cpu = unsafe { &mut *cpu };
            match cpu.read::<$ty>(addr) {
                Ok(loaded) => loaded as u64,
                Err(exception) => unsafe {
                    JitContext::raise(context, guest_pc, icount, exception, addr)
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
            icount: u64,
        ) {
            let cpu = unsafe { &mut *cpu };
            if let Err(exception) = cpu.write::<$ty>(addr, value as $ty) {
                unsafe { JitContext::raise(context, guest_pc, icount, exception, addr) }
            }
        }
    };
}

define_load_helper!(helper_load_u8, u8);
define_load_helper!(helper_load_u16, u16);
define_load_helper!(helper_load_u32, u32);
define_load_helper!(helper_load_u64, u64);

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
    pub(super) fn helper(self) -> u64 {
        let helper = match self {
            Self::SignedByte | Self::UnsignedByte => helper_load_u8 as *const (),
            Self::SignedHalf | Self::UnsignedHalf => helper_load_u16 as *const (),
            Self::SignedWord | Self::UnsignedWord => helper_load_u32 as *const (),
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
