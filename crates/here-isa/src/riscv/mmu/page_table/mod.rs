#[cfg(feature = "riscv64")]
pub mod riscv64;
#[cfg(feature = "riscv64")]
pub use riscv64::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageTableError {
    AlignFault,
    PageFault,
    PrivilegeFault,
}

pub struct PermissionCheck {
    /// At least one of the flags in `any_of` must be set in the PTE, used for `MXR`.
    /// If `any_of` is empty, this condition is ignored.
    pub any_of: PTEFlags,
    pub exact_mask: PTEFlags,
    pub exact_flags: PTEFlags,
}
