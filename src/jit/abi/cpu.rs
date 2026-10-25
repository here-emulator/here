use crate::config::arch_config::WordType;
use crate::isa::riscv::executor::RVCPU;

#[inline]
pub fn guest_reg_offset(guest_reg: u8) -> u32 {
    let offset = std::mem::offset_of!(RVCPU, reg_file) + guest_reg as usize * size_of::<WordType>();
    let offset = u32::try_from(offset).expect("RVCPU register offset is abnormal large");
    offset
}
