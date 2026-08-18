use crate::{
    config::arch_config::WordType,
    isa::riscv::{executor::RVCPU, trap::Exception},
    utils::{FloatPoint, TruncateTo, UnsignedInteger, sign_extend},
};

#[inline(always)]
pub(super) fn handle_load<T, const EXTEND: bool>(
    cpu: &mut RVCPU,
    rd: u8,
    addr: WordType,
) -> Result<(), Exception>
where
    T: UnsignedInteger,
{
    let ret = cpu.read::<T>(addr);

    match ret {
        Ok(data) => {
            let data_64: u64 = data.into();
            let mut data = data_64 as WordType;
            if EXTEND {
                data = sign_extend(data, (size_of::<T>() as u32) * 8);
            }
            cpu.reg_file.write(rd, data);
        }
        Err(err) => return Err(err),
    }
    Ok(())
}

#[inline(always)]
pub(super) fn handle_float_load<F>(cpu: &mut RVCPU, addr: WordType, rd: u8) -> Result<(), Exception>
where
    F: FloatPoint,
{
    let rst = cpu.read::<F::BitsType>(addr);

    match rst {
        Ok(data) => {
            cpu.fpu.store_raw::<F>(rd, data.truncate_to());
            Ok(())
        }
        Err(err) => Err(err),
    }
}

#[inline(always)]
pub(super) fn handle_store<T>(
    cpu: &mut RVCPU,
    addr: WordType,
    data: WordType,
) -> Result<(), Exception>
where
    T: UnsignedInteger,
{
    cpu.write(addr, T::truncate_from(data))
}

#[inline(always)]
pub(super) fn handle_float_store<F>(
    cpu: &mut RVCPU,
    addr: WordType,
    data: F::BitsType,
) -> Result<(), Exception>
where
    F: FloatPoint,
{
    cpu.write(addr, data)
}
