use std::cmp;
use std::sync::atomic::{self, Ordering};

use crate::utils::WordTrait;
use crate::{
    config::arch_config::WordType,
    isa::riscv::{
        csr_reg::csr_macro::Minstret, executor::RVCPU, instruction::RVInstrInfo, trap::Exception,
    },
    utils::{TruncateFrom, UnsignedInteger},
};

// ----------------------------------
// Atomic Memory Operation Traits
// ----------------------------------
pub(super) trait AMOTrait<T>
where
    T: UnsignedInteger,
{
    fn exec(a: &T::AtomicType, b: T, order: atomic::Ordering) -> Result<T, Exception>;
}

pub(super) struct ExecAmoAdd {}
impl AMOTrait<u64> for ExecAmoAdd {
    fn exec(
        lhs: &<u64 as UnsignedInteger>::AtomicType,
        rhs: u64,
        order: atomic::Ordering,
    ) -> Result<u64, Exception> {
        Ok(lhs.fetch_add(rhs, order))
    }
}
impl AMOTrait<u32> for ExecAmoAdd {
    fn exec(
        lhs: &<u32 as UnsignedInteger>::AtomicType,
        rhs: u32,
        order: atomic::Ordering,
    ) -> Result<u32, Exception> {
        Ok(lhs.fetch_add(rhs, order))
    }
}

pub(super) struct ExecAmoAnd {}
impl AMOTrait<u64> for ExecAmoAnd {
    fn exec(
        lhs: &<u64 as UnsignedInteger>::AtomicType,
        rhs: u64,
        order: atomic::Ordering,
    ) -> Result<u64, Exception> {
        Ok(lhs.fetch_and(rhs, order))
    }
}
impl AMOTrait<u32> for ExecAmoAnd {
    fn exec(
        lhs: &<u32 as UnsignedInteger>::AtomicType,
        rhs: u32,
        order: atomic::Ordering,
    ) -> Result<u32, Exception> {
        Ok(lhs.fetch_and(rhs, order))
    }
}

pub(super) struct ExecAmoOr {}
impl AMOTrait<u64> for ExecAmoOr {
    fn exec(
        lhs: &<u64 as UnsignedInteger>::AtomicType,
        rhs: u64,
        order: atomic::Ordering,
    ) -> Result<u64, Exception> {
        Ok(lhs.fetch_or(rhs, order))
    }
}
impl AMOTrait<u32> for ExecAmoOr {
    fn exec(
        lhs: &<u32 as UnsignedInteger>::AtomicType,
        rhs: u32,
        order: atomic::Ordering,
    ) -> Result<u32, Exception> {
        Ok(lhs.fetch_or(rhs, order))
    }
}

pub(super) struct ExecAmoXor {}
impl AMOTrait<u64> for ExecAmoXor {
    fn exec(
        lhs: &<u64 as UnsignedInteger>::AtomicType,
        rhs: u64,
        order: atomic::Ordering,
    ) -> Result<u64, Exception> {
        Ok(lhs.fetch_xor(rhs, order))
    }
}
impl AMOTrait<u32> for ExecAmoXor {
    fn exec(
        lhs: &<u32 as UnsignedInteger>::AtomicType,
        rhs: u32,
        order: atomic::Ordering,
    ) -> Result<u32, Exception> {
        Ok(lhs.fetch_xor(rhs, order))
    }
}

pub(super) struct ExecAmoMax {}
impl AMOTrait<u64> for ExecAmoMax {
    fn exec(
        lhs: &<u64 as UnsignedInteger>::AtomicType,
        rhs: u64,
        order: atomic::Ordering,
    ) -> Result<u64, Exception> {
        lhs.try_update(order, Ordering::Relaxed, |v| {
            Some(cmp::max(v as i64, rhs as i64) as u64)
        })
        .map_err(|_| Exception::StoreFault)
    }
}
impl AMOTrait<u32> for ExecAmoMax {
    fn exec(
        lhs: &<u32 as UnsignedInteger>::AtomicType,
        rhs: u32,
        order: atomic::Ordering,
    ) -> Result<u32, Exception> {
        lhs.try_update(order, Ordering::Relaxed, |v| {
            Some(cmp::max(v as i32, rhs as i32) as u32)
        })
        .map_err(|_| Exception::StoreFault)
    }
}

pub(super) struct ExecAmoMin {}
impl AMOTrait<u64> for ExecAmoMin {
    fn exec(
        lhs: &<u64 as UnsignedInteger>::AtomicType,
        rhs: u64,
        order: atomic::Ordering,
    ) -> Result<u64, Exception> {
        lhs.try_update(order, Ordering::Relaxed, |v| {
            Some(cmp::min(v as i64, rhs as i64) as u64)
        })
        .map_err(|_| Exception::StoreFault)
    }
}
impl AMOTrait<u32> for ExecAmoMin {
    fn exec(
        lhs: &<u32 as UnsignedInteger>::AtomicType,
        rhs: u32,
        order: atomic::Ordering,
    ) -> Result<u32, Exception> {
        lhs.try_update(order, Ordering::Relaxed, |v| {
            Some(cmp::min(v as i32, rhs as i32) as u32)
        })
        .map_err(|_| Exception::StoreFault)
    }
}

pub(super) struct ExecAmoMaxU {}
impl AMOTrait<u64> for ExecAmoMaxU {
    fn exec(
        lhs: &<u64 as UnsignedInteger>::AtomicType,
        rhs: u64,
        order: atomic::Ordering,
    ) -> Result<u64, Exception> {
        Ok(lhs.fetch_max(rhs, order))
    }
}
impl AMOTrait<u32> for ExecAmoMaxU {
    fn exec(
        lhs: &<u32 as UnsignedInteger>::AtomicType,
        rhs: u32,
        order: atomic::Ordering,
    ) -> Result<u32, Exception> {
        Ok(lhs.fetch_max(rhs, order))
    }
}

pub(super) struct ExecAmoMinU {}
impl AMOTrait<u64> for ExecAmoMinU {
    fn exec(
        lhs: &<u64 as UnsignedInteger>::AtomicType,
        rhs: u64,
        order: atomic::Ordering,
    ) -> Result<u64, Exception> {
        Ok(lhs.fetch_min(rhs, order))
    }
}
impl AMOTrait<u32> for ExecAmoMinU {
    fn exec(
        lhs: &<u32 as UnsignedInteger>::AtomicType,
        rhs: u32,
        order: atomic::Ordering,
    ) -> Result<u32, Exception> {
        Ok(lhs.fetch_min(rhs, order))
    }
}

pub(super) struct ExecAmoSwap {}
impl AMOTrait<u64> for ExecAmoSwap {
    fn exec(
        lhs: &<u64 as UnsignedInteger>::AtomicType,
        rhs: u64,
        order: atomic::Ordering,
    ) -> Result<u64, Exception> {
        Ok(lhs.swap(rhs, order))
    }
}
impl AMOTrait<u32> for ExecAmoSwap {
    fn exec(
        lhs: &<u32 as UnsignedInteger>::AtomicType,
        rhs: u32,
        order: atomic::Ordering,
    ) -> Result<u32, Exception> {
        Ok(lhs.swap(rhs, order))
    }
}

// ----------------------------------
// Atomic Memory Operation executor
// ----------------------------------
fn get_amo_order(aq: bool, rl: bool) -> std::sync::atomic::Ordering {
    match (aq, rl) {
        (false, false) => std::sync::atomic::Ordering::Relaxed,
        (true, false) => std::sync::atomic::Ordering::Acquire,
        (false, true) => std::sync::atomic::Ordering::Release,
        (true, true) => std::sync::atomic::Ordering::AcqRel,
    }
}

/// let t = mem[x[rs1]];  
/// x[rd] = t;
/// mem[x[rs1]] = t OP x[rs2];
pub(crate) fn exec_atomic_memory_operation<F, T>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    T: UnsignedInteger + WordTrait,
    F: AMOTrait<T>,
{
    let RVInstrInfo::A {
        rs1,
        rs2,
        rd,
        aq,
        rl,
    } = info
    else {
        unreachable!()
    };

    let (val1, val2) = cpu.reg_file.read(rs1, rs2);
    let order = get_amo_order(aq, rl);
    let res = cpu.fetch_and_op_amo(val1, T::truncate_from(val2), |l, r| F::exec(l, r, order))?;

    let res = res.sign_extend_to_wordtype();

    cpu.reg_file.write(rd, res);

    cpu.pc = cpu.pc.wrapping_add(4);
    cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);

    Ok(())
}

pub(super) fn exec_lr<T, const EXTEND: bool>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    T: UnsignedInteger + WordTrait,
{
    let RVInstrInfo::A { rs1, rd, .. } = info else {
        unreachable!()
    };

    let addr = cpu.reg_file[rs1 as usize];

    let res = cpu.load_reserved::<T>(addr)?;

    let res = if EXTEND {
        res.sign_extend_to_wordtype()
    } else {
        res.into()
    };

    cpu.reg_file.write(rd, res);
    cpu.pc = cpu.pc.wrapping_add(4);
    cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
    Ok(())
}

pub(super) fn exec_sc<T>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    T: UnsignedInteger + TruncateFrom<WordType>,
{
    if let RVInstrInfo::A { rs1, rs2, rd, .. } = info {
        let (addr, val) = cpu.reg_file.read(rs1, rs2);
        let val_t = T::truncate_from(val);

        let success = cpu.store_conditional(addr, val_t)?;

        cpu.reg_file.write(rd, if success { 0 } else { 1 });
        cpu.pc = cpu.pc.wrapping_add(4);
        cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
        Ok(())
    } else {
        unreachable!()
    }
}
