// TODO: most of this module is duplicate code.
// Maybe we can reuse common exec function once we find a good way to handle the PC increment.

use crate::{
    config::arch_config::WordType,
    debug_unreachable,
    isa::riscv::{
        csr_reg::csr_macro::Minstret,
        executor::RVCPU,
        instruction::{
            RVInstrInfo,
            exec_function::{ExecAdd, ExecTrait},
            normal_compress_exec, normal_compress_float_exec,
        },
        trap::Exception,
    },
    utils::{FloatPoint, TruncateTo, UnsignedInteger, wrapping_add_as_signed},
};

pub(super) fn exec_compress_arith<F>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    F: ExecTrait<Result<WordType, Exception>>,
{
    normal_compress_exec(cpu, |cpu| {
        let (rd, rst) = match info {
            RVInstrInfo::CR { rd_rs1, rs2 } | RVInstrInfo::CA { rd_rs1, rs2 } => {
                let (val1, val2) = cpu.reg_file.read(rd_rs1, rs2);
                (rd_rs1, F::exec(val1, val2)?)
            }
            RVInstrInfo::CI { rd_rs1, imm } | RVInstrInfo::CB { rd_rs1, imm } => {
                let val1 = cpu.reg_file.read(rd_rs1, 0).0;
                (rd_rs1, F::exec(val1, imm)?)
            }
            _ => debug_unreachable!(),
        };

        cpu.reg_file.write(rd, rst);
        Ok(())
    })
}

pub(super) fn exec_compress_branch<F>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    F: ExecTrait<bool>,
{
    if let RVInstrInfo::CB { rd_rs1: rs1, imm } = info {
        let val1 = cpu.reg_file.read(rs1, 0).0;

        if F::exec(val1, 0) {
            let target = cpu.pc.wrapping_add(imm);
            cpu.pc = target;
        } else {
            cpu.pc = cpu.pc.wrapping_add(2);
        }
    } else {
        debug_unreachable!();
    }

    cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
    Ok(())
}

pub(super) fn exec_compress_jump<const LINK: bool>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    let RVInstrInfo::CJ { target } = info else {
        debug_unreachable!();
    };
    let target = cpu.pc.wrapping_add(target);
    if LINK {
        let link = cpu.pc.wrapping_add(2);
        cpu.reg_file.write(1, link); // C.JAL always write to ra(x1)
    }
    cpu.pc = target;

    cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
    Ok(())
}

pub(super) fn exec_compress_jump_reg<const LINK: bool>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    let RVInstrInfo::CR {
        rd_rs1: rs1,
        rs2: _rs2, // rs2 is always 0 in C.JR and C.JALR
    } = info
    else {
        debug_unreachable!();
    };

    let t = cpu.pc + 2;
    let val = cpu.reg_file.read(rs1, 0).0;
    let target: WordType = val & !1; // imm has been sign_extended

    cpu.pc = target;
    if LINK {
        cpu.reg_file.write(1, t);
    }
    cpu.csr.get_by_type_existing::<Minstret>().wrapping_add(1);
    Ok(())
}

pub(super) fn exec_compress_load_sp<T, const EXTEND: bool>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    T: UnsignedInteger,
{
    normal_compress_exec(cpu, |cpu| {
        let RVInstrInfo::CI { rd_rs1: rd, imm } = info else {
            debug_unreachable!();
        };
        let val = cpu.reg_file.read(2, 0).0; // read from sp(x2)
        let addr = wrapping_add_as_signed(val, imm);
        super::exec_core::handle_load::<T, EXTEND>(cpu, rd, addr)
    })
}

pub(super) fn exec_compress_load<T, const EXTEND: bool>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    T: UnsignedInteger,
{
    normal_compress_exec(cpu, |cpu| {
        let RVInstrInfo::CL { rd, rs1, imm } = info else {
            debug_unreachable!();
        };
        let val = cpu.reg_file.read(rs1, 0).0;
        let addr = wrapping_add_as_signed(val, imm);
        super::exec_core::handle_load::<T, EXTEND>(cpu, rd, addr)
    })
}

pub(super) fn exec_compress_store_sp<T>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    T: UnsignedInteger,
{
    normal_compress_exec(cpu, |cpu| {
        let RVInstrInfo::CSS { rs2, imm } = info else {
            debug_unreachable!();
        };
        let (val1, data) = cpu.reg_file.read(2, rs2);
        let addr = wrapping_add_as_signed(val1, imm);
        super::exec_core::handle_store::<T>(cpu, addr, data)
    })
}

pub(super) fn exec_compress_store<T>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    T: UnsignedInteger,
{
    normal_compress_exec(cpu, |cpu| {
        let RVInstrInfo::CS { rs1, rs2, imm } = info else {
            debug_unreachable!();
        };
        let (val1, val2) = cpu.reg_file.read(rs1, rs2);
        let addr = wrapping_add_as_signed(val1, imm);
        super::exec_core::handle_store::<T>(cpu, addr, val2)
    })
}

pub(super) fn exec_compress_float_store_sp<F>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    F: FloatPoint,
{
    normal_compress_exec(cpu, |cpu| {
        let RVInstrInfo::CSS { rs2, imm } = info else {
            debug_unreachable!();
        };
        let addr = cpu.reg_file.read(2, 0).0;
        let val: F::BitsType = cpu.fpu.load_raw(rs2).truncate_to();
        let addr = wrapping_add_as_signed(addr, imm); // imm has been sign_extended
        super::exec_core::handle_float_store::<F>(cpu, addr, val)
    })
}

pub(super) fn exec_compress_float_store<F>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    F: FloatPoint,
{
    normal_compress_exec(cpu, |cpu| {
        let RVInstrInfo::CS { rs1, rs2, imm } = info else {
            debug_unreachable!();
        };

        let addr = cpu.reg_file.read(rs1, 0).0;
        let val: F::BitsType = cpu.fpu.load_raw(rs2).truncate_to();
        let addr = wrapping_add_as_signed(addr, imm); // imm has been sign_extended
        super::exec_core::handle_float_store::<F>(cpu, addr, val)
    })
}

pub(super) fn exec_compress_float_load_sp<F>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    F: FloatPoint,
{
    normal_compress_float_exec(cpu, |cpu| {
        let RVInstrInfo::CI { rd_rs1: rd, imm } = info else {
            debug_unreachable!();
        };
        let val = cpu.reg_file.read(2, 0).0; // read from sp(x2)
        let addr = wrapping_add_as_signed(val, imm);
        super::exec_core::handle_float_load::<F>(cpu, addr, rd)
    })
}

pub(super) fn exec_compress_float_load<F>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    F: FloatPoint,
{
    normal_compress_float_exec(cpu, |cpu| {
        let RVInstrInfo::CL { rd, rs1, imm } = info else {
            debug_unreachable!();
        };
        let val = cpu.reg_file.read(rs1, 0).0;
        let addr = wrapping_add_as_signed(val, imm);
        super::exec_core::handle_float_load::<F>(cpu, addr, rd)
    })
}

pub(super) fn exec_compress_nop(_info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception> {
    normal_compress_exec(cpu, |_| Ok(()))
}

// special instructions for C extension

pub(super) fn exec_compress_mv(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception> {
    let RVInstrInfo::CR { rd_rs1: rd, rs2 } = info else {
        debug_unreachable!();
    };

    normal_compress_exec(cpu, |cpu| {
        let val = cpu.reg_file.read(rs2, 0).0;
        cpu.reg_file.write(rd, val);
        Ok(())
    })
}

pub(super) fn exec_compress_li(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception> {
    let RVInstrInfo::CI { rd_rs1: rd, imm } = info else {
        debug_unreachable!();
    };

    normal_compress_exec(cpu, |cpu| {
        cpu.reg_file.write(rd, imm);
        Ok(())
    })
}

pub(super) fn exec_addi4spn(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception> {
    let RVInstrInfo::CIW { rd, imm } = info else {
        debug_unreachable!();
    };

    normal_compress_exec(cpu, |cpu| {
        let val1 = cpu.reg_file.read(2, 0).0; // always read x2/sp
        let rst = ExecAdd::exec(val1, imm)?;
        cpu.reg_file.write(rd, rst);
        Ok(())
    })
}
