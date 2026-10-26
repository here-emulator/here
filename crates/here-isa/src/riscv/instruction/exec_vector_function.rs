use crate::{
    config::arch_config::WordType,
    isa::{
        DebugTarget,
        riscv::{
            csr_reg::{
                NamedCsrReg,
                csr_macro::{Vcsr, Vl, Vtype},
            },
            executor::RVCPU,
            instruction::{RVInstrInfo, exec_function::ExecMove, normal_vector_exec},
            trap::Exception::{self, IllegalInstruction},
            vector::{
                arithmetic::{
                    VectorOpAdc, VectorOpBitVV, VectorOpCompressVm, VectorOpCpopM, VectorOpFirstM,
                    VectorOpFixedPointNarrowingVX, VectorOpFixedPointNarrowingWV,
                    VectorOpFixedPointVV, VectorOpFixedPointVX, VectorOpIdV, VectorOpIntegerMaskVV,
                    VectorOpIntegerMaskVX, VectorOpIntegerV, VectorOpIntegerVV, VectorOpIntegerVVM,
                    VectorOpIntegerVVV, VectorOpIntegerVX, VectorOpIntegerVXM, VectorOpIntegerVXV,
                    VectorOpIotaM, VectorOpMadc, VectorOpMerge, VectorOpMsbc, VectorOpMsbfM,
                    VectorOpMsifM, VectorOpMsofM, VectorOpNsra, VectorOpNsrl,
                    VectorOpRGatherEI16VV, VectorOpRGatherVI, VectorOpRGatherVV, VectorOpRGatherVX,
                    VectorOpSlide1Down, VectorOpSlide1Up, VectorOpSlideDown, VectorOpSlideUp,
                    VectorOpWideningIntegerVV, VectorOpWideningIntegerVVV,
                    VectorOpWideningIntegerVX, VectorOpWideningIntegerVXV,
                    VectorOpWideningIntegerWV, VectorOpWideningIntegerWX,
                },
                types::{FixedPointRoundingMode, Vsew},
            },
        },
    },
    utils::sign_extend,
};

pub(super) struct VectorConfigField {
    vtype: WordType,
    input_len: WordType,
}

pub(super) trait VectorConfigFieldExtractor {
    fn exec(imm: WordType, rs1: u8, cpu: &mut RVCPU) -> VectorConfigField;
}

pub(super) fn exec_vector_config<T>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    T: VectorConfigFieldExtractor,
{
    normal_vector_exec(cpu, |cpu, _vstart| {
        let mut vtype_csr = cpu.csr.get_by_type::<Vtype>().unwrap();
        if let RVInstrInfo::V {
            rs1,
            rs2,
            rd: vd,
            vm,
            func6,
        } = info
        {
            let imm = (func6 as WordType) << 6 | (vm as WordType) << 5 | (rs2 as WordType);
            let configfield = T::exec(imm, rs1, cpu);

            if let Some(maxvl) = vtype_csr.vsetvl(configfield.vtype) {
                let vl = configfield.input_len.min(maxvl);
                let vlmul = ((configfield.vtype & 0b111) as u8).into();
                let vsew = (((configfield.vtype >> 3) & 0b111) as u8).into();
                let vta = ((configfield.vtype >> 6) & 1) != 0;
                let vma = ((configfield.vtype >> 7) & 1) != 0;

                cpu.csr
                    .write_directly(Vl::get_index(), vl)
                    .then_some(())
                    .unwrap();
                cpu.vector.set_config((vlmul, vsew, vta, vma, vl as u16));
                cpu.write_reg(vd, vl);

                Ok(())
            } else {
                cpu.write_reg(vd, 0);
                Err(Exception::IllegalInstruction)
            }
        } else {
            std::unreachable!();
        }
    })
}

pub(super) struct VsetivliFieldExtractor {}
impl VectorConfigFieldExtractor for VsetivliFieldExtractor {
    fn exec(imm: WordType, rs1: u8, _cpu: &mut RVCPU) -> VectorConfigField {
        debug_assert!((imm & 0b1100_0000_0000) == 0b1100_0000_0000);
        let vtype = imm & !0b1100_0000_0000;
        let input_len = rs1 as WordType;
        VectorConfigField { vtype, input_len }
    }
}

pub(super) struct VsetvliFieldExtractor {}
impl VectorConfigFieldExtractor for VsetvliFieldExtractor {
    fn exec(imm: WordType, rs1: u8, cpu: &mut RVCPU) -> VectorConfigField {
        debug_assert!((imm & 0b1000_0000_0000) == 0b0000_0000_0000);
        let vtype = imm & !0b1000_0000_0000;
        let input_len = cpu.read_reg(rs1);
        VectorConfigField { vtype, input_len }
    }
}

pub(super) struct VsetvlFieldExtractor {}
impl VectorConfigFieldExtractor for VsetvlFieldExtractor {
    fn exec(imm: WordType, rs1: u8, cpu: &mut RVCPU) -> VectorConfigField {
        debug_assert!(
            (imm & 0b1111_1110_0000) == 0b1000_0000_0000,
            "imm = {:b}",
            imm
        );
        let rs2 = (imm as u8) & 0b11111;
        let vtype = cpu.read_reg(rs2);
        let input_len = cpu.read_reg(rs1);
        VectorConfigField { vtype, input_len }
    }
}

// ---------------------------------------
//          Vector Load/Store
// ---------------------------------------
struct Func6Uop {
    nf: u8,
    mew: u8,
    mop: u8,
}

#[inline(always)]
fn load_store_func6_decode(func6: u8) -> Func6Uop {
    let nf = (func6 >> 3) & 0b111;
    let mew = (func6 >> 2) & 0b1;
    let mop = func6 & 0b11;
    Func6Uop { nf, mew, mop }
}

pub(super) fn vector_unit_stride_load<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: lumop,
            rd: vd,
            vm,
            func6,
        } = info
        {
            let Func6Uop { nf, mop, .. } = load_store_func6_decode(func6);
            debug_assert_eq!(mop, 0b00);
            debug_assert_eq!(lumop, 0b00000);
            let base_addr = cpu.reg_file.read(rs1, 0).0;
            cpu.vector.stride_load(
                vd,
                EEW.into(),
                nf + 1,
                None,
                !vm,
                vstart,
                base_addr,
                &mut cpu.memory.mmio,
            )
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_whole_register_load<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: lumop,
            rd: vd,
            vm,
            func6,
        } = info
        {
            let Func6Uop { nf, mop, .. } = load_store_func6_decode(func6);
            debug_assert_eq!(mop, 0b00);
            debug_assert_eq!(lumop, 0b01000);
            debug_assert!(vm);
            let _ = EEW;
            let base_addr = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .load_whole_register(vd, nf, vstart, base_addr, &mut cpu.memory.mmio)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_mask_load<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: lumop,
            rd: vd,
            vm,
            func6,
        } = info
        {
            let Func6Uop { nf, mew, mop } = load_store_func6_decode(func6);
            debug_assert_eq!(EEW, 0);
            debug_assert_eq!((nf, mew, mop, lumop, vm), (0, 0, 0b00, 0b01011, true));
            let base_addr = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .mask_load(vd, vstart, base_addr, &mut cpu.memory.mmio)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_unit_stride_fault_only_first_load<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |_cpu, _vstart| -> Result<(), Exception> {
        if let RVInstrInfo::V {
            rs2: lumop, func6, ..
        } = info
        {
            let Func6Uop { mop, .. } = load_store_func6_decode(func6);
            debug_assert_eq!(mop, 0b00);
            debug_assert_eq!(lumop, 0b10000);
            let _ = EEW;
            unimplemented!()
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_indexed_unordered_load<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |_cpu, _vstart| -> Result<(), Exception> {
        if let RVInstrInfo::V { func6, .. } = info {
            let Func6Uop { mop, .. } = load_store_func6_decode(func6);
            debug_assert_eq!(mop, 0b01);
            let _ = EEW;
            unimplemented!()
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_constant_stride_load<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2,
            rd: vd,
            vm,
            func6,
        } = info
        {
            let Func6Uop { nf, mop, .. } = load_store_func6_decode(func6);
            debug_assert_eq!(mop, 0b10);
            let (base_addr, stride) = cpu.reg_file.read(rs1, rs2);
            cpu.vector.stride_load(
                vd,
                EEW.into(),
                nf + 1,
                Some(stride),
                !vm,
                vstart,
                base_addr,
                &mut cpu.memory.mmio,
            )
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_indexed_ordered_load<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: base_addr,
            rs2: index_arr_base,
            rd: vd,
            vm,
            func6,
        } = info
        {
            let Func6Uop { nf, mop, .. } = load_store_func6_decode(func6);
            debug_assert_eq!(mop, 0b11);
            let (base_addr, index_arr_base) = cpu.reg_file.read(base_addr, index_arr_base);
            cpu.vector.indexed_ordered_load(
                vd,
                EEW.into(),
                nf + 1,
                index_arr_base,
                !vm,
                vstart,
                base_addr,
                &mut cpu.memory.mmio,
            )
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_unit_stride_store<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: sumop,
            rd: vs3,
            vm,
            func6,
        } = info
        {
            let Func6Uop { nf, mop, .. } = load_store_func6_decode(func6);
            debug_assert_eq!(mop, 0b00);
            debug_assert_eq!(sumop, 0b00000);
            let base_addr = cpu.reg_file.read(rs1, 0).0;
            cpu.vector.stride_store(
                vs3,
                EEW.into(),
                nf + 1,
                None,
                !vm,
                vstart,
                base_addr,
                &mut cpu.memory.mmio,
            )
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_whole_register_store<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: sumop,
            rd: vs3,
            vm,
            func6,
        } = info
        {
            let Func6Uop { nf, mew, mop } = load_store_func6_decode(func6);
            debug_assert_eq!((mew, mop, sumop, vm), (0, 0b00, 0b01000, true));
            let _ = EEW;
            let base_addr = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .store_whole_register(vs3, nf, vstart, base_addr, &mut cpu.memory.mmio)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_mask_store<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: sumop,
            rd: vs3,
            vm,
            func6,
        } = info
        {
            let Func6Uop { nf, mew, mop } = load_store_func6_decode(func6);
            debug_assert_eq!(EEW, 0);
            debug_assert_eq!((nf, mew, mop, sumop, vm), (0, 0, 0b00, 0b01011, true));
            let base_addr = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .mask_store(vs3, vstart, base_addr, &mut cpu.memory.mmio)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_indexed_unordered_store<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |_cpu, _vstart| -> Result<(), Exception> {
        if let RVInstrInfo::V { func6, .. } = info {
            let Func6Uop { mop, .. } = load_store_func6_decode(func6);
            debug_assert_eq!(mop, 0b01);
            let _ = EEW;
            unimplemented!()
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_constant_stride_store<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2,
            rd: vs3,
            vm,
            func6,
        } = info
        {
            let Func6Uop { nf, mop, .. } = load_store_func6_decode(func6);
            debug_assert_eq!(mop, 0b10);
            let (base_addr, stride) = cpu.reg_file.read(rs1, rs2);
            cpu.vector.stride_store(
                vs3,
                EEW.into(),
                nf + 1,
                Some(stride),
                !vm,
                vstart,
                base_addr,
                &mut cpu.memory.mmio,
            )
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vector_indexed_ordered_store<const EEW: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: base_addr,
            rs2: index_arr_base,
            rd: vs3,
            vm,
            func6,
        } = info
        {
            let Func6Uop { nf, mop, .. } = load_store_func6_decode(func6);
            debug_assert_eq!(mop, 0b11);
            let (base_addr, index_arr_base) = cpu.reg_file.read(base_addr, index_arr_base);
            cpu.vector.indexed_ordered_store(
                vs3,
                EEW.into(),
                nf + 1,
                index_arr_base,
                !vm,
                vstart,
                base_addr,
                &mut cpu.memory.mmio,
            )
        } else {
            unreachable!()
        }
    })
}

// ---------------------------------------
//          Vector Integer
// ---------------------------------------
pub(super) fn vec_integer_op_vv<OpIVV>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    OpIVV: VectorOpIntegerVV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: vs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            cpu.vector
                .exec_integer_vv::<OpIVV>(vs1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_bit_op_vv<OpIVV>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    OpIVV: VectorOpBitVV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: vs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            cpu.vector.exec_bit_vv::<OpIVV>(vs1, vs2, vd, !vm, vstart)
        } else {
            std::unreachable!();
        }
    })
}

pub(super) fn vec_integer_op_vx<OpIVX>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    OpIVX: VectorOpIntegerVX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let x1 = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .exec_integer_vx::<OpIVX>(x1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_integer_op_vvv<OpIVV>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    OpIVV: VectorOpIntegerVVV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: vs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            cpu.vector
                .exec_integer_vvv::<OpIVV>(vs1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_integer_op_vxv<OpIVX>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    OpIVX: VectorOpIntegerVXV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let x1 = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .exec_integer_vxv::<OpIVX>(x1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_widening_integer_op_vv<OpIVV>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIVV: VectorOpWideningIntegerVV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: vs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            cpu.vector
                .exec_widening_integer_vv::<OpIVV>(vs1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_widening_integer_op_vvv<OpIVV>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIVV: VectorOpWideningIntegerVVV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: vs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            cpu.vector
                .exec_widening_integer_vvv::<OpIVV>(vs1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_widening_integer_op_vxv<OpIVX>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIVX: VectorOpWideningIntegerVXV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let x1 = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .exec_widening_integer_vxv::<OpIVX>(x1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_widening_integer_op_vx<OpIVX>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIVX: VectorOpWideningIntegerVX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let x1 = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .exec_widening_integer_vx::<OpIVX>(x1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_widening_integer_op_wv<OpIVV>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIVV: VectorOpWideningIntegerWV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: vs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            cpu.vector
                .exec_widening_integer_wv::<OpIVV>(vs1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_widening_integer_op_wx<OpIVX>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIVX: VectorOpWideningIntegerWX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let x1 = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .exec_widening_integer_wx::<OpIVX>(x1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_integer_op_vi_signed<OpIVX>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIVX: VectorOpIntegerVX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: simm5,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let imm = sign_extend(simm5 as WordType, 5);
            cpu.vector
                .exec_integer_vx::<OpIVX>(imm, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_integer_op_vi_unsigned<OpIVX>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIVX: VectorOpIntegerVX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: uimm,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            cpu.vector
                .exec_integer_vx::<OpIVX>(uimm as WordType, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

fn fixed_point_rounding_mode(cpu: &mut RVCPU) -> FixedPointRoundingMode {
    match cpu.csr.get_by_type_existing::<Vcsr>().get_vxrm() {
        0 => FixedPointRoundingMode::RoundToNearestUp,
        1 => FixedPointRoundingMode::RoundToNearestEven,
        2 => FixedPointRoundingMode::RoundDown,
        3 => FixedPointRoundingMode::RoundToOdd,
        _ => unreachable!(),
    }
}

fn finish_fixed_point_op(cpu: &mut RVCPU, saturated: bool) {
    if saturated {
        cpu.csr.get_by_type_existing::<Vcsr>().set_vxsat_directly(1);
    }
}

pub(super) fn vec_fixed_point_op_vv<Op>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    Op: VectorOpFixedPointVV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: vs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let round = fixed_point_rounding_mode(cpu);
            let saturated = cpu
                .vector
                .exec_fixed_point_vv::<Op>(vs1, vs2, vd, !vm, round, vstart)?;
            finish_fixed_point_op(cpu, saturated);
            Ok::<(), Exception>(())
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_fixed_point_op_vx<Op>(info: RVInstrInfo, cpu: &mut RVCPU) -> Result<(), Exception>
where
    Op: VectorOpFixedPointVX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let x1 = cpu.reg_file.read(rs1, 0).0;
            let round = fixed_point_rounding_mode(cpu);
            let saturated = cpu
                .vector
                .exec_fixed_point_vx::<Op>(x1, vs2, vd, !vm, round, vstart)?;
            finish_fixed_point_op(cpu, saturated);
            Ok::<(), Exception>(())
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_fixed_point_op_vi<Op, const SIGNED: bool>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    Op: VectorOpFixedPointVX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: imm5,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let imm = if SIGNED {
                sign_extend(imm5 as WordType, 5)
            } else {
                imm5 as WordType
            };
            let round = fixed_point_rounding_mode(cpu);
            let saturated = cpu
                .vector
                .exec_fixed_point_vx::<Op>(imm, vs2, vd, !vm, round, vstart)?;
            finish_fixed_point_op(cpu, saturated);
            Ok::<(), Exception>(())
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_fixed_point_narrowing_op_wv<Op>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    Op: VectorOpFixedPointNarrowingWV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: vs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let round = fixed_point_rounding_mode(cpu);
            let saturated = cpu
                .vector
                .exec_fixed_point_narrowing_wv::<Op>(vs1, vs2, vd, !vm, round, vstart)?;
            finish_fixed_point_op(cpu, saturated);
            Ok::<(), Exception>(())
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_fixed_point_narrowing_op_wx<Op>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    Op: VectorOpFixedPointNarrowingVX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let x1 = cpu.reg_file.read(rs1, 0).0;
            let round = fixed_point_rounding_mode(cpu);
            let saturated = cpu
                .vector
                .exec_fixed_point_narrowing_vx::<Op>(x1, vs2, vd, !vm, round, vstart)?;
            finish_fixed_point_op(cpu, saturated);
            Ok::<(), Exception>(())
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_fixed_point_narrowing_op_wi<Op>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    Op: VectorOpFixedPointNarrowingVX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: uimm,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let round = fixed_point_rounding_mode(cpu);
            let saturated = cpu.vector.exec_fixed_point_narrowing_vx::<Op>(
                uimm as WordType,
                vs2,
                vd,
                !vm,
                round,
                vstart,
            )?;
            finish_fixed_point_op(cpu, saturated);
            Ok::<(), Exception>(())
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_integer_op_vvm<OpIVVM>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIVVM: VectorOpIntegerVVM,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: vs1,
            rs2: vs2,
            rd: vd,
            ..
        } = info
        {
            cpu.vector
                .exec_integer_vvm::<OpIVVM>(vs1, vs2, 0, vd, false, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_integer_op_vxm<OpIVXM>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIVXM: VectorOpIntegerVXM,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: vs2,
            rd: vd,
            ..
        } = info
        {
            let x1 = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .exec_integer_vxm::<OpIVXM>(x1, vs2, 0, vd, false, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_whole_register_move_op_v<const LMUL: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    normal_vector_exec(cpu, |cpu, vstart| {
        let expected_rs1 = match LMUL {
            1 => 0,
            2 => 1,
            4 => 3,
            8 => 7,
            _ => return Err(Exception::IllegalInstruction),
        };
        if let RVInstrInfo::V {
            rs1,
            rs2: vs2,
            rd: vd,
            ..
        } = info
        {
            if rs1 != expected_rs1 {
                return Err(Exception::IllegalInstruction);
            }
            cpu.vector
                .exec_whole_register_move::<ExecMove<u64>>(vs2, vd, LMUL, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_integer_mask_op_vv<OpIMVV>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIMVV: VectorOpIntegerMaskVV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: vs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            cpu.vector
                .exec_integer_mask_vv::<OpIMVV>(vs1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_integer_mask_op_vx<OpIMVX>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIMVX: VectorOpIntegerMaskVX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let x1 = cpu.reg_file.read(rs1, 0).0;
            cpu.vector
                .exec_integer_mask_vx::<OpIMVX>(x1, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_integer_mask_op_vi<OpIMVX>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIMVX: VectorOpIntegerMaskVX,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1: simm5,
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            let imm = sign_extend(simm5 as WordType, 5);
            cpu.vector
                .exec_integer_mask_vx::<OpIMVX>(imm, vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

pub(super) fn vec_integer_ext_op_v<OpIV, const FACTOR: u8>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception>
where
    OpIV: VectorOpIntegerV,
{
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs2: vs2,
            rd: vd,
            vm,
            ..
        } = info
        {
            cpu.vector
                .exec_integer_v_ext::<OpIV, FACTOR>(vs2, vd, !vm, vstart)
        } else {
            unreachable!()
        }
    })
}

// Spec-only vector instructions that do not justify a dedicated wrapper.
//
// These ids are `usize` constants to keep `vec_integer_spec_op` usable on stable
// const generics; using a Rust enum as the const parameter would need extra
// language support. Add a new id here only when the instruction has low reuse
// and has instruction-specific legality or operand decoding.
pub(super) mod vector_spec_instr {
    pub(in super::super) const SLIDEUP_VI: usize = 1;
    pub(in super::super) const SLIDEDOWN_VI: usize = 2;
    pub(in super::super) const GATHER_VI: usize = 3;
    pub(in super::super) const COMPRESS_VM: usize = 4;
    pub(in super::super) const CPOP_M: usize = 5;
    pub(in super::super) const FIRST_M: usize = 6;
    pub(in super::super) const MSBF_M: usize = 7;
    pub(in super::super) const MSOF_M: usize = 8;
    pub(in super::super) const MSIF_M: usize = 9;
    pub(in super::super) const IOTA_M: usize = 10;
    pub(in super::super) const ID_V: usize = 11;
    pub(in super::super) const SLIDEUP_VX: usize = 12;
    pub(in super::super) const SLIDEDOWN_VX: usize = 13;
    pub(in super::super) const GATHER_VV: usize = 14;
    pub(in super::super) const GATHER_VX: usize = 15;
    pub(in super::super) const GATHER_EI16_VV: usize = 16;
    pub(in super::super) const MOVE_V: usize = 17;
    pub(in super::super) const MOVE_VX: usize = 18;
    pub(in super::super) const MOVE_VI: usize = 19;
    pub(in super::super) const NSRL_WV: usize = 20;
    pub(in super::super) const NSRA_WV: usize = 21;
    pub(in super::super) const NSRL_WX: usize = 22;
    pub(in super::super) const NSRA_WX: usize = 23;
    pub(in super::super) const NSRL_WI: usize = 24;
    pub(in super::super) const NSRA_WI: usize = 25;
    pub(in super::super) const MADC_VVM: usize = 26;
    pub(in super::super) const MSBC_VVM: usize = 27;
    pub(in super::super) const MADC_VXM: usize = 28;
    pub(in super::super) const MSBC_VXM: usize = 29;
    pub(in super::super) const MADC_VIM: usize = 30;
    pub(in super::super) const ADC_VIM: usize = 31;
    pub(in super::super) const MERGE_VIM: usize = 32;
    pub(in super::super) const MOVE_SX: usize = 33;
    pub(in super::super) const MOVE_XS: usize = 34;
    pub(in super::super) const SLIDE1UP_VX: usize = 35;
    pub(in super::super) const SLIDE1DOWN_VX: usize = 36;
}

pub(super) fn vec_integer_spec_op<const VINSTR: usize>(
    info: RVInstrInfo,
    cpu: &mut RVCPU,
) -> Result<(), Exception> {
    // Low-reuse instructions share the same RVInstrInfo decoding shape, but
    // differ in immediate handling, scalar register reads, and legality checks.
    // Keep those differences in this match so exec_mapping stays table-like
    // without reintroducing one-off wrapper functions.
    normal_vector_exec(cpu, |cpu, vstart| {
        if let RVInstrInfo::V {
            rs1, rs2, rd, vm, ..
        } = info
        {
            match VINSTR {
                vector_spec_instr::SLIDEUP_VI => {
                    let (uimm, vs2, vd) = (rs1, rs2, rd);
                    cpu.vector.exec_integer_slideup::<VectorOpSlideUp>(
                        uimm as WordType,
                        vs2,
                        vd,
                        !vm,
                        vstart,
                    )
                }
                vector_spec_instr::SLIDEUP_VX => {
                    let (vs2, vd) = (rs2, rd);
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    if vd == vs2 {
                        return Err(Exception::IllegalInstruction);
                    }
                    cpu.vector
                        .exec_integer_slideup::<VectorOpSlideUp>(x1, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::SLIDEDOWN_VI => {
                    let (uimm, vs2, vd) = (rs1, rs2, rd);
                    cpu.vector.exec_integer_slidedown::<VectorOpSlideDown>(
                        uimm as WordType,
                        vs2,
                        vd,
                        !vm,
                        vstart,
                    )
                }
                vector_spec_instr::SLIDEDOWN_VX => {
                    let (vs2, vd) = (rs2, rd);
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    cpu.vector
                        .exec_integer_slidedown::<VectorOpSlideDown>(x1, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::SLIDE1UP_VX => {
                    let (vs2, vd) = (rs2, rd);
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    cpu.vector
                        .exec_integer_slideup::<VectorOpSlide1Up>(x1, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::SLIDE1DOWN_VX => {
                    let (vs2, vd) = (rs2, rd);
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    cpu.vector
                        .exec_integer_slidedown::<VectorOpSlide1Down>(x1, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::GATHER_VV => {
                    let (vs1, vs2, vd) = (rs1, rs2, rd);
                    if vd == vs2 || vd == vs1 {
                        return Err(Exception::IllegalInstruction);
                    }
                    cpu.vector
                        .exec_integer_gather_vv::<VectorOpRGatherVV>(vs1, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::GATHER_VX => {
                    let (vs2, vd) = (rs2, rd);
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    if vd == vs2 {
                        return Err(Exception::IllegalInstruction);
                    }
                    cpu.vector
                        .exec_integer_gather_vx::<VectorOpRGatherVX>(x1, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::GATHER_VI => {
                    let (imm, vs2, vd) = (rs1, rs2, rd);
                    if vd == vs2 {
                        return Err(Exception::IllegalInstruction);
                    }
                    cpu.vector.exec_integer_gather_vx::<VectorOpRGatherVI>(
                        imm as WordType,
                        vs2,
                        vd,
                        !vm,
                        vstart,
                    )
                }
                vector_spec_instr::GATHER_EI16_VV => {
                    let (vs1, vs2, vd) = (rs1, rs2, rd);
                    if vd == vs2 || vd == vs1 {
                        return Err(Exception::IllegalInstruction);
                    }
                    cpu.vector
                        .exec_integer_gather_ei16_vv::<VectorOpRGatherEI16VV>(
                            vs1, vs2, vd, !vm, vstart,
                        )
                }
                vector_spec_instr::MOVE_V => {
                    let (vs1, vs2, vd) = (rs1, rs2, rd);
                    if vs2 != 0 {
                        return Err(Exception::IllegalInstruction);
                    }
                    cpu.vector.exec_integer_v::<ExecMove<u64>>(
                        vs1,
                        vd,
                        Vsew::E64,
                        Vsew::E64,
                        false,
                        vstart,
                    )
                }
                vector_spec_instr::MOVE_VX => {
                    let (vs2, vd) = (rs2, rd);
                    if vs2 != 0 {
                        return Err(Exception::IllegalInstruction);
                    }
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    cpu.vector
                        .exec_integer_scalar_move::<ExecMove<u64>, u64>(x1 as u64, vd, vstart)
                }
                vector_spec_instr::MOVE_VI => {
                    let (simm5, vs2, vd) = (rs1, rs2, rd);
                    if vs2 != 0 {
                        return Err(Exception::IllegalInstruction);
                    }
                    let imm = sign_extend(simm5 as WordType, 5);
                    cpu.vector
                        .exec_integer_scalar_move::<ExecMove<u64>, u64>(imm as u64, vd, vstart)
                }
                vector_spec_instr::MOVE_SX => {
                    let (rs1, vd) = (rs1, rd);
                    if rs2 != 0b00000 {
                        return Err(IllegalInstruction);
                    }
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    cpu.vector.exec_move_scalar_to_element(x1, vd, vstart)
                }
                vector_spec_instr::MOVE_XS => {
                    let (vs2, rd) = (rs2, rd);
                    if rs1 != 0b00000 {
                        return Err(IllegalInstruction);
                    }
                    if let Some(value) = cpu.vector.exec_move_element_to_scalar(vs2, vstart)? {
                        cpu.write_reg(rd, value);
                    }
                    Ok(())
                }
                vector_spec_instr::NSRL_WV => {
                    let (vs1, vs2, vd) = (rs1, rs2, rd);
                    cpu.vector
                        .exec_integer_narrowing_wv::<VectorOpNsrl>(vs1, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::NSRA_WV => {
                    let (vs1, vs2, vd) = (rs1, rs2, rd);
                    cpu.vector
                        .exec_integer_narrowing_wv::<VectorOpNsra>(vs1, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::NSRL_WX => {
                    let (vs2, vd) = (rs2, rd);
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    cpu.vector
                        .exec_integer_narrowing_vx::<VectorOpNsrl>(x1, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::NSRA_WX => {
                    let (vs2, vd) = (rs2, rd);
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    cpu.vector
                        .exec_integer_narrowing_vx::<VectorOpNsra>(x1, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::NSRL_WI => {
                    let (uimm, vs2, vd) = (rs1, rs2, rd);
                    cpu.vector.exec_integer_narrowing_vx::<VectorOpNsrl>(
                        uimm as WordType,
                        vs2,
                        vd,
                        !vm,
                        vstart,
                    )
                }
                vector_spec_instr::NSRA_WI => {
                    let (simm5, vs2, vd) = (rs1, rs2, rd);
                    let imm = sign_extend(simm5 as WordType, 5);
                    cpu.vector
                        .exec_integer_narrowing_vx::<VectorOpNsra>(imm, vs2, vd, !vm, vstart)
                }
                vector_spec_instr::MADC_VVM => {
                    let (vs1, vs2, vd) = (rs1, rs2, rd);
                    cpu.vector
                        .exec_integer_mask_vvm::<VectorOpMadc>(vs1, vs2, 0, vd, false, vstart)
                }
                vector_spec_instr::MSBC_VVM => {
                    let (vs1, vs2, vd) = (rs1, rs2, rd);
                    cpu.vector
                        .exec_integer_mask_vvm::<VectorOpMsbc>(vs1, vs2, 0, vd, false, vstart)
                }
                vector_spec_instr::MADC_VXM => {
                    let (vs2, vd) = (rs2, rd);
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    cpu.vector
                        .exec_integer_mask_vxm::<VectorOpMadc>(x1, vs2, 0, vd, false, vstart)
                }
                vector_spec_instr::MSBC_VXM => {
                    let (vs2, vd) = (rs2, rd);
                    let x1 = cpu.reg_file.read(rs1, 0).0;
                    cpu.vector
                        .exec_integer_mask_vxm::<VectorOpMsbc>(x1, vs2, 0, vd, false, vstart)
                }
                vector_spec_instr::MADC_VIM => {
                    let (simm5, vs2, vd) = (rs1, rs2, rd);
                    let imm = sign_extend(simm5 as WordType, 5);
                    cpu.vector
                        .exec_integer_mask_vxm::<VectorOpMadc>(imm, vs2, 0, vd, false, vstart)
                }
                vector_spec_instr::ADC_VIM => {
                    let (simm5, vs2, vd) = (rs1, rs2, rd);
                    let imm = sign_extend(simm5 as WordType, 5);
                    cpu.vector
                        .exec_integer_vxm::<VectorOpAdc>(imm, vs2, 0, vd, false, vstart)
                }
                vector_spec_instr::MERGE_VIM => {
                    let (simm5, vs2, vd) = (rs1, rs2, rd);
                    let imm = sign_extend(simm5 as WordType, 5);
                    cpu.vector
                        .exec_integer_vxm::<VectorOpMerge>(imm, vs2, 0, vd, false, vstart)
                }
                vector_spec_instr::COMPRESS_VM => {
                    let (vs1, vs2, vd) = (rs1, rs2, rd);
                    if !vm {
                        return Err(Exception::IllegalInstruction);
                    }
                    cpu.vector
                        .exec_compress::<VectorOpCompressVm>(vs1, vs2, vd, vstart)
                }
                vector_spec_instr::CPOP_M => {
                    let (vs2, rd) = (rs2, rd);
                    let value = cpu
                        .vector
                        .exec_mask_to_x::<VectorOpCpopM>(vs2, !vm, vstart)?;
                    cpu.write_reg(rd, value);
                    Ok(())
                }
                vector_spec_instr::FIRST_M => {
                    let (vs2, rd) = (rs2, rd);
                    let value = cpu
                        .vector
                        .exec_mask_to_x::<VectorOpFirstM>(vs2, !vm, vstart)?;
                    cpu.write_reg(rd, value);
                    Ok(())
                }
                vector_spec_instr::MSBF_M => {
                    let (vs2, vd) = (rs2, rd);
                    cpu.vector
                        .exec_mask_unary::<VectorOpMsbfM>(vs2, vd, !vm, vstart)
                }
                vector_spec_instr::MSOF_M => {
                    let (vs2, vd) = (rs2, rd);
                    cpu.vector
                        .exec_mask_unary::<VectorOpMsofM>(vs2, vd, !vm, vstart)
                }
                vector_spec_instr::MSIF_M => {
                    let (vs2, vd) = (rs2, rd);
                    cpu.vector
                        .exec_mask_unary::<VectorOpMsifM>(vs2, vd, !vm, vstart)
                }
                vector_spec_instr::IOTA_M => {
                    let (vs2, vd) = (rs2, rd);
                    cpu.vector
                        .exec_mask_to_vector::<VectorOpIotaM>(vs2, vd, !vm, vstart)
                }
                vector_spec_instr::ID_V => {
                    let vd = rd;
                    cpu.vector.exec_index::<VectorOpIdV>(vd, !vm, vstart)
                }
                _ => {
                    unreachable!()
                }
            }
        } else {
            unreachable!()
        }
    })
}

#[cfg(test)]
mod test {
    use std::{cell::UnsafeCell, rc::Rc};

    use crate::{
        device::mmio::MemoryMapIO,
        isa::riscv::{
            cpu_tester::{CPUChecker, TestCPUBuilder, run_test_exec, run_test_exec_decode},
            csr_reg::{
                NamedCsrReg,
                csr_macro::{Mstatus, Vstart},
            },
            instruction::instr_table::RiscvInstr,
            mmu::VirtAddrManager,
            vector::{
                VLEN_BYTE,
                types::{Vlmul, Vsew},
            },
        },
        ram::Ram,
        ram_config::BASE_ADDR,
    };

    use super::*;

    const TEST_DATA_ADDR_OFFSET: WordType = 0x1000;
    const TEST_DATA_BASE: WordType = BASE_ADDR + TEST_DATA_ADDR_OFFSET;

    fn u32_vec_to_bytes(data: &[u32]) -> Vec<u8> {
        data.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn vector_add_vv_test_cpu() {
        let vs1 = [1_u32, 2, 3, u32::MAX];
        let vs2 = [10_u32, 20, 30, 1];
        let expected = [11_u32, 22, 33, 0];

        run_test_exec(
            RiscvInstr::VADD_VV,
            RVInstrInfo::V {
                rs1: 1,
                rs2: 2,
                rd: 3,
                vm: true,
                func6: 0,
            },
            |builder| {
                builder
                    .vector_status(Vlmul::M1, Vsew::E32, false, false)
                    .reg_vec(1, 1, &u32_vec_to_bytes(&vs1))
                    .reg_vec(1, 2, &u32_vec_to_bytes(&vs2))
                    .pc(0x2000)
            },
            |checker| checker.reg_vec(3, &expected).pc(0x2004),
        );
    }

    #[test]
    fn vector_arithmetic_respects_vstart_and_clears_it() {
        let vs1 = [1_u32, 2, 3, 4];
        let vs2 = [10_u32, 20, 30, 40];
        let old_vd = [0xaaaa_0001_u32, 0xaaaa_0002, 0xaaaa_0003, 0xaaaa_0004];
        let expected = [old_vd[0], old_vd[1], 33, 44];

        run_test_exec(
            RiscvInstr::VADD_VV,
            RVInstrInfo::V {
                rs1: 1,
                rs2: 2,
                rd: 3,
                vm: true,
                func6: 0,
            },
            |builder| {
                builder
                    .vector_status(Vlmul::M1, Vsew::E32, false, false)
                    .csr(Vstart::get_index(), 2)
                    .reg_vec(1, 1, &u32_vec_to_bytes(&vs1))
                    .reg_vec(1, 2, &u32_vec_to_bytes(&vs2))
                    .reg_vec(1, 3, &u32_vec_to_bytes(&old_vd))
                    .pc(0x2000)
            },
            |checker| {
                checker
                    .reg_vec(3, &expected)
                    .csr(Vstart::get_index(), 0)
                    .pc(0x2004)
            },
        );
    }

    #[test]
    fn vector_add_vx_test_cpu() {
        let vs2 = [10_u32, 20, 30, u32::MAX];
        let expected = [15_u32, 25, 35, 4];

        run_test_exec(
            RiscvInstr::VADD_VX,
            RVInstrInfo::V {
                rs1: 5,
                rs2: 2,
                rd: 3,
                vm: true,
                func6: 0,
            },
            |builder| {
                builder
                    .vector_status(Vlmul::M1, Vsew::E32, false, false)
                    .reg(5, 5)
                    .reg_vec(1, 2, &u32_vec_to_bytes(&vs2))
                    .pc(0x2000)
            },
            |checker| checker.reg_vec(3, &expected).pc(0x2004),
        );
    }

    #[test]
    fn vector_add_vi_test_cpu() {
        let vs2 = [10_u32, 20, 30, 0];
        let expected = [9_u32, 19, 29, u32::MAX];

        run_test_exec(
            RiscvInstr::VADD_VI,
            RVInstrInfo::V {
                rs1: 0b11111,
                rs2: 2,
                rd: 3,
                vm: true,
                func6: 0,
            },
            |builder| {
                builder
                    .vector_status(Vlmul::M1, Vsew::E32, false, false)
                    .reg_vec(1, 2, &u32_vec_to_bytes(&vs2))
                    .pc(0x2000)
            },
            |checker| checker.reg_vec(3, &expected).pc(0x2004),
        );
    }

    #[test]
    fn whole_register_move_rejects_bad_rs1_encoding() {
        let mut cpu = TestCPUBuilder::new()
            .vector_status(Vlmul::M1, Vsew::E8, false, false)
            .pc(0x2000)
            .build();

        let err = vec_whole_register_move_op_v::<2>(
            RVInstrInfo::V {
                rs1: 0,
                rs2: 8,
                rd: 16,
                vm: true,
                func6: 0,
            },
            &mut cpu,
        )
        .unwrap_err();

        assert_eq!(err, Exception::IllegalInstruction);
        assert_eq!(cpu.pc, 0x2000);
    }

    #[test]
    fn vsetvli_preserves_tail_policy_in_vector_config() {
        let vs1 = [1_u32, 2, 3, 4];
        let vs2 = [10_u32, 20, 30, 40];
        let old_vd = [0xdead_beefu32; 4];
        let expected = [11_u32, 0xdead_beef, 0xdead_beef, 0xdead_beef];
        let mut cpu = TestCPUBuilder::new()
            .reg(1, 1)
            .reg_vec(1, 1, &u32_vec_to_bytes(&vs1))
            .reg_vec(1, 2, &u32_vec_to_bytes(&vs2))
            .reg_vec(1, 3, &u32_vec_to_bytes(&old_vd))
            .pc(0x2000)
            .build();

        exec_vector_config::<VsetvliFieldExtractor>(
            RVInstrInfo::V {
                rs1: 1,
                rs2: (Vsew::E32 as u8) << 3,
                rd: 5,
                vm: false,
                func6: Vlmul::M1 as u8,
            },
            &mut cpu,
        )
        .unwrap();

        cpu.execute(
            RiscvInstr::VADD_VV,
            RVInstrInfo::V {
                rs1: 1,
                rs2: 2,
                rd: 3,
                vm: true,
                func6: 0,
            },
        )
        .unwrap();

        CPUChecker::new(&mut cpu)
            .reg(5, 1)
            .reg_vec(3, &expected)
            .pc(0x2008);
    }

    #[test]
    fn unit_stride_load_test() {
        const TOTAL_DATA_LEN: WordType = 128;
        const TEST_VSEW: Vsew = Vsew::E32;
        const TEST_VLMUL: Vlmul = Vlmul::M8;
        type ElemType = u32;
        let ram_ref = Rc::new(UnsafeCell::new(Ram::new()));
        for i in 0..TOTAL_DATA_LEN {
            let addr = TEST_DATA_ADDR_OFFSET + i * size_of::<ElemType>() as WordType;
            unsafe {
                ram_ref
                    .as_mut_unchecked()
                    .write(addr, i as ElemType + 1)
                    .unwrap();
            }
        }
        let mmio = MemoryMapIO::from_ram(ram_ref.clone());
        let mut cpu = RVCPU::from_vaddr_manager(VirtAddrManager::from_ram_and_mmio(ram_ref, mmio));
        cpu.csr.get_by_type_existing::<Mstatus>().set_fs(1); // Enable FPU by default for convienience
        cpu.csr.get_by_type_existing::<Mstatus>().set_vs_directly(1);
        cpu.vector.set_config((
            TEST_VLMUL,
            TEST_VSEW,
            false,
            false,
            VLEN_BYTE as u16 * TEST_VLMUL.get_lmul() as u16 / TEST_VSEW.into_byte_width() as u16,
        ));

        let instr_info = RVInstrInfo::V {
            rs1: 1,
            rs2: 0, // lumop
            rd: 0,
            vm: !false, // disable mask
            func6: 0b000000,
        };
        cpu.reg_file.write(1, TEST_DATA_BASE);
        vector_unit_stride_load::<2>(instr_info, &mut cpu).unwrap();
        let vreg = cpu.vector.read_as_type::<ElemType>(0).unwrap();
        assert_eq!(
            vreg.len(),
            VLEN_BYTE * TEST_VLMUL.get_lmul() as usize / size_of::<ElemType>()
        );
        for i in 0..vreg.len() {
            assert_eq!(vreg[i], i as ElemType + 1);
        }
    }

    #[test]
    fn unit_stride_load_test_cpu() {
        let data: Vec<u16> = (0..(VLEN_BYTE / size_of::<u16>()))
            .map(|i| (i + 1000) as u16)
            .collect();
        run_test_exec_decode(
            0b000_0_00_1_00000_00001_101_00000_0000111,
            |builder| {
                builder
                    .vector_status(Vlmul::M1, Vsew::E16, false, false)
                    .reg(1, TEST_DATA_BASE)
                    .mem_range(0..128, |i| {
                        (
                            TEST_DATA_BASE + (i * size_of::<u16>()) as WordType,
                            (i + 1000) as u16,
                        )
                    })
                    .pc(0x2000)
            },
            |checker| checker.reg_vec(0, data.as_slice()).pc(0x2004),
        );
    }

    #[test]
    fn mask_load_uses_ceil_vl_over_8_bytes_and_restores_config() {
        let base_addr = TEST_DATA_BASE + 0x500;
        let initial = vec![0xffu8; VLEN_BYTE * Vlmul::M4.get_lmul() as usize];
        let ram_ref = Rc::new(UnsafeCell::new(Ram::new()));
        for (i, value) in [0b1010_0101u8, 0b0001_0011u8, 0xeeu8]
            .into_iter()
            .enumerate()
        {
            unsafe {
                ram_ref
                    .as_mut_unchecked()
                    .write(TEST_DATA_ADDR_OFFSET + 0x500 + i as WordType, value)
                    .unwrap();
            }
        }
        let mmio = MemoryMapIO::from_ram(ram_ref.clone());
        let mut cpu = RVCPU::from_vaddr_manager(VirtAddrManager::from_ram_and_mmio(ram_ref, mmio));
        cpu.csr.get_by_type_existing::<Mstatus>().set_fs(1);
        cpu.csr.get_by_type_existing::<Mstatus>().set_vs_directly(1);
        cpu.vector
            .set_config((Vlmul::M4, Vsew::E32, false, false, 13));
        cpu.vector
            .write_as_type::<u8>(Vlmul::M4.get_lmul(), 8, &initial);

        cpu.vector
            .mask_load(8, 0, base_addr, &mut cpu.memory.mmio)
            .unwrap();

        let got = cpu.vector.read_as_type::<u8>(8).unwrap();
        assert_eq!(got.len(), VLEN_BYTE * Vlmul::M4.get_lmul() as usize);
        assert_eq!(&got[..3], &[0b1010_0101, 0b0001_0011, 0xff]);
        assert_eq!(got[3], 0xff);
        assert_eq!(
            cpu.vector.read_as_type::<u32>(8).unwrap().len(),
            VLEN_BYTE * 4 / 4
        );
    }

    #[test]
    fn vector_load_fault_records_vstart_and_stops() {
        let mut cpu = TestCPUBuilder::new()
            .vector_status(Vlmul::M1, Vsew::E32, false, false)
            .reg(1, TEST_DATA_BASE)
            .reg(2, 1)
            .mem::<u32>(TEST_DATA_BASE, 0x1234_5678)
            .pc(0x2000)
            .build();
        cpu.vector
            .write_as_type(1, 4, &[0xaaaa_aaaau32; VLEN_BYTE / 4]);

        let result = vector_constant_stride_load::<2>(
            RVInstrInfo::V {
                rs1: 1,
                rs2: 2,
                rd: 4,
                vm: true,
                func6: 0b000010,
            },
            &mut cpu,
        );

        assert_eq!(result, Err(Exception::LoadMisaligned));
        CPUChecker::new(&mut cpu)
            .csr(Vstart::get_index(), 1)
            .reg_vec(4, &[0x1234_5678u32, 0xaaaa_aaaa, 0xaaaa_aaaa, 0xaaaa_aaaa])
            .pc(0x2000);
    }

    #[test]
    fn vector_load_resumes_from_vstart_and_clears_it() {
        let data: Vec<u32> = (0..(VLEN_BYTE / size_of::<u32>()))
            .map(|i| (i + 0x40) as u32)
            .collect();
        let mut builder = TestCPUBuilder::new()
            .vector_status(Vlmul::M1, Vsew::E32, false, false)
            .csr(Vstart::get_index(), 2)
            .reg(1, TEST_DATA_BASE);
        for (index, value) in data.iter().enumerate() {
            builder = builder.mem(
                TEST_DATA_BASE + (index * size_of::<u32>()) as WordType,
                *value,
            );
        }
        let mut cpu = builder.pc(0x2000).build();
        cpu.vector
            .write_as_type(1, 4, &[0xaaaa_aaaau32; VLEN_BYTE / 4]);

        vector_unit_stride_load::<2>(
            RVInstrInfo::V {
                rs1: 1,
                rs2: 0,
                rd: 4,
                vm: true,
                func6: 0,
            },
            &mut cpu,
        )
        .unwrap();

        let mut expected = data;
        expected[0] = 0xaaaa_aaaa;
        expected[1] = 0xaaaa_aaaa;
        CPUChecker::new(&mut cpu)
            .csr(Vstart::get_index(), 0)
            .reg_vec(4, &expected)
            .pc(0x2004);
    }

    #[test]
    fn slide1up_vx_decode_execute() {
        const VS2: u8 = 16;
        const VD: u8 = 24;
        const RS1: u8 = 5;
        let elem_count = VLEN_BYTE / size_of::<u16>();
        let vs2: Vec<u16> = (0..elem_count).map(|i| 0x100 + i as u16).collect();
        let old_vd: Vec<u16> = (0..elem_count).map(|i| 0x900 + i as u16).collect();
        let vs2_bytes: Vec<u8> = vs2.iter().flat_map(|value| value.to_le_bytes()).collect();
        let old_vd_bytes: Vec<u8> = old_vd
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut expected = old_vd.clone();
        expected[0] = 0x2345;
        expected[1..elem_count].copy_from_slice(&vs2[..elem_count - 1]);
        let raw_instr = 0x3800_6057
            | (1 << 25)
            | ((VS2 as u32) << 20)
            | ((RS1 as u32) << 15)
            | ((VD as u32) << 7);

        run_test_exec_decode(
            raw_instr,
            |builder| {
                builder
                    .vector_status(Vlmul::M1, Vsew::E16, false, false)
                    .reg(RS1, 0x12345)
                    .reg_vec(1, VS2, &vs2_bytes)
                    .reg_vec(1, VD, &old_vd_bytes)
                    .pc(0x2000)
            },
            |checker| checker.reg_vec(VD, &expected).pc(0x2004),
        );
    }

    #[test]
    fn slide1down_vx_decode_execute() {
        const VS2: u8 = 16;
        const VD: u8 = 24;
        const RS1: u8 = 5;
        let elem_count = VLEN_BYTE / size_of::<u16>();
        let vs2: Vec<u16> = (0..elem_count).map(|i| 0x100 + i as u16).collect();
        let old_vd: Vec<u16> = (0..elem_count).map(|i| 0x900 + i as u16).collect();
        let vs2_bytes: Vec<u8> = vs2.iter().flat_map(|value| value.to_le_bytes()).collect();
        let old_vd_bytes: Vec<u8> = old_vd
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut expected = old_vd.clone();
        expected[..elem_count - 1].copy_from_slice(&vs2[1..elem_count]);
        expected[elem_count - 1] = 0x2345;
        let raw_instr = 0x3c00_6057
            | (1 << 25)
            | ((VS2 as u32) << 20)
            | ((RS1 as u32) << 15)
            | ((VD as u32) << 7);

        run_test_exec_decode(
            raw_instr,
            |builder| {
                builder
                    .vector_status(Vlmul::M1, Vsew::E16, false, false)
                    .reg(RS1, 0x12345)
                    .reg_vec(1, VS2, &vs2_bytes)
                    .reg_vec(1, VD, &old_vd_bytes)
                    .pc(0x2000)
            },
            |checker| checker.reg_vec(VD, &expected).pc(0x2004),
        );
    }

    #[test]
    fn const_stride_load_test() {
        const TOTAL_DATA_LEN: WordType = 128;
        const TEST_VSEW: Vsew = Vsew::E32;
        const TEST_VLMUL: Vlmul = Vlmul::M8;
        const STRIDE: WordType = (size_of::<u32>() as WordType) * 2;
        type ElemType = u32;
        let ram_ref = Rc::new(UnsafeCell::new(Ram::new()));
        for i in 0..TOTAL_DATA_LEN {
            let addr = TEST_DATA_ADDR_OFFSET + i * STRIDE;
            unsafe {
                ram_ref
                    .as_mut_unchecked()
                    .write(addr, i as ElemType + 1)
                    .unwrap();
            }
        }
        let mmio = MemoryMapIO::from_ram(ram_ref.clone());
        let mut cpu = RVCPU::from_vaddr_manager(VirtAddrManager::from_ram_and_mmio(ram_ref, mmio));
        cpu.csr.get_by_type_existing::<Mstatus>().set_fs(1); // Enable FPU by default for convienience
        cpu.csr.get_by_type_existing::<Mstatus>().set_vs_directly(1);
        cpu.vector.set_config((
            TEST_VLMUL,
            TEST_VSEW,
            false,
            false,
            VLEN_BYTE as u16 * TEST_VLMUL.get_lmul() as u16 / TEST_VSEW.into_byte_width() as u16,
        ));

        let instr_info = RVInstrInfo::V {
            rs1: 1,
            rs2: 2,
            rd: 0,
            vm: !false, // disable mask
            func6: 0b000010,
        };
        cpu.reg_file.write(1, TEST_DATA_BASE);
        cpu.reg_file.write(2, STRIDE);
        vector_constant_stride_load::<2>(instr_info, &mut cpu).unwrap();
        let vreg = cpu.vector.read_as_type::<ElemType>(0).unwrap();
        assert_eq!(
            vreg.len(),
            VLEN_BYTE * TEST_VLMUL.get_lmul() as usize / size_of::<ElemType>()
        );
        for i in 0..vreg.len() {
            assert_eq!(vreg[i], i as ElemType + 1);
        }
    }

    #[test]
    fn indexed_ordered_load_test() {
        const TEST_VSEW: Vsew = Vsew::E32;
        const TEST_VLMUL: Vlmul = Vlmul::M4;
        const TEST_SEG: usize = 2;
        type ElemType = u32;

        let ram_ref = Rc::new(UnsafeCell::new(Ram::new()));
        let index_base = TEST_DATA_BASE;
        let data_base = TEST_DATA_BASE + 0x4000;

        let mmio = MemoryMapIO::from_ram(ram_ref.clone());
        let mut cpu = RVCPU::from_vaddr_manager(VirtAddrManager::from_ram_and_mmio(ram_ref, mmio));
        cpu.csr.get_by_type_existing::<Mstatus>().set_fs(1);
        cpu.csr.get_by_type_existing::<Mstatus>().set_vs_directly(1);
        cpu.vector.set_config((
            TEST_VLMUL,
            TEST_VSEW,
            false,
            false,
            VLEN_BYTE as u16 * TEST_VLMUL.get_lmul() as u16 / TEST_VSEW.into_byte_width() as u16,
        ));

        let vl = VLEN_BYTE * TEST_VLMUL.get_lmul() as usize / size_of::<ElemType>();
        let elem_cnt = vl * TEST_SEG;
        let mut expected = vec![0 as ElemType; elem_cnt];

        // 非线性索引，避免退化成 unit-stride；为 seg=2 准备总计 vl*2 个索引
        for i in 0..elem_cnt {
            let idx_addr = index_base + (i * size_of::<ElemType>()) as WordType;
            let data_index = ((i * 5 + 3) % elem_cnt) as WordType;
            let data_off = data_index * size_of::<ElemType>() as WordType;
            let data_addr = data_base + data_off;
            let val = ((i as ElemType) * 17 + 101) ^ 0x5A5A_1234;

            cpu.write::<ElemType>(idx_addr, data_off as ElemType)
                .unwrap();
            cpu.write::<ElemType>(data_addr, val).unwrap();
            expected[i] = val;
        }

        let instr_info = RVInstrInfo::V {
            rs1: 1,
            rs2: 2,
            rd: 0,
            vm: !false, // disable mask
            // nf=1(seg=2), mop=0b11(indexed ordered), mew=0
            func6: 0b001011,
        };
        cpu.reg_file.write(1, data_base);
        cpu.reg_file.write(2, index_base);
        vector_indexed_ordered_load::<2>(instr_info, &mut cpu).unwrap();

        let vreg = cpu
            .vector
            .read_with_seg(0, TEST_VSEW, TEST_SEG as u8)
            .unwrap();
        let actual: Vec<ElemType> = vreg.iter().map(|e| e.get::<ElemType>()).collect();
        assert_eq!(actual.len(), elem_cnt, "vreg len mismatch for seg=2");
        for i in 0..elem_cnt {
            assert_eq!(
                actual[i], expected[i],
                "Load mismatch at flattened index {}",
                i
            );
        }
    }

    #[test]
    fn unit_stride_store_test() {
        const TOTAL_DATA_LEN: WordType = 128;
        const TEST_VSEW: Vsew = Vsew::E32;
        const TEST_VLMUL: Vlmul = Vlmul::M8;
        type ElemType = u32;
        let ram_ref = Rc::new(UnsafeCell::new(Ram::new()));
        let mmio = MemoryMapIO::from_ram(ram_ref.clone());
        let mut cpu = RVCPU::from_vaddr_manager(VirtAddrManager::from_ram_and_mmio(ram_ref, mmio));
        cpu.csr.get_by_type_existing::<Mstatus>().set_fs(1);
        cpu.csr.get_by_type_existing::<Mstatus>().set_vs_directly(1);
        cpu.vector.set_config((
            TEST_VLMUL,
            TEST_VSEW,
            false,
            false,
            VLEN_BYTE as u16 * TEST_VLMUL.get_lmul() as u16 / TEST_VSEW.into_byte_width() as u16,
        ));

        let vreg_len = VLEN_BYTE * TEST_VLMUL.get_lmul() as usize / size_of::<ElemType>();
        let data: Vec<ElemType> = (0..vreg_len).map(|i| (i * 7 + 13) as ElemType).collect();
        cpu.vector
            .write_as_type::<ElemType>(TEST_VLMUL.get_lmul(), 0, &data);

        let instr_info = RVInstrInfo::V {
            rs1: 1,
            rs2: 0,
            rd: 0,
            vm: !false, // disable mask
            func6: 0b000000,
        };
        cpu.reg_file.write(1, TEST_DATA_BASE);
        vector_unit_stride_store::<2>(instr_info, &mut cpu).unwrap();

        for i in 0..vreg_len {
            let addr = TEST_DATA_BASE + (i * size_of::<ElemType>()) as WordType;
            let expected = data[i];
            let read: ElemType = cpu.read::<ElemType>(addr).unwrap();
            assert_eq!(read, expected, "Memory mismatch at index {}", i);
        }
    }

    #[test]
    fn unit_stride_store_test_cpu() {
        let data: Vec<u16> = (0..(VLEN_BYTE / size_of::<u16>()))
            .map(|i| (i + 2000) as u16)
            .collect();
        let byte_data: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let raw_instr = 0b000_0_00_1_00000_00001_101_00000_0100111;
        run_test_exec_decode(
            raw_instr,
            |builder| {
                builder
                    .vector_status(Vlmul::M1, Vsew::E16, false, false)
                    .reg(1, TEST_DATA_BASE)
                    .reg_vec(1, 0, byte_data.as_slice())
                    .pc(0x2000)
            },
            |mut checker| {
                let stride = size_of::<u16>() as WordType;
                for (i, &val) in data.iter().enumerate() {
                    let addr = TEST_DATA_BASE + (i as WordType) * stride;
                    checker = checker.mem::<u16>(addr, val as WordType);
                }
                checker.pc(0x2004)
            },
        );
    }

    #[test]
    fn mask_store_uses_ceil_vl_over_8_bytes_and_restores_config() {
        let base_addr = TEST_DATA_BASE + 0x700;
        let ram_ref = Rc::new(UnsafeCell::new(Ram::new()));
        for i in 0..3 {
            unsafe {
                ram_ref
                    .as_mut_unchecked()
                    .write(TEST_DATA_ADDR_OFFSET + 0x700 + i as WordType, 0xeeu8)
                    .unwrap();
            }
        }
        let mmio = MemoryMapIO::from_ram(ram_ref.clone());
        let mut cpu = RVCPU::from_vaddr_manager(VirtAddrManager::from_ram_and_mmio(ram_ref, mmio));
        cpu.csr.get_by_type_existing::<Mstatus>().set_fs(1);
        cpu.csr.get_by_type_existing::<Mstatus>().set_vs_directly(1);
        cpu.vector
            .set_config((Vlmul::M4, Vsew::E32, false, false, 13));
        let mut source = vec![0u8; VLEN_BYTE * Vlmul::M4.get_lmul() as usize];
        source[..3].copy_from_slice(&[0b1010_0101, 0b0001_0011, 0x77]);
        cpu.vector
            .write_as_type::<u8>(Vlmul::M4.get_lmul(), 8, &source);

        cpu.vector
            .mask_store(8, 0, base_addr, &mut cpu.memory.mmio)
            .unwrap();

        assert_eq!(cpu.read::<u8>(base_addr).unwrap(), 0b1010_0101);
        assert_eq!(cpu.read::<u8>(base_addr + 1).unwrap(), 0b0001_0011);
        assert_eq!(cpu.read::<u8>(base_addr + 2).unwrap(), 0xee);
        assert_eq!(
            cpu.vector.read_as_type::<u32>(8).unwrap().len(),
            VLEN_BYTE * 4 / 4
        );
    }

    #[test]
    fn mask_load_and_store_decode_execute() {
        let source_mask = [0b1010_0101u8, 0b0101_1010u8];
        let mut source_reg = vec![0u8; VLEN_BYTE];
        source_reg[..source_mask.len()].copy_from_slice(&source_mask);
        run_test_exec_decode(
            0x02b0_8407, // vlm.v v8, (x1)
            |builder| {
                builder
                    .vector_status(Vlmul::M1, Vsew::E8, false, false)
                    .reg(1, TEST_DATA_BASE)
                    .mem::<u8>(TEST_DATA_BASE, source_mask[0])
                    .mem::<u8>(TEST_DATA_BASE + 1, source_mask[1])
                    .pc(0x2000)
            },
            |checker| {
                let mut expected = vec![0u8; VLEN_BYTE];
                expected[..source_mask.len()].copy_from_slice(&source_mask);
                checker.reg_vec(8, expected.as_slice()).pc(0x2004)
            },
        );

        run_test_exec_decode(
            0x02b0_8427, // vsm.v v8, (x1)
            |builder| {
                builder
                    .vector_status(Vlmul::M1, Vsew::E8, false, false)
                    .reg(1, TEST_DATA_BASE)
                    .reg_vec(1, 8, &source_reg)
                    .pc(0x2000)
            },
            |checker| {
                checker
                    .mem::<u8>(TEST_DATA_BASE, source_mask[0] as WordType)
                    .mem::<u8>(TEST_DATA_BASE + 1, source_mask[1] as WordType)
                    .pc(0x2004)
            },
        );
    }

    #[test]
    fn const_stride_store_test() {
        const TEST_VSEW: Vsew = Vsew::E32;
        const TEST_VLMUL: Vlmul = Vlmul::M8;
        const STRIDE: WordType = (size_of::<u32>() as WordType) * 2;
        type ElemType = u32;
        let ram_ref = Rc::new(UnsafeCell::new(Ram::new()));
        let mmio = MemoryMapIO::from_ram(ram_ref.clone());
        let mut cpu = RVCPU::from_vaddr_manager(VirtAddrManager::from_ram_and_mmio(ram_ref, mmio));
        cpu.csr.get_by_type_existing::<Mstatus>().set_fs(1);
        cpu.csr.get_by_type_existing::<Mstatus>().set_vs_directly(1);
        cpu.vector.set_config((
            TEST_VLMUL,
            TEST_VSEW,
            false,
            false,
            VLEN_BYTE as u16 * TEST_VLMUL.get_lmul() as u16 / TEST_VSEW.into_byte_width() as u16,
        ));

        let vreg_len = VLEN_BYTE * TEST_VLMUL.get_lmul() as usize / size_of::<ElemType>();
        let data: Vec<ElemType> = (0..vreg_len).map(|i| (i * 7 + 13) as ElemType).collect();
        cpu.vector
            .write_as_type::<ElemType>(TEST_VLMUL.get_lmul(), 0, &data);

        let instr_info = RVInstrInfo::V {
            rs1: 1,
            rs2: 2,
            rd: 0,
            vm: !false, // disable mask
            func6: 0b000010,
        };
        cpu.reg_file.write(1, TEST_DATA_BASE);
        cpu.reg_file.write(2, STRIDE);
        vector_constant_stride_store::<2>(instr_info, &mut cpu).unwrap();

        for i in 0..vreg_len {
            let addr = TEST_DATA_BASE + (i as WordType) * STRIDE;
            let expected = data[i];
            let read: ElemType = cpu.read::<ElemType>(addr).unwrap();
            assert_eq!(read, expected, "Memory mismatch at index {}", i);
        }
    }

    #[test]
    fn indexed_ordered_store_test() {
        const TEST_VSEW: Vsew = Vsew::E32;
        const TEST_VLMUL: Vlmul = Vlmul::M1;
        type ElemType = u32;

        let ram_ref = Rc::new(UnsafeCell::new(Ram::new()));
        let index_base = TEST_DATA_BASE;
        let data_base = TEST_DATA_BASE + 0x1000;

        let mmio = MemoryMapIO::from_ram(ram_ref.clone());
        let mut cpu = RVCPU::from_vaddr_manager(VirtAddrManager::from_ram_and_mmio(ram_ref, mmio));
        cpu.csr.get_by_type_existing::<Mstatus>().set_fs(1);
        cpu.csr.get_by_type_existing::<Mstatus>().set_vs_directly(1);
        cpu.vector.set_config((
            TEST_VLMUL,
            TEST_VSEW,
            false,
            false,
            VLEN_BYTE as u16 * TEST_VLMUL.get_lmul() as u16 / TEST_VSEW.into_byte_width() as u16,
        ));

        let elem_cnt = VLEN_BYTE * TEST_VLMUL.get_lmul() as usize / size_of::<ElemType>();
        let data: Vec<ElemType> = (0..elem_cnt).map(|i| (i as ElemType) * 3 + 7).collect();
        cpu.vector
            .write_as_type::<ElemType>(TEST_VLMUL.get_lmul(), 0, &data);

        for i in 0..elem_cnt {
            let idx_addr = index_base + (i * size_of::<ElemType>()) as WordType;
            let data_off = (i * size_of::<ElemType>()) as WordType;
            cpu.write::<ElemType>(idx_addr, data_off as u32).unwrap();
        }

        let instr_info = RVInstrInfo::V {
            rs1: 1,
            rs2: 2,
            rd: 0,
            vm: !false, // disable mask
            func6: 0b000011,
        };
        cpu.reg_file.write(1, data_base);
        cpu.reg_file.write(2, index_base);
        vector_indexed_ordered_store::<2>(instr_info, &mut cpu).unwrap();

        for (i, expected) in data.iter().enumerate() {
            let addr = data_base + (i * size_of::<ElemType>()) as WordType;
            let read: ElemType = cpu.read::<ElemType>(addr).unwrap();
            assert_eq!(read, *expected, "Store mismatch at index {}", i);
        }
    }
}
