use super::*;
use crate::isa::riscv::vector::{
    tester::{VectorBuilder, VectorChecker},
    types::{FixedPointRoundingMode, Vlmul, Vsew},
};

const VL_E8_M1: u16 = 16;

fn run_test_fixed_point_vv<Op, F, G>(
    param: TestOpParameter,
    round: FixedPointRoundingMode,
    build: F,
    check: G,
) where
    Op: VectorOpFixedPointVV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker, bool) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    vector.set_fixed_point_rounding_mode(round);
    vector.clear_fixed_point_accrued_saturation_flag();
    let saturated = Op::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio), saturated);
}

fn run_test_fixed_point_vx<Op, F, G>(
    param: TestOpParameter,
    round: FixedPointRoundingMode,
    build: F,
    check: G,
) where
    Op: VectorOpFixedPointVX,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker, bool) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    vector.set_fixed_point_rounding_mode(round);
    vector.clear_fixed_point_accrued_saturation_flag();
    let saturated = Op::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio), saturated);
}

fn run_test_fixed_point_narrowing_wv<Op, F, G>(
    param: TestOpParameter,
    round: FixedPointRoundingMode,
    build: F,
    check: G,
) where
    Op: VectorOpFixedPointNarrowingWV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker, bool) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    vector.set_fixed_point_rounding_mode(round);
    vector.clear_fixed_point_accrued_saturation_flag();
    let saturated = Op::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio), saturated);
}

fn run_test_fixed_point_narrowing_vx<Op, F, G>(
    param: TestOpParameter,
    round: FixedPointRoundingMode,
    build: F,
    check: G,
) where
    Op: VectorOpFixedPointNarrowingVX,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker, bool) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    vector.set_fixed_point_rounding_mode(round);
    vector.clear_fixed_point_accrued_saturation_flag();
    let saturated = Op::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio), saturated);
}

#[test]
fn test_vector_op_saddu_vv_saturates() {
    let param = TestOpParameter::new_vv(8, 16, 24);
    let vs1 = [10u8, 2, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let vs2 = [250u8, 1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let expected = [255u8, 3, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    run_test_fixed_point_vv::<VectorOpSaddu, _, _>(
        param,
        FixedPointRoundingMode::RoundToNearestUp,
        |builder| {
            builder
                .config(Vlmul::M1, Vsew::E8, false, false, VL_E8_M1)
                .reg(1, param.vs1(), &vs1)
                .reg(1, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(saturated);
            assert!(checker.vector.fixed_point_accrued_saturation_flag());
            checker.reg(1, param.vd(), &expected)
        },
    );
}

#[test]
fn test_vector_op_sadd_vx_saturates_signed_min() {
    let param = TestOpParameter::new_vx(0xf0, 16, 24);
    let vs2 = [0x85u8, 0x10, 0x7f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let expected = [
        0x80u8, 0x00, 0x6f, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
        0xf0,
    ];

    run_test_fixed_point_vx::<VectorOpSadd, _, _>(
        param,
        FixedPointRoundingMode::RoundToNearestUp,
        |builder| {
            builder
                .config(Vlmul::M1, Vsew::E8, false, false, VL_E8_M1)
                .reg(1, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(saturated);
            checker.reg(1, param.vd(), &expected)
        },
    );
}

#[test]
fn test_vector_op_aaddu_obeys_rounding_mode() {
    let param = TestOpParameter::new_vv(8, 16, 24);
    let vs1 = [0u8, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let vs2 = [1u8, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    run_test_fixed_point_vv::<VectorOpAaddu, _, _>(
        param,
        FixedPointRoundingMode::RoundToNearestUp,
        |builder| {
            builder
                .config(Vlmul::M1, Vsew::E8, false, false, VL_E8_M1)
                .reg(1, param.vs1(), &vs1)
                .reg(1, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(!saturated);
            checker.reg(
                1,
                param.vd(),
                &[1u8, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
        },
    );

    run_test_fixed_point_vv::<VectorOpAaddu, _, _>(
        param,
        FixedPointRoundingMode::RoundDown,
        |builder| {
            builder
                .config(Vlmul::M1, Vsew::E8, false, false, VL_E8_M1)
                .reg(1, param.vs1(), &vs1)
                .reg(1, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(!saturated);
            checker.reg(
                1,
                param.vd(),
                &[0u8, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
        },
    );
}

#[test]
fn test_vector_op_asub_keeps_extra_precision_before_rounding() {
    let param = TestOpParameter::new_vv(8, 16, 24);
    let vs1 = [0x80u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let vs2 = [0x7fu8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let expected = [0x7fu8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    run_test_fixed_point_vv::<VectorOpAsub, _, _>(
        param,
        FixedPointRoundingMode::RoundDown,
        |builder| {
            builder
                .config(Vlmul::M1, Vsew::E8, false, false, VL_E8_M1)
                .reg(1, param.vs1(), &vs1)
                .reg(1, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(!saturated);
            checker.reg(1, param.vd(), &expected)
        },
    );
}

#[test]
fn test_vector_op_asubu_keeps_negative_difference_before_rounding() {
    let param = TestOpParameter::new_vv(8, 16, 24);
    let vs1 = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let vs2 = [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let expected = [0xffu8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    run_test_fixed_point_vv::<VectorOpAsubu, _, _>(
        param,
        FixedPointRoundingMode::RoundDown,
        |builder| {
            builder
                .config(Vlmul::M1, Vsew::E8, false, false, VL_E8_M1)
                .reg(1, param.vs1(), &vs1)
                .reg(1, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(!saturated);
            checker.reg(1, param.vd(), &expected)
        },
    );
}

#[test]
fn test_vector_op_smul_rounds_and_saturates() {
    let param = TestOpParameter::new_vv(8, 16, 24);
    let vs1 = [0x80u8, 0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let vs2 = [0x80u8, 0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let expected = [0x7fu8, 0x20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    run_test_fixed_point_vv::<VectorOpSmul, _, _>(
        param,
        FixedPointRoundingMode::RoundToNearestUp,
        |builder| {
            builder
                .config(Vlmul::M1, Vsew::E8, false, false, VL_E8_M1)
                .reg(1, param.vs1(), &vs1)
                .reg(1, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(saturated);
            checker.reg(1, param.vd(), &expected)
        },
    );
}

#[test]
fn test_vector_op_ssrl_vi_rounds() {
    let param = TestOpParameter::new_vx(1, 16, 24);
    let vs2 = [3u8, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let expected = [2u8, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    run_test_fixed_point_vx::<VectorOpSsrl, _, _>(
        param,
        FixedPointRoundingMode::RoundToNearestUp,
        |builder| {
            builder
                .config(Vlmul::M1, Vsew::E8, false, false, VL_E8_M1)
                .reg(1, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(!saturated);
            checker.reg(1, param.vd(), &expected)
        },
    );
}

#[test]
fn test_vector_op_nclipu_wv_saturates() {
    let param = TestOpParameter::new_vv(8, 16, 24);
    let vs1 = [0u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let vs2 = [
        0x01ffu16, 0x01ff, 0x007f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let expected = [0xffu8, 0xff, 0x7f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    run_test_fixed_point_narrowing_wv::<VectorOpNclipu, _, _>(
        param,
        FixedPointRoundingMode::RoundToNearestUp,
        |builder| {
            builder
                .config(Vlmul::M1, Vsew::E8, false, false, VL_E8_M1)
                .reg(1, param.vs1(), &vs1)
                .reg(2, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(saturated);
            checker.reg(1, param.vd(), &expected)
        },
    );
}

#[test]
fn test_vector_op_nclip_wx_saturates_signed() {
    let param = TestOpParameter::new_vx(0, 16, 24);
    let vs2 = [
        200u16,
        (-200i16) as u16,
        64u16,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    let expected = [0x7fu8, 0x80, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    run_test_fixed_point_narrowing_vx::<VectorOpNclip, _, _>(
        param,
        FixedPointRoundingMode::RoundDown,
        |builder| {
            builder
                .config(Vlmul::M1, Vsew::E8, false, false, VL_E8_M1)
                .reg(2, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(saturated);
            checker.reg(1, param.vd(), &expected)
        },
    );
}

#[test]
fn test_vector_op_nclipu_fractional_lmul_overlap_check() {
    let param = TestOpParameter::new_vv(8, 2, 3);
    let vs1 = [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let vs2 = [1u16, 2, 3, 4, 0, 0, 0, 0];
    let expected = [1u8, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    run_test_fixed_point_narrowing_wv::<VectorOpNclipu, _, _>(
        param,
        FixedPointRoundingMode::RoundDown,
        |builder| {
            builder
                .config(Vlmul::Mf2, Vsew::E8, false, false, 4)
                .reg(1, param.vs1(), &vs1)
                .reg(1, param.vs2(), &vs2)
        },
        |checker, saturated| {
            assert!(!saturated);
            checker.reg(1, param.vd(), &expected)
        },
    );
}
