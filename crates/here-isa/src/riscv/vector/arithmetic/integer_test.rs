use super::*;
use crate::isa::riscv::vector::{
    VLEN_BYTE,
    tester::{VectorBuilder, VectorChecker},
    types::{Vlmul, Vsew},
};

fn run_test_integer_vv<OpIVV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVV: VectorOpIntegerVV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_gather_vv<OpIVV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVV: VectorOpIntegerGatherVV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_gather_ei16_vv<OpIVV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVV: VectorOpIntegerGatherEI16VV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_compress<Op, F, G>(param: TestOpParameter, build: F, check: G)
where
    Op: VectorOpCompress,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    Op::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_bit_vv<OpIVV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVV: VectorOpBitVV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_vx<OpIVX, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVX: VectorOpIntegerVX,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVX::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_vvv<OpIVV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVV: VectorOpIntegerVVV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_vxv<OpIVX, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVX: VectorOpIntegerVXV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVX::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_widening_integer_vv<OpIVV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVV: VectorOpWideningIntegerVV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_widening_integer_vvv<OpIVV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVV: VectorOpWideningIntegerVVV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_widening_integer_vxv<OpIVX, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVX: VectorOpWideningIntegerVXV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVX::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_widening_integer_vx<OpIVX, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVX: VectorOpWideningIntegerVX,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVX::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_widening_integer_wv<OpIVV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVV: VectorOpWideningIntegerWV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_widening_integer_wx<OpIVX, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVX: VectorOpWideningIntegerWX,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVX::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_narrowing_wv<OpIVV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVV: VectorOpIntegerNarrowingWV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_narrowing_vx<OpIVX, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVX: VectorOpIntegerNarrowingVX,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVX::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_v<OpIV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIV: VectorOpIntegerV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

#[test]
fn test_vector_op_move_v_uses_u64_width() {
    const LMUL: Vlmul = Vlmul::M2;
    let param = TestOpParameter::new_v(8, 24, Vsew::E64, Vsew::E64);
    let source = [
        0x0102_0304_0506_0708_u64,
        0x1112_1314_1516_1718,
        0x2122_2324_2526_2728,
        0x3132_3334_3536_3738,
    ];

    run_test_integer_v::<ExecMove<u64>, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, Vsew::E8, false, false, source.len() as u16)
                .reg(LMUL.get_lmul(), param.vs2(), &source)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &source),
    );
}

#[test]
fn test_vector_scalar_move_uses_u64_width() {
    const LMUL: Vlmul = Vlmul::M1;
    let value = 0xfeed_face_cafe_beef_u64;
    let expected = [value; VLEN_BYTE / size_of::<u64>()];
    let (mut vector, mut mmio) = VectorBuilder::new()
        .config(LMUL, Vsew::E8, false, false, expected.len() as u16)
        .build();

    vector
        .exec_integer_scalar_move::<ExecMove<u64>, u64>(value, 24, 0)
        .unwrap();

    VectorChecker::new(&mut vector, &mut mmio).reg(LMUL.get_lmul(), 24, &expected);
}

#[test]
fn test_vector_move_scalar_to_element_only_writes_element_zero() {
    const LMUL: Vlmul = Vlmul::M4;
    const SEW: Vsew = Vsew::E16;
    let old_vd = [
        0x1111_u16, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888,
    ];
    let mut expected = old_vd;
    expected[0] = 0xcafe;

    let (mut vector, mut mmio) = VectorBuilder::new()
        .config(LMUL, SEW, false, false, old_vd.len() as u16)
        .reg(1, 3, &old_vd)
        .build();

    vector
        .exec_move_scalar_to_element(0xdead_beef_cafe, 3, 0)
        .unwrap();

    VectorChecker::new(&mut vector, &mut mmio).reg(1, 3, &expected);
}

#[test]
fn test_vector_move_element_to_scalar_sign_extends_element_zero() {
    const LMUL: Vlmul = Vlmul::M4;
    const SEW: Vsew = Vsew::E8;
    let mut source = [0_u8; VLEN_BYTE];
    source[0] = 0x80;
    source[1] = 0x7f;
    let (vector, _) = VectorBuilder::new()
        .config(LMUL, SEW, false, false, source.len() as u16)
        .reg(1, 5, &source)
        .build();

    assert_eq!(
        vector.exec_move_element_to_scalar(5, 0).unwrap(),
        Some(WordType::MAX - 0x7f)
    );
}

#[test]
fn test_vector_whole_register_move_copies_group() {
    const LMUL: Vlmul = Vlmul::M4;
    let source: Vec<u64> = (0..(VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u64>()))
        .map(|i| 0x1000_0000_0000_0000_u64 + i as u64)
        .collect();
    let (mut vector, mut mmio) = VectorBuilder::new()
        .config(Vlmul::M1, Vsew::E8, false, false, 0)
        .reg(LMUL.get_lmul(), 8, &source)
        .build();

    vector
        .exec_whole_register_move::<ExecMove<u64>>(8, 16, LMUL.get_lmul(), 0)
        .unwrap();

    VectorChecker::new(&mut vector, &mut mmio).reg(LMUL.get_lmul(), 16, &source);
}

fn run_test_integer_vvm<OpIVVM, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVVM: VectorOpIntegerVVM,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVVM::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_vxm<OpIVXM, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIVXM: VectorOpIntegerVXM,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIVXM::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_mask_to_x<Op, F>(param: TestOpParameter, build: F) -> WordType
where
    Op: VectorOpMaskToX,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
{
    let (mut vector, _mmio) = build(VectorBuilder::new()).build();
    Op::test(&mut vector, param).unwrap()
}

fn run_test_mask_unary<Op, F, G>(param: TestOpParameter, build: F, check: G)
where
    Op: VectorOpMaskUnary,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    Op::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_mask_to_vector<Op, F, G>(param: TestOpParameter, build: F, check: G)
where
    Op: VectorOpMaskToVector,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    Op::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_index<Op, F, G>(param: TestOpParameter, build: F, check: G)
where
    Op: VectorOpIndex,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    Op::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_mask_vv<OpIMVV, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIMVV: VectorOpIntegerMaskVV,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIMVV::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_mask_vx<OpIMVX, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIMVX: VectorOpIntegerMaskVX,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIMVX::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn run_test_integer_mask_vvm<OpIMVVM, F, G>(param: TestOpParameter, build: F, check: G)
where
    OpIMVVM: VectorOpIntegerMaskVVM,
    F: FnOnce(VectorBuilder) -> VectorBuilder,
    G: FnOnce(VectorChecker) -> VectorChecker,
{
    let (mut vector, mut mmio) = build(VectorBuilder::new()).build();
    OpIMVVM::test(&mut vector, param).unwrap();
    check(VectorChecker::new(&mut vector, &mut mmio));
}

fn mask_from_bits<I>(bits: I) -> Vec<u8>
where
    I: IntoIterator<Item = bool>,
{
    let mut mask = vec![0; VLEN_BYTE];
    for (index, bit) in bits.into_iter().enumerate() {
        write_mask_bit(&mut mask, index, bit);
    }
    mask
}

fn mask_bit(mask: &[u8], index: usize) -> bool {
    read_mask_bit(mask, index)
}

fn signed_div_u64(lhs: u64, rhs: u64) -> u64 {
    if rhs == 0 {
        u64::MAX
    } else {
        as_signed_i128(lhs).wrapping_div(as_signed_i128(rhs)) as u64
    }
}

fn signed_rem_u64(lhs: u64, rhs: u64) -> u64 {
    if rhs == 0 {
        lhs
    } else {
        as_signed_i128(lhs).wrapping_rem(as_signed_i128(rhs)) as u64
    }
}

#[test]
fn test_vector_op_merge_vvm_selects_by_v0() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vvm(8, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vl = 5;
    let vs1: Vec<u16> = (0..elem_count).map(|i| 0x1000_u16 + i as u16).collect();
    let vs2: Vec<u16> = (0..elem_count).map(|i| 0x2000_u16 + i as u16).collect();
    let old_vd: Vec<u16> = (0..elem_count).map(|i| 0x9000_u16 + i as u16).collect();
    let merge_mask = mask_from_bits([true, false, true, false, true, false, true, false]);
    let mut expected = old_vd.clone();
    for index in 0..vl {
        expected[index] = if mask_bit(&merge_mask, index) {
            vs1[index]
        } else {
            vs2[index]
        };
    }

    run_test_integer_vvm::<VectorOpMerge, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vl as u16)
                .reg(1, param.v0(), &merge_mask)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_merge_vxm_selects_scalar_by_v0() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let scalar = 0xaaaa_5555_u32 as WordType;
    let param = TestOpParameter::new_vxm(scalar, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let vs2: Vec<u32> = (0..elem_count)
        .map(|i| 0x3000_0000_u32 + i as u32)
        .collect();
    let merge_mask = mask_from_bits([false, true, true, false]);
    let expected: Vec<u32> = (0..elem_count)
        .map(|index| {
            if mask_bit(&merge_mask, index) {
                scalar as u32
            } else {
                vs2[index]
            }
        })
        .collect();

    run_test_integer_vxm::<VectorOpMerge, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(1, param.v0(), &merge_mask)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_merge_vim_uses_signed_immediate_value() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let imm = (-3_i8) as WordType;
    let param = TestOpParameter::new_vxm(imm, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u8>();
    let vs2: Vec<u8> = (0..elem_count).map(|i| 0x40_u8 + i as u8).collect();
    let merge_mask = mask_from_bits((0..elem_count).map(|index| index % 2 == 0));
    let expected: Vec<u8> = (0..elem_count)
        .map(|index| {
            if mask_bit(&merge_mask, index) {
                imm as u8
            } else {
                vs2[index]
            }
        })
        .collect();

    run_test_integer_vxm::<VectorOpMerge, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(1, param.v0(), &merge_mask)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_cpop_and_first_m_honor_mask() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_v(16, 0, SEW, SEW).with_enable_mask(true);
    let source = mask_from_bits([false, true, false, true, true, false, true, false]);
    let pred_mask = mask_from_bits([true, false, true, true, false, true, true, true]);
    let build = |builder: VectorBuilder| {
        builder
            .config(LMUL, SEW, false, false, 8)
            .reg(1, param.v0(), &pred_mask)
            .reg(1, param.vs2(), &source)
    };

    let count = run_test_mask_to_x::<VectorOpCpopM, _>(param, build);
    assert_eq!(count, 2);

    let first = run_test_mask_to_x::<VectorOpFirstM, _>(param, |builder| {
        builder
            .config(LMUL, SEW, false, false, 8)
            .reg(1, param.v0(), &pred_mask)
            .reg(1, param.vs2(), &source)
    });
    assert_eq!(first, 3);
}

#[test]
fn test_vector_op_first_m_returns_minus_one_when_no_match() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_v(16, 0, SEW, SEW);
    let source = mask_from_bits([false; 8]);

    let first = run_test_mask_to_x::<VectorOpFirstM, _>(param, |builder| {
        builder
            .config(LMUL, SEW, false, false, 8)
            .reg(1, param.vs2(), &source)
    });
    assert_eq!(first, WordType::MAX);
}

#[test]
fn test_vector_op_mask_before_first_family() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_v(16, 24, SEW, SEW);
    let source = mask_from_bits([false, false, true, false, true, false, false, false]);
    let expected_msbf = mask_from_bits([true, true, false, false, false, false, false, false]);
    let expected_msif = mask_from_bits([true, true, true, false, false, false, false, false]);
    let expected_msof = mask_from_bits([false, false, true, false, false, false, false, false]);

    run_test_mask_unary::<VectorOpMsbfM, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, 8)
                .reg(1, param.vs2(), &source)
        },
        |checker| checker.reg(1, param.vd(), &expected_msbf),
    );

    run_test_mask_unary::<VectorOpMsifM, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, 8)
                .reg(1, param.vs2(), &source)
        },
        |checker| checker.reg(1, param.vd(), &expected_msif),
    );

    run_test_mask_unary::<VectorOpMsofM, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, 8)
                .reg(1, param.vs2(), &source)
        },
        |checker| checker.reg(1, param.vd(), &expected_msof),
    );
}

#[test]
fn test_vector_op_mask_before_first_initializes_seen_first_from_vstart() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_v(16, 24, SEW, SEW);
    let source = mask_from_bits([false, true, false, false, false, false, false, false]);
    let old_vd = mask_from_bits([true, true, true, true, true, true, true, true]);
    let expected = mask_from_bits([true, true, true, false, false, false, false, false]);

    let (mut vector, mut mmio) = VectorBuilder::new()
        .config(LMUL, SEW, false, false, 8)
        .reg(1, param.vs2(), &source)
        .reg(1, param.vd(), &old_vd)
        .build();

    vector
        .exec_mask_unary::<VectorOpMsbfM>(param.vs2(), param.vd(), false, 3)
        .unwrap();

    VectorChecker::new(&mut vector, &mut mmio).reg(1, param.vd(), &expected);
}

#[test]
fn test_vector_op_iota_m_honors_mask() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_v(16, 24, SEW, SEW).with_enable_mask(true);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let source = mask_from_bits([true, false, true, true, false, true, false, false]);
    let pred_mask = mask_from_bits([true, false, true, true, true, false, true, true]);
    let old_vd: Vec<u16> = (0..elem_count).map(|index| 0x9000 + index as u16).collect();
    let mut expected = old_vd.clone();
    expected[0] = 0;
    expected[2] = 1;
    expected[3] = 2;
    expected[4] = 3;
    expected[6] = 4;
    expected[7] = 4;

    run_test_mask_to_vector::<VectorOpIotaM, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, 8)
                .reg(1, param.v0(), &pred_mask)
                .reg(1, param.vs2(), &source)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_iota_m_counts_source_bits_before_vstart() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_v(16, 24, SEW, SEW);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let source = mask_from_bits([true, false, true, true, false, false, false, false]);
    let old_vd: Vec<u16> = (0..elem_count).map(|index| 0x9000 + index as u16).collect();
    let mut expected = old_vd.clone();
    expected[4] = 3;
    expected[5] = 3;
    expected[6] = 3;
    expected[7] = 3;

    let (mut vector, mut mmio) = VectorBuilder::new()
        .config(LMUL, SEW, false, false, 8)
        .reg(1, param.vs2(), &source)
        .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        .build();

    vector
        .exec_mask_to_vector::<VectorOpIotaM>(param.vs2(), param.vd(), false, 4)
        .unwrap();

    VectorChecker::new(&mut vector, &mut mmio).reg(LMUL.get_lmul(), param.vd(), &expected);
}

#[test]
fn test_vector_op_mask_special_rejects_destination_source_overlap() {
    let mut vector = VectorBuilder::new()
        .config(Vlmul::M1, Vsew::E16, false, false, 8)
        .build()
        .0;

    assert_eq!(
        vector.exec_mask_unary::<VectorOpMsbfM>(8, 8, false, 0),
        Err(Exception::IllegalInstruction)
    );
    assert_eq!(
        vector.exec_mask_to_vector::<VectorOpIotaM>(8, 8, false, 0),
        Err(Exception::IllegalInstruction)
    );
}

#[test]
fn test_vector_op_id_v_honors_mask() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_v(0, 24, SEW, SEW).with_enable_mask(true);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let pred_mask = mask_from_bits([true, false, true, true]);
    let old_vd: Vec<u32> = (0..elem_count)
        .map(|index| 0x8000_0000 + index as u32)
        .collect();
    let mut expected = old_vd.clone();
    expected[0] = 0;
    expected[2] = 2;
    expected[3] = 3;

    run_test_index::<VectorOpIdV, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(1, param.v0(), &pred_mask)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

fn run_vv_binary_u32<OpIVV>(vs1: &[u32], vs2: &[u32], expected: &[u32])
where
    OpIVV: VectorOpIntegerVV,
{
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vv(8, 16, 24);

    run_test_integer_vv::<OpIVV, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs1(), vs1)
                .reg(LMUL.get_lmul(), param.vs2(), vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), expected),
    );
}

fn run_vv_binary_u64<OpIVV>(vs1: &[u64], vs2: &[u64], expected: &[u64])
where
    OpIVV: VectorOpIntegerVV,
{
    const LMUL: Vlmul = Vlmul::M4;
    const SEW: Vsew = Vsew::E64;
    let param = TestOpParameter::new_vv(8, 16, 24);

    run_test_integer_vv::<OpIVV, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs1(), vs1)
                .reg(LMUL.get_lmul(), param.vs2(), vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), expected),
    );
}

fn run_vx_binary_u32<OpIVX>(scalar: WordType, vs2: &[u32], expected: &[u32])
where
    OpIVX: VectorOpIntegerVX,
{
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vx(scalar, 8, 24);

    run_test_integer_vx::<OpIVX, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs2(), vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), expected),
    );
}

fn run_vx_binary_u64<OpIVX>(scalar: WordType, vs2: &[u64], expected: &[u64])
where
    OpIVX: VectorOpIntegerVX,
{
    const LMUL: Vlmul = Vlmul::M4;
    const SEW: Vsew = Vsew::E64;
    let param = TestOpParameter::new_vx(scalar, 8, 24);

    run_test_integer_vx::<OpIVX, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs2(), vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), expected),
    );
}

fn run_vvm_binary_u8<OpIVVM>(vs1: &[u8], vs2: &[u8], carry: &[u8], expected: &[u8])
where
    OpIVVM: VectorOpIntegerVVM,
{
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_vvm(8, 16, 24);

    run_test_integer_vvm::<OpIVVM, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(1, param.v0(), carry)
                .reg(LMUL.get_lmul(), param.vs1(), vs1)
                .reg(LMUL.get_lmul(), param.vs2(), vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), expected),
    );
}

fn run_mask_vv_u8<OpIMVV>(vs1: &[u8], vs2: &[u8], expected: &[u8])
where
    OpIMVV: VectorOpIntegerMaskVV,
{
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_mask_vv(8, 16, 24);

    run_test_integer_mask_vv::<OpIMVV, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vs2.len() as u16)
                .reg(LMUL.get_lmul(), param.vs1(), vs1)
                .reg(LMUL.get_lmul(), param.vs2(), vs2)
        },
        |checker| checker.reg(1, param.vd(), expected),
    );
}

fn run_mask_vx_u8<OpIMVX>(scalar: WordType, vs2: &[u8], expected: &[u8])
where
    OpIMVX: VectorOpIntegerMaskVX,
{
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_mask_vx(scalar, 8, 24);

    run_test_integer_mask_vx::<OpIMVX, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vs2.len() as u16)
                .reg(LMUL.get_lmul(), param.vs2(), vs2)
        },
        |checker| checker.reg(1, param.vd(), expected),
    );
}

fn run_mask_vvm_u8<OpIMVVM>(vs1: &[u8], vs2: &[u8], carry: &[u8], expected: &[u8])
where
    OpIMVVM: VectorOpIntegerMaskVVM,
{
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_mask_vvm(8, 16, 24);

    run_test_integer_mask_vvm::<OpIMVVM, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vs2.len() as u16)
                .reg(1, param.v0(), carry)
                .reg(LMUL.get_lmul(), param.vs1(), vs1)
                .reg(LMUL.get_lmul(), param.vs2(), vs2)
        },
        |checker| checker.reg(1, param.vd(), expected),
    );
}

fn run_widening_vv_i16_to_i32<OpIVV>(vs1: &[i16], vs2: &[i16], expected: &[i32])
where
    OpIVV: VectorOpWideningIntegerVV,
{
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vv(8, 16, 24);

    run_test_widening_integer_vv::<OpIVV, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs1(), vs1)
                .reg(LMUL.get_lmul(), param.vs2(), vs2)
        },
        |checker| checker.reg(LMUL.get_lmul() * 2, param.vd(), expected),
    );
}

fn u32_bits_to_i32(values: Vec<u32>) -> Vec<i32> {
    values.into_iter().map(|value| value as i32).collect()
}

fn run_widening_vx_i16_to_i32<OpIVX>(x1: WordType, vs2: &[i16], expected: &[i32])
where
    OpIVX: VectorOpWideningIntegerVX,
{
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vx(x1, 16, 24);

    run_test_widening_integer_vx::<OpIVX, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs2(), vs2)
        },
        |checker| checker.reg(LMUL.get_lmul() * 2, param.vd(), expected),
    );
}

fn run_widening_wv_i16_to_i32<OpIVV>(vs1: &[i16], vs2: &[i32], expected: &[i32])
where
    OpIVV: VectorOpWideningIntegerWV,
{
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vv(8, 16, 24);

    run_test_widening_integer_wv::<OpIVV, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs1(), vs1)
                .reg(LMUL.get_lmul() * 2, param.vs2(), vs2)
        },
        |checker| checker.reg(LMUL.get_lmul() * 2, param.vd(), expected),
    );
}

fn run_widening_wx_i16_to_i32<OpIVX>(x1: WordType, vs2: &[i32], expected: &[i32])
where
    OpIVX: VectorOpWideningIntegerWX,
{
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vx(x1, 16, 24);

    run_test_widening_integer_wx::<OpIVX, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul() * 2, param.vs2(), vs2)
        },
        |checker| checker.reg(LMUL.get_lmul() * 2, param.vd(), expected),
    );
}

#[test]
fn test_vector_op_add_vv() {
    const LMUL: Vlmul = Vlmul::M2;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vv(8, 16, 24);

    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let vs1: Vec<u32> = (0..elem_count).map(|i| (i as u32) * 7 + 3).collect();
    let vs2: Vec<u32> = (0..elem_count)
        .map(|i| u32::MAX.wrapping_sub((i as u32) * 5))
        .collect();
    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(lhs, rhs)| lhs.wrapping_add(*rhs))
        .collect();

    run_test_integer_vv::<VectorOpAdd, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_widening_add_sub() {
    let vs1 = [-5_i16, 7, i16::MAX, -1, 0, 1234, -3000, 99];
    let vs2 = [10_i16, -20, 1, i16::MIN, -9, -1234, 4000, -100];

    let expected: Vec<i32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| (*lhs as i32).wrapping_add(*rhs as i32))
        .collect();
    run_widening_vv_i16_to_i32::<VectorOpWadd>(&vs1, &vs2, &expected);

    let expected = u32_bits_to_i32(
        vs1.iter()
            .zip(vs2.iter())
            .map(|(rhs, lhs)| (*lhs as u16 as u32).wrapping_add(*rhs as u16 as u32))
            .collect(),
    );
    run_widening_vv_i16_to_i32::<VectorOpWaddu>(&vs1, &vs2, &expected);

    let expected: Vec<i32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| (*lhs as i32).wrapping_sub(*rhs as i32))
        .collect();
    run_widening_vv_i16_to_i32::<VectorOpWsub>(&vs1, &vs2, &expected);

    let wide_vs2 = [100_i32, -200, i32::MAX, i32::MIN, 77, -88, 9000, -9000];
    let expected: Vec<i32> = vs1
        .iter()
        .zip(wide_vs2.iter())
        .map(|(rhs, lhs)| lhs.wrapping_add(*rhs as i32))
        .collect();
    run_widening_wv_i16_to_i32::<VectorOpWadd>(&vs1, &wide_vs2, &expected);

    let expected: Vec<i32> = wide_vs2
        .iter()
        .map(|lhs| lhs.wrapping_sub(-3_i16 as i32))
        .collect();
    run_widening_wx_i16_to_i32::<VectorOpWsub>(-3_i16 as WordType, &wide_vs2, &expected);

    let expected: Vec<i32> = vs2
        .iter()
        .map(|lhs| (*lhs as i32).wrapping_add(9_i16 as i32))
        .collect();
    run_widening_vx_i16_to_i32::<VectorOpWadd>(9, &vs2, &expected);
}

#[test]
fn test_vector_op_rgather_vv() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vv(8, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let vs1 = [3_u32, 0, 5, 7];
    let vs2 = [0x1000_0000_u32, 0x2000_0000, 0x3000_0000, 0x4000_0000];
    let expected = [vs2[3], vs2[0], 0, 0];

    run_test_integer_gather_vv::<VectorOpRGatherVV, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_rgather_vv_honors_mask_and_tail() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vv(8, 16, 24).with_enable_mask(true);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vs1: Vec<u16> = (0..elem_count)
        .map(|i| [1_u16, 0, 3, 99, 2, 4, 5, 6][i])
        .collect();
    let vs2: Vec<u16> = (0..elem_count)
        .map(|i| [10_u16, 20, 30, 40, 50, 60, 70, 80][i])
        .collect();
    let old_vd: Vec<u16> = (0..elem_count).map(|i| 0x9000_u16 + i as u16).collect();
    let pred_mask = mask_from_bits([true, false, true, true, true, true, true, true]);
    let vl = 5;
    let mut expected = old_vd.clone();
    expected[0] = vs2[1];
    expected[2] = vs2[3];
    expected[3] = 0;
    expected[4] = vs2[2];

    run_test_integer_gather_vv::<VectorOpRGatherVV, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vl as u16)
                .reg(1, param.v0(), &pred_mask)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_rgatherei16_vv_uses_16_bit_indices() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vv(8, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let vs1 = [2_u16, 4, 0, 7, 1, 1, 1, 1];
    let vs2 = [0x11_u32, 0x22, 0x33, 0x44];
    let expected = [vs2[2], 0, vs2[0], 0];

    run_test_integer_gather_ei16_vv::<VectorOpRGatherEI16VV, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_rgatherei16_vv_uses_widened_index_group_for_e8() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_vv(8, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u8>();
    let vs1: Vec<u16> = (0..elem_count).rev().map(|index| index as u16).collect();
    let vs2: Vec<u8> = (0..elem_count).map(|index| 0x10 + index as u8).collect();
    let expected: Vec<u8> = vs1.iter().map(|index| vs2[*index as usize]).collect();

    run_test_integer_gather_ei16_vv::<VectorOpRGatherEI16VV, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(2, param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_rgather_vx() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E64;
    let param = TestOpParameter::new_vx(1, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u64>();
    let vs2 = [0x10_u64, 0x20, 0x30, 0x40];
    let expected = [vs2[1]; 2];

    run_test_integer_vx::<VectorOpRGatherVX, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_rgather_vi() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_vx(5, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u8>();
    let vs2: Vec<u8> = (0..elem_count).map(|i| i as u8 + 1).collect();
    let expected = [6_u8; VLEN_BYTE];

    run_test_integer_vx::<VectorOpRGatherVI, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_rgather_rejects_destination_source_overlap() {
    const LMUL: Vlmul = Vlmul::M2;
    const SEW: Vsew = Vsew::E32;
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let data: Vec<u32> = (0..elem_count).map(|i| i as u32).collect();

    let (mut vector, _) = VectorBuilder::new()
        .config(LMUL, SEW, false, false, elem_count as u16)
        .reg(LMUL.get_lmul(), 8, &data)
        .reg(LMUL.get_lmul(), 12, &data)
        .build();

    assert_eq!(
        vector.exec_integer_gather_vv::<VectorOpRGatherVV>(8, 12, 8, false, 0),
        Err(Exception::IllegalInstruction)
    );
    assert_eq!(
        vector.exec_integer_gather_vv::<VectorOpRGatherVV>(8, 12, 12, false, 0),
        Err(Exception::IllegalInstruction)
    );
    assert_eq!(
        vector.exec_integer_gather_ei16_vv::<VectorOpRGatherEI16VV>(8, 12, 8, false, 0),
        Err(Exception::IllegalInstruction)
    );
}

#[test]
fn test_vector_op_rgatherei16_vv_rejects_widened_index_overlap() {
    let mut vector = VectorBuilder::new()
        .config(Vlmul::M1, Vsew::E8, false, false, 16)
        .build()
        .0;

    assert_eq!(
        vector.exec_integer_gather_ei16_vv::<VectorOpRGatherEI16VV>(2, 8, 3, false, 0),
        Err(Exception::IllegalInstruction)
    );
}

#[test]
fn test_vector_op_rgather_scalar_forms_reject_destination_source_overlap() {
    const LMUL: Vlmul = Vlmul::M2;
    const SEW: Vsew = Vsew::E16;
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let data: Vec<u16> = (0..elem_count).map(|i| i as u16).collect();
    let (mut vector, _) = VectorBuilder::new()
        .config(LMUL, SEW, false, false, elem_count as u16)
        .reg(LMUL.get_lmul(), 16, &data)
        .build();

    assert_eq!(
        vector.exec_integer_gather_vx::<VectorOpRGatherVX>(0, 16, 17, false, 0),
        Err(Exception::IllegalInstruction)
    );
    assert_eq!(
        vector.exec_integer_gather_vx::<VectorOpRGatherVI>(0, 16, 17, false, 0),
        Err(Exception::IllegalInstruction)
    );
}

#[test]
fn test_vector_op_compress_vm() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vv(8, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vl = 6;
    let select_mask = mask_from_bits([false, true, true, false, true, false, false, false]);
    let vs2: Vec<u16> = (0..elem_count).map(|i| 0x1000_u16 + i as u16).collect();
    let old_vd: Vec<u16> = (0..elem_count).map(|i| 0x9000_u16 + i as u16).collect();
    let mut expected = old_vd.clone();
    expected[0] = vs2[1];
    expected[1] = vs2[2];
    expected[2] = vs2[4];

    run_test_compress::<VectorOpCompressVm, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vl as u16)
                .reg(1, param.vs1(), &select_mask)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_compress_vm_tail_agnostic() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vv(8, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let select_mask = mask_from_bits([true, false, true, false]);
    let vs2: Vec<u32> = (0..elem_count)
        .map(|i| 0x2000_0000_u32 + i as u32)
        .collect();
    let old_vd: Vec<u32> = (0..elem_count)
        .map(|i| 0x9000_0000_u32 + i as u32)
        .collect();
    let mut expected = vec![0; elem_count];
    expected[0] = vs2[0];
    expected[1] = vs2[2];

    run_test_compress::<VectorOpCompressVm, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, true, elem_count as u16)
                .reg(1, param.vs1(), &select_mask)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_compress_vm_rejects_overlap_and_nonzero_vstart() {
    const LMUL: Vlmul = Vlmul::M2;
    const SEW: Vsew = Vsew::E16;
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let select_mask = mask_from_bits((0..elem_count).map(|i| i % 2 == 0));
    let data: Vec<u16> = (0..elem_count).map(|i| i as u16).collect();
    let (mut vector, _) = VectorBuilder::new()
        .config(LMUL, SEW, false, false, elem_count as u16)
        .reg(1, 4, &select_mask)
        .reg(LMUL.get_lmul(), 8, &data)
        .build();

    assert_eq!(
        vector.exec_compress::<VectorOpCompressVm>(4, 8, 9, 0),
        Err(Exception::IllegalInstruction)
    );
    assert_eq!(
        vector.exec_compress::<VectorOpCompressVm>(4, 8, 4, 0),
        Err(Exception::IllegalInstruction)
    );
    assert_eq!(
        vector.exec_compress::<VectorOpCompressVm>(4, 8, 12, 1),
        Err(Exception::IllegalInstruction)
    );
}

#[test]
fn test_vector_op_slideup_vx() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vx(3, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vs2: Vec<u16> = (0..elem_count).map(|i| 0x100_u16 + i as u16).collect();
    let old_vd: Vec<u16> = (0..elem_count).map(|i| 0x900_u16 + i as u16).collect();
    let mut expected = old_vd.clone();
    for index in 3..elem_count {
        expected[index] = vs2[index - 3];
    }

    run_test_integer_vxv::<VectorOpSlideUp, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_slideup_honors_mask_and_tail() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vx(2, 16, 24).with_enable_mask(true);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let vl = 3;
    let vs2: Vec<u32> = (0..elem_count).map(|i| 0x10_u32 + i as u32).collect();
    let old_vd: Vec<u32> = (0..elem_count).map(|i| 0x80_u32 + i as u32).collect();
    let pred_mask = mask_from_bits([true, true, false, true]);
    let mut expected = old_vd.clone();
    expected[2] = old_vd[2];

    run_test_integer_vxv::<VectorOpSlideUp, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vl as u16)
                .reg(1, param.v0(), &pred_mask)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_slidedown_vx() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vx(3, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vs2: Vec<u16> = (0..elem_count).map(|i| 0x100_u16 + i as u16).collect();
    let old_vd: Vec<u16> = (0..elem_count).map(|i| 0x900_u16 + i as u16).collect();
    let mut expected = old_vd.clone();
    for index in 0..elem_count {
        expected[index] = vs2.get(index + 3).copied().unwrap_or_default();
    }

    run_test_integer_vx::<VectorOpSlideDown, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_slidedown_honors_mask_and_tail() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vx(2, 16, 24).with_enable_mask(true);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let vl = 3;
    let vs2: Vec<u32> = (0..elem_count).map(|i| 0x10_u32 + i as u32).collect();
    let old_vd: Vec<u32> = (0..elem_count).map(|i| 0x80_u32 + i as u32).collect();
    let pred_mask = mask_from_bits([true, true, false, true]);
    let mut expected = old_vd.clone();
    expected[0] = vs2[2];
    expected[1] = vs2[3];

    run_test_integer_vx::<VectorOpSlideDown, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vl as u16)
                .reg(1, param.v0(), &pred_mask)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_slide1up_vx() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vx(0x12345, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vs2: Vec<u16> = (0..elem_count).map(|i| 0x100_u16 + i as u16).collect();
    let old_vd: Vec<u16> = (0..elem_count).map(|i| 0x900_u16 + i as u16).collect();
    let mut expected = old_vd.clone();
    expected[0] = 0x2345;
    expected[1..elem_count].copy_from_slice(&vs2[..elem_count - 1]);

    run_test_integer_vxv::<VectorOpSlide1Up, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_slide1up_honors_mask_and_tail() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vx(0x55, 16, 24).with_enable_mask(true);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let vl = 3;
    let vs2: Vec<u32> = (0..elem_count).map(|i| 0x10_u32 + i as u32).collect();
    let old_vd: Vec<u32> = (0..elem_count).map(|i| 0x80_u32 + i as u32).collect();
    let pred_mask = mask_from_bits([false, true, true, true]);
    let mut expected = old_vd.clone();
    expected[1] = vs2[0];
    expected[2] = vs2[1];

    run_test_integer_vxv::<VectorOpSlide1Up, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vl as u16)
                .reg(1, param.v0(), &pred_mask)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_slide1down_vx() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vx(0x12345, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vs2: Vec<u16> = (0..elem_count).map(|i| 0x100_u16 + i as u16).collect();
    let old_vd: Vec<u16> = (0..elem_count).map(|i| 0x900_u16 + i as u16).collect();
    let mut expected = old_vd.clone();
    expected[..elem_count - 1].copy_from_slice(&vs2[1..elem_count]);
    expected[elem_count - 1] = 0x2345;

    run_test_integer_vx::<VectorOpSlide1Down, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_slide1down_honors_mask_and_tail() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vx(0x55, 16, 24).with_enable_mask(true);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u32>();
    let vl = 3;
    let vs2: Vec<u32> = (0..elem_count).map(|i| 0x10_u32 + i as u32).collect();
    let old_vd: Vec<u32> = (0..elem_count).map(|i| 0x80_u32 + i as u32).collect();
    let pred_mask = mask_from_bits([true, false, true, true]);
    let mut expected = old_vd.clone();
    expected[0] = vs2[1];
    expected[2] = 0x55;

    run_test_integer_vx::<VectorOpSlide1Down, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vl as u16)
                .reg(1, param.v0(), &pred_mask)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_narrowing_shift_right_wv() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_vv(8, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u8>();
    let vs1: Vec<u8> = (0..elem_count).map(|i| (i as u8).wrapping_mul(3)).collect();
    let vs2: Vec<u16> = (0..elem_count)
        .map(|i| 0x0100_u16.wrapping_add((i as u16) << 4))
        .collect();
    let expected: Vec<u8> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(shift, value)| (*value >> (*shift as u32 & 7)) as u8)
        .collect();

    run_test_integer_narrowing_wv::<VectorOpNsrl, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul() * 2, param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_narrowing_shift_right_wx() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vx(3, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vs2: Vec<u32> = (0..elem_count)
        .map(|i| 0x8000_0000_u32 >> (i % 4))
        .collect();
    let expected: Vec<u16> = vs2.iter().map(|value| (value >> 3) as u16).collect();

    run_test_integer_narrowing_vx::<VectorOpNsrl, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul() * 2, param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_narrowing_shift_right_wi_signed() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vx((-1_i32) as WordType, 16, 24);
    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vs2: Vec<u32> = (0..elem_count)
        .map(|i| {
            if i % 2 == 0 {
                0x8000_0000_u32
            } else {
                0x7fff_ff00_u32
            }
        })
        .collect();
    let expected: Vec<u16> = vs2
        .iter()
        .map(|value| ((*value as i32) >> 15) as u16)
        .collect();

    run_test_integer_narrowing_vx::<VectorOpNsra, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul() * 2, param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_narrowing_shift_rejects_destination_source_overlap() {
    let mut vector = VectorBuilder::new()
        .config(Vlmul::M1, Vsew::E8, false, false, 16)
        .build()
        .0;

    assert_eq!(
        vector.exec_integer_narrowing_wv::<VectorOpNsrl>(8, 2, 3, false, 0),
        Err(Exception::IllegalInstruction)
    );
    assert_eq!(
        vector.exec_integer_narrowing_vx::<VectorOpNsrl>(1, 2, 3, false, 0),
        Err(Exception::IllegalInstruction)
    );
}

#[test]
fn test_vector_op_narrowing_shift_fractional_lmul_overlap_check() {
    const LMUL: Vlmul = Vlmul::Mf2;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_vv(8, 2, 3);
    let vs1 = [0u8; VLEN_BYTE];
    let vs2 = [1u16, 2, 3, 4, 0, 0, 0, 0];
    let expected = [1u8, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    run_test_integer_narrowing_wv::<VectorOpNsrl, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, 4)
                .reg(1, param.vs1(), &vs1)
                .reg(1, param.vs2(), &vs2)
        },
        |checker| checker.reg(1, param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_widening_mul() {
    let vs1 = [-5_i16, 7, -1, 0x7fff, -300, 123, -32768, 11];
    let vs2 = [10_i16, -20, -3, 2, 400, -321, -2, 0x7fff];

    let expected: Vec<i32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| (*lhs as i32).wrapping_mul(*rhs as i32))
        .collect();
    run_widening_vv_i16_to_i32::<VectorOpWmul>(&vs1, &vs2, &expected);

    let expected = u32_bits_to_i32(
        vs1.iter()
            .zip(vs2.iter())
            .map(|(rhs, lhs)| (*lhs as u16 as u32).wrapping_mul(*rhs as u16 as u32))
            .collect(),
    );
    run_widening_vv_i16_to_i32::<VectorOpWmulu>(&vs1, &vs2, &expected);

    let expected: Vec<i32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| (*lhs as i32).wrapping_mul(*rhs as u16 as i32))
        .collect();
    run_widening_vv_i16_to_i32::<VectorOpWmulsu>(&vs1, &vs2, &expected);

    let expected: Vec<i32> = vs2
        .iter()
        .map(|lhs| (*lhs as i32).wrapping_mul((-2_i16) as i32))
        .collect();
    run_widening_vx_i16_to_i32::<VectorOpWmul>(-2_i16 as WordType, &vs2, &expected);
}

#[test]
fn test_vector_op_integer_multiply_add() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E32;
    let param = TestOpParameter::new_vv(8, 16, 24);
    let vs1 = [3_u32, 0xffff_fffe, 7, 0x8000_0000];
    let vs2 = [10_u32, 5, 0xffff_ffff, 2];
    let old_vd = [100_u32, 9, 20, 0x8000_0001];

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .zip(old_vd.iter())
        .map(|((vs1, vs2), vd)| vs1.wrapping_mul(*vs2).wrapping_add(*vd))
        .collect();
    run_test_integer_vvv::<VectorOpMacc, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .zip(old_vd.iter())
        .map(|((vs1, vs2), vd)| vs1.wrapping_mul(*vs2).wrapping_neg().wrapping_add(*vd))
        .collect();
    run_test_integer_vvv::<VectorOpNmsac, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .zip(old_vd.iter())
        .map(|((vs1, vs2), vd)| vs1.wrapping_mul(*vd).wrapping_add(*vs2))
        .collect();
    run_test_integer_vvv::<VectorOpMadd, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );

    let scalar_param = TestOpParameter::new_vx(0xffff_fffd, 16, 24);
    let scalar = scalar_param.x1() as u32;
    let expected: Vec<u32> = vs2
        .iter()
        .zip(old_vd.iter())
        .map(|(vs2, vd)| scalar.wrapping_mul(*vd).wrapping_neg().wrapping_add(*vs2))
        .collect();
    run_test_integer_vxv::<VectorOpNmsub, _, _>(
        scalar_param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), scalar_param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), scalar_param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), scalar_param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_integer_multiply_add_honors_mask_and_tail() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vv(8, 16, 24).with_enable_mask(true);
    let vs1 = [2_i16, 3, 4, 5, 6, 7, 8, 9];
    let vs2 = [10_i16, 20, 30, 40, 50, 60, 70, 80];
    let old_vd = [100_i16, 101, 102, 103, 104, 105, 106, 107];
    let pred_mask = mask_from_bits([true, false, true, false, true, true, true, true]);
    let vl = 5;
    let mut expected = old_vd;
    for index in 0..vl {
        if mask_bit(&pred_mask, index) {
            expected[index] = vs1[index]
                .wrapping_mul(vs2[index])
                .wrapping_add(old_vd[index]);
        }
    }

    run_test_integer_vvv::<VectorOpMacc, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vl as u16)
                .reg(1, param.v0(), &pred_mask)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul(), param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_widening_multiply_add() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_vv(8, 16, 24);
    let vs1 = [-5_i16, 7, -1, 0x7fff, -300, 123, -32768, 11];
    let vs2 = [10_i16, -20, -3, 2, 400, -321, -2, 0x7fff];
    let old_vd = [100_i32, -200, i32::MAX, i32::MIN, 77, -88, 9000, -9000];

    let expected: Vec<i32> = vs1
        .iter()
        .zip(vs2.iter())
        .zip(old_vd.iter())
        .map(|((vs1, vs2), vd)| (*vs1 as i32).wrapping_mul(*vs2 as i32).wrapping_add(*vd))
        .collect();
    run_test_widening_integer_vvv::<VectorOpWmacc, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul() * 2, param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul() * 2, param.vd(), &expected),
    );

    let expected = u32_bits_to_i32(
        vs1.iter()
            .zip(vs2.iter())
            .zip(old_vd.iter())
            .map(|((vs1, vs2), vd)| {
                (*vs1 as u16 as u32)
                    .wrapping_mul(*vs2 as u16 as u32)
                    .wrapping_add(*vd as u32)
            })
            .collect(),
    );
    run_test_widening_integer_vvv::<VectorOpWmaccu, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
                .reg(LMUL.get_lmul() * 2, param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul() * 2, param.vd(), &expected),
    );

    let scalar_param = TestOpParameter::new_vx((-3_i16) as WordType, 16, 24);
    let expected: Vec<i32> = vs2
        .iter()
        .zip(old_vd.iter())
        .map(|(vs2, vd)| (-3_i16 as i32).wrapping_mul(*vs2 as i32).wrapping_add(*vd))
        .collect();
    run_test_widening_integer_vxv::<VectorOpWmacc, _, _>(
        scalar_param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), scalar_param.vs2(), &vs2)
                .reg(LMUL.get_lmul() * 2, scalar_param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul() * 2, scalar_param.vd(), &expected),
    );

    let scalar_param = TestOpParameter::new_vx((-3_i16) as WordType, 16, 24);
    let expected: Vec<i32> = vs2
        .iter()
        .zip(old_vd.iter())
        .map(|(vs2, vd)| {
            (-3_i16 as i32)
                .wrapping_mul(*vs2 as u16 as i32)
                .wrapping_add(*vd)
        })
        .collect();
    run_test_widening_integer_vxv::<VectorOpWmaccus, _, _>(
        scalar_param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, expected.len() as u16)
                .reg(LMUL.get_lmul(), scalar_param.vs2(), &vs2)
                .reg(LMUL.get_lmul() * 2, scalar_param.vd(), &old_vd)
        },
        |checker| checker.reg(LMUL.get_lmul() * 2, scalar_param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_integer_vv_binary() {
    let elem_count = VLEN_BYTE / size_of::<u32>();
    let vs1: Vec<u32> = (0..elem_count)
        .map(|i| 0x8000_0011u32.wrapping_add((i as u32) * 0x1020_304))
        .collect();
    let vs2: Vec<u32> = (0..elem_count)
        .map(|i| 0xfedc_ba98u32.wrapping_sub((i as u32) * 0x0101_0101))
        .collect();

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(lhs, rhs)| lhs.wrapping_add(*rhs))
        .collect();
    run_vv_binary_u32::<VectorOpAddu>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| lhs.wrapping_sub(*rhs))
        .collect();
    run_vv_binary_u32::<VectorOpSub>(&vs1, &vs2, &expected);
    run_vv_binary_u32::<VectorOpSubu>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(lhs, rhs)| lhs & rhs)
        .collect();
    run_vv_binary_u32::<VectorOpAnd>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| lhs.wrapping_shl(rhs & 31))
        .collect();
    run_vv_binary_u32::<VectorOpSll>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| lhs.wrapping_shr(rhs & 31))
        .collect();
    run_vv_binary_u32::<VectorOpSrl>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| (as_signed_i128(*lhs) >> (rhs & 31)) as u32)
        .collect();
    run_vv_binary_u32::<VectorOpSra>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(lhs, rhs)| {
            if as_signed_i128(*lhs) > as_signed_i128(*rhs) {
                *lhs
            } else {
                *rhs
            }
        })
        .collect();
    run_vv_binary_u32::<VectorOpMax>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(lhs, rhs)| if lhs > rhs { *lhs } else { *rhs })
        .collect();
    run_vv_binary_u32::<VectorOpMaxu>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(lhs, rhs)| {
            if as_signed_i128(*lhs) < as_signed_i128(*rhs) {
                *lhs
            } else {
                *rhs
            }
        })
        .collect();
    run_vv_binary_u32::<VectorOpMin>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(lhs, rhs)| if lhs < rhs { *lhs } else { *rhs })
        .collect();
    run_vv_binary_u32::<VectorOpMinu>(&vs1, &vs2, &expected);
}

fn run_bit_binary_u32<OpIVV>(vs1: &[u32], vs2: &[u32], expected: &[u32])
where
    OpIVV: VectorOpBitVV,
{
    let param = TestOpParameter::new_vv(3, 5, 7);
    run_test_bit_vv::<OpIVV, _, _>(
        param,
        |builder| {
            builder
                .config(Vlmul::M8, Vsew::E64, false, false, (VLEN_BYTE * 8) as u16)
                .reg(1, param.vs1(), vs1)
                .reg(1, param.vs2(), vs2)
        },
        |checker| checker.reg(1, param.vd(), expected),
    );
}

#[test]
fn test_vector_op_bit_vv_binary() {
    let elem_count = VLEN_BYTE / size_of::<u32>();
    let vs1: Vec<u32> = (0..elem_count)
        .map(|i| 0x55aa_00ffu32.rotate_left((i * 5) as u32))
        .collect();
    let vs2: Vec<u32> = (0..elem_count)
        .map(|i| 0xcc33_f0f0u32.rotate_right((i * 3) as u32))
        .collect();

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(vs1, vs2)| vs2 & vs1)
        .collect();
    run_bit_binary_u32::<VectorOpAnd>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(vs1, vs2)| !(vs2 & vs1))
        .collect();
    run_bit_binary_u32::<VectorOpNand>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(vs1, vs2)| vs2 & !vs1)
        .collect();
    run_bit_binary_u32::<VectorOpAndn>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(vs1, vs2)| vs2 ^ vs1)
        .collect();
    run_bit_binary_u32::<VectorOpXor>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(vs1, vs2)| vs2 | vs1)
        .collect();
    run_bit_binary_u32::<VectorOpOr>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(vs1, vs2)| !(vs2 | vs1))
        .collect();
    run_bit_binary_u32::<VectorOpNor>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(vs1, vs2)| vs2 | !vs1)
        .collect();
    run_bit_binary_u32::<VectorOpOrn>(&vs1, &vs2, &expected);

    let expected: Vec<u32> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(vs1, vs2)| !(vs2 ^ vs1))
        .collect();
    run_bit_binary_u32::<VectorOpXnor>(&vs1, &vs2, &expected);
}

#[test]
fn test_vector_op_bit_vv_honors_mask_tail_and_vstart() {
    let param = TestOpParameter::new_vv(3, 5, 7).with_enable_mask(true);
    let vs1 = [0x1234_5678u32, 0xf0f0_0f0f, 0xaaaa_5555, 0x0101_8080];
    let vs2 = [0xffff_0000u32, 0x3333_cccc, 0x0f0f_f0f0, 0x8000_0001];
    let old_vd = [0x5555_aaaau32, 0x0123_4567, 0xffff_ffff, 0xdead_beef];
    let mask = mask_from_bits((0..VLEN_BYTE * 8).map(|index| index % 3 != 0));
    let mut expected = old_vd;
    for index in 0..VLEN_BYTE * 8 {
        let chunk = index / u32::BITS as usize;
        let bit = 1u32 << (index % u32::BITS as usize);
        let value = (vs2[chunk] & bit) != 0 && (vs1[chunk] & bit) == 0;

        if index < 17 {
            continue;
        } else if index >= 70 {
            expected[chunk] &= !bit;
        } else if mask_bit(&mask, index) {
            if value {
                expected[chunk] |= bit;
            } else {
                expected[chunk] &= !bit;
            }
        }
    }

    let (mut vector, mut mmio) = VectorBuilder::new()
        .config(Vlmul::M8, Vsew::E64, false, true, 70)
        .reg(1, param.v0(), &mask)
        .reg(1, param.vs1(), &vs1)
        .reg(1, param.vs2(), &vs2)
        .reg(1, param.vd(), &old_vd)
        .build();
    vector
        .exec_bit_vv::<VectorOpAndn>(param.vs1(), param.vs2(), param.vd(), true, 17)
        .unwrap();
    VectorChecker::new(&mut vector, &mut mmio).reg(1, param.vd(), &expected);
}

#[test]
fn test_vector_op_add_vx() {
    const LMUL: Vlmul = Vlmul::M2;
    const SEW: Vsew = Vsew::E16;
    const SCALAR: WordType = 0x12f0;
    let param = TestOpParameter::new_vx(SCALAR, 8, 10);

    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vs2: Vec<u16> = (0..elem_count)
        .map(|i| u16::MAX.wrapping_sub((i as u16) * 17))
        .collect();
    let scalar = param.x1() as u16;
    let expected: Vec<u16> = vs2.iter().map(|value| value.wrapping_add(scalar)).collect();

    run_test_integer_vx::<VectorOpAdd, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_integer_vx_binary() {
    let elem_count = VLEN_BYTE / size_of::<u32>();
    let vs2: Vec<u32> = (0..elem_count)
        .map(|i| 0x8000_00f0u32.wrapping_add((i as u32) * 0x0110_0101))
        .collect();
    let scalar = 0x1234_0005u64;
    let scalar_u32 = scalar as u32;

    let expected: Vec<u32> = vs2
        .iter()
        .map(|value| value.wrapping_add(scalar_u32))
        .collect();
    run_vx_binary_u32::<VectorOpAddu>(scalar, &vs2, &expected);

    let expected: Vec<u32> = vs2
        .iter()
        .map(|value| value.wrapping_sub(scalar_u32))
        .collect();
    run_vx_binary_u32::<VectorOpSub>(scalar, &vs2, &expected);
    run_vx_binary_u32::<VectorOpSubu>(scalar, &vs2, &expected);

    let expected: Vec<u32> = vs2
        .iter()
        .map(|value| scalar_u32.wrapping_sub(*value))
        .collect();
    run_vx_binary_u32::<VectorOpRevSub>(scalar, &vs2, &expected);

    let expected: Vec<u32> = vs2.iter().map(|value| value & scalar_u32).collect();
    run_vx_binary_u32::<VectorOpAnd>(scalar, &vs2, &expected);

    let shift = scalar_u32 & 31;
    let expected: Vec<u32> = vs2.iter().map(|value| value.wrapping_shl(shift)).collect();
    run_vx_binary_u32::<VectorOpSll>(scalar, &vs2, &expected);

    let expected: Vec<u32> = vs2.iter().map(|value| value.wrapping_shr(shift)).collect();
    run_vx_binary_u32::<VectorOpSrl>(scalar, &vs2, &expected);

    let expected: Vec<u32> = vs2
        .iter()
        .map(|value| (as_signed_i128(*value) >> shift) as u32)
        .collect();
    run_vx_binary_u32::<VectorOpSra>(scalar, &vs2, &expected);

    let expected: Vec<u32> = vs2
        .iter()
        .map(|value| {
            if as_signed_i128(*value) > as_signed_i128(scalar_u32) {
                *value
            } else {
                scalar_u32
            }
        })
        .collect();
    run_vx_binary_u32::<VectorOpMax>(scalar, &vs2, &expected);

    let expected: Vec<u32> = vs2
        .iter()
        .map(|value| {
            if *value > scalar_u32 {
                *value
            } else {
                scalar_u32
            }
        })
        .collect();
    run_vx_binary_u32::<VectorOpMaxu>(scalar, &vs2, &expected);

    let expected: Vec<u32> = vs2
        .iter()
        .map(|value| {
            if as_signed_i128(*value) < as_signed_i128(scalar_u32) {
                *value
            } else {
                scalar_u32
            }
        })
        .collect();
    run_vx_binary_u32::<VectorOpMin>(scalar, &vs2, &expected);

    let expected: Vec<u32> = vs2
        .iter()
        .map(|value| {
            if *value < scalar_u32 {
                *value
            } else {
                scalar_u32
            }
        })
        .collect();
    run_vx_binary_u32::<VectorOpMinu>(scalar, &vs2, &expected);
}

#[test]
fn test_vector_op_mul_div_e64() {
    let vs2 = [
        0,
        1,
        2,
        u64::MAX,
        0x8000_0000_0000_0000,
        0x7fff_ffff_ffff_ffff,
        0x1234_5678_9abc_def0,
        0xfedc_ba98_7654_3210,
    ];
    let vs1 = [
        0,
        1,
        u64::MAX,
        2,
        u64::MAX,
        0x8000_0000_0000_0000,
        3,
        0x1000_0000_0000_0001,
    ];

    let expected: Vec<u64> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(lhs, rhs)| lhs.wrapping_mul(*rhs))
        .collect();
    run_vv_binary_u64::<VectorOpMul>(&vs1, &vs2, &expected);

    let expected: Vec<u64> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(lhs, rhs)| ((as_signed_i128(*lhs) * as_signed_i128(*rhs)) >> 64) as u64)
        .collect();
    run_vv_binary_u64::<VectorOpMulh>(&vs1, &vs2, &expected);

    let expected: Vec<u64> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(lhs, rhs)| (((*lhs as u128) * (*rhs as u128)) >> 64) as u64)
        .collect();
    run_vv_binary_u64::<VectorOpMulhu>(&vs1, &vs2, &expected);

    let expected: Vec<u64> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| ((as_signed_i128(*lhs) * (*rhs as u128 as i128)) >> 64) as u64)
        .collect();
    run_vv_binary_u64::<VectorOpMulhsu>(&vs1, &vs2, &expected);

    let expected: Vec<u64> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| signed_div_u64(*lhs, *rhs))
        .collect();
    run_vv_binary_u64::<VectorOpDiv>(&vs1, &vs2, &expected);

    let expected: Vec<u64> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| if *rhs == 0 { u64::MAX } else { lhs / rhs })
        .collect();
    run_vv_binary_u64::<VectorOpDivu>(&vs1, &vs2, &expected);

    let expected: Vec<u64> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| signed_rem_u64(*lhs, *rhs))
        .collect();
    run_vv_binary_u64::<VectorOpRem>(&vs1, &vs2, &expected);

    let expected: Vec<u64> = vs1
        .iter()
        .zip(vs2.iter())
        .map(|(rhs, lhs)| if *rhs == 0 { *lhs } else { lhs % rhs })
        .collect();
    run_vv_binary_u64::<VectorOpRemu>(&vs1, &vs2, &expected);

    let scalar = 0xffff_ffff_ffff_fffb;
    let expected: Vec<u64> = vs2.iter().map(|value| value.wrapping_mul(scalar)).collect();
    run_vx_binary_u64::<VectorOpMul>(scalar, &vs2, &expected);

    let expected: Vec<u64> = vs2
        .iter()
        .map(|value| ((as_signed_i128(*value) * as_signed_i128(scalar)) >> 64) as u64)
        .collect();
    run_vx_binary_u64::<VectorOpMulh>(scalar, &vs2, &expected);

    let expected: Vec<u64> = vs2
        .iter()
        .map(|value| (((*value as u128) * (scalar as u128)) >> 64) as u64)
        .collect();
    run_vx_binary_u64::<VectorOpMulhu>(scalar, &vs2, &expected);

    let expected: Vec<u64> = vs2
        .iter()
        .map(|value| ((as_signed_i128(*value) * (scalar as u128 as i128)) >> 64) as u64)
        .collect();
    run_vx_binary_u64::<VectorOpMulhsu>(scalar, &vs2, &expected);

    let expected: Vec<u64> = vs2
        .iter()
        .map(|value| signed_div_u64(*value, scalar))
        .collect();
    run_vx_binary_u64::<VectorOpDiv>(scalar, &vs2, &expected);

    let expected: Vec<u64> = vs2.iter().map(|value| value / scalar).collect();
    run_vx_binary_u64::<VectorOpDivu>(scalar, &vs2, &expected);

    let expected: Vec<u64> = vs2
        .iter()
        .map(|value| signed_rem_u64(*value, scalar))
        .collect();
    run_vx_binary_u64::<VectorOpRem>(scalar, &vs2, &expected);

    let expected: Vec<u64> = vs2.iter().map(|value| value % scalar).collect();
    run_vx_binary_u64::<VectorOpRemu>(scalar, &vs2, &expected);
}

#[test]
fn test_vector_op_adc_sbc_vvm() {
    let elem_count = VLEN_BYTE;
    let vs1: Vec<u8> = (0..elem_count)
        .map(|i| 0x11u8.wrapping_add((i as u8).wrapping_mul(13)))
        .collect();
    let vs2: Vec<u8> = (0..elem_count)
        .map(|i| 0xf0u8.wrapping_sub((i as u8).wrapping_mul(7)))
        .collect();
    let carry = mask_from_bits((0..elem_count).map(|i| i % 3 == 1));

    let expected: Vec<u8> = vs1
        .iter()
        .zip(vs2.iter())
        .enumerate()
        .map(|(index, (vs1, vs2))| {
            vs2.wrapping_add(*vs1)
                .wrapping_add(mask_bit(&carry, index) as u8)
        })
        .collect();
    run_vvm_binary_u8::<VectorOpAdc>(&vs1, &vs2, &carry, &expected);

    let expected: Vec<u8> = vs1
        .iter()
        .zip(vs2.iter())
        .enumerate()
        .map(|(index, (vs1, vs2))| {
            vs2.wrapping_sub(*vs1)
                .wrapping_sub(mask_bit(&carry, index) as u8)
        })
        .collect();
    run_vvm_binary_u8::<VectorOpSbc>(&vs1, &vs2, &carry, &expected);
}

#[test]
fn test_vector_op_mask_vv() {
    let elem_count = VLEN_BYTE;
    let vs1: Vec<u8> = (0..elem_count)
        .map(|i| [3, 7, 7, 10, 0xf0, 0x80, 0xff, 1][i % 8])
        .collect();
    let vs2: Vec<u8> = (0..elem_count)
        .map(|i| [3, 8, 6, 10, 0x10, 0x7f, 0xfe, 2][i % 8])
        .collect();

    let expected = mask_from_bits(vs1.iter().zip(vs2.iter()).map(|(vs1, vs2)| vs2 == vs1));
    run_mask_vv_u8::<VectorOpMseq>(&vs1, &vs2, &expected);

    let expected = mask_from_bits(vs1.iter().zip(vs2.iter()).map(|(vs1, vs2)| vs2 != vs1));
    run_mask_vv_u8::<VectorOpMsne>(&vs1, &vs2, &expected);

    let expected = mask_from_bits(vs1.iter().zip(vs2.iter()).map(|(vs1, vs2)| vs2 < vs1));
    run_mask_vv_u8::<VectorOpMsltu>(&vs1, &vs2, &expected);
}

#[test]
fn test_vector_op_mask_vv_honors_mask_and_tail() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_mask_vv(8, 16, 24).with_enable_mask(true);
    let elem_count = VLEN_BYTE;
    let vl = 8;
    let vs1: Vec<u8> = (0..elem_count)
        .map(|i| [1, 2, 3, 4, 5, 6, 7, 8][i % 8])
        .collect();
    let vs2: Vec<u8> = (0..elem_count)
        .map(|i| [1, 9, 3, 0, 5, 0, 7, 0][i % 8])
        .collect();
    let pred_mask = mask_from_bits((0..elem_count).map(|i| i % 2 == 0));
    let mut expected = vec![0xff; VLEN_BYTE];
    for index in 0..vl {
        if mask_bit(&pred_mask, index) {
            write_mask_bit(&mut expected, index, vs2[index] == vs1[index]);
        }
    }

    run_test_integer_mask_vv::<VectorOpMseq, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, false, vl as u16)
                .reg(1, param.v0(), &pred_mask)
                .reg(1, param.vd(), &[0xffu8; VLEN_BYTE])
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(1, param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_mask_vx() {
    let elem_count = VLEN_BYTE;
    let vs2: Vec<u8> = (0..elem_count)
        .map(|i| [0x10, 0x40, 0x80, 0xff, 0x7f, 0x01, 0x40, 0x41][i % 8])
        .collect();

    let scalar = 0x40;
    let expected = mask_from_bits(vs2.iter().map(|value| *value == scalar as u8));
    run_mask_vx_u8::<VectorOpMseq>(scalar, &vs2, &expected);

    let expected = mask_from_bits(vs2.iter().map(|value| *value != scalar as u8));
    run_mask_vx_u8::<VectorOpMsne>(scalar, &vs2, &expected);

    let expected = mask_from_bits(vs2.iter().map(|value| *value < scalar as u8));
    run_mask_vx_u8::<VectorOpMsltu>(scalar, &vs2, &expected);

    let expected = mask_from_bits(vs2.iter().map(|value| (scalar as u8) > *value));
    run_mask_vx_u8::<VectorOpMsgtu>(scalar, &vs2, &expected);

    let signed_scalar = 0x7f;
    let expected = mask_from_bits(
        vs2.iter()
            .map(|value| as_signed_i128(signed_scalar as u8) > as_signed_i128(*value)),
    );
    run_mask_vx_u8::<VectorOpMsgt>(signed_scalar, &vs2, &expected);
}

#[test]
fn test_vector_op_madc_msbc_mask_vvm() {
    let elem_count = VLEN_BYTE;
    let vs1: Vec<u8> = (0..elem_count)
        .map(|i| [1, 2, 0xff, 0x80, 0x7f, 0x10, 0xf0, 0x55][i % 8])
        .collect();
    let vs2: Vec<u8> = (0..elem_count)
        .map(|i| [0xff, 1, 1, 0x80, 0x80, 0xef, 0x20, 0x54][i % 8])
        .collect();
    let carry = mask_from_bits((0..elem_count).map(|i| i % 2 == 0));

    let expected = mask_from_bits(vs1.iter().zip(vs2.iter()).enumerate().map(
        |(index, (vs1, vs2))| {
            (*vs2 as u16 + *vs1 as u16 + mask_bit(&carry, index) as u16) > u8::MAX as u16
        },
    ));
    run_mask_vvm_u8::<VectorOpMadc>(&vs1, &vs2, &carry, &expected);

    let expected = mask_from_bits(
        vs1.iter()
            .zip(vs2.iter())
            .enumerate()
            .map(|(index, (vs1, vs2))| {
                (*vs2 as u16) < (*vs1 as u16 + mask_bit(&carry, index) as u16)
            }),
    );
    run_mask_vvm_u8::<VectorOpMsbc>(&vs1, &vs2, &carry, &expected);
}

#[test]
fn test_vector_op_adc_vvm_honors_tail() {
    const LMUL: Vlmul = Vlmul::M1;
    const SEW: Vsew = Vsew::E8;
    let param = TestOpParameter::new_vvm(8, 16, 24);
    let elem_count = VLEN_BYTE;
    let vl = 5;
    let vs1: Vec<u8> = (0..elem_count).map(|i| i as u8).collect();
    let vs2: Vec<u8> = (0..elem_count)
        .map(|i| 0x80u8.wrapping_add(i as u8))
        .collect();
    let carry = mask_from_bits((0..elem_count).map(|i| i % 2 == 1));
    let mut expected = vec![0xaa; elem_count];
    for index in 0..vl {
        expected[index] = vs2[index]
            .wrapping_add(vs1[index])
            .wrapping_add(mask_bit(&carry, index) as u8);
    }
    for value in expected.iter_mut().skip(vl) {
        *value = 0;
    }

    run_test_integer_vvm::<VectorOpAdc, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, SEW, false, true, vl as u16)
                .reg(1, param.v0(), &carry)
                .reg(LMUL.get_lmul(), param.vd(), &[0xaau8; VLEN_BYTE])
                .reg(LMUL.get_lmul(), param.vs1(), &vs1)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}

#[test]
fn test_vector_op_zext_vf2() {
    const LMUL: Vlmul = Vlmul::M1;
    const SRC_SEW: Vsew = Vsew::E8;
    const DST_SEW: Vsew = Vsew::E16;
    let param = TestOpParameter::new_v(8, 10, SRC_SEW, DST_SEW);

    let elem_count = VLEN_BYTE * LMUL.get_lmul() as usize / size_of::<u16>();
    let vs2: Vec<u8> = (0..VLEN_BYTE * LMUL.get_lmul() as usize)
        .map(|i| 0x80u8.wrapping_add(i as u8))
        .collect();
    let expected: Vec<u16> = vs2
        .iter()
        .take(elem_count)
        .map(|value| *value as u16)
        .collect();

    run_test_integer_v::<VectorOpZextVf2, _, _>(
        param,
        |builder| {
            builder
                .config(LMUL, DST_SEW, false, false, elem_count as u16)
                .reg(LMUL.get_lmul(), param.vs2(), &vs2)
        },
        |checker| checker.reg(LMUL.get_lmul(), param.vd(), &expected),
    );
}
