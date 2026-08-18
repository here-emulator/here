#![cfg(test)]
use std::{cell::UnsafeCell, fmt::Debug, rc::Rc};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

use crate::{
    config::arch_config::{REGFILE_CNT, WordType},
    device::mmio::MemoryMapIO,
    isa::riscv::{
        csr_reg::{
            NamedCsrReg,
            csr_macro::{Mstatus, Vl, Vtype},
        },
        decoder::{DecodeInstr, Decoder},
        executor::RVCPU,
        instruction::{RVInstrInfo, instr_table::RiscvInstr},
        isa_builder::TEST_ISA,
        mmu::VirtAddrManager,
        vector::types::{Vlmul, Vsew},
    },
    ram::Ram,
    ram_config::{self, BASE_ADDR},
    utils::{UnsignedInteger, sign_extend},
};

pub(super) struct TestCPUBuilder {
    cpu: Box<RVCPU>,
}

impl TestCPUBuilder {
    /// Build a RISC-V CPU, only has RAM, don't have other devices.
    pub(super) fn new() -> Self {
        let ram_ref = Rc::new(UnsafeCell::new(Ram::new()));
        let mmio = MemoryMapIO::from_ram(ram_ref.clone());
        let decoder = Decoder::from_isa_str(TEST_ISA).expect("TEST_ISA must be valid");
        let mut cpu = Box::new(RVCPU::from_decoder(
            decoder,
            VirtAddrManager::from_ram_and_mmio(ram_ref, mmio),
        ));
        cpu.csr.get_by_type_existing::<Mstatus>().set_fs(1); // Enable FPU by default for convienience
        cpu.csr.get_by_type_existing::<Mstatus>().set_vs_directly(1); // Enable vector unit by default for convenience
        Self { cpu: cpu }
    }

    pub(super) fn reg(mut self, idx: u8, value: WordType) -> Self {
        self.cpu.reg_file.write(idx, value);
        self
    }

    // Explicit float type in name to avoid deducing that may cause accidental error.
    pub(super) fn reg_f32(mut self, idx: u8, value: f32) -> Self {
        self.cpu.fpu.store(idx, value);
        self
    }

    pub(super) fn reg_f64(mut self, idx: u8, value: f64) -> Self {
        self.cpu.fpu.store(idx, value);
        self
    }

    pub(super) fn reg_vec(mut self, lmul: u8, idx: u8, value: &[u8]) -> Self {
        self.cpu.vector.write_as_type(lmul, idx, value);
        self
    }

    pub(super) fn vector_status(
        mut self,
        vlmul: Vlmul,
        vsew: Vsew,
        tail_agnostic: bool,
        mask_agnostic: bool,
    ) -> Self {
        let new_vtype = vlmul as WordType
            | (vsew as WordType) << 3
            | (tail_agnostic as WordType) << 6
            | (mask_agnostic as WordType) << 7;
        let vl = self
            .cpu
            .csr
            .get_by_type::<Vtype>()
            .unwrap()
            .vsetvl(new_vtype)
            .unwrap(); // set vtype csr
        let _ = self.cpu.csr.write_directly(Vl::get_index(), vl); // set vl csr
        self.cpu
            .vector
            .set_config((vlmul, vsew, tail_agnostic, mask_agnostic, vl as u16)); // set vector config
        self
    }

    pub(super) fn pc(mut self, value: WordType) -> Self {
        self.cpu.pc = value;
        self
    }

    pub(super) fn mem<T: UnsignedInteger>(mut self, addr: WordType, value: T) -> Self {
        self.cpu.write(addr, value).unwrap();
        self
    }

    pub(super) fn mem_range<It: Iterator, T: UnsignedInteger>(
        mut self,
        indexs: It,
        f: fn(usize) -> (WordType, T),
    ) -> Self
    where
        It: Iterator<Item = usize>,
    {
        for i in indexs {
            let (addr, data) = f(i);
            self.cpu.write(addr, data).unwrap();
        }
        self
    }

    pub(super) fn mem_base<T: UnsignedInteger>(mut self, addr: WordType, value: T) -> Self {
        self.cpu.write(BASE_ADDR + addr, value).unwrap();
        self
    }

    pub(super) fn program(mut self, instrs: &[u32]) -> Self {
        let mut addr = BASE_ADDR;
        for instr in instrs {
            self.cpu.write(addr, *instr).unwrap();
            addr += 4;
        }
        self
    }

    pub(super) fn csr(mut self, csr_addr: WordType, value: WordType) -> Self {
        self.cpu.csr.write_uncheck_privilege(csr_addr, value);
        self
    }

    pub(super) fn build(self) -> Box<RVCPU> {
        self.cpu
    }
}

pub(super) struct CPUChecker<'a> {
    pub(super) cpu: &'a mut RVCPU,
}

impl<'a> CPUChecker<'a> {
    pub(super) fn new(cpu: &'a mut RVCPU) -> Self {
        Self { cpu }.reg(0, 0) // x0 is always 0
    }

    pub(super) fn reg(self, idx: u8, value: WordType) -> Self {
        assert_eq!(
            self.cpu.reg_file.read(idx, 0).0,
            value,
            "Register #{} incorrect",
            idx,
        );
        self
    }

    pub(super) fn reg_f32(self, idx: u8, value: f32) -> Self {
        let reg_val: f32 = self.cpu.fpu.load(idx);
        assert_eq!(reg_val, value, "Float Register #{} incorrect", idx);
        self
    }

    pub(super) fn reg_f64(self, idx: u8, value: f64) -> Self {
        let reg_val: f64 = self.cpu.fpu.load(idx);
        assert_eq!(reg_val, value, "Float Register #{} incorrect", idx);
        self
    }

    pub(super) fn reg_vec<T>(self, idx: u8, value: &[T]) -> Self
    where
        T: Eq + Debug,
    {
        let reg_val = self.cpu.vector.read_as_type::<T>(idx).unwrap();
        assert_eq!(value.len(), reg_val.len());
        for i in 0..reg_val.len() {
            assert_eq!(
                reg_val[i], value[i],
                "Vector Register #{idx} [{i}] incorrect"
            );
        }
        self
    }

    pub(super) fn pc(self, value: WordType) -> Self {
        assert_eq!(self.cpu.pc, value, "PC incorrect");
        self
    }

    pub(super) fn mem<T>(self, addr: WordType, value: WordType) -> Self
    where
        T: UnsignedInteger,
    {
        assert_eq!(
            self.cpu.read::<T>(addr).unwrap().into(),
            value,
            "Memory value incorrect at pos {}",
            addr
        );
        self
    }

    pub(super) fn mem_base<T>(self, addr: WordType, value: WordType) -> Self
    where
        T: UnsignedInteger,
    {
        self.mem::<T>(BASE_ADDR + addr, value)
    }

    pub(super) fn csr(self, addr: WordType, value: WordType) -> Self {
        assert_eq!(
            self.cpu.csr.read_uncheck_privilege(addr).unwrap(),
            value,
            "Csr value incorrect at addr 0x{:0x}",
            addr
        );
        self
    }

    pub(super) fn customized<F>(self, f: F) -> Self
    where
        F: FnOnce(Self) -> Self,
    {
        f(self)
    }
}

fn assert_register_state_eq(label: &str, expected: &RVCPU, actual: &RVCPU) {
    for register in 0..REGFILE_CNT {
        assert_eq!(
            actual.reg_file[register], expected.reg_file[register],
            "{label}: x{register} differs"
        );
    }
}

fn assert_execution_state_eq(label: &str, expected: &mut RVCPU, actual: &mut RVCPU) {
    assert_eq!(actual.pc, expected.pc, "{label}: PC differs");
    assert_register_state_eq(label, expected, actual);
}

#[cfg(all(target_arch = "x86_64", feature = "riscv64"))]
fn run_codegen_if_supported<F>(instr: RiscvInstr, info: RVInstrInfo, build: &F, interpreter: &RVCPU)
where
    F: Fn(TestCPUBuilder) -> TestCPUBuilder,
{
    let mut jit_cpu = build(TestCPUBuilder::new()).build();
    if !crate::jit::engine::try_execute_instruction_for_test(&mut jit_cpu, instr, info) {
        return;
    }

    assert_register_state_eq("jit/interpreter", interpreter, &jit_cpu);
}

pub(super) fn run_test_exec<F, G>(instr: RiscvInstr, info: RVInstrInfo, build: F, check: G)
where
    F: Fn(TestCPUBuilder) -> TestCPUBuilder,
    G: Fn(CPUChecker) -> CPUChecker,
{
    let mut cpu = build(TestCPUBuilder::new()).build();
    cpu.execute(instr, info).unwrap();
    check(CPUChecker::new(&mut cpu));

    #[cfg(all(target_arch = "x86_64", feature = "riscv64"))]
    run_codegen_if_supported(instr, info, &build, &cpu);
}

pub(super) fn run_test_exec_decode<F, G>(raw_instr: u32, build: F, check: G)
where
    F: Fn(TestCPUBuilder) -> TestCPUBuilder,
    G: Fn(CPUChecker) -> CPUChecker,
{
    let mut cpu = build(TestCPUBuilder::new()).build();
    let decoded = cpu.decoder.decode(raw_instr.into()).unwrap();
    let DecodeInstr { instr, info, .. } = decoded;
    // FIXME: [`RVCPU::execute`] don't handle the raised exception
    cpu.execute(instr, info).unwrap();
    check(CPUChecker::new(&mut cpu));

    #[cfg(all(target_arch = "x86_64", feature = "riscv64"))]
    run_codegen_if_supported(instr, info, &build, &cpu);
}

pub(super) fn run_test_cpu_step<F, G>(raw_instrs: &[u32], build: F, check: G)
where
    F: Fn(TestCPUBuilder) -> TestCPUBuilder,
    G: Fn(CPUChecker) -> CPUChecker,
{
    let make_cpu = || {
        let mut builder = build(TestCPUBuilder::new());
        for (i, inst) in raw_instrs.iter().enumerate() {
            builder = builder.mem(
                (size_of::<u32>() * i) as WordType + ram_config::BASE_ADDR,
                *inst,
            );
        }
        builder.build()
    };

    let mut cpu = make_cpu();
    for _ in 0..raw_instrs.len() {
        cpu.step();
    }
    check(CPUChecker::new(&mut cpu));

    #[cfg(all(target_arch = "x86_64", feature = "riscv64"))]
    {
        use crate::isa::riscv::executor::ExecutorBackend;

        let mut jit_cpu = make_cpu();
        let mut jit = crate::jit::engine::RvJitEngine::new();
        jit.step_batch(&mut jit_cpu, raw_instrs.len() as u64);
        check(CPUChecker::new(&mut jit_cpu));
        assert_execution_state_eq("JIT executor/interpreter", &mut cpu, &mut jit_cpu);
    }
}

pub(super) struct ExecTester {
    rng: ChaCha12Rng,
}

impl ExecTester {
    pub(super) fn new() -> Self {
        Self {
            rng: ChaCha12Rng::seed_from_u64(0721),
        }
    }

    pub(super) fn rand_imm12(&mut self) -> WordType {
        self.rng.random_range(0..=4095) as WordType
    }

    pub(super) fn rand_word(&mut self) -> WordType {
        self.rng.random_range(0..=WordType::MAX)
    }

    pub(super) fn rand_word2(&mut self) -> (WordType, WordType) {
        (self.rand_word(), self.rand_word())
    }

    pub(super) fn rand_reg_idx(&mut self) -> u8 {
        self.rng.random_range(1..REGFILE_CNT) as u8
    }

    pub(super) fn rand_reg_idx2(&mut self) -> (u8, u8) {
        (self.rand_reg_idx(), self.rand_reg_idx())
    }

    pub(super) fn rand_unique_reg_idx2(&mut self) -> (u8, u8) {
        let idx1 = self.rand_reg_idx();
        let mut idx2 = self.rand_reg_idx();
        while idx1 == idx2 {
            idx2 = self.rand_reg_idx();
        }
        (idx1, idx2)
    }

    pub(super) fn test_rand_r_with(
        &mut self,
        instr: RiscvInstr,
        lhs: WordType,
        rhs: WordType,
        expected: WordType,
    ) {
        let rd = self.rand_reg_idx();
        let (rs1, rs2) = self.rand_unique_reg_idx2();
        let info = RVInstrInfo::R { rd, rs1, rs2 };

        run_test_exec(
            instr,
            info,
            |builder| builder.reg(rs1, lhs).reg(rs2, rhs).pc(0x1000),
            |checker| checker.reg(rd, expected).pc(0x1004),
        );
    }

    pub(super) fn test_rand_r<F>(&mut self, instr: RiscvInstr, calc: F)
    where
        F: FnOnce(WordType, WordType) -> WordType,
    {
        let (val1, val2) = self.rand_word2();
        self.test_rand_r_with(instr, val1, val2, calc(val1, val2));
    }

    pub(super) fn test_rand_i_with(
        &mut self,
        instr: RiscvInstr,
        lhs: WordType,
        imm: WordType,
        expected: WordType,
    ) {
        let (rd, rs1) = self.rand_reg_idx2();
        let info = RVInstrInfo::I { rd, rs1, imm };

        run_test_exec(
            instr,
            info,
            |builder| builder.reg(rs1, lhs).pc(0x1000),
            |checker| checker.reg(rd, expected).pc(0x1004),
        );
    }

    pub(super) fn test_rand_i<F>(&mut self, instr: RiscvInstr, calc: F)
    where
        F: FnOnce(WordType, WordType) -> WordType,
    {
        let val = self.rand_word();
        let imm = self.rand_imm12();
        self.test_rand_i_with(
            instr,
            val,
            sign_extend(imm, 12),
            calc(val, sign_extend(imm, 12)),
        );
    }
}
