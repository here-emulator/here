use std::hint::cold_path;

use libc::{c_int, c_void};

unsafe extern "C" {
    /// XXX: Make sure you know what you're doing.
    ///
    /// rustc doesn't support wired function that "return twice",
    /// so calling this is actually an UB, and llvm cannot recognize this hidden control flow,
    /// handle it carefully:
    ///
    /// - Use `#[inline(never)]` for every function that calls `setjmp`.
    /// - Do not use local mutable variable before `setjmp`.
    /// - Check the local variable lifetime so that stack-slot reuse won't break anything.
    fn setjmp(env: *mut c_void) -> c_int;
}

use crate::{isa::riscv::instruction::instr_table::RiscvInstr, jit::old_backend::CodeBuf};

#[cfg(test)]
use crate::isa::riscv::instruction::RVInstrInfo;

use crate::{
    config::arch_config::WordType,
    isa::{
        cache::*,
        riscv::executor::{ExecutorBackend, RVCPU},
    },
};

use super::{
    jit_buffer::*,
    jit_function::{JitContext, JitFunction, JitInfo},
    old_translator::{TranslateResult, X86CodeGen},
    stats::Stats,
};

const MAX_BASIC_BLOCK_INSTRS: u64 = 1024;

#[derive(Clone, Copy)]
pub(super) enum TranslationStop {
    DecodeFailure,
    UnsupportedInstruction(RiscvInstr),
    InstructionLimit,
    ControlFlow,
}

impl Cacheable for BBCacheEntry {
    const ADDR_SHIFT_BITS: usize = 1;
}

pub struct RawBasicBlock {
    instr_cnt: u64,
    buf: CodeBuf,
}

#[derive(Clone, Copy)]
struct BasicBlock {
    instr_cnt: u64,
    func: JitFunction,
}

#[derive(Clone, Copy)]
enum BBCacheEntry {
    Compiled(BasicBlock),
    InterpreterOnly,
}

struct BasicBlockManager {
    // TODO: BB cache eviction drops JIT function pointers,
    // leaving old translated blocks unused and causing high JIT memory usage.
    // `SetCache` is used to improve hit rates and ease this issue.
    // - use a quick hashmap (maybe `rustc_hash::FxHashMap`) for secondary lookup
    // - consider if we should reset jit buffer if its too large
    cache: Cache<SetCache<BBCacheEntry, 4096, 8>>,
    jit_buffer: JitBuffer,
}

impl BasicBlockManager {
    fn new() -> Self {
        Self {
            cache: Cache::new(),
            jit_buffer: JitBuffer::new(),
        }
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.jit_buffer.reset();
    }

    fn get(&self, pc: WordType) -> Option<BBCacheEntry> {
        self.cache.get(pc)
    }

    fn put(&mut self, pc: WordType, entry: BBCacheEntry) {
        self.cache.put(pc, entry);
    }

    unsafe fn emit_code(&mut self, code: &[u8]) -> JitFunction {
        unsafe { self.jit_buffer.emit_code(code) }
    }

    fn hit_count(&self) -> u64 {
        self.cache.hit_count()
    }

    fn access_count(&self) -> u64 {
        self.cache.access_count()
    }

    fn translate(
        &mut self,
        cpu: &mut RVCPU,
        context: &mut JitContext,
        stats: &mut Stats,
        mut addr: WordType,
    ) -> Option<RawBasicBlock> {
        let context = context as *mut JitContext;
        // `misa.C` determines IALIGN and is baked into each translated block. We assume misa
        // remains unchanged while JIT code is cached; any future writable-misa support must
        // invalidate the JIT buffer when misa changes.
        let ialign = cpu.instruction_alignment();
        let mut code_gen = X86CodeGen::new(context);
        let mut instr_cnt = 0;
        let stop = loop {
            if instr_cnt == MAX_BASIC_BLOCK_INSTRS {
                break TranslationStop::InstructionLimit;
            }

            let Some(decoded) = cpu.decode_at(addr) else {
                break TranslationStop::DecodeFailure;
            };

            let info = JitInfo {
                instr_pc: addr,
                instr_cnt,
                instr_len: decoded.len,
                ialign,
            };
            match code_gen.translate(decoded, info) {
                TranslateResult::Continue => {
                    instr_cnt += 1;
                    addr = addr.wrapping_add(decoded.len);
                }
                TranslateResult::ControlFlow => {
                    instr_cnt += 1;
                    addr = addr.wrapping_add(decoded.len);
                    break TranslationStop::ControlFlow;
                }
                TranslateResult::Unsupported => {
                    break TranslationStop::UnsupportedInstruction(decoded.instr);
                }
            }
        };

        if instr_cnt == 0 {
            return None;
        }

        stats.record_translation(instr_cnt, stop);
        Some(RawBasicBlock {
            instr_cnt,
            buf: code_gen.build((!matches!(stop, TranslationStop::ControlFlow)).then_some(addr)),
        })
    }

    fn get_or_translate(
        &mut self,
        cpu: &mut RVCPU,
        context: &mut JitContext,
        stats: &mut Stats,
    ) -> Option<BasicBlock> {
        let pc = cpu.pc;
        let cached = self.get(pc);
        match cached {
            Some(BBCacheEntry::Compiled(bb)) => {
                stats.record_compiled_cache_hit();
                return Some(bb);
            }
            Some(BBCacheEntry::InterpreterOnly) => return None,
            None => {}
        }

        let Some(RawBasicBlock { instr_cnt, buf }) = self.translate(cpu, context, stats, pc) else {
            self.put(pc, BBCacheEntry::InterpreterOnly);
            return None;
        };

        unsafe {
            let func = self.emit_code(buf.as_slice());
            let bb = BasicBlock { instr_cnt, func };
            self.put(pc, BBCacheEntry::Compiled(bb));
            Some(bb)
        }
    }
}

pub(crate) struct RvJitEngine {
    basic_blocks: BasicBlockManager,
    icache_epoch: u64,
    stats: Stats,
    context: Box<JitContext>,
}

impl Drop for RvJitEngine {
    fn drop(&mut self) {
        self.stats.log(
            self.basic_blocks.hit_count(),
            self.basic_blocks.access_count(),
        );
    }
}

impl RvJitEngine {
    pub fn new() -> Self {
        Self {
            basic_blocks: BasicBlockManager::new(),
            icache_epoch: 0,
            stats: Stats::default(),
            context: Box::new(JitContext::default()),
        }
    }

    fn sync_icache_epoch(&mut self, cpu: &RVCPU) {
        let epoch = cpu.icache_epoch();
        if epoch == self.icache_epoch {
            return;
        }

        cold_path();
        self.basic_blocks.clear();
        self.icache_epoch = epoch;
    }

    fn consume_cycles(&mut self, cycles: u64) {
        self.context.remaining_cycles -= cycles;
    }

    fn run_interpreter(&mut self, cpu: &mut RVCPU, cycles: u64) {
        cpu.step_batch_no_interrupt(cycles);
        self.consume_cycles(cycles);
    }

    fn run_jit_block(&mut self, cpu: &mut RVCPU, bb: BasicBlock) {
        debug_assert!(bb.instr_cnt <= self.context.remaining_cycles);

        let next_pc = unsafe { bb.func.call(cpu as *mut RVCPU) };
        cpu.finish_jit_block(next_pc, bb.instr_cnt, None);
        self.consume_cycles(bb.instr_cnt);
        self.stats.record_execution(bb.instr_cnt);
    }

    fn handle_exception_exit(&mut self, cpu: &mut RVCPU) {
        let (exception, tval) = self
            .context
            .exception
            .take()
            .expect("a JIT longjmp must record an exception");
        let instr_count = self.context.icount;
        let consumed_cycles = instr_count + 1;

        self.consume_cycles(consumed_cycles);
        self.stats.record_exception_exit(instr_count, exception);
        self.stats.record_execution(instr_count);
        cpu.finish_jit_block(self.context.guest_pc, instr_count, Some((exception, tval)));
    }

    #[cfg(test)]
    fn compile_instruction_for_test(
        &mut self,
        cpu: &mut RVCPU,
        instr: RiscvInstr,
        info: RVInstrInfo,
    ) -> Option<BasicBlock> {
        let instr_len = instr.len();
        let instr_pc = cpu.pc;
        let jit_info = JitInfo {
            instr_pc,
            instr_cnt: 0,
            instr_len,
            ialign: cpu.instruction_alignment(),
        };

        let mut codegen = X86CodeGen::new(self.context.as_mut());
        let next_pc = match codegen.translate_instruction(instr, info, jit_info) {
            TranslateResult::Continue => Some(instr_pc.wrapping_add(instr_len)),
            TranslateResult::ControlFlow => None,
            TranslateResult::Unsupported => return None,
        };

        let code = codegen.build(next_pc);
        let func = unsafe { self.basic_blocks.emit_code(&code) };
        Some(BasicBlock { instr_cnt: 1, func })
    }
}

impl ExecutorBackend for RvJitEngine {
    #[inline(never)] // See `setjmp`
    fn step_batch(&mut self, cpu: &mut RVCPU, cycles: u64) {
        if cycles == 0 {
            return;
        }

        debug_assert_eq!(self.context.exception, None);
        self.context.remaining_cycles = cycles;

        // Don't use `cycles` directly after this because longjmp may break the value.
        #[allow(unused)]
        let cycles = "DO NOT USE";

        if cpu.try_take_pending_interrupt() {
            self.consume_cycles(1);
        }

        if unsafe { setjmp(self.context.jmp_buf_ptr()) != 0 } {
            self.handle_exception_exit(cpu);
            // fall through, run remaining cycles.
        }

        while self.context.remaining_cycles > 0 {
            self.sync_icache_epoch(cpu);

            let Some(bb) =
                self.basic_blocks
                    .get_or_translate(cpu, &mut self.context, &mut self.stats)
            else {
                self.run_interpreter(cpu, 1);
                continue;
            };

            let remaining_cycles = self.context.remaining_cycles;
            if bb.instr_cnt > remaining_cycles {
                self.run_interpreter(cpu, remaining_cycles);
                break;
            }

            self.run_jit_block(cpu, bb);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::UnsafeCell, rc::Rc};

    use super::*;
    use crate::{
        device::mmio::MemoryMapIO,
        isa::{
            DebugTarget,
            riscv::{
                RiscvTypes,
                csr_reg::{
                    NamedCsrReg,
                    csr_macro::{Mcause, Mcycle, Mepc, Minstret, Mtval},
                },
                debugger::Address,
                decoder::Decoder,
                instruction::{RVInstrInfo, instr_table::RiscvInstr},
                mmu::VirtAddrManager,
            },
        },
        ram::Ram,
        ram_config::BASE_ADDR,
    };

    fn cpu_with_instructions(raw_instrs: &[u32]) -> RVCPU {
        cpu_with_isa_instructions("RV64GC", raw_instrs)
    }

    fn cpu_with_isa_instructions(isa: &str, raw_instrs: &[u32]) -> RVCPU {
        let ram = Rc::new(UnsafeCell::new(Ram::new()));
        let mmio = MemoryMapIO::from_ram(ram.clone());
        let memory = VirtAddrManager::from_ram_and_mmio(ram, mmio);
        let decoder = Decoder::from_isa_str(isa).expect("test ISA must parse");
        let mut cpu = RVCPU::from_decoder(decoder, memory);
        let pc = cpu.pc;
        for (index, raw_instr) in raw_instrs.iter().enumerate() {
            <RVCPU as DebugTarget<RiscvTypes>>::write_memory(
                &mut cpu,
                Address::Virt(pc + (index * size_of::<u32>()) as WordType),
                *raw_instr,
            )
            .unwrap();
        }
        cpu
    }

    #[test]
    fn caches_interpreter_only_pcs_until_icache_invalidation() {
        let mut cpu = cpu_with_instructions(&[0x0220_80b3]); // mul x1, x1, x2
        let mut engine = RvJitEngine::new();

        assert!(
            engine
                .basic_blocks
                .get_or_translate(&mut cpu, &mut engine.context, &mut engine.stats)
                .is_none()
        );
        assert_eq!(engine.basic_blocks.access_count(), 1);
        assert_eq!(engine.basic_blocks.hit_count(), 0);

        assert!(
            engine
                .basic_blocks
                .get_or_translate(&mut cpu, &mut engine.context, &mut engine.stats)
                .is_none()
        );
        assert_eq!(engine.basic_blocks.access_count(), 2);
        assert_eq!(engine.basic_blocks.hit_count(), 1);

        cpu.flush_icache();
        engine.sync_icache_epoch(&cpu);
        assert!(
            engine
                .basic_blocks
                .get_or_translate(&mut cpu, &mut engine.context, &mut engine.stats)
                .is_none()
        );
        assert_eq!(engine.basic_blocks.access_count(), 3);
        assert_eq!(engine.basic_blocks.hit_count(), 1);
    }

    #[test]
    fn engine_commits_successful_memory_block_counters() {
        // addi x5, x0, 7; sd x5, 0(x1); ld x6, 0(x1); mul (block terminator)
        let mut cpu = cpu_with_instructions(&[0x0070_0293, 0x0050_b023, 0x0000_b303, 0x0220_80b3]);
        let target = BASE_ADDR + 0x380;
        cpu.reg_file.write(1, target);
        let mut engine = RvJitEngine::new();

        engine.step_batch(&mut cpu, 3);

        assert_eq!(cpu.pc, BASE_ADDR + 12);
        assert_eq!(cpu.reg_file[5], 7);
        assert_eq!(cpu.reg_file[6], 7);
        assert_eq!(cpu.read::<u64>(target).unwrap(), 7);
        assert_eq!(cpu.read_csr(Mcycle::get_index()).unwrap(), 3);
        assert_eq!(cpu.read_csr(Minstret::get_index()).unwrap(), 3);
    }

    #[test]
    fn engine_debits_the_faulting_instruction_cycle() {
        // addi x5, x0, 7; ld x6, 1(x1); mul (block terminator)
        let mut cpu = cpu_with_instructions(&[0x0070_0293, 0x0010_b303, 0x0220_80b3]);
        cpu.reg_file.write(1, BASE_ADDR);
        let fault_pc = BASE_ADDR + 4;
        let mut engine = RvJitEngine::new();

        engine.step_batch(&mut cpu, 2);

        assert_eq!(cpu.reg_file[5], 7);
        assert_eq!(cpu.read_csr(Mepc::get_index()).unwrap(), fault_pc);
        assert_eq!(cpu.read_csr(Mcause::get_index()).unwrap(), 4);
        assert_eq!(cpu.read_csr(Mtval::get_index()).unwrap(), BASE_ADDR + 1);
        assert_eq!(cpu.read_csr(Mcycle::get_index()).unwrap(), 2);
        assert_eq!(cpu.read_csr(Minstret::get_index()).unwrap(), 1);
    }

    #[test]
    fn jit_control_flow_ends_a_block_and_uses_the_dynamic_pc() {
        // addi x1, x0, 1; bne x1, x0, 8; addi x2, x0, 2; addi x3, x0, 3
        let mut cpu = cpu_with_instructions(&[0x0010_0093, 0x0000_9463, 0x0020_0113, 0x0030_0193]);
        let start = cpu.pc;
        let mut engine = RvJitEngine::new();

        engine.step_batch(&mut cpu, 3);

        assert_eq!(cpu.pc, start + 16);
        assert_eq!(cpu.reg_file[1], 1);
        assert_eq!(cpu.reg_file[2], 0);
        assert_eq!(cpu.reg_file[3], 3);
        assert_eq!(cpu.read_csr(Minstret::get_index()).unwrap(), 3);
    }

    #[test]
    fn jit_jalr_preserves_the_target_when_rd_equals_rs1() {
        // jalr x1, 0(x1)
        let mut cpu = cpu_with_instructions(&[0x0000_80e7]);
        let start = cpu.pc;
        cpu.reg_file.write(1, start + 0x11);
        let mut engine = RvJitEngine::new();

        engine.step_batch(&mut cpu, 1);

        assert_eq!(cpu.pc, start + 0x10);
        assert_eq!(cpu.reg_file[1], start + 4);
    }

    #[test]
    fn jit_jal_returns_its_target_and_writes_the_link() {
        // jal x5, 8
        let mut cpu = cpu_with_instructions(&[0x0080_02ef]);
        let start = cpu.pc;
        let mut engine = RvJitEngine::new();

        engine.step_batch(&mut cpu, 1);

        assert_eq!(cpu.pc, start + 8);
        assert_eq!(cpu.reg_file[5], start + 4);
        assert_eq!(cpu.read_csr(Mcycle::get_index()).unwrap(), 1);
        assert_eq!(cpu.read_csr(Minstret::get_index()).unwrap(), 1);
    }

    #[test]
    fn jit_compressed_control_flow_uses_two_byte_fallthrough_and_link() {
        use RiscvInstr::*;

        let mut branch = cpu_with_instructions(&[]);
        let start = branch.pc;
        branch.reg_file.write(8, 1);
        assert!(try_execute_instruction_for_test(
            &mut branch,
            C_BEQZ,
            RVInstrInfo::CB { rd_rs1: 8, imm: 8 },
        ));
        assert_eq!(branch.pc, start + 2);

        branch.pc = start;
        branch.reg_file.write(8, 0);
        assert!(try_execute_instruction_for_test(
            &mut branch,
            C_BEQZ,
            RVInstrInfo::CB { rd_rs1: 8, imm: 8 },
        ));
        assert_eq!(branch.pc, start + 8);

        let mut jump = cpu_with_instructions(&[]);
        assert!(try_execute_instruction_for_test(
            &mut jump,
            C_J,
            RVInstrInfo::CJ { target: 10 },
        ));
        assert_eq!(jump.pc, start + 10);

        let mut jump_reg = cpu_with_instructions(&[]);
        jump_reg.reg_file.write(8, start + 0x21);
        assert!(try_execute_instruction_for_test(
            &mut jump_reg,
            C_JALR,
            RVInstrInfo::CR { rd_rs1: 8, rs2: 0 },
        ));
        assert_eq!(jump_reg.pc, start + 0x20);
        assert_eq!(jump_reg.reg_file[1], start + 2);
    }

    #[test]
    fn test_instruction_runner_finishes_the_jit_block() {
        let mut cpu = cpu_with_instructions(&[]);
        let start = cpu.pc;

        assert!(try_execute_instruction_for_test(
            &mut cpu,
            RiscvInstr::ADDI,
            RVInstrInfo::I {
                rd: 1,
                rs1: 0,
                imm: 7,
            },
        ));

        assert_eq!(cpu.pc, start + 4);
        assert_eq!(cpu.reg_file[1], 7);
        assert_eq!(cpu.read_csr(Mcycle::get_index()).unwrap(), 1);
        assert_eq!(cpu.read_csr(Minstret::get_index()).unwrap(), 1);
    }

    #[test]
    fn jit_checks_alignment_only_for_taken_branches_without_c() {
        // beq x1, x0, 2: target is misaligned for IALIGN=32.
        let raw = [0x0000_8163];
        let mut not_taken = cpu_with_isa_instructions("RV64G", &raw);
        let start = not_taken.pc;
        not_taken.reg_file.write(1, 1);
        RvJitEngine::new().step_batch(&mut not_taken, 1);
        assert_eq!(not_taken.pc, start + 4);
        assert_eq!(not_taken.read_csr(Minstret::get_index()).unwrap(), 1);
        assert_eq!(not_taken.read_csr(Mcycle::get_index()).unwrap(), 1);
        assert_eq!(not_taken.read_csr(Mcause::get_index()).unwrap(), 0);

        let mut taken = cpu_with_isa_instructions("RV64G", &raw);
        let start = taken.pc;
        let mut engine = RvJitEngine::new();
        engine.step_batch(&mut taken, 1);
        assert_eq!(taken.read_csr(Mepc::get_index()).unwrap(), start);
        assert_eq!(taken.read_csr(Mcause::get_index()).unwrap(), 0);
        assert_eq!(taken.read_csr(Mtval::get_index()).unwrap(), start + 2);
        assert_eq!(taken.read_csr(Minstret::get_index()).unwrap(), 0);
        assert_eq!(taken.read_csr(Mcycle::get_index()).unwrap(), 1);
    }

    #[test]
    fn jit_supports_all_integer_branch_conditions() {
        use RiscvInstr::*;

        let cases = [
            (BEQ, 5, 5, true),
            (BEQ, 5, 6, false),
            (BNE, 5, 6, true),
            (BNE, 5, 5, false),
            (BLT, WordType::MAX, 1, true),
            (BLT, 1, WordType::MAX, false),
            (BGE, 1, WordType::MAX, true),
            (BGE, WordType::MAX, 1, false),
            (BLTU, 1, WordType::MAX, true),
            (BLTU, WordType::MAX, 1, false),
            (BGEU, WordType::MAX, 1, true),
            (BGEU, 1, WordType::MAX, false),
        ];

        for (instr, lhs, rhs, taken) in cases {
            let mut cpu = cpu_with_instructions(&[]);
            let start = cpu.pc;
            cpu.reg_file.write(1, lhs);
            cpu.reg_file.write(2, rhs);
            assert!(try_execute_instruction_for_test(
                &mut cpu,
                instr,
                RVInstrInfo::B {
                    rs1: 1,
                    rs2: 2,
                    imm: 8,
                },
            ));
            assert_eq!(cpu.pc, start + if taken { 8 } else { 4 }, "{instr:?}");
        }
    }

    #[test]
    fn jit_checks_static_jump_alignment_without_c() {
        // jal x1, 2: an aligned JAL instruction with a misaligned target.
        let mut cpu = cpu_with_isa_instructions("RV64G", &[0x0020_00ef]);
        let start = cpu.pc;
        RvJitEngine::new().step_batch(&mut cpu, 1);

        assert_eq!(cpu.read_csr(Mepc::get_index()).unwrap(), start);
        assert_eq!(cpu.read_csr(Mcause::get_index()).unwrap(), 0);
        assert_eq!(cpu.read_csr(Mtval::get_index()).unwrap(), start + 2);
        assert_eq!(cpu.reg_file[1], 0);
        assert_eq!(cpu.read_csr(Mcycle::get_index()).unwrap(), 1);
        assert_eq!(cpu.read_csr(Minstret::get_index()).unwrap(), 0);
    }

    #[test]
    fn jit_checks_dynamic_jalr_alignment_without_c() {
        // jalr x1, 0(x2); JALR clears bit 0, leaving an IALIGN=32 violation.
        let mut cpu = cpu_with_isa_instructions("RV64G", &[0x0001_00e7]);
        let start = cpu.pc;
        cpu.reg_file.write(2, start + 3);
        RvJitEngine::new().step_batch(&mut cpu, 1);

        assert_eq!(cpu.read_csr(Mepc::get_index()).unwrap(), start);
        assert_eq!(cpu.read_csr(Mcause::get_index()).unwrap(), 0);
        assert_eq!(cpu.read_csr(Mtval::get_index()).unwrap(), start + 2);
        assert_eq!(cpu.reg_file[1], 0);
        assert_eq!(cpu.read_csr(Mcycle::get_index()).unwrap(), 1);
        assert_eq!(cpu.read_csr(Minstret::get_index()).unwrap(), 0);
    }

    #[test]
    fn jit_aligned_jalr_without_c_returns_from_the_alignment_helper() {
        // jalr x1, 0(x2)
        let mut cpu = cpu_with_isa_instructions("RV64G", &[0x0001_00e7]);
        let start = cpu.pc;
        cpu.reg_file.write(2, start + 8);
        RvJitEngine::new().step_batch(&mut cpu, 1);

        assert_eq!(cpu.pc, start + 8);
        assert_eq!(cpu.reg_file[1], start + 4);
        assert_eq!(cpu.read_csr(Mcycle::get_index()).unwrap(), 1);
        assert_eq!(cpu.read_csr(Minstret::get_index()).unwrap(), 1);
    }
}

#[cfg(test)]
#[inline(never)]
pub(crate) fn try_execute_instruction_for_test(
    cpu: &mut RVCPU,
    instr: RiscvInstr,
    info: RVInstrInfo,
) -> bool {
    let mut engine = RvJitEngine::new();
    let Some(bb) = engine.compile_instruction_for_test(cpu, instr, info) else {
        return false;
    };

    engine.context.remaining_cycles = bb.instr_cnt;
    if unsafe { setjmp(engine.context.jmp_buf_ptr()) } == 0 {
        engine.run_jit_block(cpu, bb);
    } else {
        engine.handle_exception_exit(cpu);
    }
    true
}
